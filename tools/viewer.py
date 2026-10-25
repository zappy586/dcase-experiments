"""Sample viewer: log audio samples from any experiment and browse them in outputs/viewer/index.html.

    from tools.viewer import SampleLog
    log = SampleLog("dreambooth-ToyCar", description="...")        # one 'run'
    log.reference("real target clip 990", wav16k, note="...")       # fixed clips to compare against
    log.sample(step=200, key="clip 137 | sigma 2.0 | with phrase", wav=wav16k, note="prompt ...")

Files: outputs/viewer/runs.json (index of runs), outputs/viewer/runs/<run>/manifest.json, audio/*.wav, img/*.png.
Each write is atomic, so the page never reads a half-written file. Serve with the 'viewer' entry in
.claude/launch.json (python -m http.server on port 8768, directory outputs/viewer).
"""
import json
import os
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent / "outputs" / "viewer"


def _atomic_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def _slug(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-")[:80]


class SampleLog:
    def __init__(self, run, description="", sr=16000, root=ROOT, axis="step"):
        self.run, self.sr, self.root = _slug(run), sr, Path(root)
        self.dir = self.root / "runs" / self.run
        (self.dir / "audio").mkdir(parents=True, exist_ok=True)
        (self.dir / "img").mkdir(parents=True, exist_ok=True)
        self.mpath = self.dir / "manifest.json"
        self.m = json.loads(self.mpath.read_text()) if self.mpath.exists() else {
            "run": self.run, "description": description, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "references": [], "samples": []}
        self.m["axis"] = axis
        if description:
            self.m["description"] = description
        self._mel = torchaudio.transforms.MelSpectrogram(sr, n_fft=1024, hop_length=sr // 100, n_mels=128)
        self._save()

    def _files(self, name, wav, vmax=None):
        wav = np.asarray(wav.detach().cpu() if torch.is_tensor(wav) else wav, dtype=np.float32).reshape(-1)
        stem = _slug(name)
        sf.write(self.dir / "audio" / f"{stem}.wav", wav, self.sr)
        S = 10 * torch.log10(self._mel(torch.from_numpy(wav)) + 1e-10).numpy()
        vmax = float(S.max()) if vmax is None else vmax
        fig, ax = plt.subplots(figsize=(4.2, 1.8), constrained_layout=True)
        ax.imshow(S, origin="lower", aspect="auto", cmap="magma", vmin=vmax - 70, vmax=vmax,
                  extent=[0, len(wav) / self.sr, 0, 128])
        ax.set_xlabel("time (s)", fontsize=7)
        ax.set_ylabel("mel bin", fontsize=7)
        ax.tick_params(labelsize=6)
        fig.savefig(self.dir / "img" / f"{stem}.png", dpi=90)
        plt.close(fig)
        rms = float(20 * np.log10(np.sqrt(np.mean(wav ** 2)) + 1e-12))
        return f"runs/{self.run}/audio/{stem}.wav", f"runs/{self.run}/img/{stem}.png", rms

    def reference(self, key, wav, note=""):
        """a fixed clip shown at the top of the run (e.g. a real target clip, the source clip being edited)"""
        a, i, rms = self._files(f"ref-{key}", wav)
        self.m["references"] = [r for r in self.m["references"] if r["key"] != key]
        self.m["references"].append({"key": key, "audio": a, "img": i, "note": note, "rms_db": round(rms, 1)})
        self._save()

    def sample(self, step, key, wav, note=""):
        """one generated clip at a training step; 'key' names the column (same key across steps = same row)"""
        a, i, rms = self._files(f"s{step:010.4f}-{key}", wav)
        self.m["samples"] = [s for s in self.m["samples"] if not (s["step"] == step and s["key"] == key)]
        self.m["samples"].append({"step": step, "key": key, "audio": a, "img": i, "note": note, "rms_db": round(rms, 1),
                                  "time": time.strftime("%H:%M:%S")})
        self._save()

    def _save(self):
        self.m["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _atomic_json(self.mpath, self.m)
        idx_path = self.root / "runs.json"
        idx = json.loads(idx_path.read_text()) if idx_path.exists() else []
        idx = [r for r in idx if r["run"] != self.run]
        idx.append({"run": self.run, "description": self.m["description"], "created": self.m["created"],
                    "updated": self.m["updated"], "n_samples": len(self.m["samples"])})
        _atomic_json(idx_path, idx)
