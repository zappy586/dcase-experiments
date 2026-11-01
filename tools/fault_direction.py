"""Do the synthetic faults point the same way as the real ones? (diagnostic; uses test labels for analysis only)
Frozen BEATs features (768, token average, full 10 s), standardised by the real training normals; maps from
tools/transport_v1.py fitted on real training normals + 200 synthetic normals.
  synthetic direction : mean(mapped synthetic faulty) - mean(mapped synthetic normal), all 250 each, and per fault prompt
  real direction      : mean(real test anomaly) - mean(real test normal), pooled and per domain (source / target)
Reported: cosine between them; chance level for 768 numbers is about +-0.036 (1/sqrt(768)); split-half reliability
of the real direction (cosine between the directions of two random halves of the test set, mean of 20 splits) shows how
high a cosine can get; AUC on the real test set when clips are scored by projection on the synthetic direction.
Usage (main env): uv run python tools/fault_direction.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import audiogen_ti as A  # noqa: E402
from transport_v1 import CLEAN, fit_linear, fit_ot, rt_features  # noqa: E402

SYN = ROOT / "outputs" / "audiogen" / "synth_v1"
cos = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def main():
    meta = json.loads((SYN / "meta.json").read_text())
    prompts = np.array([meta[f"faulty/{i:03d}.wav"]["prompt"] for i in range(250)])
    names = list(dict.fromkeys(prompts))
    rng = np.random.default_rng(0)
    pn = rng.permutation(250)
    for name, Fe in (("clean", dict(np.load(CLEAN))), ("rt", rt_features())):
        mu, sd = Fe["real_train"].mean(0), Fe["real_train"].std(0) + 1e-6
        z = lambda X: (X - mu) / sd
        R, Rt, Gn, Gf = z(Fe["real_train"]), z(Fe["real_test"]), z(Fe["gen_normal"]), z(Fe["gen_faulty"])
        lab, dom = Fe["test_label"], Fe["test_domain"]
        real = {"pooled": Rt[lab == 1].mean(0) - Rt[lab == 0].mean(0)}
        for d, dn in ((0, "source"), (1, "target")):
            real[dn] = Rt[(lab == 1) & (dom == d)].mean(0) - Rt[(lab == 0) & (dom == d)].mean(0)
        rel = []
        for _ in range(20):
            p = rng.permutation(len(Rt)); h1, h2 = p[:100], p[100:]
            dirs = [Rt[h][lab[h] == 1].mean(0) - Rt[h][lab[h] == 0].mean(0) for h in (h1, h2)]
            rel.append(cos(*dirs))
        A.log(f"fault_direction [{name}]: real direction split-half reliability {np.mean(rel):.3f}; "
              f"source vs target real directions {cos(real['source'], real['target']):.3f}")
        Sn = Gn[pn[50:]]
        for mname, T in (("none", lambda X: X), ("linear", fit_linear(Sn, R)), ("ot", fit_ot(Sn, R))):
            tn, tf = T(Gn), T(Gf)
            syn = {"all prompts": tf.mean(0) - tn.mean(0)}
            for p in names:
                syn[p] = tf[prompts == p].mean(0) - tn.mean(0)
            for sname, v in syn.items():
                c = " ".join(f"{k} {cos(v, r):+.3f}" for k, r in real.items())
                auc = roc_auc_score(lab, Rt @ v)
                A.log(f"fault_direction [{name} / {mname}] {sname[:45]:45s}: cos with real {c} | projection AUC {auc:.3f}")


if __name__ == "__main__":
    main()
