"""Table of final results for runs under outputs/ae_noiseaware/<name>/<machine>/summary.json, plus the generalisation
gap (test-normal gap / training-clip gap, from the FINAL log line). Usage: uv run python tools/hyp_table.py v4_all7 v13_ref_seed1 ..."""
import json, re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
MS = ["ToyCar", "fan", "gearboxEmu"] if "--all" not in sys.argv else ["ToyCar", "ToyCarEmu", "bearingEmu", "fan", "gearboxEmu", "sliderEmu", "valveEmu"]
log = (ROOT / "outputs" / "live.log").read_text(encoding="utf-8", errors="replace").splitlines()
for name in [a for a in sys.argv[1:] if not a.startswith("--")]:
    cells, offs = [], []
    for m in MS:
        f = ROOT / "outputs" / "ae_noiseaware" / name / m / "summary.json"
        if not f.exists():
            cells.append(f"{m} -"); continue
        r = json.loads(f.read_text())["final"]["gap-mean"]
        ln = [l for l in log if f"[ae-noiseaware {name}/{m}]" in l and "FINAL" in l]
        g = re.search(r"val gap ([\d.]+) \| test gap normal ([\d.]+)", ln[-1]) if ln else None
        ratio = f" x{float(g[2]) / float(g[1]):.1f}" if g else ""
        cells.append(f"{m} src {r['auc_source']:.3f} tgt {r['auc_target']:.3f} pAUC {r['pauc']:.3f} off {r['official']:.3f}{ratio}")
        offs.append(r["official"])
    mean = f"mean {sum(offs) / len(offs):.3f}" if len(offs) == len(MS) else ""
    print(f"{name:16s} {mean} | " + " | ".join(cells))
