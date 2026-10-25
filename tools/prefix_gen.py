"""Generate from REAL openings: first 2 s of 20 real normal TRAINING clips (10 source + 10 target; near-minus-far,
machine gain) continued to 10 s by AudioGen (LoRA r16 'full10' + [toycar], cfg 3) with each of the 4 prompts
(audiogen_ti.ANOMALY_PROMPTS) -> 80 clips, cached in outputs/audiogen/prefix/. The same 20 openings for every prompt.

Checks use ONLY the generated part (2-10 s), against the same 2-10 s of the 200 real ToyCar TEST clips; real clips are
sent through AudioGen's EnCodec first, so both sides carry the codec.
  A. full-length: normal-prompt continuations vs real normals: AUC (log-mel shape, EnCodec latent), level, level spread,
     low-band tone, second-by-second profile over the whole 10 s
  B. per encoder (log-mel shape, EnCodec latent, AudioGen-LM layer 24 with this LoRA, BEATs):
     1 real vs generated AUC (5-fold) | 2 classifier trained on REAL normal vs REAL anomaly -> share of each prompt
     called anomalous, + share of anomalous clips among the 5 nearest real clips | 3 generated -> real: trained on
     normal-prompt (20) vs fault-prompt (20, balanced) continuations, AUC on real clips
3 openings x 4 prompts go on the comparison page.
Runs in envs/audiocraft:  envs/audiocraft/Scripts/python tools/prefix_gen.py
"""
import glob
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT / "third_party" / "beats"))
import audiogen_ti as A  # noqa: E402
from audiogen_realism import feat  # noqa: E402

OUT = ROOT / "outputs" / "audiogen" / "prefix"
NAMES = list(A.ANOMALY_PROMPTS)
PRE, SR = 2.0, 16000


def main():
    dev = torch.device("cuda")
    OUT.mkdir(parents=True, exist_ok=True)
    A.LORA.update(rank=16, name="full10")
    m, cond, t5, tid = A.load_model(dev)
    D = A.Data("ToyCar", dev)
    L = D.wav["test"].shape[-1]
    trdom = np.array(D.meta["train"]["domain"])
    rng = np.random.default_rng(0)
    pre_ids = list(rng.choice(np.where(trdom == 0)[0], 10, replace=False)) + list(np.where(trdom == 1)[0][:10])
    prefix = D.audio("train", [int(i) for i in pre_ids])[..., : int(PRE * SR)]
    m.set_generation_params(duration=L / SR, cfg_coef=3.0)
    gen = {}
    for k, (name, prompt) in enumerate(A.ANOMALY_PROMPTS.items()):
        files = [OUT / f"p{k}_{j:02d}.wav" for j in range(len(pre_ids))]
        if not all(f.exists() for f in files):
            for b in range(0, len(pre_ids), 10):
                torch.manual_seed(900 + b)
                with torch.no_grad():
                    w = m.generate_continuation(prefix[b : b + 10], SR, [prompt] * len(prefix[b : b + 10])).cpu().numpy()[:, 0, :L]
                for j, x in enumerate(w):
                    sf.write(files[b + j], x, SR)
            print(f"generated {len(files)} continuations for {name!r}", flush=True)
        gen[name] = [sf.read(f, dtype="float32")[0][:L] for f in files]
    gk = np.repeat(np.arange(4), len(pre_ids))
    gw = [w for name in NAMES for w in gen[name]]

    lab, dom = np.array(D.meta["test"]["label"]), np.array(D.meta["test"]["domain"])
    real = []
    with torch.no_grad():
        for i in range(len(lab)):
            real.append(m.compression_model.decode(*m.compression_model.encode(D.audio("test", [i])))[0, 0, :L].cpu().numpy())
    s0 = int(PRE * SR)
    cut = lambda ws: [w[s0:L] for w in ws]

    # ---- A. full-length ----
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    clf = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000))
    cvk = StratifiedKFold(5, shuffle=True, random_state=0)
    shape = lambda X: X - X.mean(1, keepdims=True)

    hid = {}
    hook = m.lm.transformer.layers[23].register_forward_hook(lambda mod, inp, out: hid.__setitem__(0, out))

    @torch.no_grad()
    def enc_lm(ws):
        x = torch.as_tensor(np.stack(ws), device=dev)[:, None]
        E, H = [], []
        for i in range(0, len(x), 8):
            xb = x[i : i + 8]
            E.append(m.compression_model.encoder(xb).mean(-1).cpu())
            ct = A.cond_tensors(cond, t5, ["the sound of a machine"] * len(xb))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                m.lm.compute_predictions(A.encode(m, xb), [], ct)
            H.append(hid[0].float().mean(1).cpu())
        return torch.cat(E).numpy(), torch.cat(H).numpy()

    Rc, Gc = cut(real), cut(gw)
    lr_, lg_ = np.stack([feat(w) for w in Rc]), np.stack([feat(w) for w in Gc])
    er, hr = enc_lm(Rc)
    eg, hg = enc_lm(Gc)
    hook.remove()
    n0 = gk == 0
    print("\n== A. full-length: normal-prompt continuations (generated part 2-10 s) vs real normals (2-10 s) ==")
    for name, R, G in (("log-mel shape", shape(lr_)[lab == 0], shape(lg_)[n0]), ("EnCodec latent", er[lab == 0], eg[n0])):
        X = np.r_[R, G]; y = np.r_[np.zeros(len(R)), np.ones(len(G))]
        p = cross_val_predict(clf(), X, y, cv=cvk, method="predict_proba")[:, 1]
        print(f"  real normal vs generated, {name}: AUC {roc_auc_score(y, p):.3f}   (without real openings, full10 LoRA: "
              f"{'0.962' if 'log' in name else '0.944'})")
    lvl_r, lvl_g = lr_[lab == 0].mean(1), lg_[n0].mean(1)
    print(f"  level generated - real {lvl_g.mean() - lvl_r.mean():+.1f} dB; level spread across clips: generated {lvl_g.std():.1f} dB, "
          f"real {lvl_r.std():.1f} dB (without openings: 1.1 dB); low-band tone generated - real "
          f"{(lg_[n0, :8].mean(1) - lvl_g).mean() - (lr_[lab == 0, :8].mean(1) - lvl_r).mean():+.1f} dB")
    print("  second by second:   real normals level / low-band   |   normal-prompt continuation level / low-band")
    for t in range(int(L / SR)):
        fr = np.stack([feat(w[t * SR : (t + 1) * SR]) for w, l_ in zip(real, lab) if l_ == 0])
        fg = np.stack([feat(w[t * SR : (t + 1) * SR]) for w in gen[NAMES[0]]])
        print(f"    {t:2d}-{t + 1:2d} s{' (real opening)' if t < PRE else '               '}: {fr.mean():6.1f} / "
              f"{(fr[:, :8].mean(1) - fr.mean(1)).mean():+5.1f}   |   {fg.mean():6.1f} / {(fg[:, :8].mean(1) - fg.mean(1)).mean():+5.1f}")

    # ---- B. per encoder ----
    del m
    torch.cuda.empty_cache()
    from BEATs import BEATs, BEATsConfig
    from beats_realism import embed
    ck = torch.load(glob.glob(str(Path.home() / ".cache/huggingface/hub/models--lpepino--beats_ckpts/snapshots/*/BEATs_iter3.pt"))[0],
                    map_location="cpu", weights_only=False)
    bm = BEATs(BEATsConfig(ck["cfg"])); bm.load_state_dict(ck["model"]); bm = bm.to(dev).eval()
    br, bg = embed(bm, Rc, dev), embed(bm, Gc, dev)
    feats = {"log-mel shape": (shape(lr_), shape(lg_)), "EnCodec latent": (er, eg),
             "AudioGen LM layer 24 (LoRA)": (hr, hg), "BEATs": (br, bg)}
    bal = np.r_[np.where(gk == 0)[0], rng.choice(np.where(gk > 0)[0], 20, replace=False)]
    print("\n== B. per encoder (generated part 2-10 s vs real 2-10 s, both through the codec) ==")
    print(f"{'encoder':28s} {'1 real vs gen':>13s} | {'2 ref AUC':>9s}  called anomalous (normal / broken / rattle / grinding)"
          f"  [5 nearest real: % faulty]   real n / real a | {'3 gen->real':>11s}")
    for name, (R, G) in feats.items():
        X = np.r_[R, G]; y = np.r_[np.zeros(len(R)), np.ones(len(G))]
        a1 = roc_auc_score(y, cross_val_predict(clf(), X, y, cv=cvk, method="predict_proba")[:, 1])
        cv = cross_val_predict(clf(), R, lab, cv=cvk, method="predict_proba")[:, 1]
        pg = clf().fit(R, lab).predict_proba(G)[:, 1]
        sd = R.std(0) + 1e-6
        d = (((G[:, None] - R[None]) / sd) ** 2).sum(-1)
        nn5 = lab[np.argsort(d, 1)[:, :5]].mean(1)
        sh = " / ".join(f"{(pg[gk == k] > .5).mean():4.0%}" for k in range(4))
        nn = " / ".join(f"{nn5[gk == k].mean():4.0%}" for k in range(4))
        g2r = roc_auc_score(lab, clf().fit(G[bal], (gk[bal] > 0).astype(int)).predict_proba(R)[:, 1])
        print(f"{name:28s} {a1:13.3f} | {roc_auc_score(lab, cv):9.3f}  {sh}  [{nn}]   {(cv[lab == 0] > .5).mean():4.0%} / "
              f"{(cv[lab == 1] > .5).mean():4.0%} | {g2r:11.3f}")
    # paired: does the fault prompt move a continuation towards anomalies compared with the normal prompt from the SAME opening?
    R, G = feats["log-mel shape"]
    ref = clf().fit(R, lab)
    p = ref.predict_proba(G)[:, 1].reshape(4, -1)
    print("\npaired (same opening): fault-prompt minus normal-prompt anomaly probability, log-mel reference: "
          + ", ".join(f"{NAMES[k]} {np.mean(p[k] - p[0]):+.2f} (higher for {np.mean(p[k] > p[0]):.0%} of openings)" for k in (1, 2, 3)))

    # ---- page ----
    ref_vmax = float(np.median([A.mel_db(w).max() for w in real[:10]]))
    entries = []
    for j in (0, 1, 10):
        base = {"clip": f"AudioGen prefix {j + 1}", "machine": "ToyCar AudioGen (full10, real opening)", "label": "generated",
                "file": f"opening = first 2 s of training clip {pre_ids[j]} ({'target' if trdom[pre_ids[j]] else 'source'})",
                "domain": "target" if trdom[pre_ids[j]] else "source"}
        for name in NAMES:
            entries.append(A.page_entry(base, f"opening + {name}", gen[name][j], ref_vmax,
                                        f"0-2 s real opening, 2-10 s AudioGen with '{A.ANOMALY_PROMPTS[name]}'",
                                        mic="near minus far (generated)"))
    A.page_add(entries)


if __name__ == "__main__":
    main()
