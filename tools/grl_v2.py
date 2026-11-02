"""v2 domain-adversarial classifier on ToyCar, trained end to end: the backbone itself is tuned.
Backbone: BEATs iter3 (90M, frozen) with LoRA (rank --rank) on q_proj, v_proj, out_proj, fc1, fc2 of all 12 layers
(--full: train every BEATs weight instead). Tokens are averaged -> 768 numbers -> two heads:
  head 1: normal vs faulty (MLP 768 -> 128 -> 1)
  head 2: real vs synthetic, behind a gradient-reversal layer (Ganin et al. JMLR 2016), lambda ramp 2/(1+exp(-10p))-1 x --lam
Data (all near-minus-far, machine gain): 1000 real ToyCar training normals; synthetic set v1 (AudioGen), 200 normal +
200 faulty for training, 50 + 50 held out. Training: random 5 s crops, batch = 16 real + 8 synthetic normal + 8 synthetic
faulty, class- and domain-balanced losses. Test: full 10 s, score = head-1 logit on the 200 real test clips (labels only
for the metrics). Every --eval steps it logs real-test metrics, held-out synthetic AUC and the domain head's accuracy.
Usage (main env): uv run python tools/grl_v2.py --lam 1
"""
import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT / "third_party" / "beats"))
import audiogen_ti as A  # noqa: E402
from BEATs import BEATs, BEATsConfig  # noqa: E402
from dcase_ae.evaluation import compute_metrics  # noqa: E402

SYN = ROOT / "outputs" / "audiogen" / "synth_v1"
TARGETS = ("q_proj", "v_proj", "out_proj", "fc1", "fc2")


class LoRA(nn.Module):
    def __init__(self, lin, r):
        super().__init__()
        self.lin, self.s = lin, 1.0
        self.a = nn.Parameter(torch.randn(r, lin.in_features) / lin.in_features ** 0.5)
        self.b = nn.Parameter(torch.zeros(lin.out_features, r))

    def forward(self, x):
        return self.lin(x) + self.s * ((x @ self.a.t()) @ self.b.t())


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class Net(nn.Module):
    def __init__(self, rank, full):
        super().__init__()
        ck = torch.load(glob.glob(str(Path.home() / ".cache/huggingface/hub/models--lpepino--beats_ckpts/snapshots/*/BEATs_iter3.pt"))[0],
                        map_location="cpu", weights_only=False)
        self.beats = BEATs(BEATsConfig(ck["cfg"]))
        self.beats.load_state_dict(ck["model"])
        if not full:
            for p in self.beats.parameters():
                p.requires_grad_(False)
            for name, mod in list(self.beats.named_modules()):
                for t in TARGETS:
                    if isinstance(getattr(mod, t, None), nn.Linear):
                        setattr(mod, t, LoRA(getattr(mod, t), rank))
        self.cls = nn.Sequential(nn.Linear(768, 128), nn.GELU(), nn.Linear(128, 1))
        self.dom = nn.Sequential(nn.Linear(768, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, x, lam=0.0, return_h=False):
        h = self.beats.extract_features(x, padding_mask=torch.zeros_like(x, dtype=torch.bool))[0].mean(1)
        hd = h if getattr(self, "wmat", None) is None else (h.float() - self.wmu) @ self.wmat  # --white critic input
        out = self.cls(h)[:, 0], self.dom(GradReverse.apply(hd, lam))[:, 0]
        return (*out, h) if return_h else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--eval", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rt", action="store_true", help="real train/test clips sent through AudioGen's EnCodec and back "
                    "(cache/toycar_rt_*.npy from tools/roundtrip_real.py)")
    ap.add_argument("--confusion", action="store_true", help="instead of gradient reversal: domain head trained on "
                    "detached features, backbone trained to make it output 0.5 (weight --lam)")
    ap.add_argument("--probe", action="store_true", help="linear probe on backbone features at step 0 and every --eval steps")
    ap.add_argument("--head_lr", type=float, default=1e-3, help="learning rate of the real-vs-synthetic head")
    ap.add_argument("--head_every", type=int, default=1, help="(--confusion) update that head only every N steps")
    ap.add_argument("--dom_smooth", type=float, default=0.0, help="(--confusion) label smoothing of that head's targets")
    ap.add_argument("--dom_normals", action="store_true", help="adversary sees only normals: real normal vs synthetic "
                    "normal (class-conditional alignment; avoids mapping synthetic faulty onto real normals)")
    ap.add_argument("--wd", action="store_true", help="classifier 2 = Wasserstein critic with gradient penalty (WDGRL, "
                    "Shen et al. AAAI 2018): critic maximises W = mean f(synthetic) - mean f(real) - gp * GP on detached "
                    "features (--n_critic updates per step); the backbone gets -W through the same gradient-reversal layer")
    ap.add_argument("--gp", type=float, default=10.0)
    ap.add_argument("--n_critic", type=int, default=5)
    ap.add_argument("--white", type=float, default=0.0, help="(--wd) whiten the critic's input with real-normal "
                    "statistics, covariance shrinkage = this value (0 = raw features)")
    ap.add_argument("--inoise", type=float, default=0.0, help="cross-entropy domain head with annealed instance noise: "
                    "Gaussian noise of this std (whitened units with --white) on its input, falling linearly to 0")
    ap.add_argument("--crop_len", type=int, default=80000, help="training clip length in samples (160000 = full 10 s, "
                    "the length used at evaluation and by the probes)")
    ap.add_argument("--syn_dir", default=str(SYN), help="folder with normal/*.wav and faulty/*.wav synthetic clips")
    ap.add_argument("--n_val", type=int, default=50, help="synthetic clips per class held out for evaluation")
    ap.add_argument("--lam_const", action="store_true", help="GRL coefficient fixed at --lam from step 1 (no DANN ramp)")
    ap.add_argument("--inoise_end", type=float, default=0.75, help="fraction of training at which the noise reaches 0")
    ap.add_argument("--white_ema", type=float, default=0.0, help="(--white) recompute the whitening every step from a "
                    "running mean/covariance of the current backbone's real features with this momentum (0 = fixed at step 0)")
    args = ap.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    tag = (f"{'wd' if args.wd else 'conf' if args.confusion else 'lam'} {args.lam:g}{'c' if args.lam_const else ''} {'full' if args.full else f'lora r{args.rank}'}"
           f"{' rt' if args.rt else ''}{' domN' if args.dom_normals else ''}{f' white{args.white:g}' if args.white else ''}{f' ema{args.white_ema:g}' if args.white_ema else ''}{f' inoise{args.inoise:g}' if args.inoise else ''}{f' crop{args.crop_len}' if args.crop_len != 80000 else ''}"
           f"{'' if Path(args.syn_dir) == SYN else f' syn-{Path(args.syn_dir).name}'}{f' head/{args.head_every} hlr {args.head_lr:g} sm {args.dom_smooth:g}' if args.head_every > 1 else ''} seed {args.seed}")
    log = lambda m: A.log(f"grl_v2 [{tag}]: {m}")

    D = A.Data("ToyCar", dev)
    real = torch.cat([D.audio("train", list(range(i, i + 25)))[:, 0, :160000].cpu() for i in range(0, 1000, 25)])
    test = torch.cat([D.audio("test", list(range(i, i + 25)))[:, 0, :160000].cpu() for i in range(0, 200, 25)])
    if args.rt:
        real = torch.from_numpy(np.load(ROOT / "cache" / "toycar_rt_train.npy").astype(np.float32))
        test = torch.from_numpy(np.load(ROOT / "cache" / "toycar_rt_test.npy").astype(np.float32))
    lab, dom = np.array(D.meta["test"]["label"]), np.array(D.meta["test"]["domain"])
    sdir = Path(args.syn_dir)
    gen = {c: torch.as_tensor(np.stack([sf.read(f, dtype="float32")[0][:160000] for f in sorted((sdir / c).glob("*.wav"))]))
           for c in ("normal", "faulty")}
    perm = {c: rng.permutation(len(gen[c])) for c in gen}
    tr = {c: gen[c][perm[c][args.n_val:]] for c in gen}; va = {c: gen[c][perm[c][:args.n_val]] for c in gen}
    log(f"data: {len(real)} real train normals, {len(test)} real test clips | synthetic from {sdir.name}: "
        f"train {len(tr['normal'])} normal + {len(tr['faulty'])} faulty, held out {args.n_val} + {args.n_val}")

    net = Net(args.rank, args.full).to(dev)
    bb = [p for p in net.beats.parameters() if p.requires_grad]
    log(f"trainable backbone params {sum(p.numel() for p in bb) / 1e6:.2f}M")
    groups = [{"params": bb, "lr": args.lr}, {"params": list(net.cls.parameters()), "lr": 1e-3}]
    if not args.wd:
        groups.append({"params": list(net.dom.parameters()), "lr": args.head_lr})
    opt = torch.optim.AdamW(groups, weight_decay=1e-2)
    # WGAN-GP critic has its own optimiser (Adam, betas 0.5/0.9 as in Gulrajani et al. 2017)
    crit_opt = torch.optim.Adam(net.dom.parameters(), lr=args.head_lr, betas=(0.5, 0.9)) if args.wd else None

    def wdist(f_r, f_s):
        return f_s.mean() - f_r.mean()

    def grad_pen(hr, hs):
        n = min(len(hr), len(hs))
        hr, hs = hr[torch.randperm(len(hr))[:n]], hs[torch.randperm(len(hs))[:n]]
        a = torch.rand(n, 1, device=hr.device)
        xi = (a * hr + (1 - a) * hs).requires_grad_(True)
        g = torch.autograd.grad(net.dom(xi).sum(), xi, create_graph=True)[0]
        return ((g.norm(dim=1) - 1) ** 2).mean()

    def crop(X, n, L=None):
        L = L or args.crop_len
        idx = torch.randint(len(X), (n,))
        if L >= X.shape[1]:  # full clip, same length as evaluation; random circular shift as augmentation
            return torch.stack([torch.roll(X[i], int(s)) for i, s in zip(idx.tolist(), torch.randint(0, X.shape[1], (n,)))])
        st = torch.randint(0, X.shape[1] - L + 1, (n,))
        return torch.stack([X[i, s : s + L] for i, s in zip(idx.tolist(), st.tolist())])

    @torch.no_grad()
    def score(X):
        net.eval(); out = []
        for i in range(0, len(X), 16):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                c, d = net(X[i : i + 16].to(dev))
            out.append(torch.stack([c, d], 1).float().cpu())
        net.train()
        return torch.cat(out).numpy()

    def evaluate(step):
        st, sv = score(test), score(torch.cat([va["normal"], va["faulty"]]))
        sr = score(real[rng.choice(1000, 100, replace=False)])
        m = compute_metrics(st[:, 0], lab, dom)
        syn = roc_auc_score(np.r_[np.zeros(50), np.ones(50)], sv[:, 0])
        dacc = ((sr[:, 1] < 0).mean() + (sv[: 50 if args.dom_normals else 100, 1] > 0).mean()) / 2
        dom_txt = (f"critic W estimate (held-out synthetic normal - real train) {sv[:50, 1].mean() - sr[:, 1].mean():.3f}"
                   if args.wd else f"domain head acc {dacc:.3f} (real test called synthetic {(st[:, 1] > 0).mean():.2f})")
        log(f"step {step}: real test src {m['auc_source']:.3f} tgt {m['auc_target']:.3f} pAUC {m['pauc']:.3f} "
            f"official {m['official']:.3f} | synthetic held-out AUC {syn:.3f} | {dom_txt}")
        if args.probe:
            probe(step)

    @torch.no_grad()
    def embed(X):
        net.eval(); out = []
        for i in range(0, len(X), 16):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(net(X[i : i + 16].to(dev), return_h=True)[2].float().cpu())
        net.train()
        return torch.cat(out).numpy()

    def probe(step):
        """Fresh logistic regression on the backbone features (5-fold CV AUC; 0.5 = cannot tell them apart), all clips
        unseen in training: real test normals vs held-out synthetic normals, real test anomalies vs held-out synthetic
        faulty, and real test normal vs real test anomaly for reference."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_predict
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        # domain-matched: real TRAINING normals (same source/target mix as the synthetic clips' model), not test normals
        H = embed(test); Hr, Hn = embed(real[probe_ids]), embed(va["normal"])
        def cv_auc(X0, X1):
            X, yy = np.r_[X0, X1], np.r_[np.zeros(len(X0)), np.ones(len(X1))]
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
            return roc_auc_score(yy, cross_val_predict(clf, X, yy, cv=5, method="predict_proba")[:, 1])
        from clora_align import probes  # linear / RBF-SVM / MLP / kNN, 5-fold CV AUC
        pr = probes(Hr, Hn)
        log(f"probe step {step}: real train normal vs synthetic normal  " + "  ".join(f"{k} {v:.3f}" for k, v in pr.items())
            + f" | real test normal vs real test anomaly (reference) {cv_auc(H[lab == 0], H[lab == 1]):.3f}")

    probe_ids = rng.choice(1000, 200, replace=False)

    if args.probe:
        probe(0)

    wmu = wmat = None
    if (args.wd or args.inoise) and args.white:
        # fixed whitening of the critic's input from plain-BEATs features (LoRA B = 0 here) of 400 random 5 s crops of
        # real training normals outside the probe set, with shrinkage: the critic's 1-Lipschitz distance is then
        # measured in the real-normal (Mahalanobis) geometry, where the low-variance real-vs-synthetic directions count
        Hw = torch.as_tensor(embed(crop(real[np.setdiff1d(np.arange(1000), probe_ids)], 400)), dtype=torch.float64)
        C = torch.cov(Hw.T); C = C + args.white * torch.trace(C) / len(C) * torch.eye(len(C), dtype=C.dtype)
        ev, U = torch.linalg.eigh(C)
        wmu, wmat = Hw.mean(0).float().to(dev), (U @ torch.diag(ev.rsqrt()) @ U.T).float().to(dev)
        net.wmu, net.wmat = wmu, wmat
        m_ema = Hw.mean(0).to(dev)                                       # running real stats start from plain BEATs
        s_ema = (Hw.T @ Hw / len(Hw)).to(dev)
        log(f"critic input whitened (shrinkage {args.white:g}{f', running stats momentum {args.white_ema:g}' if args.white_ema else ', fixed'}), "
            f"eigenvalue range {ev.min():.2e}..{ev.max():.2e}")

    y = torch.cat([torch.zeros(24), torch.ones(8)]).to(dev)                  # 1 = faulty
    d = torch.cat([torch.zeros(16), torch.ones(16)]).to(dev)                 # 1 = synthetic
    w = torch.cat([torch.full((24,), 0.5 / 24), torch.full((8,), 0.5 / 8)]).to(dev)
    # adversary on normals only (real normal vs synthetic normal): first 24 of the batch, both domains weighted equally
    nd = 24 if args.dom_normals else 32
    wd = (torch.cat([torch.full((16,), 0.5 / 16), torch.full((8,), 0.5 / 8)]) if args.dom_normals
          else torch.full((32,), 1 / 32)).to(dev)
    dsel = torch.cat([torch.zeros(16), torch.ones(8)]).to(dev) if args.dom_normals else d
    bce_d = lambda logit, target: (F.binary_cross_entropy_with_logits(logit[:nd], target, reduction="none") * wd).sum()
    t0 = time.time()
    for s in range(1, args.steps + 1):
        lam = args.lam if args.lam_const else args.lam * (2 / (1 + np.exp(-10 * s / args.steps)) - 1)
        x = torch.cat([crop(real, 16), crop(tr["normal"], 8), crop(tr["faulty"], 8)]).to(dev)
        opt.zero_grad()
        sig = args.inoise * max(0.0, 1 - s / (args.steps * args.inoise_end)) if args.inoise else 0.0
        if args.wd or args.inoise:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lc, _, h = net(x, 0.0, return_h=True)
            lc, h = lc.float(), h.float()
            loss_c = (F.binary_cross_entropy_with_logits(lc, y, reduction="none") * w).sum()
            hn, isr = h[:nd], dsel == 0                           # adversary on the normals only with --dom_normals
            if wmat is not None and args.white_ema:
                # running whitening from the CURRENT backbone's real features (batch rows 0-15 are real normals), so
                # shrinking a direction for both domains is undone by the whitening instead of lowering the distance
                hr = h[:16].detach().double()
                m_ema.mul_(args.white_ema).add_((1 - args.white_ema) * hr.mean(0))
                s_ema.mul_(args.white_ema).add_((1 - args.white_ema) * (hr.T @ hr) / len(hr))
                C = s_ema - torch.outer(m_ema, m_ema)
                C = C + args.white * torch.trace(C) / len(C) * torch.eye(len(C), dtype=C.dtype, device=dev)
                ev, U = torch.linalg.eigh(C)
                wmu, wmat = m_ema.float(), (U @ torch.diag(ev.clamp_min(1e-12).rsqrt()) @ U.T).float()
                net.wmu, net.wmat = wmu, wmat
            if wmat is not None:
                hn = (hn - wmu) @ wmat
        if args.inoise:
            # instance noise (Sonderby et al. 2016; Arjovsky & Bottou 2017): Gaussian noise on the domain head's input,
            # annealed to 0, so the two domains overlap at first and the cross-entropy head's gradient does not vanish
            hz = hn + sig * torch.randn_like(hn)
            ld = net.dom(GradReverse.apply(hz, lam))[:, 0]
            loss_d = (F.binary_cross_entropy_with_logits(ld, dsel, reduction="none") * wd).sum()
            (loss_c + loss_d).backward()
        elif args.wd:
            hd = hn.detach()
            for _ in range(args.n_critic):                       # critic: maximise W - gp * GP on detached features
                fd = net.dom(hd)[:, 0]
                loss_crit = -wdist(fd[isr], fd[~isr]) + args.gp * grad_pen(hd[isr], hd[~isr])
                crit_opt.zero_grad(); loss_crit.backward(); crit_opt.step()
            fr = net.dom(GradReverse.apply(hn, lam))[:, 0]        # same GRL as before; critic objective = -W
            loss_d = -wdist(fr[isr], fr[~isr])
            (loss_c + loss_d).backward()
            for p in net.dom.parameters():                        # critic is updated only by its own optimiser
                p.grad = None
        elif args.confusion:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lc, lconf, h = net(x, 0.0, return_h=True)
                ld = net.dom(h.detach().float())[:, 0]
            lc, lconf, ld = lc.float(), lconf.float(), ld.float()
            loss_c = (F.binary_cross_entropy_with_logits(lc, y, reduction="none") * w).sum()
            loss_conf = bce_d(lconf, torch.full((nd,), 0.5, device=dev))
            (loss_c + lam * loss_conf).backward()
            for p in net.dom.parameters():  # the confusion term trains only the backbone
                p.grad = None
            loss_d = bce_d(ld, dsel * (1 - 2 * args.dom_smooth) + args.dom_smooth)
            if s % args.head_every == 0:  # the real-vs-synthetic head learns only every N steps
                loss_d.backward()
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lc, ld = net(x, lam)
            lc, ld = lc.float(), ld.float()
            loss_c = (F.binary_cross_entropy_with_logits(lc, y, reduction="none") * w).sum()
            loss_d = bce_d(ld, dsel)
            (loss_c + loss_d).backward()
        opt.step()
        if s % 50 == 0:
            log(f"step {s}: loss cls {loss_c.item():.3f} dom {loss_d.item():.3f} lam {lam:.2f}"
                f"{f' noise {sig:.1f}' if args.inoise else ''}, {(time.time() - t0) / s:.2f} s/step")
        if s % args.eval == 0:
            evaluate(s)
    if args.probe:
        probe(args.steps)
    name = tag.replace(" ", "_")
    torch.save({n: p.detach().cpu() for n, p in net.named_parameters() if p.requires_grad}, ROOT / "outputs" / f"grl_v2_{name}.pt")


if __name__ == "__main__":
    main()
