"""Anomaly scoring (reconstruction error) and DCASE-style metrics.

The score uses nothing but the model's reconstruction of the clip itself:
  mean       mean squared error over the whole clip   <- the primary score
  max_patch  largest 16x16-patch error
  top10      mean of the largest 10% of patch errors
  nll        (models with a variance head only) mean Gaussian negative log-likelihood of the clip
No statistics of the training data, test data, domain or machine type enter the score.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import hmean
from sklearn.metrics import roc_auc_score

SCORE_TYPES = ("mean", "max_patch", "top10", "nll")


@torch.no_grad()
def reconstruct(model, specs: torch.Tensor, device, batch_size=100, amp=True):
    """Returns (recon, squared_error, nll, normalised_error) on CPU, float32, at the model's output
    resolution (the input resolution unless model.target_pool > 1). Errors are against model.target(x).
    nll / normalised_error (= squared error / predicted variance) are None for models without a variance head."""
    model.eval()
    recons, logvars = [], []
    for i in range(0, len(specs), batch_size):
        x = specs[i : i + batch_size].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            r, lv, _ = model(x)
        recons.append(r.float().cpu())
        if lv is not None:
            logvars.append(lv.float().cpu())
    recon = torch.cat(recons)
    target = model.target(specs)
    err = (recon - target) ** 2
    if not logvars:
        return recon, err, None, None
    lv = torch.cat(logvars)
    return recon, err, model.nll_map(recon, lv, target), err * torch.exp(-lv)


def clip_scores(err: torch.Tensor, patch: tuple[int, int], nll: torch.Tensor | None = None) -> dict[str, np.ndarray]:
    """err: (N,C,F,T) squared error -> per-clip anomaly scores."""
    patch_err = F.avg_pool2d(err.mean(1, keepdim=True), patch).flatten(1)  # (N, Fp*Tp)
    k = max(1, int(round(0.1 * patch_err.shape[1])))
    out = {
        "mean": err.flatten(1).mean(1).numpy(),
        "max_patch": patch_err.max(1).values.numpy(),
        "top10": patch_err.topk(k, dim=1).values.mean(1).numpy(),
    }
    if nll is not None:
        out["nll"] = nll.flatten(1).mean(1).numpy()
    return out


def compute_metrics(scores: np.ndarray, labels: np.ndarray, domains: np.ndarray, max_fpr=0.1) -> dict:
    """AUC / pAUC following the DCASE protocol; domains: 0=source, 1=target.

    auc_source / auc_target compare a domain's anomalies against its own normals;
    pauc is over the pooled test set; `official` is the harmonic mean of the three.
    """
    out = {}
    for name, d in (("source", 0), ("target", 1)):
        m = domains == d
        out[f"auc_{name}"] = float(roc_auc_score(labels[m], scores[m])) if len(np.unique(labels[m])) > 1 else float("nan")
    out["auc_all"] = float(roc_auc_score(labels, scores))
    out["pauc"] = float(roc_auc_score(labels, scores, max_fpr=max_fpr))
    out["official"] = float(hmean([out["auc_source"], out["auc_target"], out["pauc"]]))
    return out


def evaluate(model, ds, device, batch_size=100, amp=True, max_fpr=0.1) -> dict:
    """Returns {'metrics': {score_type: {...}}, 'scores': {score_type: array}, 'recon', 'err'}.
    'recon' and 'err' are upsampled to the input resolution for plotting only; scores use the raw ones."""
    recon, err, nll, zerr = reconstruct(model, ds.specs, device, batch_size, amp)
    scores = clip_scores(err, model.out_patch, nll)
    if zerr is not None:
        err = zerr  # plot the variance-normalised error
    if model.target_pool != (1, 1):
        up = lambda t: t.repeat_interleave(model.target_pool[0], dim=2).repeat_interleave(model.target_pool[1], dim=3)
        recon, err = up(recon), up(err)
    labels, domains = ds.labels.numpy(), ds.domains.numpy()
    metrics = {k: compute_metrics(v, labels, domains, max_fpr) for k, v in scores.items()}
    return {"metrics": metrics, "scores": scores, "recon": recon, "err": err}


def format_metrics(metrics: dict, primary: str) -> str:
    m = metrics[primary]
    alt = " | ".join(f"{k}:{v['auc_all']:.3f}" for k, v in metrics.items() if k != primary)
    return (
        f"AUC {m['auc_all']:.4f} (src {m['auc_source']:.3f} / tgt {m['auc_target']:.3f}) "
        f"pAUC {m['pauc']:.4f} official {m['official']:.4f} [{primary}] | other AUC {alt}"
    )
