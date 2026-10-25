"""Training loop: progress bar, periodic validation, reconstruction plots, checkpoints.

The reported result is always the LAST step of a fixed-length run. No checkpoint is chosen
using the labelled test clips, so the same settings can be used for every machine type.
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from .config import to_plain
from .data import build_datasets, infinite_loader
from .evaluation import evaluate, format_metrics
from .models import build_model
from .visualize import save_step_plots, select_indices


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_optimizer(model, cfg):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.ndim < 2 or n.endswith(("pos", "query", "mem_pos")) else decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr,
        betas=(0.9, 0.95),
    )


def lr_at(step, cfg):
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    t = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * t))


def run_dir(cfg) -> Path:
    d = Path(cfg.output.dir) / cfg.name / time.strftime("%Y%m%d-%H%M%S")
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_ckpt(path, model, step, cfg, metrics):
    torch.save({"model": model.state_dict(), "step": step, "cfg": to_plain(cfg), "metrics": metrics}, path)


def train(cfg):
    """Trains for cfg.train.steps and returns the metrics of the final step (primary score type)."""
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    out = run_dir(cfg)
    (out / "config.yaml").write_text(yaml.safe_dump(to_plain(cfg), sort_keys=False))
    print(f"Run directory: {out}")

    train_ds, val_ds = build_datasets(cfg)
    C, n_mels, n_frames = train_ds.specs.shape[1:]
    n_norm = int((val_ds.labels == 0).sum())
    print(
        f"machine={cfg.data.machine}  train={len(train_ds)}  val={len(val_ds)} "
        f"({n_norm} normal / {len(val_ds) - n_norm} anomaly)  input=({C},{n_mels},{n_frames})"
    )

    model = build_model(cfg, C, (n_mels, n_frames)).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(
        f"model: {cfg.model.encoder.name} -> {cfg.model.bottleneck.name} -> {cfg.model.decoder.name}  "
        f"({n_params:.2f}M params, {model.bottleneck.num_tokens} bottleneck tokens x {model.bottleneck.out_dim})"
    )
    opt = make_optimizer(model, cfg.train)

    vis_idx = select_indices(val_ds.labels.numpy(), val_ds.domains.numpy(), cfg.vis.n_samples, cfg.seed)
    vis_dir = out / "recon"
    log_path = out / "metrics.jsonl"
    primary = cfg.eval.score
    last = {}

    def validate(step, loss):
        nonlocal last
        res = evaluate(model, val_ds, device, cfg.eval.batch_size, cfg.train.amp, cfg.eval.pauc_max_fpr)
        tqdm.write(f"[step {step:>6}] loss {loss:.4f} | val {format_metrics(res['metrics'], primary)}")
        with open(log_path, "a") as f:
            f.write(json.dumps({"step": step, "train_loss": loss, "val": res["metrics"]}) + "\n")
        save_step_plots(
            vis_dir, step, vis_idx, val_ds, res["recon"], res["err"], res["scores"][primary], cfg.audio, cfg.vis.err_vmax
        )
        save_ckpt(out / "last.pt", model, step, cfg, res["metrics"])
        last = res["metrics"]
        model.train()

    loader = infinite_loader(train_ds, cfg.data.batch_size, cfg.data.num_workers)
    model.train()
    loss_ema = None
    if cfg.eval.at_start:
        validate(0, float("nan"))
    pbar = tqdm(range(1, cfg.train.steps + 1), desc="train", dynamic_ncols=True)
    for step in pbar:
        lr = lr_at(step - 1, cfg.train)
        for g in opt.param_groups:
            g["lr"] = lr
        x = next(loader)[0].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cfg.train.amp and device.type == "cuda"):
            recon, logvar, _ = model(x)
        loss = model.loss(recon.float(), None if logvar is None else logvar.float(), model.target(x))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        opt.step()

        if step % cfg.train.log_every == 0 or step == 1:
            v = loss.item()
            loss_ema = v if loss_ema is None else 0.9 * loss_ema + 0.1 * v
            pbar.set_postfix(loss=f"{loss_ema:.4f}", lr=f"{lr:.1e}")
        if step % cfg.eval.every == 0 or step == cfg.train.steps:
            validate(step, loss_ema if loss_ema is not None else float("nan"))
    (out / "final_metrics.json").write_text(json.dumps(last, indent=1))
    print(f"Done. Final-step metrics [{primary}]: {format_metrics(last, primary)}\nOutputs in {out}")
    return last
