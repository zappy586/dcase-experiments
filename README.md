# Noise-aware anomalous sound detection (DCASE 2026 Task 2)

Unsupervised detection of machine faults from two-microphone recordings (near + far mic), trained on normal sounds only.

## Showcase page (results, figures and audio examples)

A self-contained page in `showcase/` summarises the most important findings with charts and listenable examples.
It needs nothing but Python 3 (no packages):

```bash
git clone <this repository's URL>
cd <repository folder>
python -m http.server 8000 --directory showcase
```

Then open http://localhost:8000 in a browser. (On Windows use `py` instead of `python` if `python` isn't found.)

## Repository contents

- `ae_noiseaware.py` - main model: noise-aware autoencoder (two-mic input, near-minus-far target), incl. the
  target-domain LoRA expert (`--domain_lora`)
- `tools/` - probes and experiments (`*_probe.py`), AudioGen fine-tuning and realism checks (`audiogen_*.py`,
  `*_realism.py`, `prefix_gen.py`), evaluation-set pipeline (`run_eval.py`), showcase builder (`build_showcase.py`)
- `dcase_ae/` - data loading, caching, metrics
- Not included: datasets, caches, model checkpoints, third-party code and environments (see `.gitignore`).
  The data is the public DCASE 2026 Task 2 release (`urls.txt`).

---

## Earlier notes


Anomalous sound detection on the DCASE 2026 Task 2 development machine types in `dataset/*.zip`
(7 machines, 1000 normal training clips each, 200 labelled test clips, two mics: channel 1 near, channel 2 far).
All results use the DCASE metric (AUC per domain + pAUC at p = 0.1, harmonic mean) and the same settings for every machine.

## Results (development set, all 7 machines)

| Method | Overall | AUC | AUC source | AUC target | pAUC |
|---|---|---|---|---|---|
| Transformer autoencoder, mono mel, reconstruction error (`benchmark.py --name baseline`) | 0.502 | 0.512 | 0.527 | 0.512 | 0.516 |
| **Near-only spectrum + 3-scorer ensemble** (`near_only_benchmark.py`) | **0.686** | **0.734** | **0.746** | **0.749** | **0.624** |

"Overall" = harmonic mean of every machine's AUC source, AUC target and pAUC. Per-machine tables are in
`outputs/<name>/summary.txt`.

Near-only ensemble, per machine (AUC): ToyCar 0.657, ToyCarEmu 0.674, bearingEmu 0.652, fan 0.860,
gearboxEmu 0.718, sliderEmu 0.652, valveEmu 0.923.

Caveat: settings were chosen by looking at these development test labels (about 100 variants compared), so treat the
numbers as somewhat optimistic until checked on the evaluation set.

## Near-only spectrum method (`dcase_ae/nearonly.py`)

1. **Feature.** Per frequency, near-mic power minus the part the far mic can explain,
   `Pn - |X|^2 / Pf` with `X` the time-averaged near x conj(far) cross-spectrum, estimated inside each clip,
   averaged over the clip, pooled to 256 bands, in dB. Background noise from distant sources reaches both mics alike
   and is removed; the machine's close-range sound is kept. On the same pipeline this beats the raw near-mic spectrum
   (overall 0.631) and the mono mix (0.616).
2. **Scorers**, each fitted on the machine's normal training clips only:
   - `knn`: distance to the nearest training clip, divided by the typical leave-one-out distance of that clip's domain.
   - `gauss`: one shared Ledoit-Wolf covariance, separate source and target means, smaller Mahalanobis distance.
   - `synth`: a small MLP trained for a few seconds to tell training clips from copies with small Gaussian bumps on a few bands.
3. **Ensemble.** Each score is normalised by its median / interquartile range on the training clips, then the three are summed.
   No test domain labels and no other test clips are used.

```bash
uv run python near_only_benchmark.py                       # all machines -> outputs/near_only/
uv run python near_only_benchmark.py --machines fan,ToyCar --seed 1
```

Across seeds 0-2 the overall score is 0.680-0.686.

## Transformer autoencoder

```bash
uv sync
uv run python train.py                                     # configs/baseline.yaml, one machine
uv run python train.py data.machine=ToyCar train.steps=20000   # any dotted override
uv run python benchmark.py --name my_exp [--machines fan,ToyCar] [a.b=value ...]   # same config on every machine
uv run python evaluate.py outputs/<name>/<run>/last.pt     # test metrics + scores.csv + heatmaps
```

The first run extracts the chosen machine from its zip into `data/` and caches features in `cache/`.
Results are always taken at the last training step; no checkpoint is chosen with test labels.

Swap components with `model.encoder.name`, `model.bottleneck.name`, `model.decoder.name`. New components register with
`@ENCODERS.register("name")` etc. and keep the interfaces:

- encoder: `(B, C, F, T) -> ((B, Fp, Tp, D), visible mask or None)`, with `.out_dim` and `.grid`
- bottleneck: `((B, Fp, Tp, D), mask) -> (B, M, D')`, with `.num_tokens` and `.out_dim`
- decoder: `(B, M, D') -> (B, Fp*Tp, C*pf*pt)`

Options: `model.encoder.mask_ratio`, `model.predict_var` (Gaussian NLL loss, `nll` score), `model.target_pool`.
None of these beat the baseline on fan, ToyCar and bearingEmu (`outputs/v_*`).

## Layout

| Path | What |
|---|---|
| `near_only_benchmark.py`, `dcase_ae/nearonly.py` | Near-only spectrum method. |
| `configs/*.yaml`, `train.py`, `benchmark.py`, `evaluate.py` | Autoencoder experiments. |
| `dcase_ae/data/` | Zip extraction, log-mel features, cached datasets. |
| `dcase_ae/models/` | Encoders, bottlenecks, decoders, composed in `autoencoder.py`. |
| `dcase_ae/evaluation.py` | Scores and AUC / pAUC / DCASE-style harmonic mean. |
| `tools/make_listening_test.py` | Blind listening test page (`listening_test/`, served by `.claude/launch.json`). |
| `third_party/`, `pretrained/` | Downloaded model code and weights (EAT, FISHER, BEATs code); weights are git-ignored. |
