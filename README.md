# biased-CIFAR10-DDPM

Small class-conditional DDPM experiments for balanced and imbalanced CIFAR-10.
The code is organized like `/home/mdy/DenseAssociativeMemory`: YAML configs are
the source of truth, experiments live under `src/experiments`, and each run
writes a timestamped directory under `results/` with copied configs and summary
metadata.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The current environment was created with this venv workflow. Installed PyTorch:
`torch 2.11.0+cu130`, `torchvision 0.26.0+cu130`.

## Run

Balanced CIFAR-10:

```bash
.venv/bin/python -m src.cli.run_experiment --config configs/train_balanced_example.yaml
```

Head/tail imbalanced CIFAR-10:

```bash
.venv/bin/python -m src.cli.run_experiment --config configs/train_imbalanced_example.yaml
```

If the upstream CIFAR-10 host returns `HTTP Error 503`, the loader retries and
then falls back to configured mirrors. After the dataset is present under
`data/cifar-10-batches-py`, you can set `data.download: false`.

The imbalanced example uses:

```text
[5000, 5000, 5000, 5000, 5000, 500, 500, 500, 500, 500]
```

## Outputs

Each run writes:

- `summary.json`: device, seed, data counts, parameter count, last loss.
- `config.yaml` and `config.original.yaml`: reproducibility snapshot.
- `metrics.csv`: step-level loss.
- `epoch_metrics.csv`: epoch-level loss.
- `class_time_mse.csv`: class-wise/time-bin noise-prediction MSE.
- `checkpoints/epoch_XXXX.pt`: model, EMA model, optimizer state.
- `samples/epoch_XXXX_ddim.png`: class-conditioned sample grids when enabled.

The main research diagnostic is `class_time_mse.csv`. It records
`E_c(t)` after binning diffusion times into `diagnostics.time_bins`.

## A100 Starting Point

The example configs intentionally start with a compact U-Net:

```yaml
model:
  base_channels: 64
  channel_multipliers: [1, 2, 2, 4]
data:
  batch_size: 256
```

On A100 80GB, this is conservative. After confirming the logs and samples are
being written correctly, increase `data.batch_size` first, then
`model.base_channels` if image quality becomes the bottleneck.

The example configs are now set for an A100 80GB quality-check run:

```yaml
data:
  batch_size: 384
  num_workers: 16
model:
  base_channels: 96
train:
  max_steps: 50000
  lr: 0.0003
  ema_decay: 0.999
  save_every_steps: 5000
sampling:
  every_steps: 5000
  model: both
  steps: 100
```

Samples are saved as both raw and EMA model outputs, for example
`step_005000_ddim_raw.png` and `step_005000_ddim_ema.png`.
