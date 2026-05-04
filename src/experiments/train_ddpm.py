import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from ..data.cifar10 import build_cifar10_train_loader
from ..diffusion.ddpm import GaussianDiffusion
from ..io.outputs import prepare_output_dir
from ..models.unet import ClassConditionalUNet
from ..plotting.samples import save_sample_grid
from ..utils.device import resolve_device
from ..utils.seed import set_seed


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow = deepcopy(model).eval()
        for param in self.shadow.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_param, param in zip(self.shadow.parameters(), model.parameters()):
            ema_param.data.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)
        for ema_buffer, buffer in zip(self.shadow.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)


class ClassTimeMSE:
    def __init__(self, num_classes: int, num_bins: int, timesteps: int, device: torch.device):
        self.num_classes = num_classes
        self.num_bins = num_bins
        self.timesteps = timesteps
        self.eps_sum = torch.zeros(num_classes, num_bins, device=device)
        self.x0_sum = torch.zeros(num_classes, num_bins, device=device)
        self.count = torch.zeros(num_classes, num_bins, device=device)

    @torch.no_grad()
    def update(
        self,
        eps_loss: torch.Tensor,
        x0_loss: torch.Tensor,
        labels: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> None:
        bins = torch.div(timesteps * self.num_bins, self.timesteps, rounding_mode="floor").clamp(max=self.num_bins - 1)
        flat_index = labels * self.num_bins + bins
        self.eps_sum.view(-1).scatter_add_(0, flat_index, eps_loss.detach())
        self.x0_sum.view(-1).scatter_add_(0, flat_index, x0_loss.detach())
        self.count.view(-1).scatter_add_(0, flat_index, torch.ones_like(eps_loss))

    def eps_values(self) -> torch.Tensor:
        return self.eps_sum / self.count.clamp_min(1.0)

    def x0_values(self) -> torch.Tensor:
        return self.x0_sum / self.count.clamp_min(1.0)

    def rows(self, epoch: int, class_names: list[str]) -> list[dict[str, Any]]:
        eps_values = self.eps_values().detach().cpu()
        x0_values = self.x0_values().detach().cpu()
        counts = self.count.detach().cpu()
        rows = []
        bin_width = self.timesteps / self.num_bins
        for cls in range(self.num_classes):
            for bin_id in range(self.num_bins):
                rows.append(
                    {
                        "epoch": epoch,
                        "class": cls,
                        "class_name": class_names[cls] if cls < len(class_names) else str(cls),
                        "time_bin": bin_id,
                        "t_start": int(round(bin_id * bin_width)),
                        "t_end": int(round((bin_id + 1) * bin_width - 1)),
                        "eps_mse": float(eps_values[cls, bin_id]),
                        "x0_mse": float(x0_values[cls, bin_id]),
                        "count": int(counts[cls, bin_id]),
                    }
                )
        return rows


def _append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _save_checkpoint(path: Path, model: nn.Module, ema: EMA, optimizer: torch.optim.Optimizer, epoch: int, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "model": model.state_dict(),
            "ema": ema.shadow.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


@torch.no_grad()
def _sample_if_needed(
    config: dict,
    out_dir: Path,
    diffusion: GaussianDiffusion,
    model: nn.Module,
    ema_model: nn.Module,
    epoch: int,
    device: torch.device,
    step: int | None = None,
) -> list[str]:
    sampling_cfg = config.get("sampling", {})
    if not sampling_cfg.get("enabled", False):
        return []
    if step is None:
        every = int(sampling_cfg.get("every_epochs", 10))
        if epoch % every != 0:
            return []
        prefix = f"epoch_{epoch:04d}"
    else:
        every_steps = sampling_cfg.get("every_steps")
        if every_steps is None or step % int(every_steps) != 0:
            return []
        prefix = f"step_{step:06d}"
    num_per_class = int(sampling_cfg.get("num_per_class", 8))
    labels = torch.arange(10, device=device).repeat_interleave(num_per_class)
    sampler = sampling_cfg.get("sampler", "ddim")
    model_choice = sampling_cfg.get("model", "ema")
    if model_choice == "ema":
        models = [("ema", ema_model)]
    elif model_choice == "raw":
        models = [("raw", model)]
    elif model_choice == "both":
        models = [("raw", model), ("ema", ema_model)]
    else:
        raise ValueError(f"Unknown sampling model choice: {model_choice}")

    was_training = model.training
    model.eval()
    ema_model.eval()
    sample_paths = []
    for model_name, sample_model in models:
        if sampler == "ddpm":
            samples = diffusion.sample_ddpm(sample_model, labels)
        elif sampler == "ddim":
            samples = diffusion.sample_ddim(
                sample_model,
                labels,
                steps=int(sampling_cfg.get("steps", 50)),
                eta=float(sampling_cfg.get("eta", 0.0)),
            )
        else:
            raise ValueError(f"Unknown sampler: {sampler}")
        suffix = f"_{model_name}" if model_choice == "both" else ""
        sample_path = out_dir / "samples" / f"{prefix}_{sampler}{suffix}.png"
        save_sample_grid(samples, sample_path, nrow=num_per_class)
        sample_paths.append(str(sample_path))
    if was_training:
        model.train()
    return sample_paths


def run_train_ddpm(config: dict, config_path: str | None = None) -> None:
    seed = int(config.get("seed", 123))
    set_seed(seed)
    device = resolve_device(config.get("device", "auto"))
    out_dir = prepare_output_dir(config, config_path=config_path)

    loader, data_info = build_cifar10_train_loader(config, seed)
    model = ClassConditionalUNet(**config.get("model", {})).to(device)
    diffusion_cfg = config.get("diffusion", {})
    diffusion = GaussianDiffusion(
        timesteps=int(diffusion_cfg.get("timesteps", 1000)),
        beta_schedule=diffusion_cfg.get("beta_schedule", "cosine"),
        device=device,
    )
    train_cfg = config.get("train", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 2e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    ema = EMA(model, decay=float(train_cfg.get("ema_decay", 0.9999)))
    amp_enabled = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    diagnostics_cfg = config.get("diagnostics", {})
    class_names = diagnostics_cfg.get("class_names", [str(i) for i in range(10)])
    num_bins = int(diagnostics_cfg.get("time_bins", 20))
    global_step = 0
    sample_paths: list[str] = []
    epochs = int(train_cfg.get("epochs", 100))
    max_steps = train_cfg.get("max_steps")
    max_steps = int(max_steps) if max_steps is not None else None
    save_every_steps = train_cfg.get("save_every_steps")
    save_every_steps = int(save_every_steps) if save_every_steps is not None else None
    last_checkpoint: str | None = None

    summary = {
        "device": str(device),
        "seed": seed,
        "data": data_info,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "diffusion": diffusion_cfg,
        "output_dir": str(out_dir),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    stop_training = False
    for epoch in range(1, epochs + 1):
        model.train()
        tracker = ClassTimeMSE(10, num_bins, diffusion.timesteps, device)
        epoch_loss = 0.0
        epoch_batches = 0
        progress = tqdm(loader, desc=f"epoch {epoch}/{epochs}")
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            t = diffusion.sample_timesteps(images.shape[0])
            x_t, noise = diffusion.q_sample(images, t)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                pred = model(x_t, t, labels)
                eps_loss = F.mse_loss(pred, noise, reduction="none").mean(dim=(1, 2, 3))
                loss = eps_loss.mean()
            with torch.no_grad():
                pred_x0 = diffusion.predict_x0_from_eps(x_t, t, pred.detach()).clamp(-1.0, 1.0)
                x0_loss = F.mse_loss(pred_x0, images, reduction="none").mean(dim=(1, 2, 3))
            scaler.scale(loss).backward()
            if float(train_cfg.get("grad_clip", 0.0)) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            tracker.update(eps_loss, x0_loss, labels, t)
            epoch_loss += float(loss.detach())
            epoch_batches += 1
            global_step += 1
            if global_step % int(train_cfg.get("log_every", 50)) == 0:
                rows = [{"step": global_step, "epoch": epoch, "loss": float(loss.detach())}]
                _append_csv(out_dir / "metrics.csv", rows)
                progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
            if save_every_steps is not None and global_step % save_every_steps == 0:
                checkpoint_path = out_dir / "checkpoints" / f"step_{global_step:06d}.pt"
                _save_checkpoint(checkpoint_path, model, ema, optimizer, epoch, global_step)
                last_checkpoint = str(checkpoint_path)
            sample_paths.extend(
                _sample_if_needed(config, out_dir, diffusion, model, ema.shadow, epoch, device, step=global_step)
            )
            if max_steps is not None and global_step >= max_steps:
                stop_training = True
                break

        _append_csv(out_dir / "class_time_mse.csv", tracker.rows(epoch, class_names))
        avg_loss = epoch_loss / max(epoch_batches, 1)
        _append_csv(out_dir / "epoch_metrics.csv", [{"epoch": epoch, "loss": avg_loss}])

        if epoch % int(train_cfg.get("save_every_epochs", 10)) == 0 or epoch == epochs:
            checkpoint_path = out_dir / "checkpoints" / f"epoch_{epoch:04d}.pt"
            _save_checkpoint(checkpoint_path, model, ema, optimizer, epoch, global_step)
            last_checkpoint = str(checkpoint_path)
        if config.get("sampling", {}).get("every_steps") is None:
            sample_paths.extend(_sample_if_needed(config, out_dir, diffusion, model, ema.shadow, epoch, device))

        summary.update(
            {
                "last_epoch": epoch,
                "last_step": global_step,
                "last_loss": avg_loss,
                "last_checkpoint": last_checkpoint,
                "sample_paths": sample_paths,
            }
        )
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        if stop_training:
            break
