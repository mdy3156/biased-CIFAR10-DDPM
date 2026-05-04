from pathlib import Path

import torch
from torchvision.utils import save_image


def save_sample_grid(samples: torch.Tensor, path: Path, nrow: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    images = (samples.detach().cpu() + 1.0) * 0.5
    save_image(images.clamp(0.0, 1.0), path, nrow=nrow)
