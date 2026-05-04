import math

import torch


def make_beta_schedule(schedule: str, timesteps: int) -> torch.Tensor:
    if schedule == "linear":
        return torch.linspace(1e-4, 0.02, timesteps, dtype=torch.float32)
    if schedule == "cosine":
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
        alphas_cumprod = torch.cos(((x / timesteps) + 0.008) / 1.008 * math.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(1e-5, 0.999).float()
    raise ValueError(f"Unknown beta schedule: {schedule}")
