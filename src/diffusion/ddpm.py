import torch
import torch.nn.functional as F

from .schedule import make_beta_schedule


def _extract(values: torch.Tensor, timesteps: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    out = values.gather(0, timesteps)
    return out.reshape(timesteps.shape[0], *((1,) * (len(shape) - 1)))


class GaussianDiffusion:
    def __init__(self, timesteps: int = 1000, beta_schedule: str = "cosine", device: torch.device | str = "cpu"):
        self.timesteps = int(timesteps)
        self.device = torch.device(device)
        betas = make_beta_schedule(beta_schedule, self.timesteps).to(self.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        return torch.randint(0, self.timesteps, (batch_size,), device=self.device, dtype=torch.long)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)
        x_t = (
            _extract(self.sqrt_alphas_cumprod, t, x0.shape) * x0
            + _extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise
        )
        return x_t, noise

    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return (
            x_t - _extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * eps
        ) / _extract(self.sqrt_alphas_cumprod, t, x_t.shape)

    @torch.no_grad()
    def p_sample(self, model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pred_noise = model(x, t, y)
        beta_t = _extract(self.betas, t, x.shape)
        sqrt_one_minus = _extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alpha = _extract(self.sqrt_recip_alphas, t, x.shape)
        model_mean = sqrt_recip_alpha * (x - beta_t * pred_noise / sqrt_one_minus)
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().reshape(x.shape[0], *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * torch.sqrt(_extract(self.posterior_variance, t, x.shape)) * noise

    @torch.no_grad()
    def sample_ddpm(self, model: torch.nn.Module, labels: torch.Tensor, image_size: int = 32) -> torch.Tensor:
        x = torch.randn(labels.shape[0], 3, image_size, image_size, device=self.device)
        for step in reversed(range(self.timesteps)):
            t = torch.full((labels.shape[0],), step, device=self.device, dtype=torch.long)
            x = self.p_sample(model, x, t, labels)
        return x.clamp(-1.0, 1.0)

    @torch.no_grad()
    def sample_ddim(
        self,
        model: torch.nn.Module,
        labels: torch.Tensor,
        steps: int = 50,
        eta: float = 0.0,
        image_size: int = 32,
    ) -> torch.Tensor:
        x = torch.randn(labels.shape[0], 3, image_size, image_size, device=self.device)
        times = torch.linspace(self.timesteps - 1, 0, int(steps), device=self.device).long()
        prev_times = torch.cat([times[1:], torch.full((1,), -1, device=self.device, dtype=torch.long)])
        for time, prev_time in zip(times, prev_times):
            t = torch.full((labels.shape[0],), int(time.item()), device=self.device, dtype=torch.long)
            eps = model(x, t, labels)
            x0 = self.predict_x0_from_eps(x, t, eps).clamp(-1.0, 1.0)
            alpha = _extract(self.alphas_cumprod, t, x.shape)
            if int(prev_time.item()) < 0:
                alpha_prev = torch.ones_like(alpha)
            else:
                t_prev = torch.full((labels.shape[0],), int(prev_time.item()), device=self.device, dtype=torch.long)
                alpha_prev = _extract(self.alphas_cumprod, t_prev, x.shape)
            sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev))
            direction = torch.sqrt((1 - alpha_prev - sigma**2).clamp(min=0.0)) * eps
            noise = sigma * torch.randn_like(x) if eta > 0 else 0.0
            x = torch.sqrt(alpha_prev) * x0 + direction + noise
        return x.clamp(-1.0, 1.0)
