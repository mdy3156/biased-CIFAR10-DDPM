import math

import torch
from torch import nn
import torch.nn.functional as F


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(0, half, device=timesteps.device, dtype=torch.float32) / half
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_ch: int, dropout: float):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_ch, out_ch)
        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb_proj(F.silu(emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        flat = self.norm(x).reshape(b, c, h * w)
        q, k, v = self.qkv(flat).chunk(3, dim=1)
        scale = c ** -0.5
        weight = torch.einsum("bct,bcs->bts", q * scale, k).softmax(dim=-1)
        out = torch.einsum("bts,bcs->bct", weight, v)
        out = self.proj(out).reshape(b, c, h, w)
        return x + out


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class ClassConditionalUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 128,
        channel_multipliers: tuple[int, ...] | list[int] = (1, 2, 2, 4),
        num_res_blocks: int = 2,
        dropout: float = 0.1,
        num_classes: int = 10,
    ):
        super().__init__()
        emb_ch = base_channels * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, emb_ch),
            nn.SiLU(),
            nn.Linear(emb_ch, emb_ch),
        )
        self.class_emb = nn.Embedding(num_classes, emb_ch)
        self.input = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        channels = [base_channels]
        self.down = nn.ModuleList()
        ch = base_channels
        for level, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.down.append(ResBlock(ch, out_ch, emb_ch, dropout))
                ch = out_ch
                if level == 1:
                    self.down.append(AttentionBlock(ch))
                channels.append(ch)
            if level != len(channel_multipliers) - 1:
                self.down.append(Downsample(ch))
                channels.append(ch)

        self.mid1 = ResBlock(ch, ch, emb_ch, dropout)
        self.mid_attn = AttentionBlock(ch)
        self.mid2 = ResBlock(ch, ch, emb_ch, dropout)

        self.up = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_multipliers))):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks + 1):
                self.up.append(ResBlock(ch + channels.pop(), out_ch, emb_ch, dropout))
                ch = out_ch
                if level == 1:
                    self.up.append(AttentionBlock(ch))
            if level != 0:
                self.up.append(Upsample(ch))

        self.out = nn.Sequential(
            nn.GroupNorm(min(32, ch), ch),
            nn.SiLU(),
            nn.Conv2d(ch, in_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        emb = self.time_mlp(timestep_embedding(t, self.input.out_channels)) + self.class_emb(y)
        h = self.input(x)
        skips = [h]
        for module in self.down:
            if isinstance(module, ResBlock):
                h = module(h, emb)
                skips.append(h)
            elif isinstance(module, AttentionBlock):
                h = module(h)
            else:
                h = module(h)
                skips.append(h)
        h = self.mid1(h, emb)
        h = self.mid_attn(h)
        h = self.mid2(h, emb)
        for module in self.up:
            if isinstance(module, ResBlock):
                h = torch.cat([h, skips.pop()], dim=1)
                h = module(h, emb)
            else:
                h = module(h)
        return self.out(h)
