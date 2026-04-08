import math

import torch
import torch.nn as nn


def n_features(cfg: dict) -> int:
    """Resolve input feature size from config."""
    return int(cfg.get("n_features", 1))


def output_size(cfg: dict) -> int:
    """
    Resolve model output size from config.
    autoregressive=True  -> 1  (1-step ahead, recursive rollout at test)
    autoregressive=False -> horizon  (direct multi-step)
    """
    if cfg.get("autoregressive", True):
        return 1
    return int(cfg["horizon"])


def get_num_heads(hidden_size: int) -> int:
    for h in [8, 4, 2, 1]:
        if hidden_size % h == 0:
            return h
    return 1


def rounded_hidden_size(raw_hidden: int) -> int:
    return max(8, (int(raw_hidden) // 8) * 8)


class ResidualForecastHead(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.block(x))


class HorizonConditionedHead(nn.Module):
    def __init__(self, hidden_size: int, horizon: int, dropout: float):
        super().__init__()
        self.horizon = horizon
        self.horizon_emb = nn.Parameter(
            torch.randn(horizon, hidden_size) / math.sqrt(hidden_size)
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=get_num_heads(hidden_size),
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.res = ResidualForecastHead(hidden_size, dropout)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, enc_out: torch.Tensor) -> torch.Tensor:
        bsz, _, hidden_size = enc_out.shape
        queries = self.horizon_emb.unsqueeze(0).expand(bsz, -1, -1)
        attn_out, _ = self.attn(query=queries, key=enc_out, value=enc_out)
        attn_out = self.norm(attn_out + queries)
        attn_out = self.res(attn_out)
        attn_out = self.out(attn_out).squeeze(-1)
        return attn_out

