import torch.nn as nn
import torch
import math


from utils.optimisation import OptimisationConfig

# N_FEATURES is now dynamic — determined by feature_set at runtime.
# Builders read n_features from cfg["n_features"] instead of a global.
def _n_features(cfg: dict) -> int:
    """Resolve input feature size from config."""
    return int(cfg.get("n_features", 1))

def _output_size(cfg: dict) -> int:
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
        # enc_out: (B, T, H)
        B, _, H = enc_out.shape
        queries = self.horizon_emb.unsqueeze(0).expand(B, -1, -1)   # (B, horizon, H)
        attn_out, _ = self.attn(query=queries, key=enc_out, value=enc_out)
        attn_out = self.norm(attn_out + queries)
        attn_out = self.res(attn_out)
        attn_out = self.out(attn_out).squeeze(-1)
        return attn_out  # (B, horizon)


class SalesGRU(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, dropout=0.2, horizon=28):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.decoder = HorizonConditionedHead(hidden_size, horizon, dropout)

    def forward(self, x):
        x = self.input_proj(x)
        enc_out, _ = self.gru(x)           # (B, T, H)
        enc_out = self.norm(enc_out)
        return self.decoder(enc_out)


def build_gru(cfg):
    """
    Builder for deterministic GRU baseline.
    Returns (model, criterion, optimiser, training_kwargs).
    """
    from utils.common import device, rmse, mae, mape, r2

    model = SalesGRU(
        input_size  = _n_features(cfg),
        hidden_size = max(8, (int(cfg["hidden"]) // 8) * 8),
        num_layers  = int(cfg["layers"]),
        dropout     = cfg["dropout"],
        horizon     = _output_size(cfg),
    ).to(device)
    criterion = nn.MSELoss()
    optimiser = OptimisationConfig.configure_optimiser(model, cfg)
    training_kwargs = OptimisationConfig.configure_training_kwargs(optimiser, cfg)
    training_kwargs.setdefault("clip_grad_norm", 1.0)
    training_kwargs["extra_metrics"] = {"val_rmse": rmse, "val_mae": mae, "val_mape": mape, "val_r2": r2}
    return model, criterion, optimiser, training_kwargs




class SalesLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, dropout=0.2, horizon=28):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.decoder = HorizonConditionedHead(hidden_size, horizon, dropout)

    def forward(self, x):
        x = self.input_proj(x)
        enc_out, _ = self.lstm(x)
        enc_out = self.norm(enc_out)
        return self.decoder(enc_out)


def build_lstm(cfg):
    """
    Builder for deterministic LSTM baseline.
    Returns (model, criterion, optimiser, training_kwargs).
    """
    from utils.common import device, rmse, mae, mape, r2
    model = SalesLSTM(
        input_size  = _n_features(cfg),
        hidden_size = max(8, (int(cfg["hidden"]) // 8) * 8),
        num_layers  = int(cfg["layers"]),
        dropout     = cfg["dropout"],
        horizon     = _output_size(cfg),
    ).to(device)
    criterion = nn.MSELoss()
    optimiser = OptimisationConfig.configure_optimiser(model, cfg)
    training_kwargs = OptimisationConfig.configure_training_kwargs(optimiser, cfg)
    training_kwargs.setdefault("clip_grad_norm", 1.0)
    training_kwargs["extra_metrics"] = {"val_rmse": rmse, "val_mae": mae, "val_mape": mape, "val_r2": r2}
    return model, criterion, optimiser, training_kwargs



class ProbGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, horizon):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )

        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.norm = nn.LayerNorm(hidden_size)

        self.mu_decoder = HorizonConditionedHead(hidden_size, horizon, dropout)
        self.sigma_decoder = HorizonConditionedHead(hidden_size, horizon, dropout)

        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.input_proj(x)
        enc_out, _ = self.gru(x)
        enc_out = self.norm(enc_out)
        mu = self.mu_decoder(enc_out)
        sigma = self.softplus(self.sigma_decoder(enc_out))
        sigma = torch.clamp(sigma, min=1e-3, max=5.0)
        return mu, sigma

def build_prob_gru(cfg):
    from utils.common import device, rmse, mae, mape, r2, gaussian_nll_loss
    model = ProbGRU(
        input_size  = _n_features(cfg),
        hidden_size = max(8, (int(cfg["hidden"]) // 8) * 8),
        num_layers  = int(cfg["layers"]),
        dropout     = cfg["dropout"],
        horizon     = _output_size(cfg),
    ).to(device)
    criterion = gaussian_nll_loss
    optimiser = OptimisationConfig.configure_optimiser(model, cfg)
    training_kwargs = OptimisationConfig.configure_training_kwargs(optimiser, cfg)
    training_kwargs.setdefault("clip_grad_norm", 0.5)
    training_kwargs["disable_amp"] = True
    training_kwargs["extra_metrics"] = {"val_rmse": rmse, "val_mae": mae, "val_mape": mape, "val_r2": r2}
    training_kwargs.setdefault("sigma_reg", cfg.get("sigma_reg", 0.0))
    return model, criterion, optimiser, training_kwargs

class ProbGRU_NB(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, horizon):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )

        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.norm = nn.LayerNorm(hidden_size)

        self.mu_decoder = HorizonConditionedHead(hidden_size, horizon, dropout)
        self.alpha_decoder = HorizonConditionedHead(hidden_size, horizon, dropout)

        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.input_proj(x)
        enc_out, _ = self.gru(x)
        enc_out = self.norm(enc_out)

        mu = self.softplus(self.mu_decoder(enc_out)) + 1e-6
        alpha = self.softplus(self.alpha_decoder(enc_out)) + 1e-6
        return mu, alpha

def build_prob_gru_nb(cfg):
    from utils.common import device, rmse, mae, mape, r2, nb_nll_loss

    model = ProbGRU_NB(
        input_size  = _n_features(cfg),
        hidden_size = max(8, (int(cfg["hidden"]) // 8) * 8),
        num_layers  = int(cfg["layers"]),
        dropout     = cfg["dropout"],
        horizon     = _output_size(cfg),
    ).to(device)
    criterion = nb_nll_loss
    optimiser = OptimisationConfig.configure_optimiser(model, cfg)
    training_kwargs = OptimisationConfig.configure_training_kwargs(optimiser, cfg)
    training_kwargs.setdefault("clip_grad_norm", 0.5)
    training_kwargs["disable_amp"] = True
    training_kwargs["extra_metrics"] = {
        "val_rmse": rmse,
        "val_mae": mae,
        "val_mape": mape,
        "val_r2": r2,
    }

    return model, criterion, optimiser, training_kwargs



class ProbLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, horizon):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.norm = nn.LayerNorm(hidden_size)

        self.mu_decoder = HorizonConditionedHead(hidden_size, horizon, dropout)
        self.sigma_decoder = HorizonConditionedHead(hidden_size, horizon, dropout)
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.input_proj(x)
        enc_out, _ = self.lstm(x)
        enc_out = self.norm(enc_out)
        mu = self.mu_decoder(enc_out)
        sigma = self.softplus(self.sigma_decoder(enc_out))
        sigma = torch.clamp(sigma, min=1e-3, max=5.0)
        return mu, sigma


def build_prob_lstm(cfg):
    from utils.common import device, rmse, mae, mape, r2, gaussian_nll_loss
    model = ProbLSTM(
        input_size  = _n_features(cfg),
        hidden_size = max(8, (int(cfg["hidden"]) // 8) * 8),
        num_layers  = int(cfg["layers"]),
        dropout     = cfg["dropout"],
        horizon     = _output_size(cfg),
    ).to(device)
    criterion = gaussian_nll_loss
    optimiser = OptimisationConfig.configure_optimiser(model, cfg)
    training_kwargs = OptimisationConfig.configure_training_kwargs(optimiser, cfg)
    training_kwargs.setdefault("clip_grad_norm", 0.5)
    training_kwargs["disable_amp"] = True
    training_kwargs["extra_metrics"] = {"val_rmse": rmse, "val_mae": mae, "val_mape": mape, "val_r2": r2}
    training_kwargs.setdefault("sigma_reg", cfg.get("sigma_reg", 0.0))
    return model, criterion, optimiser, training_kwargs



class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=get_num_heads(d_model),
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, seq_len, d_model)
        attn_out, attn_weights = self.attn(
            query=x,
            key=x,
            value=x,
            need_weights=True,
            average_attn_weights=False,
        )
        x = self.norm1(x + self.dropout1(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout2(ff_out))
        return x, attn_weights

class SalesTransformer(nn.Module):
    def __init__(self, input_size, d_model=128, n_heads=4, ff_dim=256, n_layers=2, dropout=0.1, horizon=28):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.pos_enc = PositionalEncoding(d_model=d_model, max_len=256, dropout=dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_heads=n_heads, ff_dim=ff_dim, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.decoder = HorizonConditionedHead(d_model, horizon, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        for block in self.blocks:
            x, _ = block(x)
        x = self.norm(x)
        x = self.dropout(x)
        return self.decoder(x)

def build_transformer(cfg):
    """
    Builder for deterministic Transformer model.
    Returns (model, criterion, optimiser, training_kwargs).
    """
    from utils.common import device, rmse, mae, mape, r2
    model = SalesTransformer(
        input_size = _n_features(cfg),
        d_model    = max(8, (int(cfg.get("d_model", 128)) // 8) * 8),
        n_heads    = int(cfg.get("n_heads", 4)),
        ff_dim     = int(cfg.get("ff_dim", 256)),
        n_layers   = int(cfg.get("layers", 2)),
        dropout    = cfg["dropout"],
        horizon    = _output_size(cfg),
    ).to(device)
    criterion = nn.MSELoss()
    optimiser = OptimisationConfig.configure_optimiser(model, cfg)
    training_kwargs = OptimisationConfig.configure_training_kwargs(optimiser, cfg)
    training_kwargs.setdefault("clip_grad_norm", 1.0)
    training_kwargs["disable_amp"] = True
    training_kwargs["extra_metrics"] = {"val_rmse": rmse, "val_mae": mae, "val_mape": mape, "val_r2": r2}
    return model, criterion, optimiser, training_kwargs




class ProbSalesTransformer(nn.Module):
    def __init__(self, input_size, d_model=128, n_heads=4, ff_dim=256, n_layers=2, dropout=0.1, horizon=28):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.pos_enc = PositionalEncoding(d_model=d_model, max_len=256, dropout=dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_heads=n_heads, ff_dim=ff_dim, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        self.mu_decoder = HorizonConditionedHead(d_model, horizon, dropout)
        self.sigma_decoder = HorizonConditionedHead(d_model, horizon, dropout)

        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)

        for block in self.blocks:
            x, _ = block(x)

        enc_out = self.norm(x)
        mu = self.mu_decoder(enc_out)
        sigma = self.softplus(self.sigma_decoder(enc_out))
        sigma = torch.clamp(sigma, min=1e-3, max=5.0)
        return mu, sigma


def build_prob_transformer(cfg):
    """
    Builder for probabilistic Transformer model.
    Returns (model, criterion, optimiser, training_kwargs).
    """
    from utils.common import device, rmse, mae, mape, r2, gaussian_nll_loss
    model = ProbSalesTransformer(
        input_size = _n_features(cfg),
        d_model    = max(8, (int(cfg.get("d_model", 128)) // 8) * 8),
        n_heads    = int(cfg.get("n_heads", 4)),
        ff_dim     = int(cfg.get("ff_dim", 256)),
        n_layers   = int(cfg.get("layers", 2)),
        dropout    = cfg["dropout"],
        horizon    = _output_size(cfg),
    ).to(device)
    criterion = gaussian_nll_loss
    optimiser = OptimisationConfig.configure_optimiser(model, cfg)
    training_kwargs = OptimisationConfig.configure_training_kwargs(optimiser, cfg)
    training_kwargs.setdefault("clip_grad_norm", 0.5)
    training_kwargs["extra_metrics"] = {"val_rmse": rmse, "val_mae": mae, "val_mape": mape, "val_r2": r2}
    training_kwargs.setdefault("sigma_reg", cfg.get("sigma_reg", 0.0))
    return model, criterion, optimiser, training_kwargs