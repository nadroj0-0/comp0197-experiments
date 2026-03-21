import torch.nn as nn
import torch
import math


from utils.data import N_FEATURES
from utils.optimisation import OptimisationConfig



class SalesGRU(nn.Module):
    """
    Multi-layer GRU for deterministic sales point prediction.

    Takes a sequence of historical sales (and optionally extra features)
    and predicts the next timestep's sales as a single scalar.

    Args:
        input_size  (int): Number of features per timestep (1 if sales only).
        hidden_size (int): Number of units in each GRU layer.
        num_layers  (int): Number of stacked GRU layers.
        dropout     (float): Dropout probability between GRU layers.
    """
    def __init__(self, input_size=N_FEATURES, hidden_size=128, num_layers=2, dropout=0.2, horizon=28,
                 use_temporal_head=False):
        super().__init__()
        self.use_temporal_head = use_temporal_head
        self.horizon = horizon
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,   # input shape: (batch, seq_len, features)
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.norm = nn.LayerNorm(hidden_size)
        if self.use_temporal_head:
            self.temporal_head = nn.GRU(
                input_size=hidden_size,
                hidden_size=hidden_size,
                batch_first=True
            )
            self.output_layer = nn.Linear(hidden_size, 1)
        else:
            self.head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, horizon)
            )

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        x = self.input_proj(x)
        out, _ = self.gru(x)
        out = self.norm(out)
        # take only the final timestep's hidden state
        last = out[:, -1, :]  # (batch, hidden)
        if not self.use_temporal_head:
            return self.head(last)
        hidden_seq = last.unsqueeze(1).repeat(1, self.horizon, 1)
        # (batch, horizon, hidden)
        temporal_out, _ = self.temporal_head(hidden_seq)
        # (batch, horizon, hidden)
        out = self.output_layer(temporal_out).squeeze(-1)
        # (batch, horizon)
        return out


def build_gru(cfg):
    """
    Builder for deterministic GRU baseline.
    Returns (model, criterion, optimiser, training_kwargs).
    """
    from utils.common import device, rmse, mae, mape, r2
    model = SalesGRU(
        input_size=N_FEATURES,
        hidden_size=int(cfg["hidden"]),
        num_layers=int(cfg["layers"]),
        dropout=cfg["dropout"],
        horizon=int(cfg["horizon"]),
        use_temporal_head=cfg.get("use_temporal_head", False),
    ).to(device)
    criterion = nn.MSELoss()
    optimiser = OptimisationConfig.configure_optimiser(model, cfg)
    training_kwargs = OptimisationConfig.configure_training_kwargs(optimiser, cfg)
    training_kwargs.setdefault("clip_grad_norm", 1.0)
    training_kwargs["extra_metrics"] = {"val_rmse": rmse, "val_mae": mae, "val_mape": mape, "val_r2": r2}
    return model, criterion, optimiser, training_kwargs




class SalesLSTM(nn.Module):
    """
    Multi-layer LSTM for deterministic multi-step sales forecasting.

    Input  : (batch, seq_len, N_FEATURES)
    Output : (batch, horizon)

    Args
    ----
    input_size  : Number of features per timestep. Defaults to N_FEATURES (13).
    hidden_size : LSTM hidden units per layer.
    num_layers  : Number of stacked LSTM layers.
    dropout     : Dropout probability between LSTM layers (ignored if num_layers=1).
    horizon     : Number of future timesteps to predict.
    """

    def __init__(self,input_size:  int   = N_FEATURES,hidden_size: int   = 128,num_layers:  int   = 2,
                 dropout:     float = 0.2,horizon:     int   = 28,use_temporal_head: bool = False):
        super().__init__()
        self.use_temporal_head = use_temporal_head
        self.horizon = horizon
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        if self.use_temporal_head:
            self.temporal_head = nn.GRU(
                input_size=hidden_size,
                hidden_size=hidden_size,
                batch_first=True
            )
            self.output_layer = nn.Linear(hidden_size, 1)
        else:
            self.head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, horizon)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (batch, seq_len, input_size)
        x = self.input_proj(x)
        out, _ = self.lstm(x)
        out = self.norm(out)
        last = out[:, -1, :]  # (batch, hidden)
        if not self.use_temporal_head:
            return self.head(last)
        hidden_seq = last.unsqueeze(1).repeat(1, self.horizon, 1)
        # (batch, horizon, hidden)
        temporal_out, _ = self.temporal_head(hidden_seq)
        # (batch, horizon, hidden)
        out = self.output_layer(temporal_out).squeeze(-1)
        # (batch, horizon)
        return out        # (batch, horizon)


def build_lstm(cfg):
    """
    Builder for deterministic LSTM baseline.
    Returns (model, criterion, optimiser, training_kwargs).
    """
    from utils.common import device, rmse, mae, mape, r2
    model = SalesLSTM(
        input_size=N_FEATURES,
        hidden_size=int(cfg["hidden"]),
        num_layers=int(cfg["layers"]),
        dropout=cfg["dropout"],
        horizon=int(cfg["horizon"]),
        use_temporal_head=cfg.get("use_temporal_head", False),
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
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.shared = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.mu_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, horizon)
        )
        # self.alpha_head = nn.Sequential(
        self.sigma_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, horizon),
            nn.Softplus()
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        x = self.input_proj(x)
        out, _ = self.gru(x)
        out = self.norm(out)
        last = out[:, -1, :]
        shared = self.shared(last)
        mu = self.mu_head(shared)
        sigma = self.sigma_head(shared)
        # stability
        sigma = torch.clamp(sigma, min=1e-3, max=10.0)
        return mu, sigma

def build_prob_gru(cfg):
    from utils.common import device, rmse, mae, mape, r2, gaussian_nll_loss
    model = ProbGRU(
        input_size=N_FEATURES,
        hidden_size=int(cfg["hidden"]),
        num_layers=int(cfg["layers"]),
        dropout=cfg["dropout"],
        horizon=int(cfg["horizon"]),
    ).to(device)
    criterion = gaussian_nll_loss
    optimiser = OptimisationConfig.configure_optimiser(model, cfg)
    training_kwargs = OptimisationConfig.configure_training_kwargs(optimiser, cfg)
    training_kwargs.setdefault("clip_grad_norm", 0.5)
    training_kwargs["extra_metrics"] = {"val_rmse": rmse, "val_mae": mae, "val_mape": mape, "val_r2": r2}
    training_kwargs.setdefault("sigma_reg", cfg.get("sigma_reg", 0.0))
    return model, criterion, optimiser, training_kwargs



class ProbLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, horizon):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.shared = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.mu_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, horizon)
        )
        self.sigma_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, horizon),
            nn.Softplus()
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        x = self.input_proj(x)
        out, _ = self.lstm(x)
        out = self.norm(out)
        last = out[:, -1, :]
        shared = self.shared(last)
        mu = self.mu_head(shared)
        sigma = self.sigma_head(shared)
        # stability
        sigma = torch.clamp(sigma, min=1e-3, max=10.0)
        return mu, sigma


def build_prob_lstm(cfg):
    from utils.common import device, rmse, mae, mape, r2, gaussian_nll_loss
    model = ProbLSTM(
        input_size=N_FEATURES,
        hidden_size=int(cfg["hidden"]),
        num_layers=int(cfg["layers"]),
        dropout=cfg["dropout"],
        horizon=int(cfg["horizon"]),
    ).to(device)
    criterion = gaussian_nll_loss
    optimiser = OptimisationConfig.configure_optimiser(model, cfg)
    training_kwargs = OptimisationConfig.configure_training_kwargs(optimiser, cfg)
    training_kwargs.setdefault("clip_grad_norm", 0.5)
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
            num_heads=n_heads,
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
    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        ff_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
        horizon: int = 28,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model=d_model, max_len=64, dropout=dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                ff_dim=ff_dim,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        x = self.input_proj(x)
        x = self.pos_enc(x)
        for block in self.blocks:
            x, _ = block(x)
        x = x.mean(dim=1)
        return self.head(x)

def build_transformer(cfg):
    """
    Builder for deterministic Transformer model.
    Returns (model, criterion, optimiser, training_kwargs).
    """
    from utils.common import device, rmse, mae, mape, r2
    model = SalesTransformer(
        input_size=N_FEATURES,
        d_model=int(cfg.get("d_model", 128)),
        n_heads=int(cfg.get("n_heads", 4)),
        ff_dim=int(cfg.get("ff_dim", 256)),
        n_layers=int(cfg.get("layers", 2)),
        dropout=cfg["dropout"],
        horizon=int(cfg["horizon"]),
    ).to(device)
    criterion = nn.MSELoss()
    optimiser = OptimisationConfig.configure_optimiser(model, cfg)
    training_kwargs = OptimisationConfig.configure_training_kwargs(optimiser, cfg)
    training_kwargs.setdefault("clip_grad_norm", 1.0)
    training_kwargs["extra_metrics"] = {"val_rmse": rmse, "val_mae": mae, "val_mape": mape, "val_r2": r2}
    return model, criterion, optimiser, training_kwargs




class ProbSalesTransformer(nn.Module):
    def __init__(
        self,
        input_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        ff_dim: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
        horizon: int = 28,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model=d_model, max_len=64, dropout=dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                ff_dim=ff_dim,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        self.shared = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mu_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon),
        )
        self.sigma_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_proj(x)
        x = self.pos_enc(x)
        for block in self.blocks:
            x, _ = block(x)
        x = x.mean(dim=1)
        shared = self.shared(x)
        mu = self.mu_head(shared)
        sigma = torch.clamp(self.sigma_head(shared), min=1e-3, max=10.0)
        return mu, sigma


def build_prob_transformer(cfg):
    """
    Builder for probabilistic Transformer model.
    Returns (model, criterion, optimiser, training_kwargs).
    """
    from utils.common import device, rmse, mae, mape, r2, gaussian_nll_loss
    model = ProbSalesTransformer(
        input_size=N_FEATURES,
        d_model=int(cfg.get("d_model", 128)),
        n_heads=int(cfg.get("n_heads", 4)),
        ff_dim=int(cfg.get("ff_dim", 256)),
        n_layers=int(cfg.get("layers", 2)),
        dropout=cfg["dropout"],
        horizon=int(cfg["horizon"]),
    ).to(device)
    criterion = gaussian_nll_loss
    optimiser = OptimisationConfig.configure_optimiser(model, cfg)
    training_kwargs = OptimisationConfig.configure_training_kwargs(optimiser, cfg)
    training_kwargs.setdefault("clip_grad_norm", 0.5)
    training_kwargs["extra_metrics"] = {"val_rmse": rmse, "val_mae": mae, "val_mape": mape, "val_r2": r2}
    training_kwargs.setdefault("sigma_reg", cfg.get("sigma_reg", 0.0))
    return model, criterion, optimiser, training_kwargs