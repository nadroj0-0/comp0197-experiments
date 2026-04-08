import torch
import torch.nn as nn

from utils.training.optimisation import OptimisationConfig

from .common import HorizonConditionedHead, n_features, output_size


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
        enc_out, _ = self.gru(x)
        enc_out = self.norm(enc_out)
        return self.decoder(enc_out)


class BaselineGRU(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, dropout=0.0, output_size=1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


def build_baseline_gru(cfg):
    from utils.common import device, mae, mape, r2, rmse

    model = BaselineGRU(
        input_size=n_features(cfg),
        hidden_size=int(cfg.get("hidden", 64)),
        num_layers=int(cfg.get("layers", 1)),
        dropout=float(cfg.get("dropout", 0.0)),
        output_size=output_size(cfg),
    ).to(device)
    criterion = nn.MSELoss()
    optimiser = OptimisationConfig.configure_optimiser(model, cfg)
    training_kwargs = OptimisationConfig.configure_training_kwargs(optimiser, cfg)
    training_kwargs.setdefault("clip_grad_norm", 1.0)
    training_kwargs["extra_metrics"] = {
        "val_rmse": rmse,
        "val_mae": mae,
        "val_mape": mape,
        "val_r2": r2,
    }
    return model, criterion, optimiser, training_kwargs


class BaselineProbGRU(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, dropout=0.0, output_size=1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.mu_head = nn.Linear(hidden_size, output_size)
        self.sigma_head = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        mu = self.mu_head(last)
        sigma = torch.clamp(self.sigma_head(last), min=1e-3, max=10.0)
        return mu, sigma


def build_baseline_prob_gru(cfg):
    from utils.common import device, gaussian_nll_loss, mae, mape, r2, rmse

    model = BaselineProbGRU(
        input_size=n_features(cfg),
        hidden_size=int(cfg.get("hidden", 64)),
        num_layers=int(cfg.get("layers", 1)),
        dropout=float(cfg.get("dropout", 0.0)),
        output_size=output_size(cfg),
    ).to(device)
    criterion = gaussian_nll_loss
    optimiser = OptimisationConfig.configure_optimiser(model, cfg)
    training_kwargs = OptimisationConfig.configure_training_kwargs(optimiser, cfg)
    training_kwargs.setdefault("clip_grad_norm", 1.0)
    training_kwargs["disable_amp"] = True
    training_kwargs["extra_metrics"] = {
        "val_rmse": rmse,
        "val_mae": mae,
        "val_mape": mape,
        "val_r2": r2,
    }
    training_kwargs.setdefault("sigma_reg", cfg.get("sigma_reg", 0.0))
    return model, criterion, optimiser, training_kwargs


class BaselineProbGRU_NB(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, dropout=0.0, output_size=1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.softplus = nn.Softplus()
        self.mu_head = nn.Linear(hidden_size, output_size)
        self.alpha_head = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        mu = self.softplus(self.mu_head(last)) + 1e-6
        alpha = self.softplus(self.alpha_head(last)) + 1e-6
        return mu, alpha


def build_baseline_prob_gru_nb(cfg):
    from utils.common import device, mae, mape, nb_nll_loss, r2, rmse

    model = BaselineProbGRU_NB(
        input_size=n_features(cfg),
        hidden_size=int(cfg.get("hidden", 64)),
        num_layers=int(cfg.get("layers", 1)),
        dropout=float(cfg.get("dropout", 0.0)),
        output_size=output_size(cfg),
    ).to(device)
    criterion = nb_nll_loss
    optimiser = OptimisationConfig.configure_optimiser(model, cfg)
    training_kwargs = OptimisationConfig.configure_training_kwargs(optimiser, cfg)
    training_kwargs.setdefault("clip_grad_norm", 1.0)
    training_kwargs["disable_amp"] = True
    training_kwargs["extra_metrics"] = {
        "val_rmse": rmse,
        "val_mae": mae,
        "val_mape": mape,
        "val_r2": r2,
    }
    return model, criterion, optimiser, training_kwargs


class BaselineQuantileGRU(nn.Module):
    def __init__(
        self,
        input_size=1,
        hidden_size=64,
        num_layers=1,
        dropout=0.0,
        n_quantiles=9,
        horizon=1,
    ):
        super().__init__()
        self.n_quantiles = n_quantiles
        self.horizon = horizon
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, horizon * n_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        raw = self.fc(out[:, -1, :])
        if self.horizon == 1:
            return raw
        return raw.view(raw.size(0), self.horizon, self.n_quantiles)


QUANTILES = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975]


def build_baseline_quantile_gru(cfg):
    from utils.common import device, mae, mape, pinball_loss, r2, rmse

    quantiles = cfg.get("quantiles", QUANTILES)
    model = BaselineQuantileGRU(
        input_size=n_features(cfg),
        hidden_size=int(cfg.get("hidden", 64)),
        num_layers=int(cfg.get("layers", 1)),
        dropout=float(cfg.get("dropout", 0.0)),
        n_quantiles=len(quantiles),
        horizon=output_size(cfg),
    ).to(device)
    criterion = pinball_loss
    optimiser = OptimisationConfig.configure_optimiser(model, cfg)
    training_kwargs = OptimisationConfig.configure_training_kwargs(optimiser, cfg)
    training_kwargs.setdefault("clip_grad_norm", 1.0)
    training_kwargs["disable_amp"] = True
    training_kwargs["quantiles"] = quantiles
    training_kwargs["extra_metrics"] = {
        "val_rmse": rmse,
        "val_mae": mae,
        "val_mape": mape,
        "val_r2": r2,
    }
    return model, criterion, optimiser, training_kwargs


def build_baseline_wquantile_gru(cfg):
    from utils.common import device, mae, mape, r2, rmse, weighted_pinball_loss

    quantiles = cfg.get("quantiles", QUANTILES)
    model = BaselineQuantileGRU(
        input_size=n_features(cfg),
        hidden_size=int(cfg.get("hidden", 64)),
        num_layers=int(cfg.get("layers", 1)),
        dropout=float(cfg.get("dropout", 0.0)),
        n_quantiles=len(quantiles),
        horizon=output_size(cfg),
    ).to(device)
    criterion = weighted_pinball_loss
    optimiser = OptimisationConfig.configure_optimiser(model, cfg)
    training_kwargs = OptimisationConfig.configure_training_kwargs(optimiser, cfg)
    training_kwargs.setdefault("clip_grad_norm", 1.0)
    training_kwargs["disable_amp"] = True
    training_kwargs["quantiles"] = quantiles
    training_kwargs["extra_metrics"] = {
        "val_rmse": rmse,
        "val_mae": mae,
        "val_mape": mape,
        "val_r2": r2,
    }
    return model, criterion, optimiser, training_kwargs
