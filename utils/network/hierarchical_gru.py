import torch
import torch.nn as nn

from utils.training.optimisation import OptimisationConfig

from .baseline_gru import QUANTILES
from .common import output_size, rounded_hidden_size


class _HierarchyEmbedder(nn.Module):
    def __init__(self, vocab_sizes: dict, feature_index: dict, embed_dim: int = 8):
        super().__init__()
        self.idx = feature_index
        self.state_emb = nn.Embedding(vocab_sizes["state_id_int"], embed_dim)
        self.store_emb = nn.Embedding(vocab_sizes["store_id_int"], embed_dim)
        self.cat_emb = nn.Embedding(vocab_sizes["cat_id_int"], embed_dim)
        self.dept_emb = nn.Embedding(vocab_sizes["dept_id_int"], embed_dim)
        self.embed_dim = embed_dim
        self.cat_cols = ["state_id_int", "store_id_int", "cat_id_int", "dept_id_int"]
        self.cont_indices = [i for name, i in self.idx.items() if name not in self.cat_cols]
        self.output_dim = len(self.cont_indices) + 4 * embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_cont = x[:, :, self.cont_indices]
        state = self.state_emb(x[:, :, self.idx["state_id_int"]].long())
        store = self.store_emb(x[:, :, self.idx["store_id_int"]].long())
        cat = self.cat_emb(x[:, :, self.idx["cat_id_int"]].long())
        dept = self.dept_emb(x[:, :, self.idx["dept_id_int"]].long())
        return torch.cat([x_cont, state, store, cat, dept], dim=-1)


class HierarchicalGRU(nn.Module):
    def __init__(self, hidden_size, num_layers, dropout, output_size, vocab_sizes, feature_index, embed_dim=8):
        super().__init__()
        self.embedder = _HierarchyEmbedder(vocab_sizes, feature_index, embed_dim)
        self.gru = nn.GRU(
            input_size=self.embedder.output_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedder(x)
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


def build_hierarchical_gru(cfg: dict):
    from utils.common import device, mae, mape, r2, rmse

    model = HierarchicalGRU(
        hidden_size=rounded_hidden_size(cfg["hidden"]),
        num_layers=int(cfg["layers"]),
        dropout=float(cfg["dropout"]),
        output_size=output_size(cfg),
        vocab_sizes=cfg["vocab_sizes"],
        embed_dim=int(cfg.get("embed_dim", 8)),
        feature_index=cfg["feature_index"],
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


class HierarchicalProbGRU(nn.Module):
    def __init__(self, hidden_size, num_layers, dropout, horizon, vocab_sizes, feature_index, embed_dim=8):
        super().__init__()
        self.embedder = _HierarchyEmbedder(vocab_sizes, feature_index, embed_dim)
        self.gru = nn.GRU(
            input_size=self.embedder.output_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.mu_head = nn.Linear(hidden_size, horizon)
        self.sigma_head = nn.Linear(hidden_size, horizon)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor):
        x = self.embedder(x)
        out, _ = self.gru(x)
        last = out[:, -1, :]
        mu = self.mu_head(last)
        sigma = torch.clamp(self.softplus(self.sigma_head(last)), min=1e-3, max=5.0)
        return mu, sigma


def build_hierarchical_prob_gru(cfg: dict):
    from utils.common import device, gaussian_nll_loss, mae, mape, r2, rmse

    model = HierarchicalProbGRU(
        hidden_size=rounded_hidden_size(cfg["hidden"]),
        num_layers=int(cfg["layers"]),
        dropout=float(cfg["dropout"]),
        horizon=output_size(cfg),
        vocab_sizes=cfg["vocab_sizes"],
        embed_dim=int(cfg.get("embed_dim", 8)),
        feature_index=cfg["feature_index"],
    ).to(device)
    criterion = gaussian_nll_loss
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
    training_kwargs.setdefault("sigma_reg", cfg.get("sigma_reg", 0.0))
    return model, criterion, optimiser, training_kwargs


class HierarchicalProbGRU_NB(nn.Module):
    def __init__(self, hidden_size, num_layers, dropout, horizon, vocab_sizes, feature_index, embed_dim=8):
        super().__init__()
        self.embedder = _HierarchyEmbedder(vocab_sizes, feature_index, embed_dim)
        self.gru = nn.GRU(
            input_size=self.embedder.output_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.mu_head = nn.Linear(hidden_size, horizon)
        self.alpha_head = nn.Linear(hidden_size, horizon)
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor):
        x = self.embedder(x)
        out, _ = self.gru(x)
        last = out[:, -1, :]
        mu = self.softplus(self.mu_head(last)) + 1e-6
        alpha = self.softplus(self.alpha_head(last)) + 1e-6
        return mu, alpha


def build_hierarchical_prob_gru_nb(cfg: dict):
    from utils.common import device, mae, mape, nb_nll_loss, r2, rmse

    model = HierarchicalProbGRU_NB(
        hidden_size=rounded_hidden_size(cfg["hidden"]),
        num_layers=int(cfg["layers"]),
        dropout=float(cfg["dropout"]),
        horizon=output_size(cfg),
        vocab_sizes=cfg["vocab_sizes"],
        embed_dim=int(cfg.get("embed_dim", 8)),
        feature_index=cfg["feature_index"],
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


class HierarchicalQuantileGRU(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_layers,
        dropout,
        n_quantiles,
        vocab_sizes,
        feature_index,
        embed_dim=8,
        horizon=1,
    ):
        super().__init__()
        self.embedder = _HierarchyEmbedder(vocab_sizes, feature_index, embed_dim)
        self.n_quantiles = n_quantiles
        self.horizon = horizon
        self.gru = nn.GRU(
            input_size=self.embedder.output_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, horizon * n_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedder(x)
        out, _ = self.gru(x)
        raw = self.fc(out[:, -1, :])
        if self.horizon == 1:
            return raw
        return raw.view(raw.size(0), self.horizon, self.n_quantiles)


def build_hierarchical_quantile_gru(cfg: dict):
    from utils.common import device, mae, mape, pinball_loss, r2, rmse

    quantiles = cfg.get("quantiles", QUANTILES)
    model = HierarchicalQuantileGRU(
        hidden_size=rounded_hidden_size(cfg["hidden"]),
        num_layers=int(cfg["layers"]),
        dropout=float(cfg["dropout"]),
        n_quantiles=len(quantiles),
        vocab_sizes=cfg["vocab_sizes"],
        embed_dim=int(cfg.get("embed_dim", 8)),
        horizon=output_size(cfg),
        feature_index=cfg["feature_index"],
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


class HierarchicalWQuantileGRU(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_layers,
        dropout,
        n_quantiles,
        vocab_sizes,
        feature_index,
        embed_dim=8,
        horizon=1,
    ):
        super().__init__()
        self.embedder = _HierarchyEmbedder(vocab_sizes, feature_index, embed_dim)
        self.n_quantiles = n_quantiles
        self.horizon = horizon
        self.gru = nn.GRU(
            input_size=self.embedder.output_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, horizon * n_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedder(x)
        out, _ = self.gru(x)
        raw = self.fc(out[:, -1, :])
        if self.horizon == 1:
            return raw
        return raw.view(raw.size(0), self.horizon, self.n_quantiles)


def build_hierarchical_wquantile_gru(cfg: dict):
    from utils.common import device, mae, mape, r2, rmse, weighted_pinball_loss

    quantiles = cfg.get("quantiles", QUANTILES)
    model = HierarchicalWQuantileGRU(
        hidden_size=rounded_hidden_size(cfg["hidden"]),
        num_layers=int(cfg["layers"]),
        dropout=float(cfg["dropout"]),
        n_quantiles=len(quantiles),
        vocab_sizes=cfg["vocab_sizes"],
        embed_dim=int(cfg.get("embed_dim", 8)),
        horizon=output_size(cfg),
        feature_index=cfg["feature_index"],
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
