# =============================================================================
# test.py — Forecast Evaluation & Visualisation — V3
# COMP0197 Applied Deep Learning
# GenAI Note: Scaffolded with Claude (Anthropic). Verified by authors.
#
# Infrastructure runner — reads all config from YAML files.
# The only line you should ever edit here is RUN_NAME below.
#
# Usage:
#   python test.py
#   python test.py --run_name sales_only_top200
#   python test.py --experiment configs/experiment.yml
# =============================================================================

import argparse
import json
import sys
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[0]))

from utils.config_loader import (
    load_experiment,
    load_registry,
    load_model_config,
    resolve_registry_entry,
    get_model_run_dir,
)
from utils.data import build_dataloaders, denormalise, get_feature_cols

PROJECT_DIR     = Path(__file__).resolve().parent
EXPERIMENT_PATH = PROJECT_DIR / "configs" / "experiment.yml"
REGISTRY_PATH   = PROJECT_DIR / "configs" / "registry.yml"

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
QUANTILES = [0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975]

# =============================================================================
# THE ONLY LINE YOU EDIT — which run to evaluate
# Must match the run_name used during training
# Can also be overridden via --run_name CLI argument
# =============================================================================
RUN_NAME = "baseline_ablation_sales_only"

print(f"Using device: {DEVICE}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="M5 Test V3")
    p.add_argument("--run_name",   type=str, default=None,
                   help="Override RUN_NAME (default: value set in test.py)")
    p.add_argument("--experiment", type=str, default=str(EXPERIMENT_PATH),
                   help="Path to experiment.yml (default: configs/experiment.yml)")
    return p.parse_args()


# =============================================================================
# REVENUE WEIGHTS
# =============================================================================

def compute_item_weights(data_dir: str, top_k: int, sampling: str = "all") -> np.ndarray:
    sales_df  = pd.read_csv(f"{data_dir}/sales_train_evaluation.csv")
    prices_df = pd.read_csv(f"{data_dir}/sell_prices.csv")

    day_cols = [c for c in sales_df.columns if c.startswith("d_")]
    sales_df = sales_df.copy()

    avg_prices = prices_df.groupby(["item_id", "store_id"])["sell_price"].mean().reset_index()
    sales_df = sales_df.merge(avg_prices, on=["item_id", "store_id"], how="left")
    sales_df["sell_price"] = sales_df["sell_price"].fillna(prices_df["sell_price"].median())

    train_day_cols = day_cols[:-28]
    sales_df["total_sales_volume"] = sales_df[train_day_cols].sum(axis=1)
    item_revenues = sales_df["total_sales_volume"].values * sales_df["sell_price"].values

    # Sort to match test_loader order: alphabetical by id (item_id+store_id concatenated)
    # data.py sorts featured by ['id', 'date'] where 'id' = item_id_store_id string
    sales_df["_series_id"] = sales_df["item_id"] + "_" + sales_df["store_id"]
    sales_df["_revenue"] = item_revenues
    sales_df = sales_df.sort_values("_series_id").reset_index(drop=True)

    if sampling != "all":
        # for top-k: need to replicate trim_data's top-k-by-volume selection
        # then sort that subset alphabetically
        vol_sorted = sales_df.sort_values("total_sales_volume", ascending=False)
        sales_df = vol_sorted.head(top_k).sort_values("_series_id").reset_index(drop=True)

    weights = sales_df["_revenue"].values
    weights = weights / weights.sum()
    return weights.astype(np.float32)


# =============================================================================
# TFT EVALUATION METRICS
# WSPL (Weighted Scaled Pinball Loss) and partial WRMSSE.
# Called only from the is_tft branch of evaluate_model().
# All functions are pure numpy/pandas — no new imports needed.
# =============================================================================

def _tft_pinball(y_true, y_pred, q):
    diff = y_true - y_pred
    return np.where(diff >= 0, q * diff, (1 - q) * (-diff))


def _tft_series_scales(train_df, group_col="series_id", target_col="sales", time_col="time_idx"):
    scales = {}
    for sid, g in train_df.sort_values([group_col, time_col]).groupby(group_col):
        y = g[target_col].values
        diffs = np.abs(np.diff(y))
        scale = diffs.mean() if len(diffs) > 0 else 0.0
        scales[sid] = scale if scale > 0 else 1.0
    return scales


def _tft_series_weights(train_df, price_col="sell_price", group_col="series_id",
                        target_col="sales", time_col="time_idx", last_n=28):
    weights = {}
    for sid, g in train_df.sort_values([group_col, time_col]).groupby(group_col):
        tail = g.tail(last_n).copy()
        tail["revenue"] = tail[target_col] * tail[price_col]
        weights[sid] = tail["revenue"].sum()
    total = sum(weights.values())
    if total == 0:
        n = len(weights)
        return {k: 1.0 / n for k in weights}
    return {k: v / total for k, v in weights.items()}


def compute_wspl(pred_df, quantiles, scales, weights):
    series_scores = []
    for sid, g in pred_df.groupby("series_id"):
        scale = scales[sid]
        weight = weights[sid]
        q_scores = []
        for q in quantiles:
            pb = _tft_pinball(g["actual"].values, g[f"q_{q}"].values, q)
            q_scores.append(pb.mean() / scale)
        series_scores.append(weight * np.mean(q_scores))
    return float(np.sum(series_scores))


def _tft_aggregate_forecasts(df, group_cols):
    if len(group_cols) == 0:
        agg = df.groupby("horizon_step", as_index=False)[["actual", "point_forecast"]].sum()
        agg["agg_id"] = "TOTAL"
        return agg
    agg = df.groupby(group_cols + ["horizon_step"], as_index=False)[["actual", "point_forecast"]].sum()
    agg["agg_id"] = agg[group_cols].astype(str).agg("_".join, axis=1)
    return agg


def _tft_rmsse_scales(train_df, group_cols, target_col="sales", time_col="time_idx"):
    scales = {}
    if len(group_cols) == 0:
        s = train_df.groupby(time_col)[target_col].sum().sort_index().values
        denom = np.mean(np.diff(s) ** 2) if len(s) > 1 else 0.0
        scales["TOTAL"] = denom if denom > 0 else 1.0
        return scales
    grouped = train_df.groupby(group_cols + [time_col], as_index=False)[target_col].sum()
    for key, g in grouped.groupby(group_cols):
        y = g.sort_values(time_col)[target_col].values
        denom = np.mean(np.diff(y) ** 2) if len(y) > 1 else 0.0
        agg_id = "_".join(map(str, key)) if isinstance(key, tuple) else str(key)
        scales[agg_id] = denom if denom > 0 else 1.0
    return scales


def _tft_wrmsse_weights(train_df, group_cols, price_col="sell_price",
                        target_col="sales", time_col="time_idx", last_n=28):
    weights = {}
    tail_threshold = train_df[time_col].max() - last_n + 1
    tail = train_df[train_df[time_col] >= tail_threshold].copy()
    tail["revenue"] = tail[target_col] * tail[price_col]
    if len(group_cols) == 0:
        weights["TOTAL"] = tail["revenue"].sum()
    else:
        grouped = tail.groupby(group_cols)["revenue"].sum()
        for key, value in grouped.items():
            agg_id = "_".join(map(str, key)) if isinstance(key, tuple) else str(key)
            weights[agg_id] = value
    total = sum(weights.values())
    if total == 0:
        n = len(weights)
        return {k: 1.0 / n for k in weights}
    return {k: v / total for k, v in weights.items()}


def _tft_wrmsse_level(agg_pred_df, scales, weights):
    scores = []
    for agg_id, g in agg_pred_df.groupby("agg_id"):
        mse = np.mean((g["actual"].values - g["point_forecast"].values) ** 2)
        rmsse = np.sqrt(mse / scales[agg_id])
        scores.append(weights[agg_id] * rmsse)
    return float(np.sum(scores))


def compute_partial_wrmsse(pred_with_meta_df, train_df, levels):
    level_scores = []
    for group_cols in levels:
        agg = _tft_aggregate_forecasts(pred_with_meta_df, group_cols)
        scales = _tft_rmsse_scales(train_df, group_cols)
        weights = _tft_wrmsse_weights(train_df, group_cols)
        score = _tft_wrmsse_level(agg, scales, weights)
        level_scores.append(score)
    return float(np.mean(level_scores))


def _tft_plot_quantiles(actuals, preds, quantiles, save_path=None, item_name="Series"):
    """Plot quantile forecast bands for a single TFT series."""
    q_idx = {q: i for i, q in enumerate(quantiles)}
    x = np.arange(len(actuals))
    import matplotlib.pyplot as _plt
    fig, ax = _plt.subplots(figsize=(12, 5))
    ax.plot(x, actuals, label="Actual", color="black", linewidth=2)
    ax.plot(x, preds[:, q_idx[0.5]], label="Median", color="blue", linewidth=2)
    ax.fill_between(x, preds[:, q_idx[0.005]], preds[:, q_idx[0.995]], alpha=0.10, label="99% CI")
    ax.fill_between(x, preds[:, q_idx[0.025]], preds[:, q_idx[0.975]], alpha=0.15, label="95% CI")
    ax.fill_between(x, preds[:, q_idx[0.165]], preds[:, q_idx[0.835]], alpha=0.20, label="67% CI")
    ax.fill_between(x, preds[:, q_idx[0.25]], preds[:, q_idx[0.75]], alpha=0.30, label="50% CI")
    ax.set_title(f"TFT Quantile Forecast — {item_name}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Horizon step");
    ax.set_ylabel("Sales (units)")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8)
    ax.grid(alpha=0.3)
    _plt.tight_layout()
    if save_path:
        _plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    _plt.close()


# =============================================================================
# NB HELPER
# =============================================================================

def nb_params_to_quantiles(mu, alpha, quantiles, n_samples=1000):
    mu_t     = torch.tensor(mu,    dtype=torch.float32).clamp(min=1e-6)
    alpha_t  = torch.tensor(alpha, dtype=torch.float32).clamp(min=1e-6)
    variance = mu_t + alpha_t * mu_t ** 2
    p        = (mu_t / variance).clamp(min=1e-6, max=1 - 1e-6)
    r        = (mu_t * p / (1 - p)).clamp(min=1e-6)
    dist     = torch.distributions.NegativeBinomial(total_count=r, probs=p)
    samples  = dist.sample((n_samples,)).float()
    q_vals   = torch.quantile(
        samples, torch.tensor(quantiles, dtype=torch.float32), dim=0
    )
    return q_vals.permute(1, 0).cpu().numpy()

# =============================================================================
# CRPS HELPERS
# =============================================================================

SQRT_PI = np.sqrt(np.pi)
SQRT_2  = np.sqrt(2.0)

def _normal_pdf(z):
    return np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)

def _normal_cdf(z):
    from math import erf
    return 0.5 * (1 + np.vectorize(erf)(z / SQRT_2))

def crps_gaussian(mu, sigma, y):
    """
    Exact closed-form CRPS for Gaussian predictive distribution.
    mu, sigma, y: numpy arrays of same shape — must be in original (denormalised) space.
    Returns per-element CRPS, call .mean() for scalar.
    """
    sigma = np.clip(sigma, 1e-6, None)
    z     = (y - mu) / sigma
    return sigma * (z * (2 * _normal_cdf(z) - 1) + 2 * _normal_pdf(z) - 1 / SQRT_PI)

def crps_from_quantiles(y, q_preds, quantiles):
    """
    Approximate CRPS via pinball loss sum over quantile levels.
    CRPS ≈ 2 * mean_over_quantiles(pinball_loss)
    y       : (N, H)    — targets in original space
    q_preds : (N, H, Q) — predicted quantiles in original space
    quantiles: list of Q floats
    Returns scalar.
    """
    y_exp = y[:, :, np.newaxis]  # (N, H, 1)
    q = np.array(quantiles)[np.newaxis, np.newaxis, :]  # (1, 1, Q)
    diff = y_exp - q_preds  # (N, H, Q)
    loss = np.maximum(q * diff, (q - 1) * diff)  # (N, H, Q)
    weights = np.diff(np.concatenate(([0.0], quantiles)))  # (Q,) spacing
    return float((loss * weights[np.newaxis, np.newaxis, :]).sum(axis=2).mean())


# =============================================================================
# AUTOREGRESSIVE INFERENCE
# =============================================================================

def predict_autoregressive(model, seed_window, horizon, is_prob, is_nb, device,
                           is_quantile=False, quantile_median_idx=3):
    model.eval()
    current_window = seed_window.copy()
    preds, aux_list = [], []

    for _ in range(horizon):
        x_t = torch.tensor(
            current_window, dtype=torch.float32
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            if is_nb:
                mu, alpha = model(x_t)
                step_pred = float(mu.cpu().numpy()[0, 0])
                step_aux  = float(alpha.cpu().numpy()[0, 0])
            elif is_prob:
                mu, sigma = model(x_t)
                step_pred = float(mu.cpu().numpy()[0, 0])
                step_aux  = float(sigma.cpu().numpy()[0, 0])
            elif is_quantile:
                # output is (B, Q) — use median as point forecast,
                # store full quantile row for confidence intervals
                out       = model(x_t).cpu().numpy()[0]         # (Q,)
                step_pred = float(out[quantile_median_idx])
                step_aux  = out                                  # (Q,) — all quantiles
            else:
                out       = model(x_t)
                step_pred = float(out.cpu().numpy()[0, 0])
                step_aux  = None

        step_pred = max(0.0, step_pred)
        preds.append(step_pred)
        if step_aux is not None:
            aux_list.append(step_aux)

        new_row    = current_window[-1].copy()
        new_row[0] = step_pred
        if current_window.shape[1] > 5:  # sales_hierarchy_dow has 6 features
            new_row[5] = (current_window[-1, 5] + 1) % 7
        current_window = np.vstack([current_window[1:], new_row])

    return np.array(preds), (np.array(aux_list) if aux_list else None)
    # for quantile models aux is (horizon, Q) — matches direct inference shape


# =============================================================================
# PLOTTING
# =============================================================================

def plot_quantile_forecast(historical_y, true_y, pred_quantiles, quantiles_list,
                           item_name="Series", save_path=None, rmse=None):
    seq_len   = len(historical_y)
    time_hist = np.arange(seq_len)
    time_pred = np.arange(seq_len, seq_len + len(true_y))

    idx_025 = quantiles_list.index(0.025)
    idx_050 = quantiles_list.index(0.05)
    idx_250 = quantiles_list.index(0.25)
    idx_500 = quantiles_list.index(0.5)
    idx_750 = quantiles_list.index(0.75)
    idx_950 = quantiles_list.index(0.95)
    idx_975 = quantiles_list.index(0.975)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(time_hist, historical_y, label="Historical Data",
            color="black", linewidth=1.5, marker=".")
    ax.plot(time_pred, true_y, label="Actual Future",
            color="black", linestyle="--", linewidth=1.5, marker="x")
    ax.axvline(x=seq_len, color="red", linestyle=":", linewidth=2,
               label="Forecast Start (Unseen Data)")
    ax.fill_between(time_pred, pred_quantiles[:, idx_025],
                    pred_quantiles[:, idx_975], color="blue", alpha=0.10, label="95% CI")
    ax.fill_between(time_pred, pred_quantiles[:, idx_050],
                    pred_quantiles[:, idx_950], color="blue", alpha=0.20, label="90% CI")
    ax.fill_between(time_pred, pred_quantiles[:, idx_250],
                    pred_quantiles[:, idx_750], color="blue", alpha=0.35, label="50% CI")
    ax.plot(time_pred, pred_quantiles[:, idx_500],
            label="Median Forecast", color="blue", linewidth=2, marker="o")

    title = f"Quantile Forecast — {item_name}"
    if rmse is not None:
        title += f"  (RMSE={rmse:.2f})"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Days"); ax.set_ylabel("Sales (units)")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def plot_det_forecast(historical_y, true_y, pred_y,
                      item_name="Series", save_path=None, rmse=None):
    seq_len   = len(historical_y)
    time_hist = np.arange(seq_len)
    time_pred = np.arange(seq_len, seq_len + len(true_y))

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(time_hist, historical_y, label="Historical Data",
            color="black", linewidth=1.5, marker=".")
    ax.plot(time_pred, true_y, label="Actual Future",
            color="black", linestyle="--", linewidth=1.5, marker="x")
    ax.axvline(x=seq_len, color="red", linestyle=":", linewidth=2,
               label="Forecast Start (Unseen Data)")
    ax.plot(time_pred, pred_y, label="Prediction",
            color="blue", linewidth=2, marker="o")

    title = f"Deterministic Forecast — {item_name}"
    if rmse is not None:
        title += f"  (RMSE={rmse:.2f})"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Days"); ax.set_ylabel("Sales (units)")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def plot_training_curves(hist_path, model_name, save_path=None):
    with open(hist_path) as f:
        saved = json.load(f)
    metrics    = saved["metrics"]["epoch_metrics"]
    epochs     = [m["epoch"]          for m in metrics]
    train_loss = [m["train_loss"]      for m in metrics]
    val_loss   = [m["validation_loss"] for m in metrics]
    has_rmse   = "val_rmse" in metrics[0]
    has_r2     = "val_r2"   in metrics[0]
    n_plots    = 1 + int(has_rmse) + int(has_r2)
    fig, axes  = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    axes[0].plot(epochs, train_loss, label="Train Loss", color="black", linewidth=2)
    axes[0].plot(epochs, val_loss,   label="Val Loss",   color="blue",
                 linewidth=2, linestyle="--")
    axes[0].set_title("Loss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    col = 1
    if has_rmse:
        axes[col].plot(epochs, [m["val_rmse"] for m in metrics], color="blue", linewidth=2)
        axes[col].set_title("Val RMSE", fontsize=12, fontweight="bold")
        axes[col].set_xlabel("Epoch"); axes[col].set_ylabel("RMSE")
        axes[col].grid(alpha=0.3); col += 1
    if has_r2:
        axes[col].plot(epochs, [m["val_r2"] for m in metrics], color="blue", linewidth=2)
        axes[col].set_title("Val R²", fontsize=12, fontweight="bold")
        axes[col].set_xlabel("Epoch"); axes[col].set_ylabel("R²")
        axes[col].grid(alpha=0.3)
    plt.suptitle(f"{model_name} — Training History", fontsize=13,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


# =============================================================================
# SINGLE MODEL EVALUATION
# =============================================================================

def evaluate_model(model_name: str, run_dir: Path,
                   exp_eval: dict, registry: dict) -> dict | None:
    """
    Evaluate a single model from a run directory.

    Data settings come from experiment.yml eval block — not the saved
    model config — so we always evaluate on the full dataset regardless
    of what subset the model was trained or searched on.
    """
    model_dir  = get_model_run_dir(run_dir, model_name)
    model_path = model_dir / f"{model_name}_model.pt"
    hist_path  = model_dir / f"{model_name}_train_history.json"

    if not model_path.exists():
        print(f"  [SKIP] {model_name} — model file not found at {model_path}")
        return None
    if not hist_path.exists():
        print(f"  [SKIP] {model_name} — history file not found at {hist_path}")
        return None

    # resolve builder and flags from registry
    if model_name not in registry:
        print(f"  [SKIP] {model_name} — not in registry.yml")
        return None
    resolved    = resolve_registry_entry(registry[model_name])
    builder     = resolved["builder"]
    is_prob     = resolved["is_prob"]
    is_nb       = resolved["is_nb"]
    is_quantile = resolved["is_quantile"]
    is_tft = resolved.get('is_tft', False)

    # load saved config for model architecture params
    with open(hist_path) as f:
        saved = json.load(f)
    cfg = saved.get("config", {})
    if not cfg:
        print(f"  [SKIP] {model_name} — no config in history JSON")
        return None

    # model params from saved config; data params from eval block
    autoregressive = bool(cfg.get("autoregressive", True))
    horizon        = int(cfg.get("horizon",         28))
    seq_len        = int(cfg.get("seq_len",          28))
    use_normalise  = bool(cfg.get("use_normalise",  False))
    # feature_set for data loading must match training — a sales_only model
    # cannot accept sales_hierarchy_dow inputs. eval block can override
    # top_k_series and sampling (which series to eval on) but NOT feature_set.
    feature_set    = str(cfg.get("feature_set", "sales_only"))
    top_k_series   = int(exp_eval.get("top_k_series",   30490))
    sampling       = str(exp_eval.get("sampling",        "all"))
    data_dir       = str(exp_eval.get("data_dir",    "./data"))

    if not is_tft:
        # load test data — autoregressive=False to get full 28-day targets
        _, _, test_loader, stats, _ = build_dataloaders(
            data_dir       = data_dir,
            seq_len        = seq_len,
            horizon        = horizon,
            batch_size     = int(cfg.get("batch_size", 256)),
            top_k_series   = top_k_series,
            feature_set    = feature_set,
            autoregressive = False,
            use_normalise  = use_normalise,
            sampling       = sampling,
            zscore_target  = not is_prob,
            max_series     = cfg.get("max_series"),
            num_workers    = 0,
            seed           = int(cfg.get("seed", 42)),
        )

    # n_features must match what the model was trained with — use saved cfg,
    # not the eval feature_set (which controls data loading only)
    train_feature_set  = str(cfg.get("feature_set", "sales_only"))

    # ------------------------------------------------------------------
    # TFT EVALUATION — separate path
    # Uses native predict() for correct index/actual reconstruction.
    # Computes standard metrics from median quantile + WSPL + WRMSSE.
    # ------------------------------------------------------------------
    if is_tft:
        from utils.data import build_tft_dataframe, build_tft_dataloaders, load_or_download_m5
        from utils.network import build_tft

        # rebuild TFT data (same function as train.py — deterministic given same cfg)
        sales_df_tft, cal_df_tft, prices_df_tft = load_or_download_m5(
            cfg.get('data_dir', './data'))
        tft_df = build_tft_dataframe(sales_df_tft, cal_df_tft, prices_df_tft, cfg)
        _, val_loader_tft, _, training_dataset = build_tft_dataloaders(tft_df, cfg)

        # reload model weights into a fresh TFT instance
        model_tft, _, _, _ = build_tft(cfg, training_dataset)
        model_tft.load_state_dict(
            torch.load(model_path, map_location=DEVICE, weights_only=True))
        model_tft.to(DEVICE)
        model_tft.eval()

        # native predict() — handles denorm, index, series alignment
        predictions = model_tft.predict(
            val_loader_tft, mode='raw', return_x=True, return_index=True)
        pred_tensor_raw = predictions.output['prediction'].detach().cpu().numpy()  # (N, H, Q)
        actuals_raw = predictions.x['decoder_target'].detach().cpu().numpy()  # (N, H)
        index = predictions.index

        # build pred_df (one row per series x horizon_step)
        quantiles = cfg.get('quantiles', [0.005, 0.025, 0.165, 0.25, 0.5,
                                          0.75, 0.835, 0.975, 0.995])
        median_idx = quantiles.index(0.5)
        rows_out = []
        N, H, Q = pred_tensor_raw.shape
        for i in range(N):
            sid = index.iloc[i]['series_id']
            for t in range(H):
                row = {'series_id': sid,
                       'time_idx': int(index.iloc[i]['time_idx']) + t,
                       'horizon_step': t + 1,
                       'actual': float(actuals_raw[i, t]),
                       'point_forecast': float(pred_tensor_raw[i, t, median_idx])}
                for j, q in enumerate(quantiles):
                    row[f'q_{q}'] = float(pred_tensor_raw[i, t, j])
                rows_out.append(row)
        pred_df = pd.DataFrame(rows_out)

        # point metrics from median quantile
        preds_orig = pred_df['point_forecast'].values
        targets_orig = pred_df['actual'].values
        rmse_val = float(np.sqrt(((preds_orig - targets_orig) ** 2).mean()))
        mae_val = float(np.abs(preds_orig - targets_orig).mean())
        mask = targets_orig > 0
        mape_val = float(np.abs((preds_orig[mask] - targets_orig[mask]) /
                                targets_orig[mask]).mean() * 100) if mask.any() else float('nan')
        ss_res = float(((targets_orig - preds_orig) ** 2).sum())
        ss_tot = float(((targets_orig - targets_orig.mean()) ** 2).sum())
        r2_val = float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
        print(f'  RMSE={rmse_val:.4f}  MAE={mae_val:.4f}  MAPE={mape_val:.2f}%  R2={r2_val:.4f}')
        metrics_dict = {'model': model_name, 'rmse': rmse_val, 'mae': mae_val,
                        'mape': mape_val, 'r2': r2_val}

        # weighted metrics
        item_weights = compute_item_weights(
            data_dir=cfg.get('data_dir', './data'),
            top_k=int(cfg.get('n_series', 200)),
            sampling='top',
        )
        per_item_mse = pred_df.groupby('series_id').apply(
            lambda g: ((g['point_forecast'] - g['actual']) ** 2).mean(), include_groups=False).values
        per_item_mae = pred_df.groupby('series_id').apply(
            lambda g: (g['point_forecast'] - g['actual']).abs().mean(), include_groups=False).values
        # item_weights has one entry per series; per_item arrays are grouped the same way
        w_rmse = float(np.sqrt((item_weights * per_item_mse).mean()))
        w_mae = float((item_weights * per_item_mae).mean())
        print(f'  W-RMSE={w_rmse:.4f}  W-MAE={w_mae:.4f}')
        metrics_dict['w_rmse'] = w_rmse
        metrics_dict['w_mae'] = w_mae

        # WSPL and partial WRMSSE
        training_cutoff = tft_df['time_idx'].max() - int(cfg.get('max_prediction_length', 28))
        train_hist_df = tft_df[tft_df.time_idx <= training_cutoff].copy()
        scales = _tft_series_scales(train_hist_df)
        weights = _tft_series_weights(train_hist_df)
        wspl = compute_wspl(pred_df, quantiles, scales, weights)
        print(f'  WSPL={wspl:.4f}')
        metrics_dict['wspl'] = wspl

        # CRPS for TFT — approximate via pinball sum over quantile grid
        # reshape pred_df into (N, H, Q) array for crps_from_quantiles
        q_cols = [f'q_{q}' for q in quantiles]
        n_series = pred_df['series_id'].nunique()
        q_arr = pred_df[q_cols].values.reshape(n_series, H, len(quantiles))  # (N, H, Q)
        t_arr = pred_df['actual'].values.reshape(n_series, H)  # (N, H)
        crps_tft = crps_from_quantiles(t_arr, q_arr, quantiles)
        print(f'  CRPS={crps_tft:.4f}')
        metrics_dict['crps'] = crps_tft

        meta_cols = ['series_id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        series_meta = tft_df[meta_cols].drop_duplicates('series_id')
        pred_meta = pred_df.merge(series_meta, on='series_id', how='left')
        wrmsse_levels = [[], ['cat_id'], ['dept_id'], ['item_id']]
        wrmsse = compute_partial_wrmsse(pred_meta, train_hist_df, wrmsse_levels)
        print(f'  Partial-WRMSSE={wrmsse:.4f}')
        metrics_dict['partial_wrmsse'] = wrmsse

        # save metrics JSON
        output_dir = model_dir / 'plots'
        output_dir.mkdir(exist_ok=True)
        with open(model_dir / f'{model_name}_test_metrics.json', 'w') as f:
            json.dump(metrics_dict, f, indent=2)

        # forecast plots — 5 representative series
        labels_plot = ['best', 'good', 'median', 'poor', 'worst']
        rmse_per = pred_df.groupby('series_id').apply(
            lambda g: np.sqrt(((g['point_forecast'] - g['actual']) ** 2).mean()), include_groups=False)
        sorted_series = rmse_per.sort_values().index.tolist()
        n_s = len(sorted_series)
        plot_series = [sorted_series[0], sorted_series[n_s // 4], sorted_series[n_s // 2],
                       sorted_series[3 * n_s // 4], sorted_series[-1]]
        for i, sid in enumerate(plot_series):
            sg = pred_df[pred_df.series_id == sid].sort_values('horizon_step')
            acts = sg['actual'].values
            q_cols = [f'q_{q}' for q in quantiles]
            pmat = sg[q_cols].values
            _tft_plot_quantiles(acts, pmat, quantiles,
                                save_path=output_dir / f'{model_name}_sample_{labels_plot[i]}.png',
                                item_name=labels_plot[i].capitalize())

        return metrics_dict  # early return — skip the rest of evaluate_model()


    cfg["n_features"]  = len(get_feature_cols(train_feature_set))
    model, _, _, _     = builder(cfg)
    model.load_state_dict(
        torch.load(model_path, map_location=DEVICE, weights_only=True)
    )
    model.to(DEVICE)
    model.eval()
    output_dir = model_dir / "plots"
    output_dir.mkdir(exist_ok=True)

    plot_training_curves(hist_path, model_name,
                         save_path=output_dir / f"{model_name}_training_curves.png")

    # quantile metadata
    quantiles = cfg.get("quantiles", QUANTILES)
    median_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2

    # inference
    all_inputs, all_preds, all_targets, all_aux = [], [], [], []

    if autoregressive:
        with torch.no_grad():
            for x, y in test_loader:
                all_inputs.append(x)
                all_targets.append(y)
                batch_preds, batch_aux = [], []
                for b in range(x.shape[0]):
                    seed = x[b].numpy()
                    p, a = predict_autoregressive(
                        model, seed, horizon, is_prob, is_nb, DEVICE,
                        is_quantile=is_quantile, quantile_median_idx=median_idx,
                    )
                    batch_preds.append(p)
                    if a is not None:
                        batch_aux.append(a)
                all_preds.append(torch.tensor(np.array(batch_preds)))
                if batch_aux:
                    all_aux.append(torch.tensor(np.array(batch_aux)))
    else:
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(DEVICE)
                if is_nb:
                    mu, alpha = model(x)
                    all_preds.append(mu.cpu())
                    all_aux.append(alpha.cpu())
                elif is_prob:
                    mu, sigma = model(x)
                    all_preds.append(mu.cpu())
                    all_aux.append(sigma.cpu())
                elif is_quantile:
                    # output (B, Q) — store full quantile tensor for coverage/plots
                    # median stored separately for point metrics
                    q_out = model(x).cpu()             # (B, Q)
                    all_preds.append(q_out[:, :, median_idx])  # (B, H) — median quantile across all horizon steps
                    all_aux.append(q_out)  # (B, H, Q) — full quantile tensor
                else:
                    all_preds.append(model(x).cpu())
                all_inputs.append(x.cpu())
                all_targets.append(y)

    inputs_t   = torch.cat(all_inputs)
    preds_np   = torch.cat(all_preds).numpy()
    targets_np = torch.cat(all_targets).numpy()
    aux_np     = torch.cat(all_aux).numpy() if all_aux else None

    # denormalise
    if use_normalise:
        if is_nb:
            targets_orig = np.clip(targets_np, 0, 1e6)
            preds_orig   = np.clip(preds_np,   0, 1e6)
        elif is_prob:
            targets_orig = np.expm1(np.clip(targets_np, 0, 12.0))
            preds_orig   = np.expm1(np.clip(preds_np,   0, 12.0))
        else:
            targets_orig = denormalise(targets_np, stats)
            preds_orig   = denormalise(preds_np,   stats)
    else:
        targets_orig = targets_np.copy()
        preds_orig   = preds_np.copy()

    targets_orig = np.clip(targets_orig, 0, 1e6)
    preds_orig   = np.clip(preds_orig,   0, 1e6)

    # standard metrics
    rmse   = float(np.sqrt(((preds_orig - targets_orig) ** 2).mean()))
    mae    = float(np.abs(preds_orig - targets_orig).mean())
    mask   = targets_orig > 0
    mape   = float(np.abs(
        (preds_orig[mask] - targets_orig[mask]) / targets_orig[mask]
    ).mean() * 100) if mask.any() else float("nan")
    ss_res = float(((targets_orig - preds_orig) ** 2).sum())
    ss_tot = float(((targets_orig - targets_orig.mean()) ** 2).sum())
    r2     = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    print(f"  RMSE={rmse:.4f}  MAE={mae:.4f}  MAPE={mape:.2f}%  R²={r2:.4f}")
    metrics_dict = {"model": model_name, "rmse": rmse, "mae": mae, "mape": mape, "r2": r2}

    # weighted metrics
    item_weights = compute_item_weights(
        data_dir = data_dir,
        top_k    = top_k_series,
        sampling = sampling,
    )
    per_item_mse = ((preds_orig - targets_orig) ** 2).mean(axis=1)
    per_item_mae = np.abs(preds_orig - targets_orig).mean(axis=1)
    w_rmse = float(np.sqrt((item_weights * per_item_mse).mean()))
    w_mae  = float((item_weights * per_item_mae).mean())
    print(f"  W-RMSE={w_rmse:.4f}  W-MAE={w_mae:.4f}")
    metrics_dict["w_rmse"] = w_rmse
    metrics_dict["w_mae"]  = w_mae

    # coverage for probabilistic models
    if is_nb and aux_np is not None:
        lower_all, upper_all, q_all_list = [], [], []
        for i in range(len(preds_np)):
            q = nb_params_to_quantiles(preds_np[i], aux_np[i], QUANTILES, n_samples=200)
            lower_all.append(q[:, QUANTILES.index(0.025)])
            upper_all.append(q[:, QUANTILES.index(0.975)])
            q_all_list.append(q)
        lower_orig = np.clip(np.array(lower_all), 0, 1e6)
        upper_orig = np.clip(np.array(upper_all), 0, 1e6)
        q_all = np.stack(q_all_list)  # (N, H, Q)
        coverage = float(((targets_orig >= lower_orig) & (targets_orig <= upper_orig)).mean())
        width = float((upper_orig - lower_orig).mean())
        crps_nb = crps_from_quantiles(targets_orig, q_all, QUANTILES)
        print(f"  Coverage(95%)={coverage:.4f}  Width={width:.4f}  CRPS={crps_nb:.4f}")
        metrics_dict["coverage_95"] = coverage
        metrics_dict["interval_width"] = width
        metrics_dict["crps"] = crps_nb

    elif is_prob and aux_np is not None:
        if use_normalise:
            lower_orig = np.clip(np.expm1((preds_np - 1.96 * aux_np).clip(0, 12.0)), 0, 1e6)
            upper_orig = np.clip(np.expm1((preds_np + 1.96 * aux_np).clip(0, 12.0)), 0, 1e6)
        else:
            lower_orig = np.clip(preds_orig - 1.96 * aux_np, 0, 1e6)
            upper_orig = np.clip(preds_orig + 1.96 * aux_np, 0, 1e6)
        coverage = float(((targets_orig >= lower_orig) & (targets_orig <= upper_orig)).mean())
        width    = float((upper_orig - lower_orig).mean())
        print(f"  Coverage(95%)={coverage:.4f}  Width={width:.4f}")
        # CRPS for Gaussian — exact closed form, computed in original space
        # aux_np is sigma in training space — need to convert to original space
        if use_normalise:
            # sigma is in log1p space; approximate in original via delta method
            sigma_orig = aux_np * np.exp(np.clip(preds_np, 0, 12.0))
        else:
            sigma_orig = aux_np
        sigma_orig = np.clip(sigma_orig, 1e-6, None)
        crps_g = crps_gaussian(preds_orig, sigma_orig, targets_orig).mean()
        print(f"  CRPS={crps_g:.4f}")
        metrics_dict["crps"] = float(crps_g)
        metrics_dict["coverage_95"]    = coverage
        metrics_dict["interval_width"] = width

    elif is_quantile and aux_np is not None:
        # aux_np shape:
        #   autoregressive : (N, horizon, Q) — full quantile per step
        #   direct         : (N, Q)          — single-step quantile output
        idx_025 = quantiles.index(0.025) if 0.025 in quantiles else 0
        idx_975 = quantiles.index(0.975) if 0.975 in quantiles else -1
        if aux_np.ndim == 3:
            # autoregressive: lower/upper per timestep then mean over horizon
            lower_orig = np.clip(aux_np[:, :, idx_025], 0, 1e6)  # (N, horizon)
            upper_orig = np.clip(aux_np[:, :, idx_975], 0, 1e6)
        else:
            # direct: single quantile value per item
            lower_orig = np.clip(aux_np[:, idx_025:idx_025+1], 0, 1e6)
            upper_orig = np.clip(aux_np[:, idx_975:idx_975+1], 0, 1e6)
        coverage = float(((targets_orig >= lower_orig) & (targets_orig <= upper_orig)).mean())
        width    = float((upper_orig - lower_orig).mean())
        print(f"  Coverage(95%)={coverage:.4f}  Width={width:.4f}")
        # CRPS for quantile — approximate via pinball sum, only when (N, H, Q)
        if aux_np.ndim == 3:
            crps_q = crps_from_quantiles(targets_orig, aux_np, quantiles)
            print(f"  CRPS={crps_q:.4f}")
            metrics_dict["crps"] = crps_q
        metrics_dict["coverage_95"]    = coverage
        metrics_dict["interval_width"] = width

    # save per-model metrics
    with open(model_dir / f"{model_name}_test_metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=2)

    # forecast plots
    labels     = ["best", "good", "median", "poor", "worst"]
    rmse_per   = np.sqrt(((preds_orig - targets_orig) ** 2).mean(axis=1))
    sorted_idx = np.argsort(rmse_per)
    plot_indices = [
        sorted_idx[0],
        sorted_idx[len(sorted_idx) // 4],
        sorted_idx[len(sorted_idx) // 2],
        sorted_idx[3 * len(sorted_idx) // 4],
        sorted_idx[-1],
    ]

    hist_raw  = inputs_t[:, :, 0].numpy()
    hist_orig = np.clip(
        denormalise(hist_raw, stats, col="sales") if (use_normalise and stats) else hist_raw,
        0, 500
    )

    for i, idx in enumerate(plot_indices):
        hist      = hist_orig[idx]
        true      = np.clip(targets_orig[idx], 0, 500)
        pred      = np.clip(preds_orig[idx],   0, 500)
        r         = float(rmse_per[idx])
        name      = labels[i].capitalize()
        save_path = output_dir / f"{model_name}_sample_{labels[i]}.png"

        if is_nb and aux_np is not None:
            q = nb_params_to_quantiles(
                preds_np[idx], aux_np[idx], QUANTILES, n_samples=500
            )
            plot_quantile_forecast(hist, true, np.clip(q, 0, 500), QUANTILES,
                                   item_name=name, save_path=save_path, rmse=r)
        elif is_prob and aux_np is not None:
            mu_i    = preds_np[idx] if use_normalise else preds_orig[idx]
            sigma_i = aux_np[idx]
            z_vals  = [-1.96, -1.645, -0.674, 0.0, 0.674, 1.645, 1.96]
            if use_normalise:
                q_orig = np.clip(
                    np.expm1(
                        np.stack([mu_i + z * sigma_i for z in z_vals], axis=1)
                        .clip(0, 12.0)
                    ), 0, 500
                )
            else:
                q_orig = np.clip(
                    np.stack([mu_i + z * sigma_i for z in z_vals], axis=1),
                    0, 500
                )
            plot_quantile_forecast(hist, true, q_orig, QUANTILES,
                                   item_name=name, save_path=save_path, rmse=r)
        elif is_quantile and aux_np is not None:
            if aux_np.ndim == 3:
                # autoregressive: aux_np[idx] is (horizon, Q)
                q_orig = np.clip(aux_np[idx], 0, 500)
            else:
                # direct: aux_np[idx] is (Q,) — reshape to (1, Q) won't plot well
                # this case shouldn't occur since direct gives (N, Q) not (N, horizon, Q)
                q_orig = np.clip(aux_np[idx].reshape(1, -1).repeat(horizon, axis=0), 0, 500)
            plot_quantile_forecast(hist, true, q_orig, quantiles,
                                   item_name=name, save_path=save_path, rmse=r)
        else:
            plot_det_forecast(hist, true, pred,
                              item_name=name, save_path=save_path, rmse=r)

    return metrics_dict


# =============================================================================
# MAIN
# =============================================================================

def main():
    args     = parse_args()
    run_name = args.run_name or RUN_NAME
    run_dir  = PROJECT_DIR / "runs" / run_name

    if not run_dir.exists():
        raise FileNotFoundError(
            f"Run directory not found: {run_dir}\n"
            f"Have you run train.py with run_name='{run_name}'?"
        )

    # load from the run snapshot — guarantees we use the exact config
    # that was used during training, not whatever is currently in configs/
    exp_cfg  = load_experiment(run_dir / "configs" / "experiment.yml")
    registry = load_registry(REGISTRY_PATH)

    exp_eval = exp_cfg.get("eval", {})
    models   = exp_cfg.get("models", [])

    if not models:
        raise ValueError(f"No models in {run_dir}/configs/experiment.yml")

    print(f"\n{'='*60}")
    print(f"  BATCH EVALUATION — {run_name}")
    print(f"  Models   : {models}")
    print(f"  Eval data: sampling={exp_eval.get('sampling','all')} | "
          f"top_k={exp_eval.get('top_k_series',30490)} | "
          f"feature_set={exp_eval.get('feature_set','sales_only')}")
    print(f"{'='*60}")

    all_results = []

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"  Evaluating: {model_name}")
        print(f"{'='*60}")

        result = evaluate_model(
            model_name = model_name,
            run_dir    = run_dir,
            exp_eval   = exp_eval,
            registry   = registry,
        )
        if result is not None:
            all_results.append(result)

    if all_results:
        print(f"\n{'='*100}")
        print("  RESULTS SUMMARY")
        print(f"{'='*100}")
        print(f"{'Model':<25} {'RMSE':>8} {'W-RMSE':>8} {'MAE':>8} {'W-MAE':>8} {'R²':>8} {'CRPS':>8} {'WSPL':>8} {'WRMSSE':>8}")
        print("-" * 100)
        for r in all_results:
            print(
                f"{r['model']:<25} "
                f"{r['rmse']:>8.4f} "
                f"{r.get('w_rmse', float('nan')):>8.4f} "
                f"{r['mae']:>8.4f} "
                f"{r.get('w_mae', float('nan')):>8.4f} "
                f"{r['r2']:>8.4f} "
                f"{r.get('crps', float('nan')):>8.4f} "
                f"{r.get('wspl', float('nan')):>8.4f} "
                f"{r.get('partial_wrmsse', float('nan')):>8.4f}"
            )
        print("=" * 100)

        out_path = run_dir / "all_test_metrics.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Combined metrics saved: {out_path}")

    print(f"\n  Done — evaluated {len(all_results)}/{len(models)} models")


if __name__ == "__main__":
    main()