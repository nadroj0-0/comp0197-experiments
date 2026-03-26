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

def compute_item_weights(data_dir: str, top_k: int,
                         sampling: str = "all") -> np.ndarray:
    """
    Compute per-item revenue weights matching the eval dataset exactly.
    sampling="all" → all series; sampling="top" → top-k by volume.
    """
    sales_df  = pd.read_csv(f"{data_dir}/sales_train_evaluation.csv")
    prices_df = pd.read_csv(f"{data_dir}/sell_prices.csv")

    day_cols = [c for c in sales_df.columns if c.startswith("d_")]
    sales_df = sales_df.copy()
    sales_df["total_sales_volume"] = sales_df[day_cols].sum(axis=1)
    sales_df = (
        sales_df.sort_values("total_sales_volume", ascending=False)
        .reset_index(drop=True)
    )

    if sampling != "all":
        sales_df = sales_df.head(top_k).reset_index(drop=True)

    avg_prices = (
        prices_df.groupby(["item_id", "store_id"])["sell_price"]
        .mean().reset_index()
    )
    sales_df = sales_df.merge(avg_prices, on=["item_id", "store_id"], how="left")
    sales_df["sell_price"] = sales_df["sell_price"].fillna(
        prices_df["sell_price"].median()
    )

    train_day_cols = day_cols[:-28]
    item_volumes   = sales_df[train_day_cols].sum(axis=1).values
    item_revenues  = item_volumes * sales_df["sell_price"].values
    item_weights   = item_revenues / item_revenues.sum()

    return item_weights.astype(np.float32)


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

    # load test data — autoregressive=False to get full 28-day targets
    _, _, test_loader, stats = build_dataloaders(
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
                    all_preds.append(q_out[:, median_idx:median_idx+1])  # (B, 1) median
                    all_aux.append(q_out)              # (B, Q) all quantiles
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
        lower_all, upper_all = [], []
        for i in range(len(preds_np)):
            q = nb_params_to_quantiles(preds_np[i], aux_np[i], QUANTILES, n_samples=200)
            lower_all.append(q[:, QUANTILES.index(0.025)])
            upper_all.append(q[:, QUANTILES.index(0.975)])
        lower_orig = np.clip(np.array(lower_all), 0, 1e6)
        upper_orig = np.clip(np.array(upper_all), 0, 1e6)
        coverage = float(((targets_orig >= lower_orig) & (targets_orig <= upper_orig)).mean())
        width    = float((upper_orig - lower_orig).mean())
        print(f"  Coverage(95%)={coverage:.4f}  Width={width:.4f}")
        metrics_dict["coverage_95"]    = coverage
        metrics_dict["interval_width"] = width

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
        print(f"\n{'='*70}")
        print("  RESULTS SUMMARY")
        print(f"{'='*70}")
        print(f"{'Model':<25} {'RMSE':>8} {'W-RMSE':>8} {'MAE':>8} {'W-MAE':>8} {'R²':>8}")
        print("-" * 70)
        for r in all_results:
            print(
                f"{r['model']:<25} "
                f"{r['rmse']:>8.4f} "
                f"{r.get('w_rmse', float('nan')):>8.4f} "
                f"{r['mae']:>8.4f} "
                f"{r.get('w_mae', float('nan')):>8.4f} "
                f"{r['r2']:>8.4f}"
            )
        print("=" * 70)

        out_path = run_dir / "all_test_metrics.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Combined metrics saved: {out_path}")

    print(f"\n  Done — evaluated {len(all_results)}/{len(models)} models")


if __name__ == "__main__":
    main()