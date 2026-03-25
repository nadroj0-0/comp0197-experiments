# =============================================================================
# test.py — Forecast Evaluation & Visualisation
# COMP0197 Applied Deep Learning
# GenAI Note: Scaffolded with Claude (Anthropic). Verified by authors.
# =============================================================================

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from utils.network import (
    build_gru, build_lstm, build_transformer,
    build_prob_gru, build_prob_lstm, build_prob_transformer,
    build_prob_gru_nb,
)
from utils.data import build_dataloaders, denormalise
from utils.training_strategies import gru_step, prob_gru_step, prob_nb_step

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_NAME = "gru_nb_direct_prob"
# Options:
#   gru_direct_det          — deterministic GRU
#   lstm_direct_det         — deterministic LSTM
#   transformer_direct_det  — deterministic Transformer
#   gru_direct_prob         — probabilistic GRU (Gaussian NLL)
#   lstm_direct_prob        — probabilistic LSTM (Gaussian NLL)
#   transformer_direct_prob — probabilistic Transformer (Gaussian NLL)
#   gru_nb_direct_prob      — probabilistic GRU (Negative Binomial NLL)

QUANTILES = [0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975]

MODEL_DIR  = Path("./models") / MODEL_NAME
MODEL_PATH = MODEL_DIR / f"{MODEL_NAME}_model.pt"
HIST_PATH  = MODEL_DIR / f"{MODEL_NAME}_train_history.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# =============================================================================
# MODEL REGISTRY — (builder, training_step, is_prob, is_nb)
# =============================================================================

MODEL_REGISTRY = {
    "gru_direct_det":           (build_gru,              gru_step,       False, False),
    "lstm_direct_det":          (build_lstm,             gru_step,       False, False),
    "transformer_direct_det":   (build_transformer,      gru_step,       False, False),
    "gru_direct_prob":          (build_prob_gru,         prob_gru_step,  True,  False),
    "lstm_direct_prob":         (build_prob_lstm,        prob_gru_step,  True,  False),
    "transformer_direct_prob":  (build_prob_transformer, prob_gru_step,  True,  False),
    "gru_nb_direct_prob":       (build_prob_gru_nb,      prob_nb_step,   True,  True),
}

# =============================================================================
# NB HELPER — (mu, alpha) parameters -> quantiles via sampling
# =============================================================================

def nb_params_to_quantiles(mu, alpha, quantiles, n_samples=1000):
    """
    Convert NB (mu, alpha) parameters to quantile predictions.
    mu, alpha : (horizon,) numpy arrays — count space, > 0
    returns   : (horizon, Q) numpy array
    """
    mu_t    = torch.tensor(mu,    dtype=torch.float32).clamp(min=1e-6)
    alpha_t = torch.tensor(alpha, dtype=torch.float32).clamp(min=1e-6)

    variance = mu_t + alpha_t * mu_t ** 2
    p = (mu_t / variance).clamp(min=1e-6, max=1 - 1e-6)
    r = (mu_t * p / (1 - p)).clamp(min=1e-6)

    dist    = torch.distributions.NegativeBinomial(total_count=r, probs=p)
    samples = dist.sample((n_samples,)).float()  # (n_samples, horizon)

    q_vals = torch.quantile(
        samples,
        torch.tensor(quantiles, dtype=torch.float32),
        dim=0,
    )  # (Q, horizon)

    return q_vals.permute(1, 0).cpu().numpy()  # (horizon, Q)


# =============================================================================
# PLOTTING — teammate's exact style
# =============================================================================

def plot_quantile_forecast(historical_y, true_y, pred_quantiles, quantiles_list,
                           item_name="Series", save_path=None, rmse=None):
    """
    Teammate's notebook style:
    - Black solid: historical data
    - Black dashed: actual future
    - Red dotted vertical: forecast start
    - Blue fills: 95%, 90%, 50% CI
    - Blue solid: median forecast
    - Legend outside right
    """
    seq_len  = len(historical_y)
    pred_len = len(true_y)
    time_hist = np.arange(seq_len)
    time_pred = np.arange(seq_len, seq_len + pred_len)

    idx_025 = quantiles_list.index(0.025)
    idx_050 = quantiles_list.index(0.05)
    idx_250 = quantiles_list.index(0.25)
    idx_500 = quantiles_list.index(0.5)
    idx_750 = quantiles_list.index(0.75)
    idx_950 = quantiles_list.index(0.95)
    idx_975 = quantiles_list.index(0.975)

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(time_hist, historical_y,
            label="Historical Data", color="black", linewidth=1.5)
    ax.plot(time_pred, true_y,
            label="Actual Future", color="black", linestyle="--", linewidth=1.5)
    ax.axvline(x=seq_len, color="red", linestyle=":",
               linewidth=2, label="Forecast Start (Unseen Data)")

    ax.fill_between(time_pred,
                    pred_quantiles[:, idx_025], pred_quantiles[:, idx_975],
                    color="blue", alpha=0.10, label="95% CI")
    ax.fill_between(time_pred,
                    pred_quantiles[:, idx_050], pred_quantiles[:, idx_950],
                    color="blue", alpha=0.20, label="90% CI")
    ax.fill_between(time_pred,
                    pred_quantiles[:, idx_250], pred_quantiles[:, idx_750],
                    color="blue", alpha=0.35, label="50% CI")
    ax.plot(time_pred, pred_quantiles[:, idx_500],
            label="Median Forecast", color="blue", linewidth=2)

    title = f"Quantile Forecast — {item_name}"
    if rmse is not None:
        title += f"  (RMSE={rmse:.2f})"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Time Steps", fontsize=11)
    ax.set_ylabel("Sales (units)", fontsize=11)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def plot_det_forecast(historical_y, true_y, pred_y,
                      item_name="Series", save_path=None, rmse=None):
    """Deterministic version of teammate's style."""
    seq_len  = len(historical_y)
    pred_len = len(true_y)
    time_hist = np.arange(seq_len)
    time_pred = np.arange(seq_len, seq_len + pred_len)

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(time_hist, historical_y,
            label="Historical Data", color="black", linewidth=1.5)
    ax.plot(time_pred, true_y,
            label="Actual Future", color="black", linestyle="--", linewidth=1.5)
    ax.axvline(x=seq_len, color="red", linestyle=":",
               linewidth=2, label="Forecast Start (Unseen Data)")
    ax.plot(time_pred, pred_y,
            label="Prediction", color="blue", linewidth=2)

    title = f"Deterministic Forecast — {item_name}"
    if rmse is not None:
        title += f"  (RMSE={rmse:.2f})"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Time Steps", fontsize=11)
    ax.set_ylabel("Sales (units)", fontsize=11)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def plot_training_curves(hist_path, save_path=None):
    """Train/val loss + extra metrics from history JSON."""
    with open(hist_path) as f:
        saved = json.load(f)

    history = saved["metrics"]
    metrics = history["epoch_metrics"]
    epochs     = [m["epoch"]          for m in metrics]
    train_loss = [m["train_loss"]      for m in metrics]
    val_loss   = [m["validation_loss"] for m in metrics]

    has_rmse = "val_rmse" in metrics[0]
    has_r2   = "val_r2"   in metrics[0]

    n_plots = 1 + int(has_rmse) + int(has_r2)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    axes[0].plot(epochs, train_loss, label="Train Loss", color="black", linewidth=2)
    axes[0].plot(epochs, val_loss,   label="Val Loss",   color="blue",  linewidth=2, linestyle="--")
    axes[0].set_title("Loss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    col = 1
    if has_rmse:
        val_rmse = [m["val_rmse"] for m in metrics]
        axes[col].plot(epochs, val_rmse, color="blue", linewidth=2)
        axes[col].set_title("Val RMSE", fontsize=12, fontweight="bold")
        axes[col].set_xlabel("Epoch"); axes[col].set_ylabel("RMSE")
        axes[col].grid(alpha=0.3)
        col += 1

    if has_r2:
        val_r2 = [m["val_r2"] for m in metrics]
        axes[col].plot(epochs, val_r2, color="blue", linewidth=2)
        axes[col].set_title("Val R²", fontsize=12, fontweight="bold")
        axes[col].set_xlabel("Epoch"); axes[col].set_ylabel("R²")
        axes[col].grid(alpha=0.3)

    plt.suptitle(f"{MODEL_NAME} — Training History",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print(f"  TESTING: {MODEL_NAME}")
    print("=" * 60)

    if MODEL_NAME not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{MODEL_NAME}'. Options: {list(MODEL_REGISTRY)}"
        )

    builder, training_step, is_prob, is_nb = MODEL_REGISTRY[MODEL_NAME]

    # -------------------------------------------------------------------------
    # 1. LOAD CONFIG
    # -------------------------------------------------------------------------
    print("\n[1/4] Loading config and data...")
    with open(HIST_PATH) as f:
        saved = json.load(f)
    cfg = saved.get("config", {})
    if not cfg:
        raise RuntimeError(
            f"No 'config' key found in {HIST_PATH}. "
            "Ensure full_train saves config into history JSON."
        )

    # -------------------------------------------------------------------------
    # 2. LOAD TEST DATA
    # -------------------------------------------------------------------------
    _, _, test_loader, stats = build_dataloaders(
        data_dir      = cfg.get("data_dir",   "./data"),
        seq_len       = cfg.get("seq_len",    56),
        horizon       = cfg.get("horizon",    28),
        batch_size    = cfg.get("batch_size", 256),
        store_id      = cfg.get("store_id",   "CA_3"),
        max_series    = cfg.get("max_series"),
        num_workers   = 0,
        seed          = cfg.get("seed",       42),
        zscore_target = not is_prob,
    )

    # -------------------------------------------------------------------------
    # 3. LOAD MODEL
    # -------------------------------------------------------------------------
    print("\n[2/4] Loading model weights...")
    model, _, _, _ = builder(cfg)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    )
    model.to(DEVICE)
    model.eval()
    print(f"  Loaded: {MODEL_PATH}")

    output_dir = MODEL_DIR / "plots"
    output_dir.mkdir(exist_ok=True)

    # training curves plot
    plot_training_curves(
        HIST_PATH,
        save_path=output_dir / f"{MODEL_NAME}_training_curves.png",
    )

    # -------------------------------------------------------------------------
    # 4. INFERENCE
    # -------------------------------------------------------------------------
    print("\n[3/4] Running inference on test set...")

    horizon = cfg.get("horizon", 28)
    seq_len = cfg.get("seq_len",  28)

    all_inputs, all_preds, all_targets = [], [], []
    all_aux = []

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
            else:
                out = model(x)
                all_preds.append(out.cpu())
            all_inputs.append(x.cpu())
            all_targets.append(y)

    inputs_t  = torch.cat(all_inputs)
    preds_t   = torch.cat(all_preds)
    targets_t = torch.cat(all_targets)
    preds_np   = preds_t.numpy()
    targets_np = targets_t.numpy()

    if is_prob or is_nb:
        aux_np = torch.cat(all_aux).numpy()

    print(f"  preds_np range: {preds_np.min():.3f} to {preds_np.max():.3f}")
    print(f"  targets_np range: {targets_np.min():.3f} to {targets_np.max():.3f}")

    # -------------------------------------------------------------------------
    # 5. DENORMALISE TO ORIGINAL SALES UNITS
    # -------------------------------------------------------------------------
    if is_nb:
        # targets are raw counts (zscore_target=False kept them untransformed)
        # mu from softplus is also raw counts
        targets_orig = np.clip(targets_np, 0, 1e6)
        preds_orig = np.clip(preds_np, 0, 1e6)
    elif is_prob:
        # Gaussian: targets and preds in log1p space
        targets_orig = np.expm1(np.clip(targets_np, 0, 12.0))
        preds_orig   = np.expm1(np.clip(preds_np,   0, 12.0))
    else:
        # deterministic: z-scored log1p — use denormalise()
        targets_orig = denormalise(targets_np, stats)
        preds_orig   = denormalise(preds_np,   stats)

    targets_orig = np.clip(targets_orig, 0, 1e6)
    preds_orig   = np.clip(preds_orig,   0, 1e6)

    # -------------------------------------------------------------------------
    # 6. METRICS
    # -------------------------------------------------------------------------
    rmse = float(np.sqrt(((preds_orig - targets_orig) ** 2).mean()))
    mae  = float(np.abs(preds_orig - targets_orig).mean())
    mask = targets_orig > 0
    mape = float(
        np.abs((preds_orig[mask] - targets_orig[mask]) / targets_orig[mask]).mean() * 100
    ) if mask.any() else float("nan")
    ss_res = float(((targets_orig - preds_orig) ** 2).sum())
    ss_tot = float(((targets_orig - targets_orig.mean()) ** 2).sum())
    r2     = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    print(f"\nTest metrics (original sales units):")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  MAPE : {mape:.2f}%")
    print(f"  R²   : {r2:.4f}")

    metrics_dict = {"model": MODEL_NAME, "rmse": rmse, "mae": mae, "mape": mape, "r2": r2}

    if is_nb:
        print("  Computing NB 95% coverage (sampling)...")
        lower_all, upper_all = [], []
        for i in range(len(preds_np)):
            q = nb_params_to_quantiles(preds_np[i], aux_np[i], QUANTILES, n_samples=200)
            lower_all.append(q[:, QUANTILES.index(0.025)])
            upper_all.append(q[:, QUANTILES.index(0.975)])
        lower_orig = np.clip(np.array(lower_all), 0, 1e6)
        upper_orig = np.clip(np.array(upper_all), 0, 1e6)
        coverage = float(((targets_orig >= lower_orig) & (targets_orig <= upper_orig)).mean())
        width    = float((upper_orig - lower_orig).mean())
        print(f"  Coverage (95%) : {coverage:.4f}  (target ≈ 0.95)")
        print(f"  Interval width : {width:.4f} units")
        metrics_dict["coverage_95"]    = coverage
        metrics_dict["interval_width"] = width

    elif is_prob:
        lower_log = preds_np - 1.96 * aux_np
        upper_log = preds_np + 1.96 * aux_np
        lower_orig = np.clip(np.expm1(lower_log.clip(0, 12.0)), 0, 1e6)
        upper_orig = np.clip(np.expm1(upper_log.clip(0, 12.0)), 0, 1e6)
        coverage = float(((targets_orig >= lower_orig) & (targets_orig <= upper_orig)).mean())
        width    = float((upper_orig - lower_orig).mean())
        print(f"  Coverage (95%) : {coverage:.4f}  (target ≈ 0.95)")
        print(f"  Interval width : {width:.4f} units")
        metrics_dict["coverage_95"]    = coverage
        metrics_dict["interval_width"] = width

    metrics_path = MODEL_DIR / f"{MODEL_NAME}_test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"\n  Metrics saved: {metrics_path}")

    # -------------------------------------------------------------------------
    # 7. FORECAST PLOTS — 5 representative series (best → worst by RMSE)
    # -------------------------------------------------------------------------
    print("\n[4/4] Generating forecast plots...")

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

    # Use lag_7 feature (index 9) as historical sales proxy for display
    # FEATURE_COLS = [sell_price(0), wday_sin(1), wday_cos(2), month_sin(3),
    #                 month_cos(4), has_event(5), snap_CA(6), snap_TX(7),
    #                 snap_WI(8), lag_7(9), lag_28(10), roll_mean_7(11), roll_mean_28(12)]
    LAG7_IDX = 9
    hist_norm = inputs_t[:, :, LAG7_IDX].numpy()
    if is_prob or is_nb:
        # lag_7 is z-scored log1p — denormalise using lag_7 stats
        hist_orig = denormalise(hist_norm, stats, col="lag_7") \
            if "lag_7" in stats else np.zeros_like(hist_norm)
    else:
        hist_orig = denormalise(hist_norm, stats, col="lag_7") \
            if "lag_7" in stats else np.zeros_like(hist_norm)
    hist_orig = np.clip(hist_orig, 0, 500)

    for i, idx in enumerate(plot_indices):
        hist = hist_orig[idx]
        true = np.clip(targets_orig[idx], 0, 500)
        pred = np.clip(preds_orig[idx],   0, 500)
        r    = float(rmse_per[idx])
        name = labels[i].capitalize()
        save_path = output_dir / f"{MODEL_NAME}_sample_{labels[i]}.png"

        if is_nb:
            q = nb_params_to_quantiles(preds_np[idx], aux_np[idx], QUANTILES, n_samples=500)
            q = np.clip(q, 0, 500)
            plot_quantile_forecast(
                hist, true, q, QUANTILES,
                item_name=name, save_path=save_path, rmse=r,
            )
        elif is_prob:
            sigma_i = aux_np[idx]
            mu_i    = preds_np[idx]
            z_scores = [-1.96, -1.645, -0.674, 0.0, 0.674, 1.645, 1.96]
            q_log  = np.stack([mu_i + z * sigma_i for z in z_scores], axis=1)
            q_orig = np.clip(np.expm1(q_log.clip(0, 12.0)), 0, 500)
            plot_quantile_forecast(
                hist, true, q_orig, QUANTILES,
                item_name=name, save_path=save_path, rmse=r,
            )
        else:
            plot_det_forecast(
                hist, true, pred,
                item_name=name, save_path=save_path, rmse=r,
            )

    print(f"\n  Plots saved in: {output_dir}/")
    print(f"\n✅  Done — {MODEL_NAME}")


# =============================================================================

if __name__ == "__main__":
    main()