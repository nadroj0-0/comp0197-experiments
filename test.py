# =============================================================================
# test.py — Forecast Evaluation & Visualisation — V3
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
    build_prob_gru_nb, build_baseline_gru, build_baseline_prob_gru,
    build_baseline_prob_gru_nb,
)
from utils.data import build_dataloaders, denormalise
from utils.training_strategies import gru_step, prob_gru_step, prob_nb_step
import pandas as pd

# =============================================================================
# CONFIGURATION — add/remove entries to control which models get evaluated
# =============================================================================

TEST_CONFIGS = [
    {"model_type": "baseline_gru", "probabilistic": False},
    {"model_type": "baseline_gru", "probabilistic": True},
    {"model_type": "gru",         "probabilistic": False},
    {"model_type": "gru",         "probabilistic": True},
    {"model_type": "lstm",        "probabilistic": False},
    {"model_type": "lstm",        "probabilistic": True},
    {"model_type": "transformer", "probabilistic": False},
    {"model_type": "transformer", "probabilistic": True},
    {"model_type": "baseline_gru_nb", "probabilistic": True},
    {"model_type": "gru_nb",      "probabilistic": True},
]

QUANTILES  = [0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975]
MODELS_DIR = Path("./models")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# =============================================================================
# MODEL REGISTRY — (builder, training_step, is_prob, is_nb)
# =============================================================================

MODEL_REGISTRY = {
    "baseline_gru_det": (build_baseline_gru,      gru_step,      False, False),
    "baseline_gru_prob":(build_baseline_prob_gru, prob_gru_step, True,  False),
    "gru_det":          (build_gru,              gru_step,       False, False),
    "lstm_det":         (build_lstm,             gru_step,       False, False),
    "transformer_det":  (build_transformer,      gru_step,       False, False),
    "gru_prob":         (build_prob_gru,         prob_gru_step,  True,  False),
    "lstm_prob":        (build_prob_lstm,        prob_gru_step,  True,  False),
    "transformer_prob": (build_prob_transformer, prob_gru_step,  True,  False),
    "baseline_gru_nb_prob": (build_baseline_prob_gru_nb, prob_nb_step, True, True),
    "gru_nb_prob":      (build_prob_gru_nb,      prob_nb_step,   True,  True),
}

def model_name_from_cfg(cfg):
    is_prob = cfg["probabilistic"]
    model_type = cfg["model_type"]
    suffix = "prob" if is_prob else "det"
    return f"{model_type}_{suffix}"


# =============================================================================
# REVENUE WEIGHTS — matches teammate's methodology exactly
# weight_i = (train_volume_i * avg_price_i) / sum(all revenues)
# =============================================================================

def compute_item_weights(data_dir: str, top_k: int) -> np.ndarray:
    """
    Compute per-item revenue weights for the top-k series.
    Matches teammate's preprocess_lstm_data_with_revenue_weights exactly:
      - Select top-k items by total sales volume
      - Merge average sell price per (item_id, store_id)
      - weight_i = volume_i * price_i, normalised to sum to 1
    Returns array of shape (top_k,) in the same order as the data pipeline.
    """
    sales_df  = pd.read_csv(f"{data_dir}/sales_train_evaluation.csv")
    prices_df = pd.read_csv(f"{data_dir}/sell_prices.csv")

    # Replicate trim_data — top-k by total volume
    day_cols = [c for c in sales_df.columns if c.startswith("d_")]
    sales_df["total_sales_volume"] = sales_df[day_cols].sum(axis=1)
    sales_df = (
        sales_df.sort_values("total_sales_volume", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )

    # Average price per (item_id, store_id) — same as teammate
    avg_prices = (
        prices_df.groupby(["item_id", "store_id"])["sell_price"]
        .mean()
        .reset_index()
    )
    sales_df = sales_df.merge(avg_prices, on=["item_id", "store_id"], how="left")
    sales_df["sell_price"] = sales_df["sell_price"].fillna(prices_df["sell_price"].median())

    # Train split volume — exclude last 28 days (test window), same as teammate
    train_day_cols = day_cols[:-28]
    item_volumes   = sales_df[train_day_cols].sum(axis=1).values
    item_revenues  = item_volumes * sales_df["sell_price"].values
    item_weights   = item_revenues / item_revenues.sum()  # normalise to sum to 1

    return item_weights.astype(np.float32)


# =============================================================================
# NB HELPER
# =============================================================================

def nb_params_to_quantiles(mu, alpha, quantiles, n_samples=1000):
    mu_t    = torch.tensor(mu,    dtype=torch.float32).clamp(min=1e-6)
    alpha_t = torch.tensor(alpha, dtype=torch.float32).clamp(min=1e-6)
    variance = mu_t + alpha_t * mu_t ** 2
    p = (mu_t / variance).clamp(min=1e-6, max=1 - 1e-6)
    r = (mu_t * p / (1 - p)).clamp(min=1e-6)
    dist   = torch.distributions.NegativeBinomial(total_count=r, probs=p)
    samples = dist.sample((n_samples,)).float()
    q_vals  = torch.quantile(
        samples,
        torch.tensor(quantiles, dtype=torch.float32),
        dim=0,
    )
    return q_vals.permute(1, 0).cpu().numpy()  # (horizon, Q)


# =============================================================================
# AUTOREGRESSIVE INFERENCE
# =============================================================================

def predict_autoregressive(model, seed_window, horizon, is_prob, is_nb, device):
    """
    Recursive 28-step rollout from a seed window.
    seed_window : (seq_len, n_features) numpy array
    Returns preds (horizon,) and aux (horizon,) or None
    """
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
            else:
                out = model(x_t)
                step_pred = float(out.cpu().numpy()[0, 0])
                step_aux  = None

        step_pred = max(0.0, step_pred)
        preds.append(step_pred)
        if step_aux is not None:
            aux_list.append(step_aux)

        # Slide window — update sales (index 0) with prediction
        new_row    = current_window[-1].copy()
        new_row[0] = step_pred
        current_window = np.vstack([current_window[1:], new_row])

    return np.array(preds), (np.array(aux_list) if aux_list else None)


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
    axes[0].plot(epochs, val_loss,   label="Val Loss",   color="blue",  linewidth=2, linestyle="--")
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

def evaluate_model(model_name, builder, is_prob, is_nb):
    model_dir  = MODELS_DIR / model_name
    model_path = model_dir / f"{model_name}_model.pt"
    hist_path  = model_dir / f"{model_name}_train_history.json"

    if not model_path.exists():
        print(f"  [SKIP] {model_name} — model file not found")
        return None
    if not hist_path.exists():
        print(f"  [SKIP] {model_name} — history file not found")
        return None

    # Load config
    with open(hist_path) as f:
        saved = json.load(f)
    cfg = saved.get("config", {})
    if not cfg:
        print(f"  [SKIP] {model_name} — no config in history JSON")
        return None

    autoregressive = cfg.get("autoregressive", False)
    horizon        = cfg.get("horizon", 28)
    seq_len        = cfg.get("seq_len",  28)
    use_normalise  = cfg.get("use_normalise", False)

    # Load test data — always autoregressive=False to get full 28-day targets
    _, _, test_loader, stats = build_dataloaders(
        data_dir       = cfg.get("data_dir",     "./data"),
        seq_len        = seq_len,
        horizon        = horizon,
        batch_size     = cfg.get("batch_size",   256),
        top_k_series   = cfg.get("top_k_series", 200),
        feature_set    = cfg.get("feature_set",  "sales_only"),
        autoregressive = False,
        use_normalise  = use_normalise,
        zscore_target  = not is_prob,
        max_series     = cfg.get("max_series"),
        num_workers    = 0,
        seed           = cfg.get("seed", 42),
    )

    # Load model
    model, _, _, _ = builder(cfg)
    model.load_state_dict(
        torch.load(model_path, map_location=DEVICE, weights_only=True)
    )
    model.to(DEVICE)
    model.eval()

    # Plots dir
    output_dir = model_dir / "plots"
    output_dir.mkdir(exist_ok=True)

    plot_training_curves(hist_path, model_name,
                         save_path=output_dir / f"{model_name}_training_curves.png")

    # Inference
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
                        model, seed, horizon, is_prob, is_nb, DEVICE
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
                else:
                    all_preds.append(model(x).cpu())
                all_inputs.append(x.cpu())
                all_targets.append(y)

    inputs_t   = torch.cat(all_inputs)
    preds_np   = torch.cat(all_preds).numpy()
    targets_np = torch.cat(all_targets).numpy()
    aux_np     = torch.cat(all_aux).numpy() if all_aux else None

    # Denormalise
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

    # Metrics
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

    # Weighted metrics — revenue weights matching teammate's methodology
    item_weights = compute_item_weights(
        data_dir = cfg.get("data_dir", "./data"),
        top_k    = cfg.get("top_k_series", 200),
    )
    # item_weights: (N,)  preds/targets: (N, horizon)
    per_item_mse = ((preds_orig - targets_orig) ** 2).mean(axis=1)  # (N,)
    per_item_mae = np.abs(preds_orig - targets_orig).mean(axis=1)  # (N,)
    w_rmse = float(np.sqrt((item_weights * per_item_mse).mean()))
    w_mae = float((item_weights * per_item_mae).mean())
    print(f"  W-RMSE={w_rmse:.4f}  W-MAE={w_mae:.4f}")
    metrics_dict["w_rmse"] = w_rmse
    metrics_dict["w_mae"]  = w_mae

    # Coverage for probabilistic models
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

    # Save metrics
    with open(model_dir / f"{model_name}_test_metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=2)

    # Forecast plots — 5 representative series
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
            q = nb_params_to_quantiles(preds_np[idx], aux_np[idx], QUANTILES, n_samples=500)
            plot_quantile_forecast(hist, true, np.clip(q, 0, 500), QUANTILES,
                                   item_name=name, save_path=save_path, rmse=r)
        elif is_prob and aux_np is not None:
            mu_i    = preds_np[idx] if use_normalise else preds_orig[idx]
            sigma_i = aux_np[idx]
            z_vals  = [-1.96, -1.645, -0.674, 0.0, 0.674, 1.645, 1.96]
            if use_normalise:
                q_orig = np.clip(
                    np.expm1(np.stack([mu_i + z * sigma_i for z in z_vals], axis=1).clip(0, 12.0)),
                    0, 500
                )
            else:
                q_orig = np.clip(
                    np.stack([mu_i + z * sigma_i for z in z_vals], axis=1),
                    0, 500
                )
            plot_quantile_forecast(hist, true, q_orig, QUANTILES,
                                   item_name=name, save_path=save_path, rmse=r)
        else:
            plot_det_forecast(hist, true, pred,
                              item_name=name, save_path=save_path, rmse=r)

    return metrics_dict


# =============================================================================
# MAIN — loops over all TEST_CONFIGS
# =============================================================================

def main():
    print("=" * 60)
    print("  BATCH EVALUATION — V3")
    print("=" * 60)

    all_results = []

    for override in TEST_CONFIGS:
        model_name = model_name_from_cfg(override)
        is_prob    = override["probabilistic"]
        is_nb      = override["model_type"] == "gru_nb"

        if model_name not in MODEL_REGISTRY:
            print(f"\n[SKIP] {model_name} — not in MODEL_REGISTRY")
            continue

        builder, _, _, _ = MODEL_REGISTRY[model_name]

        print(f"\n{'='*60}")
        print(f"  Evaluating: {model_name}")
        print(f"{'='*60}")

        result = evaluate_model(model_name, builder, is_prob, is_nb)
        if result is not None:
            all_results.append(result)

    # Summary table
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

        # Save combined results
        with open(MODELS_DIR / "all_test_metrics.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Combined metrics saved: {MODELS_DIR / 'all_test_metrics.json'}")

    print(f"\n✅  Done — evaluated {len(all_results)}/{len(TEST_CONFIGS)} models")


if __name__ == "__main__":
    main()