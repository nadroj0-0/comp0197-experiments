# =============================================================================
# test_gru_models.py — Wrapper-based evaluation runner
# COMP0197 Applied Deep Learning
#
# Evaluates every requested model in a run through the BaseModel wrappers.
#
# Usage:
#   python test_gru_models.py
#   python test_gru_models.py --run_name sales_only_top200
#   python test_gru_models.py --experiment configs/experiment.yml
# =============================================================================

import argparse
import json
import sys
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.config_loader import (
    load_experiment,
    load_registry,
    load_model_config,
    resolve_registry_entry,
    get_model_run_dir,
)
from utils.data import build_dataloaders, denormalise, get_feature_cols
from models import get_model_class, get_available_model_names

PROJECT_DIR     = Path(__file__).resolve().parents[1]
EXPERIMENT_PATH = PROJECT_DIR / "configs" / "experiment.yml"
REGISTRY_PATH   = PROJECT_DIR / "configs" / "registry.yml"

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
QUANTILES = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975]

# Default run name for quick local use. CLI can override it.
RUN_NAME = "baseline_ablation_sales_only"

print(f"Using device: {DEVICE}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="M5 Test V3")
    p.add_argument("--run_name",   type=str, default=None,
                   help="Override the default run name")
    p.add_argument("--experiment", type=str, default=str(EXPERIMENT_PATH),
                   help="Path to experiment.yml (default: configs/experiment.yml)")
    return p.parse_args()


# =============================================================================
# REVENUE WEIGHTS
# =============================================================================

def compute_item_weights(data_dir: str, top_k: int, sampling: str = "all",
                          last_n_days: int = 28) -> np.ndarray:
    """
    Compute revenue weights from the last training days, following the M5 setup.
    """
    sales_df    = pd.read_csv(f"{data_dir}/sales_train_evaluation.csv")
    calendar_df = pd.read_csv(f"{data_dir}/calendar.csv")
    prices_df   = pd.read_csv(f"{data_dir}/sell_prices.csv")

    day_cols   = [c for c in sales_df.columns if c.startswith("d_")]
    total_days = len(day_cols)

    # M5 train_end = d_1773 (same boundary as Yen)
    train_end = total_days - 6 * 28   # 1773

    # Melt wide → long for the last 28 training days only
    last28_cols = day_cols[train_end - last_n_days : train_end]
    id_cols     = [c for c in sales_df.columns if not c.startswith("d_")]
    last28_long = sales_df[id_cols + last28_cols].melt(
        id_vars=id_cols, var_name="d", value_name="sales"
    )

    # Merge wm_yr_wk from calendar so we can join prices
    last28_long = last28_long.merge(
        calendar_df[["d", "wm_yr_wk"]], on="d", how="left"
    )

    # Merge daily prices
    last28_long = last28_long.merge(
        prices_df[["store_id", "item_id", "wm_yr_wk", "sell_price"]],
        on=["store_id", "item_id", "wm_yr_wk"], how="left"
    )
    last28_long["sell_price"] = (
        last28_long.groupby("id")["sell_price"]
        .transform(lambda x: x.ffill().fillna(0.0))
    )

    # Revenue = daily sales * daily price, summed over last 28 days
    train_rev = (
        (last28_long["sales"] * last28_long["sell_price"])
        .groupby(last28_long["id"]).sum()
    )

    # Align to the series order the test_loader uses
    # WindowedM5Dataset groups by 'id' — order matches featured["id"].drop_duplicates()
    # which is alphabetical after sort_values(["id","date"])
    all_ids = sorted(sales_df["id"].unique())

    if sampling != "all":
        # top-k by total training volume, then sort alphabetically
        vol = sales_df[day_cols[:train_end]].sum(axis=1)
        sales_df["_vol"] = vol.values
        top_ids = set(
            sales_df.sort_values("_vol", ascending=False)
            .head(top_k)["id"]
        )
        all_ids = sorted(i for i in all_ids if i in top_ids)

    weights = train_rev.reindex(all_ids).fillna(0.0).values.astype(np.float32)
    total   = weights.sum()
    if total > 0:
        weights /= total
    return weights


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
    Evaluate a single trained wrapper model through the BaseModel inference API.
    """
    if model_name not in registry:
        print(f"  [SKIP] {model_name} — not in registry.yml")
        return None

    try:
        wrapper_cls = get_model_class(model_name)
    except KeyError:
        print(f"  [SKIP] {model_name} — no BaseModel wrapper — "
              f"available wrappers: {get_available_model_names()}")
        return None

    wrapper = wrapper_cls(
        data_dir   = str(exp_eval.get("data_dir", "./data")),
        output_dir = "./outputs",
        run_name   = run_dir.name,
        do_search  = False,
    )
    return wrapper.run_inference_pipeline()


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
        print(f"{'Model':<25} {'RMSE':>8} {'W-RMSE':>8} {'MAE':>8} {'W-MAE':>8} {'R²':>8} {'CRPS':>8}")
        print("-" * 80)
        for r in all_results:
            print(
                f"{r['model']:<25} "
                f"{r['rmse']:>8.4f} "
                f"{r.get('w_rmse', float('nan')):>8.4f} "
                f"{r['mae']:>8.4f} "
                f"{r.get('w_mae', float('nan')):>8.4f} "
                f"{r['r2']:>8.4f} "
                f"{r.get('crps', float('nan')):>8.4f}"
            )
        print("=" * 70)

        out_path = run_dir / "all_test_metrics.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Combined metrics saved: {out_path}")

    print(f"\n  Done — evaluated {len(all_results)}/{len(models)} models")


if __name__ == "__main__":
    main()
