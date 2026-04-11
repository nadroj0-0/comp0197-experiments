import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ..base_model import BaseModel
from utils.configs.config_loader import (
    get_model_run_dir,
    load_effective_train_config,
    load_experiment,
    load_registry,
    resolve_registry_entry,
)
from utils.data import encode_hierarchy, get_feature_cols, get_vocab_sizes

PROJECT_DIR = Path(__file__).resolve().parents[2]
REGISTRY_PATH = PROJECT_DIR / "configs" / "registry.yml"


def _build_featured_frame(self_model, feature_set: str | None = None) -> pd.DataFrame:
    featured = pd.concat([
        self_model.train_raw,
        self_model.val_raw,
        self_model.test_raw,
    ]).reset_index(drop=True)

    include_dow = str(feature_set) == "sales_hierarchy_dow"
    featured = encode_hierarchy(featured, include_dow=include_dow)
    featured["has_event"] = (
        (~featured["event_name_1"].astype(str).isin(["none", "nan", "None"]))
        .astype(np.float32)
    )

    if not pd.api.types.is_datetime64_any_dtype(featured["date"]):
        featured["date"] = pd.to_datetime(featured["date"])

    return featured.sort_values(["id", "d_num"]).reset_index(drop=True)


def _nb_params_to_quantiles(mu, alpha, quantiles, n_samples=1000):
    mu_t = torch.tensor(mu, dtype=torch.float32).clamp(min=1e-6)
    alpha_t = torch.tensor(alpha, dtype=torch.float32).clamp(min=1e-6)
    variance = mu_t + alpha_t * mu_t ** 2
    p = (mu_t / variance).clamp(min=1e-6, max=1 - 1e-6)
    r = (mu_t * p / (1 - p)).clamp(min=1e-6)
    dist = torch.distributions.NegativeBinomial(total_count=r, probs=p)
    samples = dist.sample((n_samples,)).float()
    q_vals = torch.quantile(
        samples, torch.tensor(quantiles, dtype=torch.float32), dim=0
    )
    return q_vals.movedim(0, -1).cpu().numpy()


def _gaussian_params_to_quantiles(mu, sigma, quantiles):
    mu_t = torch.tensor(mu, dtype=torch.float32)
    sigma_t = torch.tensor(sigma, dtype=torch.float32).clamp(min=1e-6)
    dist = torch.distributions.Normal(mu_t, sigma_t)
    qs = torch.tensor(quantiles, dtype=torch.float32).view(-1, 1, 1)
    q_vals = dist.icdf(qs)
    return q_vals.permute(1, 2, 0).cpu().numpy()


def _coerce_to_target_quantiles(pred_q, source_quantiles, target_quantiles):
    pred_q = np.asarray(pred_q, dtype=np.float32)
    source_quantiles = np.asarray(source_quantiles, dtype=np.float32)
    target_quantiles = np.asarray(target_quantiles, dtype=np.float32)

    if pred_q.ndim == 2:
        pred_q = pred_q[:, np.newaxis, :]

    if np.array_equal(source_quantiles, target_quantiles):
        return pred_q

    flat = pred_q.reshape(-1, pred_q.shape[-1])
    interp = np.empty((flat.shape[0], len(target_quantiles)), dtype=np.float32)
    for i, row in enumerate(flat):
        interp[i] = np.interp(target_quantiles, source_quantiles, row)
    return interp.reshape(pred_q.shape[0], pred_q.shape[1], len(target_quantiles))


def _finalise_quantiles(q_preds):
    q_preds = np.clip(np.asarray(q_preds, dtype=np.float32), 0.0, None)
    return np.maximum.accumulate(q_preds, axis=-1)


def _crps_from_quantiles(y, q_preds, quantiles):
    y_exp = y[:, :, np.newaxis]
    q = np.array(quantiles, dtype=np.float32)[np.newaxis, np.newaxis, :]
    diff = y_exp - q_preds
    loss = np.maximum(q * diff, (q - 1) * diff)
    weights = np.diff(np.concatenate(([0.0], np.array(quantiles, dtype=np.float32))))
    return float((loss * weights[np.newaxis, np.newaxis, :]).sum(axis=2).mean())


def _plot_quantile_forecast(historical_y, true_y, pred_quantiles, quantiles_list,
                            item_name="Series", save_path=None, rmse=None):
    seq_len = len(historical_y)
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
    ax.set_xlabel("Days")
    ax.set_ylabel("Sales (units)")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_det_forecast(historical_y, true_y, pred_y,
                       item_name="Series", save_path=None, rmse=None):
    seq_len = len(historical_y)
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
    ax.set_xlabel("Days")
    ax.set_ylabel("Sales (units)")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_training_curves(hist_path, model_name, save_path=None):
    if not hist_path.exists():
        return

    with open(hist_path) as f:
        saved = json.load(f)
    metrics = saved.get("metrics", {}).get("epoch_metrics", [])
    if not metrics:
        return

    epochs = [m["epoch"] for m in metrics]
    train_loss = [m["train_loss"] for m in metrics]
    val_loss = [m["validation_loss"] for m in metrics]
    has_rmse = "val_rmse" in metrics[0]
    has_r2 = "val_r2" in metrics[0]
    n_plots = 1 + int(has_rmse) + int(has_r2)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    axes[0].plot(epochs, train_loss, label="Train Loss", color="black", linewidth=2)
    axes[0].plot(epochs, val_loss, label="Val Loss", color="blue", linewidth=2, linestyle="--")
    axes[0].set_title("Loss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    col = 1
    if has_rmse:
        axes[col].plot(epochs, [m["val_rmse"] for m in metrics], color="blue", linewidth=2)
        axes[col].set_title("Val RMSE", fontsize=12, fontweight="bold")
        axes[col].set_xlabel("Epoch")
        axes[col].set_ylabel("RMSE")
        axes[col].grid(alpha=0.3)
        col += 1
    if has_r2:
        axes[col].plot(epochs, [m["val_r2"] for m in metrics], color="blue", linewidth=2)
        axes[col].set_title("Val R²", fontsize=12, fontweight="bold")
        axes[col].set_xlabel("Epoch")
        axes[col].set_ylabel("R²")
        axes[col].grid(alpha=0.3)

    plt.suptitle(f"{model_name} — Training History", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def _predict_autoregressive_quantiles(model, context_batch, future_feat_batch,
                                      horizon, is_prob, is_nb, is_quantile,
                                      quantiles, model_quantiles, device):
    sales_idx = 0
    median_idx = (
        model_quantiles.index(0.5)
        if (is_quantile and 0.5 in model_quantiles)
        else len(model_quantiles) // 2
    )

    q_forecasts = []
    with torch.no_grad():
        for seed_window, future_feats in zip(context_batch, future_feat_batch):
            current_window = seed_window.copy()
            series_q = []
            for step in range(horizon):
                x_t = torch.tensor(
                    current_window, dtype=torch.float32, device=device
                ).unsqueeze(0)

                if is_nb:
                    mu, alpha = model(x_t)
                    mu_np = mu.squeeze(0).cpu().numpy()
                    alpha_np = alpha.squeeze(0).cpu().numpy()
                    step_q = _nb_params_to_quantiles(mu_np[:1], alpha_np[:1], quantiles)[0]
                    point_pred = float(mu_np[0])
                elif is_prob:
                    mu, sigma = model(x_t)
                    mu_np = mu.squeeze(0).cpu().numpy()
                    sigma_np = sigma.squeeze(0).cpu().numpy()
                    step_q = _gaussian_params_to_quantiles(
                        mu_np[:1][np.newaxis, :], sigma_np[:1][np.newaxis, :], quantiles
                    )[0, 0]
                    point_pred = float(mu_np[0])
                elif is_quantile:
                    raw_q = model(x_t).squeeze(0).cpu().numpy()
                    if raw_q.ndim == 2:
                        raw_q = raw_q[0]
                    step_q = _coerce_to_target_quantiles(
                        raw_q[np.newaxis, np.newaxis, :], model_quantiles, quantiles
                    )[0, 0]
                    point_pred = float(raw_q[median_idx])
                else:
                    point_pred = float(model(x_t).squeeze(0).cpu().numpy()[0])
                    step_q = np.full(len(quantiles), point_pred, dtype=np.float32)

                series_q.append(step_q)
                new_row = future_feats[step].copy()
                new_row[sales_idx] = max(0.0, point_pred)
                current_window = np.vstack([current_window[1:], new_row])

            q_forecasts.append(series_q)

    return _finalise_quantiles(np.asarray(q_forecasts, dtype=np.float32))


def _load_run_training_config(model_name: str, run_name: str):
    run_dir = PROJECT_DIR / "runs" / run_name
    model_dir = get_model_run_dir(run_dir, model_name)
    hist_path = model_dir / f"{model_name}_train_history.json"
    run_model_yml = run_dir / "configs" / "models" / f"{model_name}.yml"
    run_experiment_yml = run_dir / "configs" / "experiment.yml"

    cfg = None
    if hist_path.exists():
        with open(hist_path) as f:
            saved = json.load(f)
        cfg = saved.get("config")

    if not cfg:
        exp_cfg = load_experiment(run_experiment_yml)
        cfg = load_effective_train_config(exp_cfg, run_model_yml)

    return run_dir, model_dir, hist_path, cfg


def predict_from_trained_model(self_model, model_name: str, run_name: str):
    run_dir, model_dir, _, cfg = _load_run_training_config(model_name, run_name)
    registry = load_registry(REGISTRY_PATH)
    resolved = resolve_registry_entry(registry[model_name])

    builder = resolved["builder"]
    is_prob = resolved["is_prob"]
    is_nb = resolved["is_nb"]
    is_quantile = resolved["is_quantile"]

    model_path = model_dir / f"{model_name}_model.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    feature_set = str(cfg.get("feature_set", "sales_yen_hierarchy"))
    self_model.load_and_split_data()
    featured = _build_featured_frame(self_model, feature_set=feature_set)

    feature_cols = get_feature_cols(feature_set)
    horizon = int(cfg.get("horizon", 28))
    seq_len = int(cfg.get("seq_len", 28))
    autoregressive = bool(cfg.get("autoregressive", True))
    target_quantiles = list(BaseModel.QUANTILES)
    model_quantiles = list(cfg.get("quantiles", target_quantiles))

    cfg = cfg.copy()
    cfg["n_features"] = len(feature_cols)
    cfg["vocab_sizes"] = (
        cfg.get("vocab_sizes")
        or (get_vocab_sizes(featured) if "hierarchy" in feature_set else {})
    )
    cfg["feature_index"] = cfg.get("feature_index") or {
        col: i for i, col in enumerate(feature_cols)
    }

    model, _, _, _ = builder(cfg)
    model.load_state_dict(torch.load(model_path, map_location=self_model.device, weights_only=True))
    model.to(self_model.device)
    model.eval()

    test_start = int(self_model.test_raw["d_num"].min())
    forecast_start = test_start + BaseModel.PRED_LENGTH
    forecast_end = forecast_start + horizon - 1

    needed_cols = ["id", "d_num"] + feature_cols
    featured_small = featured[needed_cols].sort_values(["id", "d_num"]).reset_index(drop=True)

    series_ids = []
    contexts = []
    future_feats = []

    for sid, sub in featured_small.groupby("id", sort=False):
        d_nums = sub["d_num"].to_numpy()
        feat_arr = sub[feature_cols].to_numpy(dtype=np.float32, copy=False)

        hist_mask = d_nums < forecast_start
        fut_mask = (d_nums >= forecast_start) & (d_nums <= forecast_end)
        history_features = feat_arr[hist_mask]
        future_features = feat_arr[fut_mask]

        if len(history_features) < seq_len:
            raise ValueError(
                f"Series {sid} has only {len(history_features)} rows before d_{forecast_start}, "
                f"but seq_len={seq_len} is required."
            )
        if len(future_features) != horizon:
            raise ValueError(f"Series {sid} has {len(future_features)} future rows, expected {horizon}.")

        series_ids.append(sid)
        contexts.append(history_features[-seq_len:])
        future_feats.append(future_features)

    context_batch = np.stack(contexts)
    future_feat_batch = np.stack(future_feats)

    if autoregressive:
        q_preds = _predict_autoregressive_quantiles(
            model=model,
            context_batch=context_batch,
            future_feat_batch=future_feat_batch,
            horizon=horizon,
            is_prob=is_prob,
            is_nb=is_nb,
            is_quantile=is_quantile,
            quantiles=target_quantiles,
            model_quantiles=model_quantiles,
            device=self_model.device,
        )
    else:
        x = torch.tensor(context_batch, dtype=torch.float32, device=self_model.device)
        with torch.no_grad():
            if is_nb:
                mu, alpha = model(x)
                q_preds = _nb_params_to_quantiles(mu.cpu().numpy(), alpha.cpu().numpy(), target_quantiles)
            elif is_prob:
                mu, sigma = model(x)
                q_preds = _gaussian_params_to_quantiles(mu.cpu().numpy(), sigma.cpu().numpy(), target_quantiles)
            elif is_quantile:
                raw_q = model(x).cpu().numpy()
                q_preds = _coerce_to_target_quantiles(raw_q, model_quantiles, target_quantiles)
            else:
                preds = model(x).cpu().numpy()
                q_preds = np.repeat(preds[:, :, np.newaxis], len(target_quantiles), axis=2)

        q_preds = _finalise_quantiles(q_preds)

    rows = []
    for sid, series_q in zip(series_ids, q_preds):
        for day_ahead, step_q in enumerate(series_q, start=1):
            row = {"id": sid, "day_ahead": day_ahead}
            for q, val in zip(target_quantiles, step_q):
                row[f"q{q}"] = float(val)
            rows.append(row)

    preds_df = pd.DataFrame(rows).sort_values(["id", "day_ahead"]).reset_index(drop=True)
    save_path = model_dir / f"{model_name}_predictions.csv"
    preds_df.to_csv(save_path, index=False)
    print(f"[predict] Saved predictions to {save_path}")

    self_model.train_processed = contexts
    self_model.test_processed = future_feats
    return preds_df


def evaluate_predictions(self_model, model_name: str, run_name: str, preds_df: pd.DataFrame):
    _, model_dir, hist_path, _ = _load_run_training_config(model_name, run_name)
    registry = load_registry(REGISTRY_PATH)
    resolved = resolve_registry_entry(registry[model_name])
    is_prob = resolved["is_prob"]
    is_nb = resolved["is_nb"]
    is_quantile = resolved["is_quantile"]

    self_model.load_and_split_data()
    forecast_start = int(self_model.test_raw["d_num"].min()) + BaseModel.PRED_LENGTH
    forecast_end = forecast_start + BaseModel.PRED_LENGTH - 1

    future_truth = (
        self_model.test_raw[
            (self_model.test_raw["d_num"] >= forecast_start) &
            (self_model.test_raw["d_num"] <= forecast_end)
        ][["id", "d_num", "sales"]]
        .copy()
    )
    future_truth["day_ahead"] = future_truth["d_num"] - forecast_start + 1

    merged = preds_df.merge(
        future_truth[["id", "day_ahead", "sales"]],
        on=["id", "day_ahead"],
        how="inner",
    ).sort_values(["id", "day_ahead"]).reset_index(drop=True)

    series_ids = merged["id"].drop_duplicates().tolist()
    quantiles = list(BaseModel.QUANTILES)
    q_cols = [f"q{q}" for q in quantiles]

    pred_q = merged[["id", "day_ahead"] + q_cols].pivot(index="id", columns="day_ahead", values=q_cols)
    q_preds = np.stack([
        pred_q[f"q{q}"].reindex(series_ids).values.astype(np.float32)
        for q in quantiles
    ], axis=2)

    targets = (
        merged[["id", "day_ahead", "sales"]]
        .pivot(index="id", columns="day_ahead", values="sales")
        .reindex(series_ids)
        .values.astype(np.float32)
    )

    median_col = "q0.5"
    preds = (
        merged[["id", "day_ahead", median_col]]
        .pivot(index="id", columns="day_ahead", values=median_col)
        .reindex(series_ids)
        .values.astype(np.float32)
    )

    preds = np.clip(preds, 0, 1e6)
    targets = np.clip(targets, 0, 1e6)
    q_preds = _finalise_quantiles(q_preds)

    rmse = float(np.sqrt(((preds - targets) ** 2).mean()))
    mae = float(np.abs(preds - targets).mean())
    mask = targets > 0
    mape = float(np.abs((preds[mask] - targets[mask]) / targets[mask]).mean() * 100) if mask.any() else float("nan")
    ss_res = float(((targets - preds) ** 2).sum())
    ss_tot = float(((targets - targets.mean()) ** 2).sum())
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    weights = self_model.item_weights.reindex(series_ids).fillna(0.0).values.astype(np.float32)
    total_w = weights.sum()
    if total_w > 0:
        weights /= total_w

    per_series_mse = ((preds - targets) ** 2).mean(axis=1)
    per_series_mae = np.abs(preds - targets).mean(axis=1)
    w_rmse = float(np.sqrt((weights * per_series_mse).sum()))
    w_mae = float((weights * per_series_mae).sum())

    metrics = {
        "model": model_name,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r2": r2,
        "w_rmse": w_rmse,
        "w_mae": w_mae,
    }

    self_model._validate_preds(preds_df)
    y_mat, q_arr, group_ids, scale = self_model._build_pinball_tensor(preds_df)
    group_metrics = {
        "model": model_name,
        "wspl": float(self_model.compute_wspl(y_mat, q_arr, group_ids, scale)),
        "crps": float(self_model.compute_crps(y_mat, q_arr, group_ids)),
        **self_model.compute_coverage(preds_df, y_mat, group_ids),
    }
    metrics.update(group_metrics)

    if is_prob or is_nb or is_quantile:
        idx_025 = quantiles.index(0.025)
        idx_975 = quantiles.index(0.975)
        lower = q_preds[:, :, idx_025]
        upper = q_preds[:, :, idx_975]
        metrics["coverage_95"] = float(((targets >= lower) & (targets <= upper)).mean())
        metrics["interval_width"] = float((upper - lower).mean())
        metrics["quantile_crps"] = _crps_from_quantiles(targets, q_preds, quantiles)

    legacy_plots_dir = model_dir / "plots"
    if legacy_plots_dir.exists():
        shutil.rmtree(legacy_plots_dir)

    if hist_path.exists():
        _plot_training_curves(
            hist_path,
            model_name,
            save_path=model_dir / f"{model_name}_training_curves.png",
        )

    with open(model_dir / f"{model_name}_test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(model_dir / f"{model_name}_basemodel_metrics.json", "w") as f:
        json.dump(group_metrics, f, indent=2)

    print(f"  RMSE={rmse:.4f}  MAE={mae:.4f}  MAPE={mape:.2f}%  R²={r2:.4f}")
    print(f"  W-RMSE={w_rmse:.4f}  W-MAE={w_mae:.4f}")
    print(
        f"  WSPL={metrics['wspl']:.4f}  "
        f"CRPS={metrics['crps']:.4f}  "
        f"CovErr95={metrics['coverage_error_95pct']:.4f}"
    )
    if "coverage_95" in metrics:
        print(
            f"  Coverage(95%)={metrics['coverage_95']:.4f}  "
            f"Width={metrics['interval_width']:.4f}  "
            f"CRPS={metrics['crps']:.4f}  "
            f"QCRPS={metrics['quantile_crps']:.4f}"
        )
    return metrics
