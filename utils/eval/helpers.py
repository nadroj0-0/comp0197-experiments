import json
from math import erf

import matplotlib.pyplot as plt
import numpy as np
import torch


SQRT_PI = np.sqrt(np.pi)
SQRT_2 = np.sqrt(2.0)


def nb_params_to_quantiles(mu, alpha, quantiles, n_samples=1000):
    mu_t = torch.tensor(mu, dtype=torch.float32).clamp(min=1e-6)
    alpha_t = torch.tensor(alpha, dtype=torch.float32).clamp(min=1e-6)
    variance = mu_t + alpha_t * mu_t**2
    p = (mu_t / variance).clamp(min=1e-6, max=1 - 1e-6)
    r = (mu_t * p / (1 - p)).clamp(min=1e-6)
    dist = torch.distributions.NegativeBinomial(total_count=r, probs=p)
    samples = dist.sample((n_samples,)).float()
    q_vals = torch.quantile(
        samples, torch.tensor(quantiles, dtype=torch.float32), dim=0
    )
    return q_vals.permute(1, 0).cpu().numpy()


def _normal_pdf(z):
    return np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)


def _normal_cdf(z):
    return 0.5 * (1 + np.vectorize(erf)(z / SQRT_2))


def crps_gaussian(mu, sigma, y):
    sigma = np.clip(sigma, 1e-6, None)
    z = (y - mu) / sigma
    return sigma * (z * (2 * _normal_cdf(z) - 1) + 2 * _normal_pdf(z) - 1 / SQRT_PI)


def crps_from_quantiles(y, q_preds, quantiles):
    y_exp = y[:, :, np.newaxis]
    q = np.array(quantiles)[np.newaxis, np.newaxis, :]
    diff = y_exp - q_preds
    loss = np.maximum(q * diff, (q - 1) * diff)
    weights = np.diff(np.concatenate(([0.0], quantiles)))
    return float((loss * weights[np.newaxis, np.newaxis, :]).sum(axis=2).mean())


def predict_autoregressive(
    model,
    seed_window,
    horizon,
    is_prob,
    is_nb,
    device,
    is_quantile=False,
    quantile_median_idx=3,
):
    model.eval()
    current_window = seed_window.copy()
    preds, aux_list = [], []

    for _ in range(horizon):
        x_t = torch.tensor(current_window, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            if is_nb:
                mu, alpha = model(x_t)
                step_pred = float(mu.cpu().numpy()[0, 0])
                step_aux = float(alpha.cpu().numpy()[0, 0])
            elif is_prob:
                mu, sigma = model(x_t)
                step_pred = float(mu.cpu().numpy()[0, 0])
                step_aux = float(sigma.cpu().numpy()[0, 0])
            elif is_quantile:
                out = model(x_t).cpu().numpy()[0]
                step_pred = float(out[quantile_median_idx])
                step_aux = out
            else:
                out = model(x_t)
                step_pred = float(out.cpu().numpy()[0, 0])
                step_aux = None

        step_pred = max(0.0, step_pred)
        preds.append(step_pred)
        if step_aux is not None:
            aux_list.append(step_aux)

        new_row = current_window[-1].copy()
        new_row[0] = step_pred
        if current_window.shape[1] > 5:
            new_row[5] = (current_window[-1, 5] + 1) % 7
        current_window = np.vstack([current_window[1:], new_row])

    return np.array(preds), (np.array(aux_list) if aux_list else None)


def plot_quantile_forecast(
    historical_y,
    true_y,
    pred_quantiles,
    quantiles_list,
    item_name="Series",
    save_path=None,
    rmse=None,
):
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
    ax.plot(time_hist, historical_y, label="Historical Data", color="black", linewidth=1.5, marker=".")
    ax.plot(time_pred, true_y, label="Actual Future", color="black", linestyle="--", linewidth=1.5, marker="x")
    ax.axvline(x=seq_len, color="red", linestyle=":", linewidth=2, label="Forecast Start (Unseen Data)")
    ax.fill_between(time_pred, pred_quantiles[:, idx_025], pred_quantiles[:, idx_975], color="blue", alpha=0.10, label="95% CI")
    ax.fill_between(time_pred, pred_quantiles[:, idx_050], pred_quantiles[:, idx_950], color="blue", alpha=0.20, label="90% CI")
    ax.fill_between(time_pred, pred_quantiles[:, idx_250], pred_quantiles[:, idx_750], color="blue", alpha=0.35, label="50% CI")
    ax.plot(time_pred, pred_quantiles[:, idx_500], label="Median Forecast", color="blue", linewidth=2, marker="o")

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
        print(f"  Saved: {save_path}")
    plt.close()


def plot_det_forecast(historical_y, true_y, pred_y, item_name="Series", save_path=None, rmse=None):
    seq_len = len(historical_y)
    time_hist = np.arange(seq_len)
    time_pred = np.arange(seq_len, seq_len + len(true_y))

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(time_hist, historical_y, label="Historical Data", color="black", linewidth=1.5, marker=".")
    ax.plot(time_pred, true_y, label="Actual Future", color="black", linestyle="--", linewidth=1.5, marker="x")
    ax.axvline(x=seq_len, color="red", linestyle=":", linewidth=2, label="Forecast Start (Unseen Data)")
    ax.plot(time_pred, pred_y, label="Prediction", color="blue", linewidth=2, marker="o")

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
        print(f"  Saved: {save_path}")
    plt.close()


def plot_training_curves(hist_path, model_name, save_path=None):
    with open(hist_path) as f:
        saved = json.load(f)
    metrics = saved["metrics"]["epoch_metrics"]
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
        print(f"  Saved: {save_path}")
    plt.close()

