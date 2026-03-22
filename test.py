# =============================================================================
# test.py — Forecast Evaluation & Visualisation
# =============================================================================

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from utils.network import build_gru
from utils.data import build_dataloaders, denormalise
from utils.training_strategies import gru_step
from utils.experiment import evaluate_model

# -----------------------------------------------------------------------------
# CONFIG (MUST MATCH TRAINING)
# -----------------------------------------------------------------------------

# replace the CONFIG block and model loading section with this

import json

MODEL_NAME = "gru_direct_det"
MODEL_DIR  = Path("./models") / MODEL_NAME
MODEL_PATH = MODEL_DIR / f"{MODEL_NAME}_model.pt"
HIST_PATH  = MODEL_DIR / f"{MODEL_NAME}_train_history.json"

# load the config that was actually used for training
with open(HIST_PATH) as f:
    saved = json.load(f)
CONFIG = saved["config"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# PLOTTING (TIME SERIES — NOT GAP PLOT)
# -----------------------------------------------------------------------------

def plot_forecast(pred, true, save_path=None, title="28-Day Forecast"):
    horizon = len(pred)
    t = np.arange(horizon)

    plt.figure(figsize=(10, 5))
    plt.plot(t, true, label="Actual", linestyle="--")
    plt.plot(t, pred, label="Prediction")

    plt.xlabel("Days Ahead")
    plt.ylabel("Sales")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved plot: {save_path}")

    plt.close()


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def main():
    print("="*50)
    print("🔍 TESTING MODEL")
    print("="*50)

    # -------------------------------------------------------------------------
    # 1. LOAD DATA
    # -------------------------------------------------------------------------
    print("\n[1/4] Loading data...")

    train_loader, val_loader, test_dataset, stats = build_dataloaders(
        data_dir=CONFIG["data_dir"],
        seq_len=CONFIG["seq_len"],
        horizon=CONFIG["horizon"],
        batch_size=CONFIG["batch_size"],
        store_id=CONFIG["store_id"],
        max_series=CONFIG["max_series"],
        num_workers=0,
        seed=CONFIG["seed"],
    )

    # -------------------------------------------------------------------------
    # 2. LOAD MODEL
    # -------------------------------------------------------------------------
    print("\n[2/4] Loading model...")

    model, _, _, _ = build_gru(CONFIG)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    print(f"✅ Loaded model from: {MODEL_PATH}")

    # -------------------------------------------------------------------------
    # 3. INFERENCE
    # -------------------------------------------------------------------------
    print("\n[3/4] Running inference...")

    _, _, preds, targets = evaluate_model(
        test_dataset,
        model,
        criterion=torch.nn.MSELoss(),
        training_step=gru_step
    )

    preds = preds.numpy()
    targets = targets.numpy()

    test_rmse = np.sqrt(((preds - targets) ** 2).mean())
    test_mae = np.abs(preds - targets).mean()
    test_r2 = 1 - ((targets - preds) ** 2).sum() / ((targets - targets.mean()) ** 2).sum()
    print(f"\nTest metrics (normalised):")
    print(f"  RMSE : {test_rmse:.4f}")
    print(f"  MAE  : {test_mae:.4f}")
    print(f"  R²   : {test_r2:.4f}")

    print(f"Pred shape: {preds.shape}")
    print(f"Target shape: {targets.shape}")

    # -------------------------------------------------------------------------
    # 4. VISUALISATION
    # -------------------------------------------------------------------------
    print("\n[4/4] Generating plots...")

    output_dir = Path("./plots")
    output_dir.mkdir(exist_ok=True)

    rmse_per_sample = np.sqrt(((preds - targets) ** 2).mean(axis=1))
    sorted_idx = np.argsort(rmse_per_sample)
    median_start = len(sorted_idx) // 2 - 2
    plot_indices = sorted_idx[median_start:median_start + 5]

    for i, idx in enumerate(plot_indices):
        pred = denormalise(preds[idx], stats)
        true = denormalise(targets[idx], stats)
        pred = np.clip(pred, 0, 500)
        true = np.clip(true, 0, 500)
        save_path = output_dir / f"{MODEL_NAME}_sample_{i}.png"
        plot_forecast(pred, true, save_path=save_path,
                      title=f"{MODEL_NAME} | Sample {i} (RMSE={rmse_per_sample[idx]:.2f})")

    print("\n✅ Done — plots saved in ./plots/")


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()