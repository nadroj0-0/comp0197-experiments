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

CONFIG = {
    "seed": 42,
    "hidden": 256,
    "layers": 2,
    "dropout": 0.15,
    "horizon": 28,

    "batch_size": 1024,
    "seq_len": 56,
    "store_id": "CA_3",
    "data_dir": "./data",
    "max_series": None,
    "num_workers": 4,

    "optimiser": "adamw",
    "optimiser_params": {
        "lr": 1e-3,
        "weight_decay": 1e-4,
    },
}

MODEL_NAME = "gru_direct_det"
MODEL_PATH = Path("./models") / MODEL_NAME / f"{MODEL_NAME}_model.pt"

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
        num_workers=CONFIG["num_workers"],
        seed=CONFIG["seed"],
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False
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
        test_loader,
        model,
        criterion=torch.nn.MSELoss(),
        training_step=gru_step
    )

    preds = preds.numpy()
    targets = targets.numpy()

    print(f"Pred shape: {preds.shape}")
    print(f"Target shape: {targets.shape}")

    # -------------------------------------------------------------------------
    # 4. VISUALISATION
    # -------------------------------------------------------------------------
    print("\n[4/4] Generating plots...")

    output_dir = Path("./plots")
    output_dir.mkdir(exist_ok=True)

    NUM_PLOTS = 5

    for i in range(NUM_PLOTS):
        pred = preds[i]
        true = targets[i]

        # 🔥 CRITICAL: denormalise
        pred = denormalise(pred, stats)
        true = denormalise(true, stats)

        save_path = output_dir / f"{MODEL_NAME}_sample_{i}.png"

        plot_forecast(
            pred,
            true,
            save_path=save_path,
            title=f"{MODEL_NAME} | Sample {i}"
        )

    print("\n✅ Done — plots saved in ./plots/")


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()