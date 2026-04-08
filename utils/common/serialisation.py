import json
import time
from pathlib import Path

import torch

from .runtime import device


def save_model(model, name, model_dir):
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f"{name}_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")
    return model_path


def save_history(history, name, stage, model, model_dir, config=None):
    model_dir.mkdir(exist_ok=True)
    history_path = model_dir / f"{name}_{stage}_history.json"
    payload = {
        "model": name,
        "architecture": str(model),
        "stage": stage,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": config or {},
        "metrics": history,
    }
    with open(history_path, "w") as f:
        json.dump(payload, f, indent=4)
    print(f"History saved to: {history_path}")
    return history_path


def load_history(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def extract_epoch_metrics(history: dict):
    metrics = history["metrics"]["epoch_metrics"]
    epochs = [m["epoch"] for m in metrics]
    train_acc = [m.get("train_accuracy") for m in metrics]
    val_acc = [m["validation_accuracy"] for m in metrics]
    return epochs, train_acc, val_acc


def load_model(dropout_prob: float, weights_path: Path) -> torch.nn.Module:
    from utils.network import SalesGRU

    model = SalesGRU(dropout=dropout_prob).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
