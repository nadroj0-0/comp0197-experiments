import torch
import numpy as np
from utils.data import build_dataloaders, denormalise

if __name__ == '__main__':
    _, _, test_loader, stats = build_dataloaders(
        data_dir="./data",
        seq_len=56,
        horizon=28,
        batch_size=1024,
        store_id="CA_3",
        num_workers=0,
        seed=42,
        zscore_target=True,
    )

    all_lag28, all_roll28, all_targets = [], [], []
    for x, y in test_loader:
        all_lag28.append(x[:, -1, 10].unsqueeze(1).expand(-1, 28))
        all_roll28.append(x[:, -1, 12].unsqueeze(1).expand(-1, 28))
        all_targets.append(y)

    lag28   = torch.cat(all_lag28).numpy()
    roll28  = torch.cat(all_roll28).numpy()
    targets = torch.cat(all_targets).numpy()

    def metrics(preds_norm, name):
        # normalised space (fair comparison with GRU val metrics)
        rmse_n = np.sqrt(((preds_norm - targets)**2).mean())
        r2_n   = 1 - ((targets - preds_norm)**2).sum() / ((targets - targets.mean())**2).sum()
        print(f"{name:20s} | RMSE(norm)={rmse_n:.4f} | R²(norm)={r2_n:.4f}")

    metrics(lag28,  "Lag-28 baseline")
    metrics(roll28, "Roll-mean-28")