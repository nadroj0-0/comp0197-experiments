import torch
from torch.utils.data import Dataset

from .specs import TARGET_COL


class WindowedM5Dataset(Dataset):
    """
    Window-first dataset. Each window is assigned to train/val/test
    by the date of its target block.

    autoregressive=True  : y is scalar (next step) — train 1-step ahead,
                           use recursive rollout at test time
    autoregressive=False : y is (horizon,) vector — direct multi-step
    """

    def __init__(
        self,
        df,
        feature_cols,
        seq_len,
        horizon,
        split,
        val_start_date,
        test_start_date,
        autoregressive=True,
        max_series=None,
        item_weights=None,
        series_ids=None,
    ):
        self.seq_len = seq_len
        self.horizon = horizon
        self.autoregressive = autoregressive
        self.feature_cols = feature_cols
        self.split = split
        self.samples = []
        self.series_data = {}
        self.item_weights = item_weights
        self._series_idx_map = {}

        grouped = df.groupby("id")

        if series_ids is None:
            series_ids = list(grouped.groups.keys())

        self.series_ids = series_ids
        if max_series is not None:
            series_ids = series_ids[:int(max_series)]

        for sid in series_ids:
            sub = grouped.get_group(sid).sort_values("date").reset_index(drop=True)
            self.series_data[sid] = {
                "features": torch.from_numpy(
                    sub[feature_cols].values.astype("float32")
                ),
                "targets": torch.from_numpy(
                    sub[TARGET_COL].values.astype("float32")
                ),
                "dates": sub["date"].values.astype("datetime64[ns]"),
            }
            dates = self.series_data[sid]["dates"]
            n = len(sub) - seq_len - horizon + 1
            if n <= 0:
                continue
            for i in range(n):
                y_start = i + seq_len
                y_end = y_start + horizon
                target_start_date = dates[y_start]
                target_end_date = dates[y_end - 1]
                if split == "train":
                    if target_end_date < val_start_date:
                        self.samples.append((sid, i))
                elif split == "val":
                    if target_start_date >= val_start_date and target_end_date < test_start_date:
                        self.samples.append((sid, i))
                elif split == "test":
                    if target_start_date >= test_start_date:
                        self.samples.append((sid, i))
                else:
                    raise ValueError(f"Unknown split: {split}")

        print(f"[data] {split}: {len(self.samples):,} windows  "
              f"(autoregressive={autoregressive})")
        self._series_idx_map = {sid: i for i, sid in enumerate(self.series_ids)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sid, i = self.samples[idx]
        sub = self.series_data[sid]
        x = sub["features"][i: i + self.seq_len]
        if self.autoregressive:
            y = sub["targets"][i + self.seq_len]
        else:
            y = sub["targets"][i + self.seq_len: i + self.seq_len + self.horizon]

        if self.item_weights is not None:
            series_idx = self._series_idx_map[sid]
            w = torch.tensor(self.item_weights[series_idx], dtype=torch.float32)
            return x, y, w

        return x, y
