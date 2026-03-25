import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# =============================================================================
# FEATURE SETS
# "sales_only"          — raw sales only (input_size=1)  matches M5.ipynb
# "sales_hierarchy"     — sales + static hierarchy IDs   matches BHM notebook
# "sales_hierarchy_dow" — sales + hierarchy + day_of_week matches LGBM notebook
# =============================================================================
FEATURE_SETS = {
    "sales_only":           ["sales"],
    "sales_hierarchy":      ["sales", "state_id_int", "store_id_int",
                             "cat_id_int", "dept_id_int"],
    "sales_hierarchy_dow":  ["sales", "state_id_int", "store_id_int",
                             "cat_id_int", "dept_id_int", "day_of_week"],
}
TARGET_COL = "sales"


def get_feature_cols(feature_set: str) -> list:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature_set '{feature_set}'. "
                         f"Options: {list(FEATURE_SETS)}")
    return FEATURE_SETS[feature_set]


# =============================================================================
# 1. LOAD RAW FILES
# =============================================================================

def load_raw_data(data_dir: str) -> tuple:
    """Load the M5 CSV files from data_dir."""
    sales_df    = pd.read_csv(os.path.join(data_dir, "sales_train_evaluation.csv"))
    calendar_df = pd.read_csv(os.path.join(data_dir, "calendar.csv"))
    prices_df   = pd.read_csv(os.path.join(data_dir, "sell_prices.csv"))
    print(f"[data] Sales    : {sales_df.shape}")
    print(f"[data] Calendar : {calendar_df.shape}")
    print(f"[data] Prices   : {prices_df.shape}")
    return sales_df, calendar_df, prices_df


def load_or_download_m5(data_dir: str):
    """Ensures M5 evaluation dataset exists locally. Downloads once if missing."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "sales":    data_dir / "sales_train_evaluation.csv",
        "calendar": data_dir / "calendar.csv",
        "prices":   data_dir / "sell_prices.csv",
    }
    urls = {
        "sales":    "https://huggingface.co/datasets/kashif/M5/resolve/main/sales_train_evaluation.csv",
        "calendar": "https://huggingface.co/datasets/kashif/M5/resolve/main/calendar.csv",
        "prices":   "https://huggingface.co/datasets/kashif/M5/resolve/main/sell_prices.csv",
    }
    for key in files:
        if not files[key].exists():
            print(f"[data] Downloading {key}...")
            df = pd.read_csv(urls[key])
            df.to_csv(files[key], index=False)
        else:
            print(f"[data] Found local {key}")
    return load_raw_data(data_dir)


def trim_data(df: pd.DataFrame, num_items: int) -> pd.DataFrame:
    """Select top-N series by total sales volume across all time."""
    if df.empty:
        return df
    df = df.copy()
    day_cols = [c for c in df.columns if c.startswith("d_")]
    df["total_sales_volume"] = df[day_cols].sum(axis=1)
    trimmed_df = (
        df.sort_values("total_sales_volume", ascending=False)
          .head(num_items)
          .drop(columns=["total_sales_volume"])
          .reset_index(drop=True)
    )
    print(f"[data] Original items: {len(df)} | Selected top-{num_items} by volume")
    print(f"[data] Trimmed shape : {trimmed_df.shape}")
    return trimmed_df


# =============================================================================
# 2. MELT SALES  —  wide -> long
# =============================================================================

def melt_sales(sales_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot wide sales table to long format: one row per (item, day)."""
    id_cols  = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in sales_df.columns if c.startswith("d_")]
    long_df = sales_df.melt(
        id_vars=id_cols,
        value_vars=day_cols,
        var_name="d",
        value_name="sales",
    )
    print(f"[data] After melt  : {long_df.shape}")
    return long_df


# =============================================================================
# 3. ENCODE HIERARCHY  —  static categorical IDs + optional day_of_week
# =============================================================================

def encode_hierarchy(df: pd.DataFrame, include_dow: bool = False) -> pd.DataFrame:
    """
    Encode static hierarchy columns as integers.
    Adds: state_id_int, store_id_int, cat_id_int, dept_id_int
    Optionally adds: day_of_week (0-6, derived from 'd' column index)
    """
    df = df.copy()
    for col in ["state_id", "store_id", "cat_id", "dept_id"]:
        df[f"{col}_int"] = df[col].astype("category").cat.codes.astype(np.float32)
    if include_dow:
        # Extract day number from 'd_N' and compute day of week
        df["d_num"] = df["d"].str.extract(r"d_(\d+)").astype(int)
        df["day_of_week"] = ((df["d_num"] + 5) % 7).astype(np.float32)
    print(f"[data] After hierarchy encoding : {df.shape}")
    return df


# =============================================================================
# 4. MERGE CALENDAR  (only needed for full feature sets)
# =============================================================================

def merge_calendar(long_df: pd.DataFrame, calendar_df: pd.DataFrame) -> pd.DataFrame:
    cal_cols = [
        "d", "date", "wm_yr_wk",
        "wday", "month", "year",
        "event_name_1", "event_type_1",
        "event_name_2", "event_type_2",
        "snap_CA", "snap_TX", "snap_WI",
    ]
    merged = long_df.merge(calendar_df[cal_cols], on="d", how="left")
    merged["date"] = pd.to_datetime(merged["date"])
    print(f"[data] After calendar merge : {merged.shape}")
    return merged


# =============================================================================
# 5. MERGE PRICES  (only needed for full feature sets)
# =============================================================================

def merge_prices(merged_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
    merged = merged_df.merge(prices_df, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    merged["sell_price"] = merged["sell_price"].fillna(0.0)
    print(f"[data] After price merge    : {merged.shape}")
    return merged


# =============================================================================
# 6. FEATURE ENGINEERING  (only needed for full feature sets)
# =============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["id", "date"]).reset_index(drop=True)
    df["wday_sin"]  = np.sin(2 * np.pi * df["wday"]  / 7).astype(np.float32)
    df["wday_cos"]  = np.cos(2 * np.pi * df["wday"]  / 7).astype(np.float32)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12).astype(np.float32)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12).astype(np.float32)
    df["has_event"] = (~df["event_name_1"].isna()).astype(np.float32)
    for col in ["snap_CA", "snap_TX", "snap_WI"]:
        df[col] = df[col].astype(np.float32)
    df["lag_7"]  = df.groupby("id")["sales"].shift(7)
    df["lag_28"] = df.groupby("id")["sales"].shift(28)
    df["roll_mean_7"]  = df.groupby("id")["sales"].transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).mean()
    )
    df["roll_mean_28"] = df.groupby("id")["sales"].transform(
        lambda x: x.shift(1).rolling(28, min_periods=1).mean()
    )
    df = df.dropna(subset=["lag_7", "lag_28"]).reset_index(drop=True)
    print(f"[data] After feature engineering : {df.shape}")
    return df


# =============================================================================
# 7. TRAIN / VAL / TEST SPLIT  (temporal — never shuffle!)
# =============================================================================

def split_data(
    df: pd.DataFrame,
    val_days:  int = 56,
    test_days: int = 28,
) -> tuple:
    all_dates  = sorted(df["date"].unique())
    test_start = all_dates[-test_days]
    val_start  = all_dates[-(test_days + val_days)]
    train_df = df[df["date"] <  val_start].copy()
    val_df   = df[(df["date"] >= val_start) & (df["date"] < test_start)].copy()
    test_df  = df[df["date"] >= test_start].copy()
    print(f"[data] Train : {train_df['date'].min().date()} -> {train_df['date'].max().date()}  "
          f"({len(train_df):,} rows)")
    print(f"[data] Val   : {val_df['date'].min().date()} -> {val_df['date'].max().date()}  "
          f"({len(val_df):,} rows)")
    print(f"[data] Test  : {test_df['date'].min().date()} -> {test_df['date'].max().date()}  "
          f"({len(test_df):,} rows)")
    return train_df, val_df, test_df


# =============================================================================
# 8. NORMALISATION  (optional — off by default to match teammate)
# =============================================================================

def fit_normalisation_stats(train_df: pd.DataFrame, zscore_target: bool = True) -> dict:
    """Fit normalisation stats on TRAIN rows only."""
    stats = {}
    tmp = train_df.copy()
    tmp["sales"] = np.log1p(tmp["sales"].clip(lower=0))
    mean = float(tmp["sales"].mean())
    std  = float(tmp["sales"].std()) + 1e-8
    stats["sales"] = {"mean": mean, "std": std, "log1p": True, "zscore": zscore_target}
    return stats


def apply_normalisation(df: pd.DataFrame, stats: dict, zscore_target: bool = True) -> pd.DataFrame:
    """Apply previously fitted TRAIN stats to any dataframe."""
    out = df.copy()
    out["sales"] = np.log1p(out["sales"].clip(lower=0))
    if zscore_target:
        out["sales"] = (out["sales"] - stats["sales"]["mean"]) / stats["sales"]["std"]
    return out


def denormalise(preds: np.ndarray, stats: dict, col: str = "sales") -> np.ndarray:
    """Reverse normalisation for predictions."""
    out = preds * stats[col]["std"] + stats[col]["mean"]
    if stats[col].get("log1p"):
        out = np.expm1(out)
    return out


# =============================================================================
# 9. PYTORCH DATASET  —  window-first, supports autoregressive and direct modes
# =============================================================================

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
        df: pd.DataFrame,
        feature_cols: list,
        seq_len: int,
        horizon: int,
        split: str,
        val_start_date,
        test_start_date,
        autoregressive: bool = True,
        max_series: int = None,
    ):
        self.seq_len        = seq_len
        self.horizon        = horizon
        self.autoregressive = autoregressive
        self.feature_cols   = feature_cols
        self.split          = split
        self.samples        = []
        self.series_data    = {}

        grouped    = df.groupby("id")
        series_ids = list(grouped.groups.keys())
        if max_series is not None:
            series_ids = series_ids[:int(max_series)]

        for sid in series_ids:
            sub = grouped.get_group(sid).sort_values("date").reset_index(drop=True)
            self.series_data[sid] = {
                "features": torch.from_numpy(
                    sub[feature_cols].values.astype(np.float32)
                ),
                "targets": torch.from_numpy(
                    sub[TARGET_COL].values.astype(np.float32)
                ),
                "dates": sub["date"].values.astype("datetime64[ns]"),
            }
            dates = self.series_data[sid]["dates"]
            n = len(sub) - seq_len - horizon + 1
            if n <= 0:
                continue
            for i in range(n):
                y_start          = i + seq_len
                y_end            = y_start + horizon
                target_start_date = dates[y_start]
                target_end_date   = dates[y_end - 1]
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sid, i = self.samples[idx]
        sub = self.series_data[sid]
        x = sub["features"][i : i + self.seq_len]
        if self.autoregressive:
            y = sub["targets"][i + self.seq_len]          # scalar
        else:
            y = sub["targets"][i + self.seq_len : i + self.seq_len + self.horizon]
        return x, y


# =============================================================================
# 10. SEED HELPERS
# =============================================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    generator = torch.Generator().manual_seed(seed)
    print(f"Random seed set to {seed}")
    return generator, seed


def init_seed(cfg: dict):
    seed = cfg.get("seed", 42)
    generator, seed = set_seed(seed)
    cfg["seed"] = seed
    return generator


# =============================================================================
# 11. MAIN ENTRY POINT
# =============================================================================

def build_dataloaders(
    data_dir:       str,
    seq_len:        int  = 28,
    horizon:        int  = 28,
    batch_size:     int  = 256,
    top_k_series:   int  = 200,
    feature_set:    str  = "sales_only",
    autoregressive: bool = True,
    use_normalise:  bool = False,
    zscore_target:  bool = True,
    max_series:     int  = None,
    num_workers:    int  = 2,
    seed:           int  = 42,
) -> tuple:
    """
    Main data pipeline for V3.

    Parameters
    ----------
    top_k_series   : Number of highest-volume series to keep (default 200).
    feature_set    : One of "sales_only", "sales_hierarchy",
                     "sales_hierarchy_dow".
    autoregressive : True  -> 1-step ahead targets (recursive rollout at test)
                     False -> horizon-length targets (direct multi-step)
    use_normalise  : Apply log1p + optional z-score to sales target.
                     False by default to match teammate's raw-count workflow.
    zscore_target  : Only used when use_normalise=True.

    Returns
    -------
    (train_loader, val_loader, test_loader, stats)
        stats is None when use_normalise=False.
    """
    generator, _ = set_seed(seed)
    feature_cols  = get_feature_cols(feature_set)
    n_features    = len(feature_cols)

    # ------------------------------------------------------------------
    # 1. Load + trim
    # ------------------------------------------------------------------
    sales_df, calendar_df, prices_df = load_or_download_m5(data_dir)
    sales_df = trim_data(sales_df, top_k_series)

    # ------------------------------------------------------------------
    # 2. Build featured dataframe — path depends on feature_set
    # ------------------------------------------------------------------
    long_df = melt_sales(sales_df)

    if feature_set == "sales_only":
        # Minimal path — only need a date column for temporal splits
        # Merge calendar just to get the date column
        cal_dates = calendar_df[["d", "date"]].copy()
        cal_dates["date"] = pd.to_datetime(cal_dates["date"])
        featured = long_df.merge(cal_dates, on="d", how="left")
        featured = featured.sort_values(["id", "date"]).reset_index(drop=True)
        print(f"[data] Sales-only pipeline: {featured.shape}")

    elif feature_set in ("sales_hierarchy", "sales_hierarchy_dow"):
        # Add static hierarchy IDs (and optionally day_of_week)
        include_dow = feature_set == "sales_hierarchy_dow"
        cal_dates = calendar_df[["d", "date"]].copy()
        cal_dates["date"] = pd.to_datetime(cal_dates["date"])
        featured = long_df.merge(cal_dates, on="d", how="left")
        featured = encode_hierarchy(featured, include_dow=include_dow)
        featured = featured.sort_values(["id", "date"]).reset_index(drop=True)
        print(f"[data] Hierarchy pipeline ({feature_set}): {featured.shape}")

    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    # ------------------------------------------------------------------
    # 3. Temporal split boundaries
    # ------------------------------------------------------------------
    train_df_raw, val_df_raw, test_df_raw = split_data(
        featured, val_days=56, test_days=28
    )
    val_start_date  = val_df_raw["date"].values.astype("datetime64[ns]")[0]
    test_start_date = test_df_raw["date"].values.astype("datetime64[ns]")[0]

    # ------------------------------------------------------------------
    # 4. Normalisation (optional)
    # ------------------------------------------------------------------
    if use_normalise:
        stats = fit_normalisation_stats(train_df_raw, zscore_target=zscore_target)
        featured = apply_normalisation(featured, stats, zscore_target=zscore_target)
        print(f"[data] Normalisation applied (zscore_target={zscore_target})")
    else:
        stats = None
        print("[data] No normalisation — using raw sales counts")

    # ------------------------------------------------------------------
    # 5. Build windowed datasets
    # ------------------------------------------------------------------
    shared = dict(
        feature_cols    = feature_cols,
        seq_len         = seq_len,
        horizon         = horizon,
        val_start_date  = val_start_date,
        test_start_date = test_start_date,
        autoregressive  = autoregressive,
        max_series      = max_series,
    )
    train_ds = WindowedM5Dataset(featured, split="train", **shared)
    val_ds   = WindowedM5Dataset(featured, split="val",   **shared)
    test_ds  = WindowedM5Dataset(featured, split="test",  **shared)

    # ------------------------------------------------------------------
    # 6. DataLoaders
    # ------------------------------------------------------------------
    _pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, generator=generator, pin_memory=_pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=_pin,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=_pin,
    )

    print(f"\n[data] Feature set  : {feature_set}  ({n_features} features)")
    print(f"[data] Input shape  : (seq_len={seq_len}, n_features={n_features})")
    print(f"[data] Autoregressive: {autoregressive}")
    print(f"[data] Train batches: {len(train_loader)}")
    print(f"[data] Val batches  : {len(val_loader)}")
    print(f"[data] Test batches : {len(test_loader)}")
    print(f"[DEBUG] Train windows: {len(train_ds):,}")
    print(f"[DEBUG] Val windows  : {len(val_ds):,}")
    print(f"[DEBUG] Test windows : {len(test_ds):,}")

    return train_loader, val_loader, test_loader, stats