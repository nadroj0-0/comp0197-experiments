import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# Features fed into the model at each timestep
# ---------------------------------------------------------------------------
# FEATURE_COLS = [
#     "sell_price",
#     "wday_sin", "wday_cos",
#     "month_sin", "month_cos",
#     "has_event",
#     "snap_CA", "snap_TX", "snap_WI",
#     "lag_7", "lag_28",
#     "roll_mean_7", "roll_mean_28",
# ]
FEATURE_COLS = [
    #"sales",
    "sell_price",
    "wday_sin", "wday_cos",
    "month_sin", "month_cos",
    "has_event",
    "snap_CA", "snap_TX", "snap_WI",
    "lag_7", "lag_28",
    "roll_mean_7", "roll_mean_28",
]
N_FEATURES = len(FEATURE_COLS)   # 13  — use this in network.py as input_size
TARGET_COL = "sales"


# =============================================================================
# 1. LOAD RAW FILES
# =============================================================================

def load_raw_data(data_dir: str) -> tuple:
    """
    Load the three core M5 CSV files from data_dir.

    Expected files
    --------------
    sales_train_validation.csv : wide-format sales (30 490 items x ~1 913 day cols)
    calendar.csv               : day -> date mapping + event / SNAP flags
    sell_prices.csv            : weekly item prices per store

    Returns
    -------
    (sales_df, calendar_df, prices_df)
    """
    sales_df    = pd.read_csv(os.path.join(data_dir, "sales_train_validation.csv"))
    calendar_df = pd.read_csv(os.path.join(data_dir, "calendar.csv"))
    prices_df   = pd.read_csv(os.path.join(data_dir, "sell_prices.csv"))

    print(f"[data] Sales    : {sales_df.shape}")
    print(f"[data] Calendar : {calendar_df.shape}")
    print(f"[data] Prices   : {prices_df.shape}")
    return sales_df, calendar_df, prices_df

def load_or_download_m5(data_dir: str):
    """
    Ensures M5 dataset exists locally. Downloads once if missing.
    Returns loaded DataFrames.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "sales": data_dir / "sales_train_validation.csv",
        "calendar": data_dir / "calendar.csv",
        "prices": data_dir / "sell_prices.csv",
    }
    # URLs (HuggingFace mirrors)
    urls = {
        "sales": "https://huggingface.co/datasets/kashif/M5/resolve/main/sales_train_validation.csv",
        "calendar": "https://huggingface.co/datasets/kashif/M5/resolve/main/calendar.csv",
        "prices": "https://huggingface.co/datasets/kashif/M5/resolve/main/sell_prices.csv",
    }
    for key in files:
        if not files[key].exists():
            print(f"[data] Downloading {key}...")
            df = pd.read_csv(urls[key])
            df.to_csv(files[key], index=False)
        else:
            print(f"[data] Found local {key}")
    # sales_df    = pd.read_csv(files["sales"])
    # calendar_df = pd.read_csv(files["calendar"])
    # prices_df   = pd.read_csv(files["prices"])
    sales_df, calendar_df, prices_df = load_raw_data(data_dir)
    return sales_df, calendar_df, prices_df

def trim_data(df: pd.DataFrame, num_items: int) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()  # Avoid mutation bugs
    # Identify all the daily sales columns (d_1, d_2, etc.)
    day_cols = [c for c in df.columns if c.startswith('d_')]
    # Calculate the total sales for every single row across all 1,900+ days
    df["total_sales_volume"] = df[day_cols].sum(axis=1)
    # Sort by total sales (highest first) and grab the top N rows
    trimmed_df = (
        df.sort_values("total_sales_volume", ascending=False)
          .head(num_items)
          .drop(columns=["total_sales_volume"])
    )
    print(f"[data] Data Selection Complete:")
    print(f"[data] Original items: {len(df)} | Selected items: {len(trimmed_df)}")
    print(f"[data] Trimmed to top {num_items} items")
    print(f"[data] Trimmed DataFrame shape: {trimmed_df.shape}")
    return trimmed_df


# =============================================================================
# 2. MELT SALES  —  wide -> long
# =============================================================================

def melt_sales(sales_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot the wide sales table so each row is one (item, day) observation.

    Wide  :  item_id | d_1 | d_2 | ... | d_1913
    Long  :  item_id | d   | sales
    """
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
# 3. MERGE CALENDAR
# =============================================================================

def merge_calendar(long_df: pd.DataFrame, calendar_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join calendar features (date, weekday, month, events, SNAP flags, wm_yr_wk)
    onto the long sales dataframe via the 'd' key.
    """
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
# 4. MERGE PRICES
# =============================================================================

def merge_prices(merged_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join weekly sell prices via (store_id, item_id, wm_yr_wk).
    Missing prices (item not yet on sale) are filled with 0.
    """
    merged = merged_df.merge(prices_df, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    merged["sell_price"] = merged["sell_price"].fillna(0.0)
    print(f"[data] After price merge    : {merged.shape}")
    return merged


# =============================================================================
# 5. FEATURE ENGINEERING
# =============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build model-ready features from the merged dataframe.

    Added features
    --------------
    wday_sin / wday_cos    : cyclical weekday encoding
    month_sin / month_cos  : cyclical month encoding
    has_event              : binary — any national event today?
    lag_7 / lag_28         : sales 7 and 28 days ago (key retail signals)
    roll_mean_7 / _28      : rolling mean (shift-1 to avoid leakage)

    Rows with NaN lags (first 28 days per series) are dropped.
    """
    df = df.copy().sort_values(["id", "date"]).reset_index(drop=True)

    # Cyclical time
    df["wday_sin"]  = np.sin(2 * np.pi * df["wday"]  / 7).astype(np.float32)
    df["wday_cos"]  = np.cos(2 * np.pi * df["wday"]  / 7).astype(np.float32)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12).astype(np.float32)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12).astype(np.float32)

    # Event flag
    df["has_event"] = (~df["event_name_1"].isna()).astype(np.float32)

    # SNAP flags
    for col in ["snap_CA", "snap_TX", "snap_WI"]:
        df[col] = df[col].astype(np.float32)

    # Lag features
    df["lag_7"]  = df.groupby("id")["sales"].shift(7)
    df["lag_28"] = df.groupby("id")["sales"].shift(28)

    # Rolling mean (shift-1 so current day's sales are never used)
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
# 6. TRAIN / VAL / TEST SPLIT  (temporal — never shuffle!)
# =============================================================================

def split_data(
    df: pd.DataFrame,
    val_days:  int = 56,
    test_days: int = 28,
) -> tuple:
    """
    Strict temporal split aligned with the M5 evaluation window.

    Test  : last `test_days` calendar days
    Val   : `val_days` days immediately before test
    Train : everything before val

    Returns (train_df, val_df, test_df)
    """
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
# 7. NORMALISATION  (fit on train only — no data leakage)
# =============================================================================

def normalise(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    zscore_target: bool = True,
) -> tuple:
    """
    Z-score normalise continuous features using TRAIN statistics only.
    Binary and cyclical features are left unchanged.

    Returns (train_df, val_df, test_df, stats)
        stats : dict  col -> {"mean": float, "std": float}
                Pass this to denormalise() when converting predictions back.
    """
    continuous = [
        "sell_price", "lag_7", "lag_28",
        "roll_mean_7", "roll_mean_28", "sales",
    ]
    stats = {}
    # normalise features — always z-scored
    for col in continuous:
        for split in (train_df, val_df, test_df):
            split[col] = np.log1p(split[col].clip(lower=0))
        mean = float(train_df[col].mean())
        std  = float(train_df[col].std()) + 1e-8
        stats[col] = {"mean": mean, "std": std, "log1p": True}
        for split in (train_df, val_df, test_df):
            split[col] = (split[col] - mean) / std

    # NOTE: sales has already been log1p + z-scored above
    # We now FIX it depending on zscore_target
    if not zscore_target:
        # Undo z-score but KEEP log1p
        mean = stats["sales"]["mean"]
        std = stats["sales"]["std"]
        for split in (train_df, val_df, test_df):
            split["sales"] = split["sales"] * std + mean
        stats["sales"] = {"mean": 0.0, "std": 1.0, "log1p": True, "zscore": False,}
    else:
        stats["sales"]["zscore"] = True

    return train_df, val_df, test_df, stats

def fit_normalisation_stats(
    train_df: pd.DataFrame,
    zscore_target: bool = True,
) -> dict:
    """
    Fit normalisation stats on TRAIN rows only.
    """
    continuous = [
        "sell_price", "lag_7", "lag_28",
        "roll_mean_7", "roll_mean_28", "sales",
    ]
    stats = {}
    tmp = train_df.copy()
    for col in continuous:
        tmp[col] = np.log1p(tmp[col].clip(lower=0))
        mean = float(tmp[col].mean())
        std = float(tmp[col].std()) + 1e-8
        stats[col] = {
            "mean": mean,
            "std": std,
            "log1p": True,
        }
    stats["sales"]["zscore"] = bool(zscore_target)
    return stats


def apply_normalisation(
    df: pd.DataFrame,
    stats: dict,
    zscore_target: bool = True,
) -> pd.DataFrame:
    """
    Apply previously fitted TRAIN stats to any dataframe.
    """
    out = df.copy()
    continuous = [
        "sell_price", "lag_7", "lag_28",
        "roll_mean_7", "roll_mean_28", "sales",
    ]
    for col in continuous:
        if col != "sales" and stats[col].get("log1p", False):
            out[col] = np.log1p(out[col].clip(lower=0))

        if col == "sales":
            if zscore_target:
                out[col] = (out[col] - stats[col]["mean"]) / stats[col]["std"]
            else:
                continue  # keep raw counts for NB
            continue
        out[col] = (out[col] - stats[col]["mean"]) / stats[col]["std"]
    return out


def denormalise(preds: np.ndarray, stats: dict, col: str = "sales") -> np.ndarray:
    """Reverse z-score normalisation for `col` (default: 'sales')."""
    out = preds * stats[col]["std"] + stats[col]["mean"]
    if stats[col].get("log1p"):
        out = np.expm1(out)
    return out


# =============================================================================
# 8. PYTORCH DATASET  —  sliding window over a single series
# =============================================================================

class M5SeriesDataset(Dataset):
    """
    Sliding-window dataset for ONE item-store time series.

    Each sample
    -----------
    x : FEATURE_COLS values for timesteps [i : i+seq_len]              (seq_len, N_FEATURES)
    y : TARGET_COL   values for timesteps [i+seq_len : i+seq_len+horizon]  (horizon,)

    Args
    ----
    series_df : DataFrame for a single series, sorted by date.
    seq_len   : Lookback window length.
    horizon   : Number of future steps to predict.
    """

    def __init__(self, series_df: pd.DataFrame, seq_len: int = 28, horizon: int = 28):
        self.seq_len  = seq_len
        self.horizon  = horizon
        self.features = series_df[FEATURE_COLS].values.astype(np.float32)
        self.targets  = series_df[TARGET_COL].values.astype(np.float32)
        self.n        = max(0, len(series_df) - seq_len - horizon + 1)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.seq_len]
        y = self.targets[idx + self.seq_len : idx + self.seq_len + self.horizon]
        return torch.tensor(x), torch.tensor(y)


# =============================================================================
# 9. MULTI-SERIES DATASET  —  all 30 490 item-store series combined
# =============================================================================

class M5Dataset(Dataset):
    """
    Aggregates sliding-window samples from every item-store series into
    one flat Dataset that DataLoader can sample uniformly.

    Args
    ----
    df         : Full featured DataFrame (all series), sorted by id + date.
    seq_len    : Lookback window length.
    horizon    : Forecast horizon.
    max_series : Limit to first N series — useful for quick debugging runs.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        seq_len:    int = 28,
        horizon:    int = 28,
        max_series: int = None,
    ):
        self.datasets = []

        series_ids = df["id"].unique()
        if max_series is not None:
            series_ids = series_ids[:int(max_series)]

        for sid in series_ids:
            sub = df[df["id"] == sid].sort_values("date").reset_index(drop=True)
            ds  = M5SeriesDataset(sub, seq_len, horizon)
            if len(ds) > 0:
                self.datasets.append(ds)

        # Flat (dataset_idx, local_window_idx) index
        self.index = [
            (d_i, s_i)
            for d_i, ds in enumerate(self.datasets)
            for s_i in range(len(ds))
        ]
        print(f"[data] {len(self.datasets)} series  ->  {len(self.index):,} total windows")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        d_i, s_i = self.index[idx]
        return self.datasets[d_i][s_i]


class WindowedM5Dataset(Dataset):
    """
    Window-first dataset built from the FULL per-series dataframe.

    Each sample is assigned to train / val / test according to the
    date of the first target timestep (or equivalently the target block).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int,
        horizon: int,
        split: str,
        val_start_date,
        test_start_date,
        max_series: int = None,
    ):
        self.seq_len = seq_len
        self.horizon = horizon
        self.split = split
        self.samples = []
        self.series_data = {}
        grouped = df.groupby("id")
        series_ids = list(grouped.groups.keys())
        if max_series is not None:
            series_ids = series_ids[:int(max_series)]
        for sid in series_ids:
            sub = grouped.get_group(sid).sort_values("date").reset_index(drop=True)
            self.series_data[sid] = {
                "features": torch.from_numpy(sub[FEATURE_COLS].values.astype(np.float32)),
                "targets": torch.from_numpy(sub[TARGET_COL].values.astype(np.float32)),
                "dates": sub["date"].values.astype("datetime64[ns]")
            }
            features = self.series_data[sid]["features"]
            targets = self.series_data[sid]["targets"]
            dates = self.series_data[sid]["dates"]
            n = len(sub) - seq_len - horizon + 1
            if n <= 0:
                continue
            for i in range(n):
                x_start = i
                x_end = i + seq_len
                y_start = x_end
                y_end = x_end + horizon
                target_start_date = dates[y_start]
                target_end_date = dates[y_end - 1]
                # assign split using the TARGET block
                if split == "train":
                    # all target dates must lie before validation starts
                    if target_end_date < val_start_date:
                        self.samples.append((sid, x_start))
                elif split == "val":
                    # target block starts in val region and ends before test starts
                    if target_start_date >= val_start_date and target_end_date < test_start_date:
                        self.samples.append((sid, x_start))
                elif split == "test":
                    # target block starts in test region
                    if target_start_date >= test_start_date:
                        self.samples.append((sid, x_start))
                else:
                    raise ValueError(f"Unknown split: {split}")
        print(f"[data] {split}: {len(self.samples):,} windows")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sid, i = self.samples[idx]
        sub = self.series_data[sid]
        x = sub["features"][i: i + self.seq_len]
        y = sub["targets"][i + self.seq_len: i + self.seq_len + self.horizon]
        return x, y



def set_seed(seed=None):
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.
    Also configures deterministic CUDA behaviour.
    """
    if seed is None:
        seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    generator = torch.Generator().manual_seed(seed)
    print(f"Random seed set to {seed}")
    return generator, seed

def init_seed(cfg):
    """
    Resolve seed from config, initialise RNGs, and record the final seed.
    Returns the dataloader generator.
    """
    seed = cfg.get("seed")
    generator, seed = set_seed(seed)
    cfg["seed"] = seed
    return generator


def build_dataloaders_old(
    data_dir:    str,
    seq_len:     int = 28,
    horizon:     int = 28,
    batch_size:  int = 256,
    store_id:    str = None,
    max_series:  int = None,
    num_workers: int = 2,
    seed:        int = 42,
    top_k_items: int = None,
    zscore_target: bool = True,
) -> tuple:
    """
    CSV files -> normalised PyTorch DataLoaders in one call.

    Args
    ----
    data_dir    : Directory containing the M5 CSV files.
    seq_len     : Input lookback window (timesteps).
    horizon     : Forecast horizon (timesteps to predict).
    batch_size  : DataLoader batch size.
    max_series  : Cap number of series (e.g. 100 for debugging; None = all 30 490).
    num_workers : DataLoader worker processes.
    seed        : Random seed.

    Returns
    -------
    (train_loader, val_loader, test_loader, stats)
        stats : normalisation stats dict — pass to denormalise() at test time.
    """
    generator, _ = set_seed(seed)

    # sales_df, calendar_df, prices_df = load_or_download_m5(data_dir)
    # if top_k_items is not None:
    #     sales_df = trim_data(sales_df, top_k_items)
    # long_df  = melt_sales(sales_df)
    sales_df, calendar_df, prices_df = load_or_download_m5(data_dir)
    if store_id is not None:
        sales_df = sales_df[sales_df["store_id"] == store_id].reset_index(drop=True)
        print(f"[data] Filtered to {store_id}: {len(sales_df)} series")
    if top_k_items is not None:
        sales_df = trim_data(sales_df, top_k_items)
    long_df = melt_sales(sales_df)
    merged   = merge_calendar(long_df, calendar_df)
    merged   = merge_prices(merged, prices_df)
    featured = engineer_features(merged)

    train_df, val_df, test_df = split_data(featured, val_days=56, test_days=28)
    train_df, val_df, test_df, stats = normalise(train_df, val_df, test_df, zscore_target=zscore_target)

    train_ds = M5Dataset(train_df, seq_len, horizon, max_series)
    val_ds   = M5Dataset(val_df,   seq_len, horizon, max_series)
    test_ds  = M5Dataset(test_df,  seq_len, horizon, max_series)
    _pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, generator=generator, pin_memory=_pin,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=_pin,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=_pin,
    )

    print(f"\n[data] Input  : (seq_len={seq_len}, n_features={N_FEATURES})")
    print(f"[data] Output : (horizon={horizon},)")
    print(f"[data] Features: {FEATURE_COLS}\n")

    return train_loader, val_loader, test_loader, stats

def build_dataloaders(data_dir: str, seq_len: int = 28, horizon: int = 28, batch_size: int = 256, store_id: str = None,
                      max_series: int = None, num_workers: int = 2, seed: int = 42, top_k_items: int = None,
                      zscore_target: bool = True) -> tuple:
    """
    Robust window-first split pipeline:
    1) build full featured dataframe
    2) get calendar boundaries
    3) fit stats on train rows only
    4) normalise full dataframe with train stats
    5) build windows from full series
    6) assign each window to train / val / test by target dates
    """
    generator, _ = set_seed(seed)

    sales_df, calendar_df, prices_df = load_or_download_m5(data_dir)

    if store_id is not None:
        sales_df = sales_df[sales_df["store_id"] == store_id].reset_index(drop=True)
        print(f"[data] Filtered to {store_id}: {len(sales_df)} series")

    if top_k_items is not None:
        sales_df = trim_data(sales_df, top_k_items)

    long_df = melt_sales(sales_df)
    merged = merge_calendar(long_df, calendar_df)
    merged = merge_prices(merged, prices_df)
    featured = engineer_features(merged)

    # Use split_data only to derive date boundaries and train rows for fitting stats
    train_df_raw, val_df_raw, test_df_raw = split_data(featured, val_days=56, test_days=28)
    val_start_date = val_df_raw["date"].values.astype("datetime64[ns]")[0]
    test_start_date = test_df_raw["date"].values.astype("datetime64[ns]")[0]

    print(type(val_start_date), val_start_date.dtype)

    # Fit stats on TRAIN ROWS ONLY
    stats = fit_normalisation_stats(train_df_raw, zscore_target=zscore_target)

    # Apply stats to FULL dataframe
    featured_norm = apply_normalisation(featured, stats, zscore_target=zscore_target)

    # Build window-first datasets
    train_ds = WindowedM5Dataset(featured_norm, seq_len=seq_len, horizon=horizon, split="train", val_start_date=val_start_date,
                                 test_start_date=test_start_date, max_series=max_series)
    val_ds = WindowedM5Dataset(featured_norm, seq_len=seq_len, horizon=horizon, split="val", val_start_date=val_start_date,
                               test_start_date=test_start_date, max_series=max_series)
    test_ds = WindowedM5Dataset(featured_norm, seq_len=seq_len, horizon=horizon, split="test", val_start_date=val_start_date,
                                test_start_date=test_start_date, max_series=max_series)
    _pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=generator,pin_memory=_pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=_pin)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=_pin)

    print(f"\n[data] Input  : (seq_len={seq_len}, n_features={N_FEATURES})")
    print(f"[data] Output : (horizon={horizon},)")
    print(f"[data] Features: {FEATURE_COLS}")
    print(f"[data] Train batches: {len(train_loader)}")
    print(f"[data] Val batches  : {len(val_loader)}")
    print(f"[data] Test batches : {len(test_loader)}\n")

    print(f"[DEBUG] Train windows: {len(train_loader.dataset):,}")
    print(f"[DEBUG] Val windows  : {len(val_loader.dataset):,}")
    print(f"[DEBUG] Test windows : {len(test_loader.dataset):,}")

    return train_loader, val_loader, test_loader, stats


# =============================================================================
# 10. BATCHED PIPELINE — loads from preprocessed pickle files
# =============================================================================

# Features available in batches that are useful for LSTM/GRU
# Matches the cookbook section 4.3 (Deterministic LSTM/GRU feature set)
BATCH_FEATURE_COLS = [
    "sell_price",
    "wday_sin", "wday_cos",
    "month_sin", "month_cos",
    "is_event_day",
    "snap",
    "sales_lag_7", "sales_lag_28",
    "sales_roll_mean_7", "sales_roll_mean_28",
]
N_BATCH_FEATURES = len(BATCH_FEATURE_COLS)  # 11


def load_batches(
        processed_dir: str,
        store_id: str = "CA_3",
        max_series: int = None,
) -> pd.DataFrame:
    """
    Load preprocessed pickle batches and filter to a single store.

    Because keep_static_in_batches=False, the batch files don't contain
    store_id. We join series_info first to get CA_3 series IDs, then
    filter each batch as we load it.

    Args
    ----
    processed_dir : Path to m5_processed_validation/ folder.
    store_id      : Store to filter to (default CA_3).
    max_series    : Cap number of series for quick runs.

    Returns
    -------
    Single concatenated DataFrame for the requested store,
    observed rows only (is_future == 0), sorted by id + d_num.
    """
    processed_dir = Path(processed_dir)
    features_dir = processed_dir / "features"
    static_dir = processed_dir / "static"

    # --- Load series info to get CA_3 IDs ---
    series_info = pd.read_pickle(
        static_dir / "validation_series_info.pkl.gz", compression="gzip"
    )
    ca3_ids = series_info[series_info["store_id"] == store_id]["id"].values
    if max_series is not None:
        ca3_ids = ca3_ids[:int(max_series)]
    ca3_id_set = set(ca3_ids)
    print(f"[data] {store_id} series : {len(ca3_ids)}")

    # --- Load batches, filtering to CA_3 as we go ---
    batch_files = sorted(features_dir.glob("validation_features_batch_*.pkl.gz"))
    chunks = []
    for bf in batch_files:
        chunk = pd.read_pickle(bf, compression="gzip")
        chunk = chunk[chunk["id"].isin(ca3_id_set)]
        if len(chunk) > 0:
            chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)

    # Keep only observed rows (drop future placeholder rows)
    df = df[df["is_future"] == 0].copy()

    # Drop rows where lag features are NaN (first 28 days of each series)
    df = df.dropna(subset=[c for c in BATCH_FEATURE_COLS if c in df.columns])

    # Sort for correct sliding window construction
    df = df.sort_values(["id", "d_num"]).reset_index(drop=True)

    print(f"[data] Loaded {len(df):,} rows for {store_id} "
          f"({df['id'].nunique()} series, {df['d_num'].nunique()} days)")
    return df


def split_data_by_dnum(
    df:        pd.DataFrame,
    train_end: int = 1857,
    val_end:   int = 1885,
    seq_len:   int = 28,
) -> tuple:
    """
    Temporal split using recommended d_num boundaries.

    Defaults match validation_recommended_splits.json fold_2:
        Train : d_1   to d_1857
        Val   : d_1858 to d_1885
        Test  : d_1886 to d_1913

    Returns (train_df, val_df, test_df)
    """
    train_df = df[df["d_num"] <= train_end].copy()
    val_df = df[(df["d_num"] > train_end - seq_len) & (df["d_num"] <= val_end)].copy()
    test_df = df[df["d_num"] > val_end - seq_len].copy()

    print(f"[data] Train : d_1 -> d_{train_end}  ({len(train_df):,} rows)")
    print(f"[data] Val   : d_{train_end + 1} -> d_{val_end}  ({len(val_df):,} rows)")
    print(f"[data] Test  : d_{val_end + 1}+  ({len(test_df):,} rows)")
    return train_df, val_df, test_df


def split_data_rolling(
    df:        pd.DataFrame,
    train_end: int = 1800,
    test_start: int = 1886,
    seq_len:   int = 28,
    horizon:   int = 28,
) -> tuple:
    """
    Rolling validation split.

    Train : d_1 to d_1800
    Val   : four rolling 28-day horizons (d_1801-1828, d_1829-1856,
            d_1857-1884, d_1885-1912) with seq_len days of context prepended
    Test  : d_1886 to d_1913 — held out, touched only once at final eval

    Returns (train_df, val_df, test_df)
    """
    # Val covers d_1801 to d_1885 with seq_len context prepended
    val_start_ctx = train_end - seq_len + 1
    val_end       = test_start - 1          # d_1885

    train_df = df[df["d_num"] <= train_end].copy()
    val_df   = df[(df["d_num"] >= val_start_ctx) & (df["d_num"] <= val_end)].copy()
    test_df  = df[df["d_num"] >= (test_start - seq_len)].copy()

    print(f"[data] Train : d_1 -> d_{train_end}  ({len(train_df):,} rows)")
    print(f"[data] Val   : d_{val_start_ctx} -> d_{val_end}  ({len(val_df):,} rows)")
    print(f"[data] Test  : d_{test_start}+  ({len(test_df):,} rows)")
    return train_df, val_df, test_df

def normalise_batched(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        zscore_target: bool = True,
) -> tuple:
    """
    zscore_target : If True (default), z-score the sales target after log1p.
                    Set False for probabilistic models using NB NLL —
                    log1p only keeps sales as non-negative counts.

    Returns (train_df, val_df, test_df, stats)
    """
    continuous = [
        "sell_price",
        "sales_lag_7", "sales_lag_28",
        "sales_roll_mean_7", "sales_roll_mean_28",
        "sales",
    ]
    stats = {}
    # normalise features — always z-scored
    for col in continuous:
        for split in (train_df, val_df, test_df):
            split[col] = np.log1p(split[col].clip(lower=0))
        mean = float(train_df[col].mean())
        std = float(train_df[col].std()) + 1e-8
        stats[col] = {"mean": mean, "std": std, "log1p": True}
        for split in (train_df, val_df, test_df):
            split[col] = (split[col] - mean) / std

    # sales target — log1p always, z-score only if zscore_target=True
    for split in (train_df, val_df, test_df):
        split["sales"] = np.log1p(split["sales"].clip(lower=0))
    if zscore_target:
        mean = float(train_df["sales"].mean())
        std = float(train_df["sales"].std()) + 1e-8
        stats["sales"] = {"mean": mean, "std": std, "log1p": True}
        for split in (train_df, val_df, test_df):
            split["sales"] = (split["sales"] - mean) / std
    else:
        stats["sales"] = {"mean": 0.0, "std": 1.0, "log1p": True, "zscore": False}

    return train_df, val_df, test_df, stats


def build_dataloaders_from_batches(
        data_dir: str,
        seq_len: int = 28,
        horizon: int = 28,
        batch_size: int = 256,
        store_id: str = "CA_3",
        max_series: int = None,
        num_workers: int = 2,
        seed: int = 42,
        zscore_target: bool = True,
) -> tuple:
    """
    Pickle batches -> normalised PyTorch DataLoaders in one call.

    Loads preprocessed features from
    data_dir/m5_processed_validation/, filters to store_id,
    applies the recommended temporal split, normalises, and
    returns DataLoaders ready for the GRU/LSTM.

    Args
    ----
    data_dir    : Group data/ folder (contains m5_processed_validation/).
    seq_len     : Input lookback window (timesteps).
    horizon     : Forecast horizon (timesteps to predict).
    batch_size  : DataLoader batch size.
    store_id    : Store to train on (default CA_3).
    max_series  : Cap series count for quick debug runs.
    num_workers : DataLoader workers.
    seed        : Random seed.

    Returns
    -------
    (train_loader, val_loader, test_loader, stats)
    """
    generator, _ = set_seed(seed)

    processed_dir = Path(data_dir) / "m5_processed_validation"

    # Load + filter to store
    df = load_batches(processed_dir, store_id=store_id, max_series=max_series)

    # Split using recommended boundaries
    #train_df, val_df, test_df = split_data_by_dnum(df, seq_len=seq_len)
    train_df, val_df, test_df = split_data_rolling(df, seq_len=seq_len, horizon=horizon)

    # Normalise (fit on train only)
    train_df, val_df, test_df, stats = normalise_batched(train_df, val_df, test_df, zscore_target=zscore_target)

    # Build datasets using existing M5Dataset/M5SeriesDataset classes
    # but pointing at BATCH_FEATURE_COLS
    # We temporarily override the global for dataset construction
    train_ds = _make_batch_dataset(train_df, seq_len, horizon)
    val_ds = _make_batch_dataset(val_df, seq_len, horizon)
    test_ds = _make_batch_dataset(test_df, seq_len, horizon)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, generator=generator,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )

    print(f"\n[data] Store    : {store_id}")
    print(f"[data] Input    : (seq_len={seq_len}, n_features={N_BATCH_FEATURES})")
    print(f"[data] Output   : (horizon={horizon},)")
    print(f"[data] Features : {BATCH_FEATURE_COLS}\n")

    return train_loader, val_loader, test_loader, stats

# class BatchSeriesDataset(Dataset):
#     """Pre-computes all windows at construction time for fast GPU feeding."""
#     def __init__(self, series_df, seq_len, horizon):
#         features = series_df[BATCH_FEATURE_COLS].values.astype(np.float32)
#         targets  = series_df["sales"].values.astype(np.float32)
#         n = max(0, len(series_df) - seq_len - horizon + 1)
#         if n == 0:
#             self.x = torch.zeros((0, seq_len, len(BATCH_FEATURE_COLS)), dtype=torch.float32)
#             self.y = torch.zeros((0, horizon), dtype=torch.float32)
#         else:
#             self.x = torch.tensor(
#                 np.stack([features[i: i + seq_len] for i in range(n)])
#             )
#             self.y = torch.tensor(
#                 np.stack([targets[i + seq_len: i + seq_len + horizon] for i in range(n)])
#             )
#
#     def __len__(self):
#         return len(self.x)
#
#     def __getitem__(self, idx):
#         return self.x[idx], self.y[idx]

class BatchSeriesDataset(Dataset):
    """On-the-fly slicing with fast torch.from_numpy — minimal RAM usage."""
    def __init__(self, series_df, seq_len, horizon):
        self.seq_len  = seq_len
        self.horizon  = horizon
        self.features = series_df[BATCH_FEATURE_COLS].values.astype(np.float32)
        self.targets  = series_df["sales"].values.astype(np.float32)
        self.n        = max(0, len(series_df) - seq_len - horizon + 1)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = self.features[idx: idx + self.seq_len].copy()
        y = self.targets[idx + self.seq_len: idx + self.seq_len + self.horizon].copy()
        return torch.from_numpy(x), torch.from_numpy(y)


# class BatchDataset(Dataset):
#     """Concatenates all series windows into one flat tensor dataset."""
#     def __init__(self, df, seq_len, horizon):
#         all_x, all_y = [], []
#         count = 0
#         for sid in df["id"].unique():
#             sub = df[df["id"] == sid].sort_values("d_num").reset_index(drop=True)
#             ds  = BatchSeriesDataset(sub, seq_len, horizon)
#             if len(ds) > 0:
#                 all_x.append(ds.x)
#                 all_y.append(ds.y)
#                 count += 1
#         self.x = torch.cat(all_x, dim=0)
#         self.y = torch.cat(all_y, dim=0)
#         print(f"[data] {count} series -> {len(self.x):,} windows")
#
#     def __len__(self):
#         return len(self.x)
#
#     def __getitem__(self, idx):
#         return self.x[idx], self.y[idx]

class BatchDataset(Dataset):
    """Index-based aggregation — no data duplication in RAM."""
    def __init__(self, df, seq_len, horizon):
        self.datasets = []
        for sid in df["id"].unique():
            sub = df[df["id"] == sid].sort_values("d_num").reset_index(drop=True)
            ds  = BatchSeriesDataset(sub, seq_len, horizon)
            if len(ds) > 0:
                self.datasets.append(ds)
        self.index = [
            (d_i, s_i)
            for d_i, ds in enumerate(self.datasets)
            for s_i in range(len(ds))
        ]
        print(f"[data] {len(self.datasets)} series -> {len(self.index):,} windows")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        d_i, s_i = self.index[idx]
        return self.datasets[d_i][s_i]


def _make_batch_dataset(df: pd.DataFrame, seq_len: int, horizon: int) -> Dataset:
    """Build a pre-computed BatchDataset. Use num_workers=0 only."""
    return BatchDataset(df, seq_len, horizon)


