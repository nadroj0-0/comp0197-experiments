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
    "sales_yen":            ["sales","sell_price","is_available","wday",
                             "month","year","snap_CA","snap_TX","snap_WI",
                             "has_event"],
    "sales_yen_hierarchy":  ["sales", "sell_price", "is_available", "wday",
                             "month", "year", "snap_CA", "snap_TX", "snap_WI",
                             "has_event", "state_id_int", "store_id_int", "cat_id_int",
                             "dept_id_int"],
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

def get_vocab_sizes(featured_df: pd.DataFrame) -> dict:
    """
    Compute vocabulary sizes for each hierarchy column.
    Returns dict e.g. {'state_id_int': 3, 'store_id_int': 10, ...}
    """
    hierarchy_cols = ['state_id_int', 'store_id_int', 'cat_id_int', 'dept_id_int']
    return {col: int(featured_df[col].max()) + 1 for col in hierarchy_cols}

def trim_data(df: pd.DataFrame, num_items: int,
              sampling: str = "top") -> pd.DataFrame:
    """
    Select series from the full dataset.

    sampling:
      "top"        — top-N by total sales volume (default, matches teammate)
      "stratified" — equal thirds from top/middle/bottom of volume ranking.
                     num_items is the per-stratum count, total = num_items * 3.
                     Used for hyperparameter search to cover full distribution.
      "all"        — entire dataset, num_items is ignored.
                     Used for Phase 2 full training.
    """
    if df.empty:
        return df
    df = df.copy()
    day_cols = [c for c in df.columns if c.startswith("d_")]
    df["total_sales_volume"] = df[day_cols].sum(axis=1)
    df_sorted = (
        df.sort_values("total_sales_volume", ascending=False)
          .reset_index(drop=True)
    )
    total = len(df_sorted)

    if sampling == "all":
        trimmed_df = df_sorted
        print(f"[data] Original items: {total} | Using full dataset (sampling=all)")

    elif sampling == "stratified":
        n         = num_items
        mid_lo    = total // 2 - n // 2
        mid_hi    = mid_lo + n
        top_df    = df_sorted.iloc[:n]
        middle_df = df_sorted.iloc[mid_lo:mid_hi]
        bottom_df = df_sorted.iloc[total - n:]
        trimmed_df = (
            pd.concat([top_df, middle_df, bottom_df])
              .drop_duplicates()
              .reset_index(drop=True)
        )
        print(f"[data] Original items: {total} | "
              f"Stratified {n}+{n}+{n} = {len(trimmed_df)} series")

    else:  # "top" — default
        trimmed_df = df_sorted.head(num_items)
        print(f"[data] Original items: {total} | Selected top-{num_items} by volume")

    trimmed_df = (
        trimmed_df
          .drop(columns=["total_sales_volume"])
          .reset_index(drop=True)
    )
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
    val_days:  int = 112,
    test_days: int = 56,
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
        item_weights: np.ndarray = None,
        series_ids=None
    ):
        self.seq_len        = seq_len
        self.horizon        = horizon
        self.autoregressive = autoregressive
        self.feature_cols   = feature_cols
        self.split          = split
        self.samples        = []
        self.series_data    = {}
        # per-series revenue weights — only set for wquantile training loader
        # None means return (x, y); array means return (x, y, weight)
        self.item_weights   = item_weights
        # build once for O(1) lookup in __getitem__ when weights are used
        self._series_idx_map: dict = {}

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

        # build series index map once — used by __getitem__ for weight lookup
        self._series_idx_map = {sid: i for i, sid in enumerate(self.series_ids)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sid, i = self.samples[idx]
        sub = self.series_data[sid]
        x = sub["features"][i : i + self.seq_len]
        if self.autoregressive:
            y = sub["targets"][i + self.seq_len]
        else:
            y = sub["targets"][i + self.seq_len : i + self.seq_len + self.horizon]

        if self.item_weights is not None:
            series_idx = self._series_idx_map[sid]
            w = torch.tensor(self.item_weights[series_idx], dtype=torch.float32)
            return x, y, w

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
    sampling:       str  = "top",
    include_weights: bool = False,
    split_protocol: str = "default",
    weight_protocol: str = "default",
) -> tuple:
    """
    Build windowed M5 loaders for training, validation, and test.

    The key switches here are the sampling regime, feature set, split
    protocol, and whether weighted training batches are needed.
    """
    generator, _ = set_seed(seed)
    feature_cols  = get_feature_cols(feature_set)
    feature_index = {col: i for i, col in enumerate(feature_cols)}
    n_features    = len(feature_cols)

    # ------------------------------------------------------------------
    # 1. Load + trim
    # ------------------------------------------------------------------
    sales_df, calendar_df, prices_df = load_or_download_m5(data_dir)
    sales_df = trim_data(sales_df, top_k_series, sampling=sampling)

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


    elif feature_set in ("sales_yen", "sales_yen_hierarchy"):
        include_hierarchy = "hierarchy" in feature_set
        # 1. Merge calendar
        featured = merge_calendar(long_df, calendar_df)
        # 2. Merge prices (no fill yet)
        featured = featured.merge(prices_df, on=["store_id", "item_id", "wm_yr_wk"], how="left")
        # 3. Create availability BEFORE filling
        featured["is_available"] = featured["sell_price"].notna().astype(np.float32)
        # 4. Create event feature BEFORE filling
        featured["has_event"] = (~featured["event_name_1"].isna()).astype(np.float32)
        # 5. Add hierarchy (if needed)
        if include_hierarchy:
            featured = encode_hierarchy(featured, include_dow=False)
        # 6. Sort before group ops
        featured = featured.sort_values(["id", "date"]).reset_index(drop=True)
        # 7. Forward fill prices per series
        featured["sell_price"] = (featured.groupby("id")["sell_price"].transform(lambda x: x.ffill()))
        # 8. Fill remaining missing prices
        featured["sell_price"] = featured["sell_price"].fillna(0.0)
        # 9. Fill event columns AFTER has_event
        for col in ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
            featured[col] = featured[col].fillna("none")
        print(f"[data] Yen-style pipeline ({feature_set}): {featured.shape}")

    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    series_ids = featured["id"].drop_duplicates().tolist()

    # ------------------------------------------------------------------
    # 3. Temporal split boundaries
    # ------------------------------------------------------------------
    # Split protocol controls the validation/test window lengths.
    if split_protocol == "yen_v1":
        _val_days, _test_days = 112, 56
        print(f"[data] Protocol: yen_v1 — val=112 days, test=56 days")
    else:
        _val_days, _test_days = 28, 28
        print(f"[data] Protocol: default — val=28 days, test=28 days")

    train_df_raw, val_df_raw, test_df_raw = split_data(
        featured, val_days=_val_days, test_days=_test_days
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
    # 5. Compute revenue weights (only for weighted pinball training)
    # ------------------------------------------------------------------
    train_item_weights = None
    if include_weights:

        if weight_protocol == "yen_v1":
            # Daily revenue weights over the last 28 training days
            dates = np.sort(train_df_raw["date"].unique())
            last28_cutoff = dates[-28]
            last28_df = train_df_raw[train_df_raw["date"] >= last28_cutoff].copy()
            if "sell_price" not in last28_df.columns:
                raise KeyError(
                    "sell_price missing from train_df_raw while computing "
                    "yen_v1 revenue weights"
                )

            train_rev = (
                (last28_df["sales"] * last28_df["sell_price"])
                .groupby(last28_df["id"]).sum()
            )

            train_rev = train_rev.reindex(series_ids).fillna(0.0)

            item_revs = train_rev.values.astype(np.float32)

        else:
            # Older fallback weighting rule
            prices_df_w = prices_df.copy()

            avg_prices = (
                prices_df_w.groupby(["item_id", "store_id"])["sell_price"]
                .mean().reset_index()
            )

            sales_w = sales_df.merge(
                avg_prices, on=["item_id", "store_id"], how="left"
            )

            sales_w["sell_price"] = sales_w["sell_price"].fillna(
                prices_df_w["sell_price"].median()
            )

            day_cols_w = [c for c in sales_df.columns if c.startswith("d_")]
            train_day_cols = day_cols_w[:-28]

            item_vols = sales_w[train_day_cols].sum(axis=1).values
            item_revs = item_vols * sales_w["sell_price"].values

        train_item_weights = (item_revs / item_revs.sum()).astype(np.float32)

        print(
            f"[data] Revenue weights computed "
            f"({len(train_item_weights)} series, protocol={weight_protocol})"
        )

    # ------------------------------------------------------------------
    # 6. Build windowed datasets
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
    # weights only on train — val and test always return (x, y)
    train_ds = WindowedM5Dataset(featured, split="train",
                                 item_weights=train_item_weights, series_ids=series_ids, **shared)
    val_ds   = WindowedM5Dataset(featured, split="val",   **shared)
    test_ds  = WindowedM5Dataset(featured, split="test",  **shared)

    # ------------------------------------------------------------------
    # 7. DataLoaders
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

    # ------------------------------------------------------------------
    # 8. Compute vocab sizes for hierarchy embedding tables
    # Only meaningful when hierarchy columns are present in feature_set.
    # Returns empty dict for sales_only — builders check before use.
    # ------------------------------------------------------------------
    hierarchy_cols = ["state_id_int", "store_id_int", "cat_id_int", "dept_id_int"]
    if any(c in feature_cols for c in hierarchy_cols):
        vocab_sizes = get_vocab_sizes(featured)
        print(f"[data] Vocab sizes  : { {k: v for k, v in vocab_sizes.items()} }")
    else:
        vocab_sizes = {}

    return train_loader, val_loader, test_loader, stats, vocab_sizes, feature_index
