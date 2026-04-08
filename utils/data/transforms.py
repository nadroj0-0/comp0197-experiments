import numpy as np
import pandas as pd


def get_vocab_sizes(featured_df: pd.DataFrame) -> dict:
    hierarchy_cols = ['state_id_int', 'store_id_int', 'cat_id_int', 'dept_id_int']
    return {col: int(featured_df[col].max()) + 1 for col in hierarchy_cols}


def trim_data(df: pd.DataFrame, num_items: int,
              sampling: str = "top") -> pd.DataFrame:
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
        n = num_items
        mid_lo = total // 2 - n // 2
        mid_hi = mid_lo + n
        top_df = df_sorted.iloc[:n]
        middle_df = df_sorted.iloc[mid_lo:mid_hi]
        bottom_df = df_sorted.iloc[total - n:]
        trimmed_df = (
            pd.concat([top_df, middle_df, bottom_df])
              .drop_duplicates()
              .reset_index(drop=True)
        )
        print(f"[data] Original items: {total} | "
              f"Stratified {n}+{n}+{n} = {len(trimmed_df)} series")
    else:
        trimmed_df = df_sorted.head(num_items)
        print(f"[data] Original items: {total} | Selected top-{num_items} by volume")

    trimmed_df = (
        trimmed_df
          .drop(columns=["total_sales_volume"])
          .reset_index(drop=True)
    )
    print(f"[data] Trimmed shape : {trimmed_df.shape}")
    return trimmed_df


def melt_sales(sales_df: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in sales_df.columns if c.startswith("d_")]
    long_df = sales_df.melt(
        id_vars=id_cols,
        value_vars=day_cols,
        var_name="d",
        value_name="sales",
    )
    print(f"[data] After melt  : {long_df.shape}")
    return long_df


def encode_hierarchy(df: pd.DataFrame, include_dow: bool = False) -> pd.DataFrame:
    df = df.copy()
    for col in ["state_id", "store_id", "cat_id", "dept_id"]:
        df[f"{col}_int"] = df[col].astype("category").cat.codes.astype(np.float32)
    if include_dow:
        df["d_num"] = df["d"].str.extract(r"d_(\d+)").astype(int)
        df["day_of_week"] = ((df["d_num"] + 5) % 7).astype(np.float32)
    print(f"[data] After hierarchy encoding : {df.shape}")
    return df


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


def merge_prices(merged_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
    merged = merged_df.merge(prices_df, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    merged["sell_price"] = merged["sell_price"].fillna(0.0)
    print(f"[data] After price merge    : {merged.shape}")
    return merged


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["id", "date"]).reset_index(drop=True)
    df["wday_sin"] = np.sin(2 * np.pi * df["wday"] / 7).astype(np.float32)
    df["wday_cos"] = np.cos(2 * np.pi * df["wday"] / 7).astype(np.float32)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12).astype(np.float32)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12).astype(np.float32)
    df["has_event"] = (~df["event_name_1"].isna()).astype(np.float32)
    for col in ["snap_CA", "snap_TX", "snap_WI"]:
        df[col] = df[col].astype(np.float32)
    df["lag_7"] = df.groupby("id")["sales"].shift(7)
    df["lag_28"] = df.groupby("id")["sales"].shift(28)
    df["roll_mean_7"] = df.groupby("id")["sales"].transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).mean()
    )
    df["roll_mean_28"] = df.groupby("id")["sales"].transform(
        lambda x: x.shift(1).rolling(28, min_periods=1).mean()
    )
    df = df.dropna(subset=["lag_7", "lag_28"]).reset_index(drop=True)
    print(f"[data] After feature engineering : {df.shape}")
    return df


def split_data(
    df: pd.DataFrame,
    val_days: int = 112,
    test_days: int = 56,
) -> tuple:
    all_dates = sorted(df["date"].unique())
    test_start = all_dates[-test_days]
    val_start = all_dates[-(test_days + val_days)]
    train_df = df[df["date"] < val_start].copy()
    val_df = df[(df["date"] >= val_start) & (df["date"] < test_start)].copy()
    test_df = df[df["date"] >= test_start].copy()
    print(f"[data] Train : {train_df['date'].min().date()} -> {train_df['date'].max().date()}  "
          f"({len(train_df):,} rows)")
    print(f"[data] Val   : {val_df['date'].min().date()} -> {val_df['date'].max().date()}  "
          f"({len(val_df):,} rows)")
    print(f"[data] Test  : {test_df['date'].min().date()} -> {test_df['date'].max().date()}  "
          f"({len(test_df):,} rows)")
    return train_df, val_df, test_df


def fit_normalisation_stats(train_df: pd.DataFrame, zscore_target: bool = True) -> dict:
    stats = {}
    tmp = train_df.copy()
    tmp["sales"] = np.log1p(tmp["sales"].clip(lower=0))
    mean = float(tmp["sales"].mean())
    std = float(tmp["sales"].std()) + 1e-8
    stats["sales"] = {"mean": mean, "std": std, "log1p": True, "zscore": zscore_target}
    return stats


def apply_normalisation(df: pd.DataFrame, stats: dict, zscore_target: bool = True) -> pd.DataFrame:
    out = df.copy()
    out["sales"] = np.log1p(out["sales"].clip(lower=0))
    if zscore_target:
        out["sales"] = (out["sales"] - stats["sales"]["mean"]) / stats["sales"]["std"]
    return out


def denormalise(preds: np.ndarray, stats: dict, col: str = "sales") -> np.ndarray:
    out = preds * stats[col]["std"] + stats[col]["mean"]
    if stats[col].get("log1p"):
        out = np.expm1(out)
    return out
