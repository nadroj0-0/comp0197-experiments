
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


STATIC_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
PRICE_FEATURE_COLS = [
    "store_id",
    "item_id",
    "wm_yr_wk",
    "sell_price",
    "release_wm_yr_wk",
    "release_d",
    "price_lag_1w",
    "price_change_1w",
    "price_pct_change_1w",
    "price_roll_mean_4w",
    "price_roll_mean_13w",
    "price_roll_mean_52w",
    "price_rel_4w",
    "price_rel_13w",
    "price_rel_52w",
    "price_rel_cat_store",
    "price_rel_dept_store",
    "price_rank_dept_store",
    "price_change_flag_1w",
]
DEFAULT_LAGS = [1, 7, 14, 28, 56]
DEFAULT_ROLL_WINDOWS = [7, 28, 56]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Complete preprocessing pipeline for the M5 Walmart forecasting data."
    )
    parser.add_argument("--sales-path", required=True, help="Path to sales_train_validation.csv or sales_train_evaluation.csv")
    parser.add_argument("--calendar-path", required=True, help="Path to calendar.csv")
    parser.add_argument("--prices-path", required=True, help="Path to sell_prices.csv")
    parser.add_argument("--output-dir", required=True, help="Directory where processed outputs will be written")
    parser.add_argument("--dataset-name", default="validation", choices=["validation", "evaluation"], help="Name used in output metadata")
    parser.add_argument("--batch-size", type=int, default=1024, help="Number of series to melt per batch")
    parser.add_argument("--output-format", default="pickle", choices=["pickle", "csv"], help="Batch file format")
    parser.add_argument("--compression", default="gzip", choices=["gzip", "bz2", "xz", "zip", "none"], help="Compression for csv/pickle outputs")
    parser.add_argument("--start-d", type=int, default=1, help="First day number to keep from the sales history")
    parser.add_argument("--end-d", type=int, default=None, help="Last observed day number to keep from the sales history")
    parser.add_argument("--add-future-days", action="store_true", help="Append rows for future known covariates up to the end of calendar.csv")
    parser.add_argument("--drop-pre-release", action="store_true", help="Drop rows before the item first appeared in sell_prices")
    parser.add_argument("--keep-static-in-batches", action="store_true", help="Keep repeated static identifiers in every output batch")
    parser.add_argument("--build-hierarchy-features", action="store_true", help="Build store/state/category-store/department-store aggregate lag features")
    parser.add_argument("--min-history-for-roll", type=int, default=1, help="Minimum periods for rolling statistics")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def maybe_compression(name: str) -> Optional[str]:
    return None if name == "none" else name


def get_suffix(fmt: str, compression: str) -> str:
    comp = maybe_compression(compression)
    if fmt == "pickle":
        if comp is None:
            return ".pkl"
        return {
            "gzip": ".pkl.gz",
            "bz2": ".pkl.bz2",
            "xz": ".pkl.xz",
            "zip": ".pkl.zip",
        }.get(comp, ".pkl")
    if fmt == "csv":
        if comp is None:
            return ".csv"
        return {
            "gzip": ".csv.gz",
            "bz2": ".csv.bz2",
            "xz": ".csv.xz",
            "zip": ".csv.zip",
        }.get(comp, ".csv")
    raise ValueError(f"Unsupported format: {fmt}")


def save_frame(df: pd.DataFrame, path: Path, fmt: str, compression: str = "gzip") -> None:
    comp = maybe_compression(compression)
    if fmt == "pickle":
        if comp is None:
            df.to_pickle(path)
        else:
            df.to_pickle(path, compression=comp)
    elif fmt == "csv":
        if comp is None:
            df.to_csv(path, index=False)
        else:
            df.to_csv(path, index=False, compression=comp)
    else:
        raise ValueError(f"Unsupported format: {fmt}")


def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcast numeric columns and convert low-cardinality object columns to categories.
    """
    df = df.copy()
    for col in df.columns:
        col_type = df[col].dtype

        if pd.api.types.is_datetime64_any_dtype(col_type):
            continue

        if pd.api.types.is_object_dtype(col_type):
            num_unique = df[col].nunique(dropna=False)
            num_total = len(df[col])
            if num_total > 0 and num_unique / num_total < 0.5:
                df[col] = df[col].astype("category")
            continue

        if pd.api.types.is_integer_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif pd.api.types.is_float_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast="float")

    return df


def build_code_map(values: Sequence[str]) -> Dict[str, int]:
    unique_vals = pd.Series(values, dtype="object").fillna("None").astype(str).drop_duplicates().tolist()
    return {v: i for i, v in enumerate(unique_vals)}


def prepare_calendar(calendar_path: str) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    cal = pd.read_csv(calendar_path)
    cal["date"] = pd.to_datetime(cal["date"])
    cal["d_num"] = cal["d"].str[2:].astype(np.int16)

    for col in ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
        cal[col] = cal[col].fillna("None").astype(str)

    cal["is_weekend"] = cal["weekday"].isin(["Saturday", "Sunday"]).astype(np.int8)
    cal["week_of_year"] = cal["date"].dt.isocalendar().week.astype(np.int16)
    cal["quarter"] = cal["date"].dt.quarter.astype(np.int8)
    cal["day_of_month"] = cal["date"].dt.day.astype(np.int8)
    cal["day_of_year"] = cal["date"].dt.dayofyear.astype(np.int16)
    cal["is_month_start"] = cal["date"].dt.is_month_start.astype(np.int8)
    cal["is_month_end"] = cal["date"].dt.is_month_end.astype(np.int8)
    cal["is_quarter_start"] = cal["date"].dt.is_quarter_start.astype(np.int8)
    cal["is_quarter_end"] = cal["date"].dt.is_quarter_end.astype(np.int8)
    cal["is_year_start"] = cal["date"].dt.is_year_start.astype(np.int8)
    cal["is_year_end"] = cal["date"].dt.is_year_end.astype(np.int8)
    cal["is_event_day"] = ((cal["event_name_1"] != "None") | (cal["event_name_2"] != "None")).astype(np.int8)
    cal["month_sin"] = np.sin(2 * np.pi * cal["month"] / 12.0).astype(np.float32)
    cal["month_cos"] = np.cos(2 * np.pi * cal["month"] / 12.0).astype(np.float32)
    cal["wday_sin"] = np.sin(2 * np.pi * cal["wday"] / 7.0).astype(np.float32)
    cal["wday_cos"] = np.cos(2 * np.pi * cal["wday"] / 7.0).astype(np.float32)

    event_maps = {
        "event_name_1": build_code_map(cal["event_name_1"].tolist()),
        "event_type_1": build_code_map(cal["event_type_1"].tolist()),
        "event_name_2": build_code_map(cal["event_name_2"].tolist()),
        "event_type_2": build_code_map(cal["event_type_2"].tolist()),
    }
    for col, mapping in event_maps.items():
        cal[f"{col}_code"] = cal[col].map(mapping).astype(np.int16)

    keep_cols = [
        "d",
        "d_num",
        "date",
        "wm_yr_wk",
        "weekday",
        "wday",
        "month",
        "year",
        "week_of_year",
        "quarter",
        "day_of_month",
        "day_of_year",
        "is_weekend",
        "is_month_start",
        "is_month_end",
        "is_quarter_start",
        "is_quarter_end",
        "is_year_start",
        "is_year_end",
        "is_event_day",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
        "event_name_1_code",
        "event_type_1_code",
        "event_name_2_code",
        "event_type_2_code",
        "snap_CA",
        "snap_TX",
        "snap_WI",
        "month_sin",
        "month_cos",
        "wday_sin",
        "wday_cos",
    ]
    cal = reduce_mem_usage(cal[keep_cols])
    return cal, event_maps


def get_day_cols_from_header(sales_path: str) -> List[str]:
    header = pd.read_csv(sales_path, nrows=0)
    return [c for c in header.columns if c.startswith("d_")]


def load_sales_wide(sales_path: str, start_d: int = 1, end_d: Optional[int] = None) -> Tuple[pd.DataFrame, List[str]]:
    all_day_cols = get_day_cols_from_header(sales_path)
    if end_d is None:
        end_d = max(int(c.split("_")[1]) for c in all_day_cols)

    keep_day_cols = [c for c in all_day_cols if start_d <= int(c.split("_")[1]) <= end_d]
    usecols = STATIC_COLS + keep_day_cols

    dtype_map = {col: "category" for col in STATIC_COLS}
    dtype_map.update({col: np.int16 for col in keep_day_cols})

    sales = pd.read_csv(sales_path, usecols=usecols, dtype=dtype_map)
    return sales, keep_day_cols


def compute_first_nonzero_day(sales: pd.DataFrame, day_cols: List[str]) -> np.ndarray:
    values = sales[day_cols].to_numpy(dtype=np.int16, copy=False)
    nz = values > 0
    has_nz = nz.any(axis=1)
    first = np.full(values.shape[0], np.nan, dtype=np.float32)
    if has_nz.any():
        first[has_nz] = nz[has_nz].argmax(axis=1) + 1
    return first


def compute_series_stats(sales: pd.DataFrame, day_cols: List[str]) -> pd.DataFrame:
    values = sales[day_cols].to_numpy(dtype=np.int16, copy=False)
    nonzero = values > 0

    series_total = values.sum(axis=1).astype(np.float32)
    series_mean = values.mean(axis=1).astype(np.float32)
    series_std = values.std(axis=1).astype(np.float32)
    zero_rate = (values == 0).mean(axis=1).astype(np.float32)
    nonzero_count = nonzero.sum(axis=1)
    mean_nonzero = np.divide(
        series_total,
        np.where(nonzero_count == 0, np.nan, nonzero_count),
    ).astype(np.float32)

    first_sale_d = compute_first_nonzero_day(sales, day_cols)
    last_sale_d = np.full(values.shape[0], np.nan, dtype=np.float32)
    has_nz = nonzero.any(axis=1)
    if has_nz.any():
        reversed_first = nonzero[has_nz][:, ::-1].argmax(axis=1)
        last_sale_d[has_nz] = len(day_cols) - reversed_first

    stats = pd.DataFrame(
        {
            "id": sales["id"].astype(str).values,
            "series_total_sales": series_total,
            "series_mean_sales": series_mean,
            "series_std_sales": series_std,
            "series_zero_rate": zero_rate,
            "series_nonzero_count": nonzero_count.astype(np.int16),
            "series_mean_nonzero_sales": mean_nonzero,
            "first_sale_d": first_sale_d,
            "last_sale_d": last_sale_d,
        }
    )
    return reduce_mem_usage(stats)


def build_static_series_table(sales: pd.DataFrame, day_cols: List[str]) -> pd.DataFrame:
    static = sales[STATIC_COLS].copy()
    for col in STATIC_COLS:
        static[col] = static[col].astype(str)

    static["id_idx"] = pd.factorize(static["id"], sort=True)[0].astype(np.int32)
    static["item_idx"] = pd.factorize(static["item_id"], sort=True)[0].astype(np.int16)
    static["dept_idx"] = pd.factorize(static["dept_id"], sort=True)[0].astype(np.int8)
    static["cat_idx"] = pd.factorize(static["cat_id"], sort=True)[0].astype(np.int8)
    static["store_idx"] = pd.factorize(static["store_id"], sort=True)[0].astype(np.int8)
    static["state_idx"] = pd.factorize(static["state_id"], sort=True)[0].astype(np.int8)

    first_sale_d = compute_first_nonzero_day(sales, day_cols)
    static["first_sale_d"] = first_sale_d
    return reduce_mem_usage(static)


def prepare_prices(prices_path: str, static_series: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    item_meta = static_series[["item_id", "dept_id", "cat_id"]].drop_duplicates("item_id").copy()
    prices = pd.read_csv(prices_path, dtype={"store_id": "category", "item_id": "category", "wm_yr_wk": np.int32, "sell_price": np.float32})
    prices["store_id"] = prices["store_id"].astype(str)
    prices["item_id"] = prices["item_id"].astype(str)

    prices = prices.merge(item_meta, on="item_id", how="left")
    prices["dept_id"] = prices["dept_id"].astype(str)
    prices["cat_id"] = prices["cat_id"].astype(str)

    prices = prices.sort_values(["store_id", "item_id", "wm_yr_wk"]).reset_index(drop=True)
    series_grp = prices.groupby(["store_id", "item_id"], sort=False)

    prices["release_wm_yr_wk"] = series_grp["wm_yr_wk"].transform("min").astype(np.int32)
    prices["price_lag_1w"] = series_grp["sell_price"].shift(1).astype(np.float32)
    prices["price_change_1w"] = (prices["sell_price"] - prices["price_lag_1w"]).astype(np.float32)
    prices["price_pct_change_1w"] = np.where(
        prices["price_lag_1w"].notna() & (prices["price_lag_1w"] != 0),
        prices["sell_price"] / prices["price_lag_1w"] - 1.0,
        np.nan,
    ).astype(np.float32)

    for window in [4, 13, 52]:
        prices[f"price_roll_mean_{window}w"] = (
            series_grp["sell_price"].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        ).astype(np.float32)
        prices[f"price_rel_{window}w"] = np.where(
            prices[f"price_roll_mean_{window}w"].notna() & (prices[f"price_roll_mean_{window}w"] != 0),
            prices["sell_price"] / prices[f"price_roll_mean_{window}w"],
            np.nan,
        ).astype(np.float32)

    cat_store_mean = prices.groupby(["cat_id", "store_id", "wm_yr_wk"], sort=False)["sell_price"].transform("mean")
    dept_store_mean = prices.groupby(["dept_id", "store_id", "wm_yr_wk"], sort=False)["sell_price"].transform("mean")
    prices["price_rel_cat_store"] = np.where(
        cat_store_mean.notna() & (cat_store_mean != 0),
        prices["sell_price"] / cat_store_mean,
        np.nan,
    ).astype(np.float32)
    prices["price_rel_dept_store"] = np.where(
        dept_store_mean.notna() & (dept_store_mean != 0),
        prices["sell_price"] / dept_store_mean,
        np.nan,
    ).astype(np.float32)

    prices["price_rank_dept_store"] = (
        prices.groupby(["dept_id", "store_id", "wm_yr_wk"], sort=False)["sell_price"]
        .rank(method="average", pct=True)
        .astype(np.float32)
    )
    prices["price_change_flag_1w"] = prices["price_change_1w"].fillna(0).ne(0).astype(np.int8)

    week_to_first_d = (
        calendar.groupby("wm_yr_wk", observed=True)["d_num"].min().astype(np.int16).reset_index().rename(columns={"d_num": "release_d"})
    )
    prices = prices.merge(week_to_first_d, left_on="release_wm_yr_wk", right_on="wm_yr_wk", how="left", suffixes=("", "_release"))
    prices["release_d"] = prices["release_d"].astype(np.float32)
    prices = prices.drop(columns=["wm_yr_wk_release"], errors="ignore")

    keep_cols = [c for c in PRICE_FEATURE_COLS if c in prices.columns]
    prices = reduce_mem_usage(prices[keep_cols])
    return prices


def build_hierarchy_feature_table(
    sales: pd.DataFrame,
    day_cols: List[str],
    group_cols: List[str],
    prefix: str,
) -> pd.DataFrame:
    """
    Build aggregate daily sales tables at a higher level and compute a small set
    of lagged / rolling-demand features that can be merged back onto the bottom level.
    """
    agg = sales.groupby(group_cols, observed=True)[day_cols].sum()
    stacked = agg.stack().reset_index()
    d_col = stacked.columns[len(group_cols)]
    value_col = stacked.columns[-1]
    stacked = stacked.rename(columns={d_col: "d", value_col: f"{prefix}_sales"})
    stacked["d_num"] = stacked["d"].str[2:].astype(np.int16)
    stacked = stacked.sort_values(group_cols + ["d_num"]).reset_index(drop=True)

    grp = stacked.groupby(group_cols, sort=False, observed=True)[f"{prefix}_sales"]
    stacked[f"{prefix}_sales_lag_7"] = grp.shift(7).astype(np.float32)
    stacked[f"{prefix}_sales_lag_28"] = grp.shift(28).astype(np.float32)
    stacked[f"{prefix}_sales_roll_mean_7"] = grp.transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean()).astype(np.float32)
    stacked[f"{prefix}_sales_roll_mean_28"] = grp.transform(lambda s: s.shift(1).rolling(28, min_periods=7).mean()).astype(np.float32)

    keep_cols = group_cols + [
        "d",
        f"{prefix}_sales_lag_7",
        f"{prefix}_sales_lag_28",
        f"{prefix}_sales_roll_mean_7",
        f"{prefix}_sales_roll_mean_28",
    ]
    stacked = reduce_mem_usage(stacked[keep_cols])
    return stacked


def append_future_rows(static_chunk: pd.DataFrame, future_days: pd.DataFrame) -> pd.DataFrame:
    if future_days.empty:
        return pd.DataFrame(columns=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "d", "sales"])

    n_series = len(static_chunk)
    n_days = len(future_days)
    repeated = static_chunk.loc[static_chunk.index.repeat(n_days), STATIC_COLS].copy().reset_index(drop=True)
    repeated["d"] = np.tile(future_days["d"].astype(str).to_numpy(), n_series)
    repeated["sales"] = np.nan
    return repeated


def add_snap_feature(df: pd.DataFrame) -> pd.DataFrame:
    state = df["state_id"].astype(str)
    df["snap"] = np.select(
        [state == "CA", state == "TX", state == "WI"],
        [df["snap_CA"], df["snap_TX"], df["snap_WI"]],
        default=np.nan,
    ).astype(np.float32)
    return df


def add_release_and_age_features(df: pd.DataFrame) -> pd.DataFrame:
    df["available_for_sale"] = np.where(
        df["release_d"].notna(),
        (df["d_num"] >= df["release_d"]).astype(np.int8),
        0,
    ).astype(np.int8)

    df["days_since_release"] = np.where(
        df["release_d"].notna() & (df["d_num"] >= df["release_d"]),
        df["d_num"] - df["release_d"],
        np.nan,
    ).astype(np.float32)

    df["age_since_first_sale"] = np.where(
        df["first_sale_d"].notna() & (df["d_num"] >= df["first_sale_d"]),
        df["d_num"] - df["first_sale_d"],
        np.nan,
    ).astype(np.float32)

    return df


def add_series_lag_features(
    df: pd.DataFrame,
    lags: Sequence[int] = DEFAULT_LAGS,
    roll_windows: Sequence[int] = DEFAULT_ROLL_WINDOWS,
    min_history_for_roll: int = 1,
) -> pd.DataFrame:
    df = df.sort_values(["id", "d_num"]).reset_index(drop=True)
    grp = df.groupby("id", sort=False)["sales"]

    for lag in lags:
        df[f"sales_lag_{lag}"] = grp.shift(lag).astype(np.float32)

    for window in roll_windows:
        df[f"sales_roll_mean_{window}"] = (
            grp.transform(lambda s: s.shift(1).rolling(window, min_periods=min_history_for_roll).mean())
            .astype(np.float32)
        )

    df["sales_roll_std_28"] = (
        grp.transform(lambda s: s.shift(1).rolling(28, min_periods=min_history_for_roll).std())
        .astype(np.float32)
    )

    positive = df["sales"].gt(0)
    df["sales_roll_nonzero_rate_28"] = (
        positive.groupby(df["id"], sort=False)
        .transform(lambda s: s.shift(1).rolling(28, min_periods=min_history_for_roll).mean())
        .astype(np.float32)
    )

    df["sale_occurrence"] = np.where(df["sales"].notna(), df["sales"].gt(0).astype(np.int8), np.nan)
    return df


def build_time_split_metadata(max_observed_d: int) -> Dict[str, Dict[str, List[int]]]:
    """
    Recommended 28-day rolling windows for this project.
    """
    if max_observed_d >= 1941:
        return {
            "fold_1": {"train": [1, 1885], "valid": [1886, 1913]},
            "fold_2": {"train": [1, 1913], "valid": [1914, 1941]},
            "final_test": {"train": [1, 1941], "predict": [1942, 1969]},
        }
    return {
        "fold_1": {"train": [1, 1857], "valid": [1858, 1885]},
        "fold_2": {"train": [1, 1885], "valid": [1886, 1913]},
        "holdout": {"train": [1, 1913], "predict": [1914, 1941]},
    }


def make_output_columns(keep_static_in_batches: bool, hierarchy_prefixes: Sequence[str]) -> List[str]:
    cols = [
        "id",
        "d",
        "d_num",
        "date",
        "wm_yr_wk",
        "sales",
        "sales_observed",
        "is_future",
        "available_for_sale",
        "days_since_release",
        "age_since_first_sale",
        "snap",
        "weekday",
        "wday",
        "month",
        "year",
        "week_of_year",
        "quarter",
        "day_of_month",
        "day_of_year",
        "is_weekend",
        "is_month_start",
        "is_month_end",
        "is_quarter_start",
        "is_quarter_end",
        "is_year_start",
        "is_year_end",
        "is_event_day",
        "event_name_1_code",
        "event_type_1_code",
        "event_name_2_code",
        "event_type_2_code",
        "month_sin",
        "month_cos",
        "wday_sin",
        "wday_cos",
        "sell_price",
        "price_lag_1w",
        "price_change_1w",
        "price_pct_change_1w",
        "price_roll_mean_4w",
        "price_roll_mean_13w",
        "price_roll_mean_52w",
        "price_rel_4w",
        "price_rel_13w",
        "price_rel_52w",
        "price_rel_cat_store",
        "price_rel_dept_store",
        "price_rank_dept_store",
        "price_change_flag_1w",
        "sales_lag_1",
        "sales_lag_7",
        "sales_lag_14",
        "sales_lag_28",
        "sales_lag_56",
        "sales_roll_mean_7",
        "sales_roll_mean_28",
        "sales_roll_mean_56",
        "sales_roll_std_28",
        "sales_roll_nonzero_rate_28",
        "sale_occurrence",
    ]

    for prefix in hierarchy_prefixes:
        cols.extend(
            [
                f"{prefix}_sales_lag_7",
                f"{prefix}_sales_lag_28",
                f"{prefix}_sales_roll_mean_7",
                f"{prefix}_sales_roll_mean_28",
            ]
        )

    if keep_static_in_batches:
        cols.extend(["item_id", "dept_id", "cat_id", "store_id", "state_id"])

    return cols


def preprocess_pipeline(
    sales_path: str,
    calendar_path: str,
    prices_path: str,
    output_dir: str,
    dataset_name: str = "validation",
    batch_size: int = 1024,
    output_format: str = "pickle",
    compression: str = "gzip",
    start_d: int = 1,
    end_d: Optional[int] = None,
    add_future_days: bool = False,
    drop_pre_release: bool = False,
    keep_static_in_batches: bool = False,
    build_hierarchy_features: bool = True,
    min_history_for_roll: int = 1,
) -> None:
    out_dir = Path(output_dir)
    static_dir = out_dir / "static"
    features_dir = out_dir / "features"
    metadata_dir = out_dir / "metadata"

    ensure_dir(static_dir)
    ensure_dir(features_dir)
    ensure_dir(metadata_dir)

    print("Loading calendar...")
    calendar, event_maps = prepare_calendar(calendar_path)
    save_frame(calendar, static_dir / f"{dataset_name}_calendar_features{get_suffix(output_format, compression)}", output_format, compression)

    print("Loading sales wide table...")
    sales, day_cols = load_sales_wide(sales_path, start_d=start_d, end_d=end_d)
    max_observed_d = max(int(c.split("_")[1]) for c in day_cols)

    print("Building static tables and series statistics...")
    static_series = build_static_series_table(sales, day_cols)
    series_stats = compute_series_stats(sales, day_cols)

    save_frame(static_series, static_dir / f"{dataset_name}_series_info{get_suffix(output_format, compression)}", output_format, compression)
    save_frame(series_stats, static_dir / f"{dataset_name}_series_stats{get_suffix(output_format, compression)}", output_format, compression)

    print("Preparing price features...")
    prices = prepare_prices(prices_path, static_series, calendar)
    save_frame(prices, static_dir / f"{dataset_name}_price_features{get_suffix(output_format, compression)}", output_format, compression)

    hierarchy_tables = {}
    hierarchy_specs = {
        "store": ["store_id"],
        "state": ["state_id"],
        "cat_store": ["cat_id", "store_id"],
        "dept_store": ["dept_id", "store_id"],
    }

    if build_hierarchy_features:
        print("Building hierarchy features...")
        for prefix, group_cols in hierarchy_specs.items():
            hierarchy_tables[prefix] = build_hierarchy_feature_table(sales, day_cols, group_cols, prefix)
            save_frame(
                hierarchy_tables[prefix],
                static_dir / f"{dataset_name}_{prefix}_aggregate_features{get_suffix(output_format, compression)}",
                output_format,
                compression,
            )

    future_days = pd.DataFrame(columns=["d"])
    if add_future_days:
        future_days = calendar.loc[calendar["d_num"] > max_observed_d, ["d"]].copy()

    static_with_stats = static_series.copy()
    full_output_cols = make_output_columns(keep_static_in_batches, list(hierarchy_tables.keys()))

    summary = {
        "dataset_name": dataset_name,
        "sales_path": sales_path,
        "calendar_path": calendar_path,
        "prices_path": prices_path,
        "n_series": int(len(static_series)),
        "start_d": int(start_d),
        "end_d": int(max_observed_d),
        "future_days_added": int(len(future_days)),
        "batch_size": int(batch_size),
        "output_format": output_format,
        "drop_pre_release": bool(drop_pre_release),
        "keep_static_in_batches": bool(keep_static_in_batches),
        "features": full_output_cols,
    }

    map_summary = {k: v for k, v in event_maps.items()}
    with open(metadata_dir / f"{dataset_name}_event_code_maps.json", "w", encoding="utf-8") as f:
        json.dump(map_summary, f, indent=2)

    split_meta = build_time_split_metadata(max_observed_d)
    with open(metadata_dir / f"{dataset_name}_recommended_splits.json", "w", encoding="utf-8") as f:
        json.dump(split_meta, f, indent=2)

    print("Processing long-format batches...")
    n_series = len(sales)
    batch_files = []

    for batch_idx, start in enumerate(range(0, n_series, batch_size)):
        stop = min(start + batch_size, n_series)
        print(f"  Batch {batch_idx:03d}: series {start} to {stop - 1}")

        sales_chunk = sales.iloc[start:stop].copy()
        static_chunk = static_with_stats.iloc[start:stop].copy()

        long_obs = sales_chunk.melt(
            id_vars=STATIC_COLS,
            value_vars=day_cols,
            var_name="d",
            value_name="sales",
        )
        long_obs["sales"] = long_obs["sales"].astype(np.float32)

        if add_future_days and not future_days.empty:
            future_chunk = append_future_rows(static_chunk, future_days)
            future_chunk = future_chunk[long_obs.columns]
            long_df = pd.concat([long_obs, future_chunk], axis=0, ignore_index=True)
        else:
            long_df = long_obs

        long_df["d_num"] = long_df["d"].str[2:].astype(np.int16)
        long_df["id"] = long_df["id"].astype(str)
        for col in ["item_id", "dept_id", "cat_id", "store_id", "state_id"]:
            long_df[col] = long_df[col].astype(str)

        batch_price_keys = static_chunk[["store_id", "item_id"]].drop_duplicates().copy()
        batch_price_keys["store_id"] = batch_price_keys["store_id"].astype(str)
        batch_price_keys["item_id"] = batch_price_keys["item_id"].astype(str)
        batch_prices = prices.merge(batch_price_keys, on=["store_id", "item_id"], how="inner")

        long_df = long_df.merge(calendar, on=["d", "d_num"], how="left")
        long_df = long_df.merge(
            batch_prices,
            on=["store_id", "item_id", "wm_yr_wk"],
            how="left",
        )
        long_df = long_df.merge(
            static_chunk[["id", "first_sale_d"]],
            on="id",
            how="left",
        )

        long_df = add_snap_feature(long_df)
        long_df["sales_observed"] = long_df["sales"].notna().astype(np.int8)
        long_df["is_future"] = long_df["sales"].isna().astype(np.int8)
        long_df = add_release_and_age_features(long_df)

        if drop_pre_release:
            keep_mask = (long_df["available_for_sale"] == 1) | (long_df["is_future"] == 1)
            long_df = long_df.loc[keep_mask].copy()

        long_df = add_series_lag_features(
            long_df,
            lags=DEFAULT_LAGS,
            roll_windows=DEFAULT_ROLL_WINDOWS,
            min_history_for_roll=min_history_for_roll,
        )

        if build_hierarchy_features:
            if "store" in hierarchy_tables:
                long_df = long_df.merge(hierarchy_tables["store"], on=["store_id", "d"], how="left")
            if "state" in hierarchy_tables:
                long_df = long_df.merge(hierarchy_tables["state"], on=["state_id", "d"], how="left")
            if "cat_store" in hierarchy_tables:
                long_df = long_df.merge(hierarchy_tables["cat_store"], on=["cat_id", "store_id", "d"], how="left")
            if "dept_store" in hierarchy_tables:
                long_df = long_df.merge(hierarchy_tables["dept_store"], on=["dept_id", "store_id", "d"], how="left")

        if not keep_static_in_batches:
            long_df = long_df.drop(columns=["snap_CA", "snap_TX", "snap_WI", "event_name_1", "event_type_1", "event_name_2", "event_type_2"], errors="ignore")
        else:
            long_df = long_df.drop(columns=["snap_CA", "snap_TX", "snap_WI"], errors="ignore")

        for col in full_output_cols:
            if col not in long_df.columns:
                long_df[col] = np.nan

        long_df = long_df[full_output_cols]
        long_df = reduce_mem_usage(long_df)

        ext = get_suffix(output_format, compression)
        batch_path = features_dir / f"{dataset_name}_features_batch_{batch_idx:03d}{ext}"
        save_frame(long_df, batch_path, output_format, compression)
        batch_files.append(str(batch_path))

        del sales_chunk, static_chunk, long_obs, long_df, batch_price_keys, batch_prices

    summary["n_batches"] = len(batch_files)
    summary["batch_files"] = batch_files

    with open(metadata_dir / f"{dataset_name}_preprocessing_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Done.")
    print(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    preprocess_pipeline(
        sales_path=args.sales_path,
        calendar_path=args.calendar_path,
        prices_path=args.prices_path,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        output_format=args.output_format,
        compression=args.compression,
        start_d=args.start_d,
        end_d=args.end_d,
        add_future_days=args.add_future_days,
        drop_pre_release=args.drop_pre_release,
        keep_static_in_batches=args.keep_static_in_batches,
        build_hierarchy_features=args.build_hierarchy_features,
        min_history_for_roll=args.min_history_for_roll,
    )


if __name__ == "__main__":
    main()
