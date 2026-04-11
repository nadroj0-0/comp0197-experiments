import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .datasets import WindowedM5Dataset
from .io import load_or_download_m5
from .specs import get_feature_cols
from .transforms import (
    apply_normalisation,
    encode_hierarchy,
    fit_normalisation_stats,
    get_vocab_sizes,
    melt_sales,
    merge_calendar,
    merge_prices,
    split_data,
    trim_data,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    generator = torch.Generator().manual_seed(seed)
    print(f"Random seed set to {seed}")
    return generator, seed


def init_seed(cfg: dict):
    seed = cfg.get("seed", 42)
    generator, seed = set_seed(seed)
    cfg["seed"] = seed
    return generator


def _build_featured_frame(
    long_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    feature_set: str,
) -> pd.DataFrame:
    if feature_set == "sales_only":
        cal_dates = calendar_df[["d", "date"]].copy()
        cal_dates["date"] = pd.to_datetime(cal_dates["date"])
        featured = long_df.merge(cal_dates, on="d", how="left")
        featured = featured.sort_values(["id", "date"]).reset_index(drop=True)
        print(f"[data] Sales-only pipeline: {featured.shape}")
        return featured

    if feature_set in ("sales_hierarchy", "sales_hierarchy_dow"):
        include_dow = feature_set == "sales_hierarchy_dow"
        cal_dates = calendar_df[["d", "date"]].copy()
        cal_dates["date"] = pd.to_datetime(cal_dates["date"])
        featured = long_df.merge(cal_dates, on="d", how="left")
        featured = encode_hierarchy(featured, include_dow=include_dow)
        featured = featured.sort_values(["id", "date"]).reset_index(drop=True)
        print(f"[data] Hierarchy pipeline ({feature_set}): {featured.shape}")
        return featured

    if feature_set in ("sales_yen", "sales_yen_hierarchy"):
        include_hierarchy = "hierarchy" in feature_set
        featured = merge_calendar(long_df, calendar_df)
        featured = merge_prices(featured, prices_df)
        featured["is_available"] = featured["sell_price"].gt(0.0).astype(np.float32)
        featured["has_event"] = (~featured["event_name_1"].isna()).astype(np.float32)
        if include_hierarchy:
            featured = encode_hierarchy(featured, include_dow=False)
        featured = featured.sort_values(["id", "date"]).reset_index(drop=True)
        featured["sell_price"] = featured.groupby("id")["sell_price"].transform(lambda x: x.ffill())
        featured["sell_price"] = featured["sell_price"].fillna(0.0)
        for col in ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
            featured[col] = featured[col].fillna("none")
        print(f"[data] Yen-style pipeline ({feature_set}): {featured.shape}")
        return featured

    raise ValueError(f"Unknown feature_set: {feature_set}")


def _get_split_days(split_protocol: str) -> tuple[int, int]:
    if split_protocol == "yen_v1":
        print("[data] Protocol: yen_v1 — val=112 days, test=56 days")
        return 112, 56

    print("[data] Protocol: default — val=28 days, test=28 days")
    return 28, 28


def _compute_train_item_weights(
    *,
    weight_protocol: str,
    train_df_raw: pd.DataFrame,
    series_ids: list[str],
    sales_df: pd.DataFrame,
    prices_df: pd.DataFrame,
) -> np.ndarray:
    if weight_protocol == "yen_v1":
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
        prices_df_w = prices_df.copy()
        avg_prices = (
            prices_df_w.groupby(["item_id", "store_id"])["sell_price"]
            .mean().reset_index()
        )
        sales_w = sales_df.merge(avg_prices, on=["item_id", "store_id"], how="left")
        sales_w["sell_price"] = sales_w["sell_price"].fillna(
            prices_df_w["sell_price"].median()
        )
        day_cols_w = [c for c in sales_df.columns if c.startswith("d_")]
        train_day_cols = day_cols_w[:-28]
        item_vols = sales_w[train_day_cols].sum(axis=1).values
        item_revs = item_vols * sales_w["sell_price"].values

    weights = item_revs / item_revs.sum()
    return weights.astype(np.float32)


def build_dataloaders(
    data_dir: str,
    seq_len: int = 28,
    horizon: int = 28,
    batch_size: int = 256,
    top_k_series: int = 200,
    feature_set: str = "sales_only",
    autoregressive: bool = True,
    use_normalise: bool = False,
    zscore_target: bool = True,
    max_series: int = None,
    num_workers: int = 2,
    seed: int = 42,
    sampling: str = "top",
    include_weights: bool = False,
    split_protocol: str = "default",
    weight_protocol: str = "default",
) -> tuple:
    generator, _ = set_seed(seed)
    feature_cols = get_feature_cols(feature_set)
    feature_index = {col: i for i, col in enumerate(feature_cols)}
    n_features = len(feature_cols)

    sales_df, calendar_df, prices_df = load_or_download_m5(data_dir)
    sales_df = trim_data(sales_df, top_k_series, sampling=sampling)

    long_df = melt_sales(sales_df)
    featured = _build_featured_frame(long_df, calendar_df, prices_df, feature_set)

    series_ids = featured["id"].drop_duplicates().tolist()
    _val_days, _test_days = _get_split_days(split_protocol)

    train_df_raw, val_df_raw, test_df_raw = split_data(
        featured, val_days=_val_days, test_days=_test_days
    )
    val_start_date = val_df_raw["date"].values.astype("datetime64[ns]")[0]
    test_start_date = test_df_raw["date"].values.astype("datetime64[ns]")[0]

    if use_normalise:
        stats = fit_normalisation_stats(train_df_raw, zscore_target=zscore_target)
        featured = apply_normalisation(featured, stats, zscore_target=zscore_target)
        print(f"[data] Normalisation applied (zscore_target={zscore_target})")
    else:
        stats = None
        print("[data] No normalisation — using raw sales counts")

    train_item_weights = None
    if include_weights:
        train_item_weights = _compute_train_item_weights(
            weight_protocol=weight_protocol,
            train_df_raw=train_df_raw,
            series_ids=series_ids,
            sales_df=sales_df,
            prices_df=prices_df,
        )
        print(
            f"[data] Revenue weights computed "
            f"({len(train_item_weights)} series, protocol={weight_protocol})"
        )

    shared = dict(
        feature_cols=feature_cols,
        seq_len=seq_len,
        horizon=horizon,
        val_start_date=val_start_date,
        test_start_date=test_start_date,
        autoregressive=autoregressive,
        max_series=max_series,
    )
    train_ds = WindowedM5Dataset(
        featured, split="train",
        item_weights=train_item_weights, series_ids=series_ids, **shared
    )
    val_ds = WindowedM5Dataset(featured, split="val", **shared)
    test_ds = WindowedM5Dataset(featured, split="test", **shared)

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

    hierarchy_cols = ["state_id_int", "store_id_int", "cat_id_int", "dept_id_int"]
    if any(c in feature_cols for c in hierarchy_cols):
        vocab_sizes = get_vocab_sizes(featured)
        print(f"[data] Vocab sizes  : { {k: v for k, v in vocab_sizes.items()} }")
    else:
        vocab_sizes = {}

    return train_loader, val_loader, test_loader, stats, vocab_sizes, feature_index
