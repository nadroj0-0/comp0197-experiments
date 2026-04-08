import os
from pathlib import Path

import pandas as pd


def load_raw_data(data_dir: str) -> tuple:
    sales_df = pd.read_csv(os.path.join(data_dir, "sales_train_evaluation.csv"))
    calendar_df = pd.read_csv(os.path.join(data_dir, "calendar.csv"))
    prices_df = pd.read_csv(os.path.join(data_dir, "sell_prices.csv"))
    print(f"[data] Sales    : {sales_df.shape}")
    print(f"[data] Calendar : {calendar_df.shape}")
    print(f"[data] Prices   : {prices_df.shape}")
    return sales_df, calendar_df, prices_df


def load_or_download_m5(data_dir: str):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "sales": data_dir / "sales_train_evaluation.csv",
        "calendar": data_dir / "calendar.csv",
        "prices": data_dir / "sell_prices.csv",
    }
    urls = {
        "sales": "https://huggingface.co/datasets/kashif/M5/resolve/main/sales_train_evaluation.csv",
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
    return load_raw_data(data_dir)
