import os
import random
import pickle
import json
import torch

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class BaseModel(ABC):

    QUANTILES   = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975]
    PRED_LENGTH = 28
    SEED        = 25
    TARGET_START = 1914

    def __init__(self, data_dir="data", output_dir="outputs"):
        random.seed(self.SEED)
        np.random.seed(self.SEED)
        torch.manual_seed(self.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_dir   = data_dir
        self.output_dir = output_dir
        os.makedirs(data_dir,   exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        self.train_raw = self.val_raw = self.test_raw = None
        self.calendar  = self.prices  = self.item_weights = None
        self.train_processed = self.val_processed = self.test_processed = None
        self.model = None

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @abstractmethod
    def preprocess(self): ...
    # Input:  self.train_raw, self.val_raw, self.test_raw
    # Output: self.train_processed, self.val_processed, self.test_processed
    #         (format is model-specific: tensors, lgb.Dataset, DataFrames, etc.)

    @abstractmethod
    def train(self): ...
    # Input:  self.train_processed, self.val_processed
    # Must:
    #   1. Set self.model
    #   2. Save weights → output_dir/{model_name}.pth (or .pkl for LGBM)

    @abstractmethod
    def predict(self): ...
    # Input: self.test_processed, self.output_dir
    # Must: 
    # 1. Load model from: output_dir/{model_name}.pth/pkl 
    # 2. Only use d_1886-d_1913 as context (if needed)
    # 3. Predict 9 quantiles for d_1914-d_1941 in preds_df: a DataFrame (30490 * 28 rows) with columns:
    #     id | day_ahead (1-28) | q0.025 | q0.05 | q0.1 | q0.25 | q0.5 | q0.75 | q0.9 | q0.95 | q0.975
    # 4. Sort predictions by id, day_ahead
    # 5. Make sure quantiles are non-decreasing and >= 0
    # 6. Save predictions → output_dir/{model_name}_predictions.csv
    # Output: preds_df

    # Shared methods

    def load_and_split_data(self):
        """
        Downloads (or loads from cache) M5 data and processes it into long-format dataframes (1 row per item and day). 
        Split into train (d_1-d_1773), val (d_1774-d_1885), and test (d_1886-d_1941).
        Save raw splits to cache for loading when training models.

        Output: self.train_raw, self.val_raw, self.test_raw, self.item_weights
        """
        # Step 1: load data
        cache = os.path.join(self.data_dir, "raw_split.pkl")

        if os.path.exists(cache):
            with open(cache, "rb") as f:
                d = pickle.load(f)
            print("Loaded cached data splits.")

        else:
            base     = "https://huggingface.co/datasets/kashif/M5/resolve/main"
            sales    = pd.read_csv(f"{base}/sales_train_evaluation.csv")
            calendar = pd.read_csv(f"{base}/calendar.csv")
            prices   = pd.read_csv(f"{base}/sell_prices.csv")
            print("Downloaded M5 data.")

            # Step 2: melt sales to long format
            id_cols  = [c for c in sales.columns if not c.startswith("d_")]
            day_cols = [c for c in sales.columns if c.startswith("d_")]
            sales_long = sales[id_cols + day_cols].melt(
                id_vars=id_cols, var_name='d', value_name='sales'
            )
            sales_long['d_num'] = sales_long['d'].str[2:].astype(int)

            # Step 3: merge calendar data and set dtypes
            for col in ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']:
                calendar[col] = calendar[col].fillna('none').astype('category')
            calendar['weekday'] = calendar['weekday'].astype('category')
            calendar['date']    = pd.to_datetime(calendar['date'])
            for col in ['snap_CA', 'snap_TX', 'snap_WI', 'wday', 'month']:
                calendar[col] = calendar[col].astype('int8')
            calendar['year'] = calendar['year'].astype('int16')  # fix year encoding to handle years 2011-2016
            sales_long = sales_long.merge(calendar, on='d', how='left')

            # Step 4: merge daily prices
            sales_long = sales_long.merge(
                prices[['store_id', 'item_id', 'wm_yr_wk', 'sell_price']],
                on=['store_id', 'item_id', 'wm_yr_wk'], how='left'
            )

            # Step 5: add is_available flag and forward-fill missing prices
            sales_long['is_available'] = sales_long['sell_price'].notna().astype('int8')
            sales_long = sales_long.sort_values(['id', 'd_num']).reset_index(drop=True)
            sales_long['sell_price'] = (
                sales_long.groupby('id')['sell_price']
                .transform(lambda x: x.ffill().fillna(0.0))
            )

            # Step 6: check missing values
            missing = sales_long.isnull().sum()
            missing = missing[missing > 0]
            assert len(missing) == 0, f"Missing values found:\n{missing}"

            # Step 7: split into train/val/test
            total_days = len(day_cols)
            train_end  = total_days - 6 * self.PRED_LENGTH   # 1773
            val_end    = total_days - 2 * self.PRED_LENGTH   # 1885

            train_raw = sales_long[sales_long['d_num'] <= train_end].reset_index(drop=True)
            val_raw   = sales_long[(sales_long['d_num'] > train_end) & (sales_long['d_num'] <= val_end)].reset_index(drop=True)
            test_raw  = sales_long[sales_long['d_num'] > val_end].reset_index(drop=True)

            assert train_raw['d_num'].nunique() == 1773
            assert val_raw['d_num'].nunique()   == 112
            assert test_raw['d_num'].nunique()  == 56
            print(f"Train: d_1-d_{train_end} | Val: d_{train_end+1}-d_{val_end} | Test: d_{val_end+1}-d_{total_days}")

            # Step 8: calculate revenue weights on last 28 training days only
            last28 = train_raw[train_raw['d_num'] > train_end - 28]
            train_rev = (last28['sales'] * last28['sell_price']).groupby(last28['id']).sum()
            item_weights = (train_rev / train_rev.sum()).rename('weight')

            # Step 9: save data splits to cache
            d = dict(train_raw=train_raw, val_raw=val_raw, test_raw=test_raw,
                        item_weights=item_weights)

            # save all splits and weights into raw_split.pkl
            with open(cache, "wb") as f:
                pickle.dump(d, f)
            print(f"Cached to {cache}")

        self.train_raw    = d["train_raw"]
        self.val_raw      = d["val_raw"]
        self.test_raw     = d["test_raw"]
        self.item_weights = d["item_weights"]

        return self.train_raw, self.val_raw, self.test_raw, self.item_weights

    # Evaluation

    def _build_pinball_tensor(self, preds_df: pd.DataFrame):
        """
        Shared setup for WSPL and CRPS. Requires self.train_raw, self.test_raw.
        Returns y_mat (N,28), q_arr (N,9,28), ids, scale.
        """
        q_cols  = [f"q{q}" for q in self.QUANTILES]
        
        # Extract test targets for d_1914-d_1941 (28 days)
        test_targets = (
            self.test_raw[self.test_raw['d_num'] >= self.TARGET_START]
            .pivot(index='id', columns='d_num', values='sales')
            .sort_index()
        )
        ids = test_targets.index.tolist()
        y_mat = test_targets.values.astype(np.float32)  # Shape: (N_series, 28)
        
        # Reshape predictions to match
        preds_pivot = (
            preds_df.set_index(['id', 'day_ahead'])[q_cols]
            .unstack('day_ahead')
            .sort_index(axis=1) 
            .loc[ids]
)
        q_arr = preds_pivot.values.reshape(len(ids), len(self.QUANTILES), self.PRED_LENGTH)
        
        # Scale from train
        train_s = self.train_raw.sort_values(['id', 'd_num']).copy()
        train_s['prev'] = train_s.groupby('id')['sales'].shift(1)
        scale = (
            train_s.dropna(subset=['prev'])
            .assign(abs_diff=lambda df: (df['sales'] - df['prev']).abs())
            .groupby('id')['abs_diff']
            .mean()
            .reindex(ids)  # Align scale to test data IDs
            .clip(lower=1e-8)
        )
        
        # Checks for debugging
        assert y_mat.shape == (len(ids), 28), f"y_mat shape mismatch: {y_mat.shape}"
        assert q_arr.shape == (len(ids), 9, 28), f"q_arr shape mismatch: {q_arr.shape}"
        
        return y_mat, q_arr, ids, scale
 
    def compute_wspl(self, y_mat, q_arr, ids, scale) -> float:
        """Weighted scaled pinball loss across 9 quantiles and 28 forecast days."""
        q_vals  = np.array(self.QUANTILES)
        errors  = y_mat[:, None, :] - q_arr
        pinball = np.maximum(q_vals[None, :, None] * errors,
                             (q_vals[None, :, None] - 1) * errors)
        wspl_per_series = pinball.mean(axis=(1, 2)) / scale.reindex(ids).values
        weights         = self.item_weights.reindex(ids).fillna(0).values
        return float(np.dot(weights, wspl_per_series))
 
    def compute_crps(self, y_mat, q_arr, ids) -> float:
        """Weighted CRPS approximated as 2x mean pinball loss across quantiles and forecast days."""
        q_vals  = np.array(self.QUANTILES)
        errors  = y_mat[:, None, :] - q_arr
        pinball = np.maximum(q_vals[None, :, None] * errors,
                            (q_vals[None, :, None] - 1) * errors)
        crps_per_series = 2 * pinball.mean(axis=(1, 2))
        weights = self.item_weights.reindex(ids).fillna(0).values
        return float(np.dot(weights, crps_per_series))
 
    def compute_coverage(self, preds_df: pd.DataFrame, y_mat, ids) -> dict:
        """
        Coverage error (actual - nominal) for 4 prediction intervals.
        Positive = over-coverage, negative = under-coverage.
        """
        intervals = {
            0.50: ("q0.25",  "q0.75"),
            0.80: ("q0.1",   "q0.9"),
            0.90: ("q0.05",  "q0.95"),
            0.95: ("q0.025", "q0.975"),
        }
        preds_indexed = (
            preds_df.set_index(['id', 'day_ahead'])
            .reindex(pd.MultiIndex.from_product(
                [ids, range(1, self.PRED_LENGTH + 1)],
                names=['id', 'day_ahead']))
        )
        coverage_errors = {}
        for nominal, (lower_col, upper_col) in intervals.items():
            lower   = preds_indexed[lower_col].values.reshape(len(ids), self.PRED_LENGTH)
            upper   = preds_indexed[upper_col].values.reshape(len(ids), self.PRED_LENGTH)
            covered = ((y_mat >= lower) & (y_mat <= upper)).mean()
            coverage_errors[f"coverage_error_{int(nominal*100)}pct"] = round(
                float(covered - nominal), 6)
        return coverage_errors
    
    def _validate_preds(self, preds_df):
        """
        Validate predictions for easier debugging of inference pipeline.
        """
        expected_ids = self.test_raw['id'].unique()
        q_cols = [f"q{q}" for q in self.QUANTILES]
        assert set(expected_ids).issubset(set(preds_df['id'])), "Missing IDs"
        assert set(preds_df['day_ahead'].unique()) == set(range(1, 29)), "day_ahead must be 1-28"
        assert all(c in preds_df.columns for c in q_cols), "Missing quantile columns"
        assert (preds_df[q_cols].diff(axis=1).iloc[:, 1:] >= 0).all().all(), "Quantiles non-monotonic"
        assert (preds_df[q_cols] >= 0).all().all(), "Negative quantile predictions"

    def evaluate(self, preds_df: pd.DataFrame) -> dict:
        """
        Compute WSPL, CRPS and coverage error from a predictions DataFrame.
        Saves results to output_dir/{model_name}_results.json.
        Requires: raw_split.pkl in data_dir. Run load_and_split_data() first.
        """
        self._validate_preds(preds_df)
        # Step 1: load data and weights
        cache = os.path.join(self.data_dir, "raw_split.pkl")
        assert os.path.exists(cache), "Run load_and_split_data() first."
        with open(cache, "rb") as f:
            d = pickle.load(f)
        self.train_raw    = d["train_raw"]
        self.test_raw     = d["test_raw"]
        self.item_weights = d["item_weights"]

        # Step 2: build pinball tensor
        y_mat, q_arr, ids, scale = self._build_pinball_tensor(preds_df)
    
        # Step 3: compute evaluation metrics
        results = {
            "model" : self.model_name,
            "wspl"  : round(self.compute_wspl(y_mat, q_arr, ids, scale), 6),
            "crps"  : round(self.compute_crps(y_mat, q_arr, ids), 6),
            **self.compute_coverage(preds_df, y_mat, ids),
        }

        # Step 4: save results to json
        out_path = os.path.join(self.output_dir, f"{self.model_name}_results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        return results

    # Pipelines

    def run_training_pipeline(self):
    # Combines loading and splitting data, preprocessing, and training into a single pipeline
        self.load_and_split_data()
        print("Finished shared data processing.")
        self.preprocess()
        print("Finished model-specific data processing.")
        self.train()
        print("Finished model training.")

    def run_inference_pipeline(self):
    # Combines inference and evaluation into a single pipeline
        self.load_and_split_data()  # needs train_raw and test_raw, if training pipeline was not run before this
        preds_df = self.predict()
        print("Finished model inference.")
        results = self.evaluate(preds_df)
        print("Finished model evaluation.")
        return results