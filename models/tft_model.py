import os
import torch
import pickle
import numpy as np
import pandas as pd
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

from .base_model import BaseModel

class TFTModel(BaseModel):
    def __init__(self, data_dir="data", output_dir="outputs"):
        super().__init__(data_dir, output_dir)
        self.max_encoder_length = 56
        self.max_prediction_length = 28
        self.batch_size = 64
        self.model = None

    @property
    def model_name(self) -> str:
        return "tft"

    def _prepare_tft_df(self, df):
        """Standardizes dataframe types for TimeSeriesDataSet."""
        df = df.copy()
        df["time_idx"] = df["d_num"].astype(int)
        
        # Ensure categorical consistency
        event_cols = ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]
        for col in event_cols:
            df[col] = df[col].astype(str).replace('nan', 'None')
            
        categorical_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "weekday"]
        for col in categorical_cols:
            df[col] = df[col].astype(str)
            
        df["sales"] = df["sales"].astype(float)
        df["sell_price"] = df["sell_price"].fillna(0).astype(float)
        return df

    def preprocess(self):
        """Converts raw splits into TFT TimeSeriesDataSets with disk caching."""
        cache_path = os.path.join(self.data_dir, "tft_dataset_cache.pkl")

        # 1. Try to load from cache
        if os.path.exists(cache_path):
            print(f"📂 Loading pre-processed TFT datasets from cache: {cache_path}")
            with open(cache_path, "rb") as f:
                cached_data = pickle.load(f)
                self.train_processed = cached_data['train']
                self.val_processed = cached_data['val']
                self.test_processed = cached_data['test']
            return

        print("🛠️  Step 1/3: Formatting DataFrames...")
        train_df = self._prepare_tft_df(self.train_raw)
        val_df = self._prepare_tft_df(self.val_raw)
        test_df = self._prepare_tft_df(self.test_raw)
        
        full_train_val = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
        training_cutoff = train_df["time_idx"].max()

        print(f"🛠️  Step 2/3: Creating Training Dataset (This takes ~15 mins)...")
        self.train_processed = TimeSeriesDataSet(
            full_train_val[full_train_val.time_idx <= training_cutoff],
            time_idx="time_idx",
            target="sales",
            group_ids=["id"],
            min_encoder_length=self.max_encoder_length // 2, 
            max_encoder_length=self.max_encoder_length,
            min_prediction_length=self.max_prediction_length,
            max_prediction_length=self.max_prediction_length,
            static_categoricals=["item_id", "dept_id", "cat_id", "store_id", "state_id"],
            time_varying_known_categoricals=["weekday", "event_name_1", "event_type_1"],
            time_varying_known_reals=["time_idx", "snap_CA", "snap_TX", "snap_WI", "sell_price"],
            time_varying_unknown_reals=["sales"],
            target_normalizer=GroupNormalizer(groups=["id"], transformation="softplus"),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        print("🛠️  Step 3/3: Generating Validation and Test sets...")
        self.val_processed = TimeSeriesDataSet.from_dataset(
            self.train_processed, full_train_val, predict=True, stop_randomization=True
        )

        full_data = pd.concat([full_train_val, test_df], axis=0).reset_index(drop=True)
        self.test_processed = TimeSeriesDataSet.from_dataset(
            self.train_processed, full_data, predict=True, stop_randomization=True
        )

        # 2. Save to cache
        print("💾 Saving processed datasets to disk for future use...")
        with open(cache_path, "wb") as f:
            pickle.dump({
                'train': self.train_processed,
                'val': self.val_processed,
                'test': self.test_processed
            }, f)
        print("✅ Preprocessing Complete.")

    def train(self, epochs=10):
        """Trains the TFT model using PyTorch Lightning."""
        # Use available CPU cores for data loading (max 4 to avoid OOM)
        num_cpus = min(os.cpu_count(), 4)
        
        train_loader = self.train_processed.to_dataloader(
            train=True, batch_size=self.batch_size, num_workers=num_cpus
        )
        val_loader = self.val_processed.to_dataloader(
            train=False, batch_size=self.batch_size, num_workers=num_cpus
        )

        self.model = TemporalFusionTransformer.from_dataset(
            self.train_processed,
            learning_rate=0.001,
            hidden_size=16,
            attention_head_size=4,
            dropout=0.1,
            hidden_continuous_size=8,
            loss=QuantileLoss(quantiles=self.QUANTILES),
        )

        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator="auto",
            devices=1,
            callbacks=[EarlyStopping(monitor="val_loss", patience=3)],
            enable_progress_bar=True
        )

        trainer.fit(self.model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, f"{self.model_name}.pth"))

    def predict(self):
        """Generates predictions and ensures monotonicity."""
        if self.model is None:
            self.preprocess()
            self.model = TemporalFusionTransformer.from_dataset(
                self.train_processed, 
                loss=QuantileLoss(quantiles=self.QUANTILES)
            )
            weights_path = os.path.join(self.output_dir, f"{self.model_name}.pth")
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        
        self.model.to(self.device)
        self.model.eval()

        test_loader = self.test_processed.to_dataloader(train=False, batch_size=self.batch_size)
        output = self.model.predict(test_loader, mode="quantiles", return_index=True)
        
        preds_array = output.output.cpu().numpy()
        index_df = output.index
        q_cols = [f"q{q}" for q in self.QUANTILES]
        
        res_list = []
        for i in range(len(index_df)):
            item_id = index_df.iloc[i]["id"]
            item_preds = pd.DataFrame(preds_array[i], columns=q_cols)
            item_preds["id"] = item_id
            item_preds["day_ahead"] = np.arange(1, 29)
            res_list.append(item_preds)

        preds_df = pd.concat(res_list).reset_index(drop=True)
        preds_df[q_cols] = preds_df[q_cols].clip(lower=0)
        preds_df[q_cols] = np.sort(preds_df[q_cols].values, axis=1)

        out_path = os.path.join(self.output_dir, f"{self.model_name}_predictions.csv")
        preds_df.to_csv(out_path, index=False)
        return preds_df