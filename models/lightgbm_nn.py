import os
import time
import gc
import pickle
import json
import joblib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm  # Ensure tqdm is installed: pip install tqdm
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from pandas.api.types import is_numeric_dtype
import lightgbm as lgb

from .base_model import BaseModel

class FeatureNN(nn.Module):
    def __init__(self, n_items, input_dim):
        super().__init__()
        self.emb = nn.Embedding(n_items, 8)
        self.backbone = nn.Sequential(
            nn.Linear(input_dim + 8, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.head = nn.Linear(32, 1)

    def forward(self, x, item):
        x = torch.cat([x, self.emb(item)], dim=1)
        features = self.backbone(x)
        out = self.head(features)
        return out.squeeze(-1), features

class LightGBM_NN(BaseModel):

    @property
    def model_name(self):
        return "LightGBM_NN_Hybrid"

    def preprocess(self):
        print("starting preprocessing")
        def add_features(df):
            df = df.copy().sort_values(["id","d_num"])
            grp = df.groupby("id")["sales"]
            for lag in [1,7,14,28]:
                df[f"lag_{lag}"] = grp.shift(lag)
            for w in [7,28]:
                df[f"roll_mean_{w}"] = grp.shift(1).rolling(w).mean()
            df["roll_std_7"] = grp.shift(1).rolling(7).std()
            df["trend"] = df["d_num"].astype(np.int16)
            df["week_of_year"] = df["wm_yr_wk"].astype(np.int16)
            df["dow_sin"] = np.sin(2*np.pi*df["wday"]/7).astype(np.float32)
            df["dow_cos"] = np.cos(2*np.pi*df["wday"]/7).astype(np.float32)
            return df

        train = add_features(self.train_raw)
        val = add_features(self.val_raw)
        test = add_features(self.test_raw)

        for dataset in [train, val, test]:
            cat_cols = ["item_id","dept_id","cat_id","store_id","state_id"]
            num_cols = [c for c in dataset.columns if c not in cat_cols]
            for col in num_cols:
                if is_numeric_dtype(dataset[col]):
                    dataset[col] = dataset[col].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)

        self.train_processed = train
        self.val_processed = val
        self.test_processed = test

    def train(self, epochs=5):  # FIX: Added epochs argument to prevent TypeError
        print(f"\nTRAINING STARTING - Neural Network for {epochs} epochs\n")
        
        quantiles = self.QUANTILES
        models = {}
        df = self.train_processed.copy()

        # Target Preparation
        df["target"] = df.groupby("id")["sales"].shift(-1)
        df["target"] = np.log1p(df["target"])
        df = df.dropna(subset=["target"])

        # Categorical Encoding
        cat_cols = ["item_id","dept_id","cat_id","store_id","state_id"]
        cat_maps = {}
        for c in cat_cols:
            uniques = df[c].unique()
            cat_maps[c] = {k:i for i,k in enumerate(uniques)}
            df[c] = df[c].map(cat_maps[c]).astype(np.int16)

        self.n_items = len(cat_maps["item_id"])

        features = [
            "lag_1","lag_7","lag_14","lag_28",
            "roll_mean_7","roll_mean_28", "roll_std_7",
            "trend","week_of_year","dow_sin","dow_cos",
            "wday","month","sell_price","is_available"
        ]
        self.features = features

        scaler = StandardScaler()
        X = df[features].values.astype(np.float32)
        X_scaled = scaler.fit_transform(X)
        self.scaler = scaler

        # NN Data Preparation
        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        item_t = torch.tensor(df["item_id"].values, dtype=torch.long)
        y_t = torch.tensor(df["target"].values.astype(np.float32))
        dataset = TensorDataset(X_t, item_t, y_t)
        loader = DataLoader(dataset, batch_size=8192, shuffle=True)

        model_nn = FeatureNN(self.n_items, len(features)).to(self.device)
        opt = torch.optim.Adam(model_nn.parameters(), lr=0.0002)

        # NN Training with Progress Bar
        for epoch in range(epochs):
            total_loss = 0
            # Wrap loader with tqdm for the progress bar
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
            
            model_nn.train()
            for xb, itemb, yb in pbar:
                xb, itemb, yb = xb.to(self.device), itemb.to(self.device), yb.to(self.device)

                pred, _ = model_nn(xb, itemb)
                loss = nn.MSELoss()(pred, yb)

                opt.zero_grad()
                loss.backward()
                opt.step()

                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            print(f"Epoch {epoch+1} Complete | Avg Loss: {total_loss/len(loader):.4f}")

        # Extracting Embeddings
        print("\nExtracting Hybrid Embeddings for LightGBM...")
        model_nn.eval()
        emb_list = []
        # Use a non-shuffled loader for consistent concatenation
        extract_loader = DataLoader(dataset, batch_size=8192, shuffle=False)
        with torch.no_grad():
            for xb, itemb, _ in tqdm(extract_loader, desc="Extracting"):
                _, emb = model_nn(xb.to(self.device), itemb.to(self.device))
                emb_list.append(emb.cpu().numpy())
        
        emb = np.vstack(emb_list)
        X_aug = np.concatenate((X_scaled, 0.4 * emb), axis=1)
        y = df["target"].values.astype(np.float32)

        # LightGBM Training
        for q in quantiles:
            print(f"Training LightGBM Quantile: {q}")
            dtrain = lgb.Dataset(X_aug, y)
            model = lgb.train(
                {
                    "objective": "quantile",
                    "alpha": q,
                    "learning_rate": 0.03,
                    "num_leaves": 64,
                    "verbose": -1
                },
                dtrain,
                num_boost_round=200
            )
            model.save_model(os.path.join(self.output_dir, f"lgb_q_{q}.txt"))
            models[q] = model

        # Save Metadata
        joblib.dump(scaler, os.path.join(self.output_dir, "scaler.pkl"))
        joblib.dump(features, os.path.join(self.output_dir, "features.pkl"))
        joblib.dump(cat_maps, os.path.join(self.output_dir, "cat_maps.pkl"))
        joblib.dump(self.n_items, os.path.join(self.output_dir, "n_items.pkl"))
        torch.save(model_nn.state_dict(), os.path.join(self.output_dir, "nn_embedding.pth"))

        self.models = models
        print("\nTRAINING COMPLETE\n")

    def predict(self) -> pd.DataFrame:
        import joblib
        import torch
        import lightgbm as lgb

        print(f"\n🚀 Running Batch Prediction for {self.model_name}...")

        # 1. Load Metadata and Models
        scaler = joblib.load(os.path.join(self.output_dir, "scaler.pkl"))
        features_list = joblib.load(os.path.join(self.output_dir, "features.pkl"))
        cat_maps = joblib.load(os.path.join(self.output_dir, "cat_maps.pkl"))
        n_items = joblib.load(os.path.join(self.output_dir, "n_items.pkl"))

        model_nn = FeatureNN(n_items, len(features_list))
        model_nn.load_state_dict(torch.load(os.path.join(self.output_dir, "LightGBM_NN_Hybrid.pth"), map_location=self.device))
        model_nn.eval().to(self.device)

        lgb_models = {
            q: lgb.Booster(model_file=os.path.join(self.output_dir, f"lgb_q_{q}.txt"))
            for q in self.QUANTILES
        }

        # 2. Prepare the Test Feature Matrix
        # Like H-LSTM, we take the processed test data (which already has lags/rolling features)
        test_df = self.test_processed.copy()
        
        # Filter for the 28-day prediction window
        mask = (test_df["d_num"] >= self.TARGET_START) & (test_df["d_num"] < self.TARGET_START + self.PRED_LENGTH)
        predict_df = test_df[mask].copy().sort_values(["id", "d_num"])

        # Map Categoricals (Item IDs)
        predict_df["item_id_encoded"] = predict_df["item_id"].map(cat_maps["item_id"]).fillna(0).astype(np.int64)
        
        # Standardize Numerical Features
        X_raw = predict_df[features_list].values.astype(np.float32)
        X_scaled = scaler.transform(X_raw)

        # 3. Batch Generate Neural Network Embeddings (The "H-LSTM" way)
        print(f"Extracting representations for {len(predict_df)} rows...")
        X_t = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        item_t = torch.tensor(predict_df["item_id_encoded"].values, dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            # Process in large batches to avoid OOM but keep speed high
            _, emb = model_nn(X_t, item_t)
            emb = emb.cpu().numpy()

        # Combine Scaled Features + NN Embeddings
        X_aug = np.concatenate((X_scaled, 0.4 * emb), axis=1)

        # 4. Generate Quantile Predictions
        print("Running LightGBM Quantile boosters...")
        q_cols = [f"q{q}" for q in self.QUANTILES]
        
        # Initialize results array (Rows, Quantiles)
        preds_matrix = np.zeros((len(predict_df), len(self.QUANTILES)))

        for i, q in enumerate(self.QUANTILES):
            # Predict for all items/days at once
            p = lgb_models[q].predict(X_aug)
            # Inverse log transform (since training used log1p)
            preds_matrix[:, i] = np.expm1(p)

        # 5. Format to Standardized Output
        predict_df[q_cols] = preds_matrix
        
        # Calculate day_ahead relative to TARGET_START
        predict_df["day_ahead"] = predict_df["d_num"] - self.TARGET_START + 1
        
        # Select final columns and sort
        preds_df = predict_df[["id", "day_ahead"] + q_cols].reset_index(drop=True)

        # MANDATORY: Enforce non-negativity and non-decreasing quantiles
        preds_df[q_cols] = preds_df[q_cols].clip(lower=0)
        # Sort quantiles across each row to ensure q0.025 <= q0.5 <= q0.975
        preds_df[q_cols] = np.sort(preds_df[q_cols].values, axis=1)

        out_path = os.path.join(self.output_dir, f"{self.model_name}_predictions.csv")
        preds_df.to_csv(out_path, index=False)
        print(f"✅ Saved → {out_path}")

        return preds_df