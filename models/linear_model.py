import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from .base_model import BaseModel

class LinearModel(BaseModel):
    @property
    def model_name(self): 
        return "linear"

    def preprocess(self):
        """Samples 50,000 rows for a fast baseline comparison."""
        for attr, df in [('train_processed', self.train_raw),
                         ('val_processed',   self.val_raw),
                         ('test_processed',  self.test_raw)]:
            # Sample for speed as it's a baseline
            sample = df.sample(n=min(50_000, len(df)), random_state=self.SEED)
            X = torch.tensor(
                sample[['wday', 'month', 'sell_price', 'is_available']].values,
                dtype=torch.float32
            )
            y = torch.tensor(
                np.log1p(sample['sales'].values), dtype=torch.float32
            ).unsqueeze(1)
            setattr(self, attr, (X, y))

    def train(self, epochs=50, lr=1e-2):
        """Trains a simple linear regression on the sample."""
        X_train, y_train = self.train_processed
        X_val,   y_val   = self.val_processed

        # Input dimension is 4 (wday, month, price, availability)
        self.model = nn.Linear(X_train.shape[1], 1).to(self.device)
        optimizer  = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn    = nn.MSELoss()

        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            loss_fn(self.model(X_train.to(self.device)), y_train.to(self.device)).backward()
            optimizer.step()

        torch.save(self.model.state_dict(),
                   os.path.join(self.output_dir, f"{self.model_name}.pth"))

    def predict(self) -> pd.DataFrame:
        """Generates deterministic quantile predictions."""
        # 1. Load weights
        weights_path = os.path.join(self.output_dir, f"{self.model_name}.pth")
        self.model   = nn.Linear(4, 1).to(self.device)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

        # 2. Build features from the target window
        target = self.test_raw[self.test_raw['d_num'] >= self.TARGET_START].copy()
        X = torch.tensor(
            target[['wday', 'month', 'sell_price', 'is_available']].values,
            dtype=torch.float32
        )

        # 3. Inference in log space
        with torch.no_grad():
            log_preds   = self.model(X.to(self.device)).cpu().numpy().flatten()
            point_preds = np.expm1(log_preds).clip(min=0)

        # 4. Standardize for all 9 quantiles
        q_cols   = [f"q{q}" for q in self.QUANTILES]
        preds_df = target[['id', 'd_num']].copy().reset_index(drop=True)
        preds_df['day_ahead'] = preds_df['d_num'] - self.TARGET_START + 1
        
        for col in q_cols:
            preds_df[col] = point_preds

        # 5. Enforce monotonicity
        preds_df[q_cols] = (
            preds_df[q_cols]
            .clip(lower=0)
            .apply(lambda row: np.sort(row.values), axis=1, result_type='expand')
            .set_axis(q_cols, axis=1)
        )

        # 6. Save and return
        out_path = os.path.join(self.output_dir, f"{self.model_name}_predictions.csv")
        preds_df.to_csv(out_path, index=False)
        return preds_df