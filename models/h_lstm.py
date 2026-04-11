import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
# New version
from .base_model import BaseModel

# --- 1. Architecture & Loss ---
class WeightedPinballLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target, weights):
        if target.dim() == 1: target = target.unsqueeze(1)
        if weights.dim() == 1: weights = weights.unsqueeze(1)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i:i+1]
            loss_q = torch.max((q - 1) * errors, q * errors)
            losses.append(loss_q)
        return (torch.cat(losses, dim=1) * weights).mean(dim=0).sum()

class HMLSTM(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, num_layers=2, num_quantiles=9, dropout_rate=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(64, num_quantiles))

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(self.dropout(lstm_out[:, -1, :]))

# --- 2. Memory-Efficient Dataset ---
class M5TimeSeriesDataset(Dataset):
    def __init__(self, df, cat_mappings, weight_dict, seq_length=28):
        self.seq_length = seq_length
        num_items = df['id'].nunique()
        num_days = len(df) // num_items
        self.item_ids = df['id'].unique()
        self.sales = torch.tensor(df['sales'].values.astype(np.float32).reshape(num_items, num_days))
        self.wday = torch.tensor(df['wday'].values.astype(np.float32).reshape(num_items, num_days))
        
        static_list = []
        for col in ['state_id', 'store_id', 'cat_id', 'dept_id']:
            static_list.append(df.iloc[0::num_days][col].map(cat_mappings[col]).fillna(0).astype(np.float32).values)
        self.static_feats = torch.tensor(np.stack(static_list, axis=1))
        self.weights = torch.tensor([weight_dict.get(i_id, 0.0) for i_id in self.item_ids], dtype=torch.float32)
        self.num_items = num_items
        self.valid_starts_per_item = max(0, num_days - self.seq_length)

    def __len__(self): return self.num_items * self.valid_starts_per_item

    def __getitem__(self, idx):
        item_idx, t = idx // self.valid_starts_per_item, idx % self.valid_starts_per_item
        X = torch.cat([self.sales[item_idx, t:t+self.seq_length].unsqueeze(1),
                       self.static_feats[item_idx].unsqueeze(0).expand(self.seq_length, -1),
                       self.wday[item_idx, t:t+self.seq_length].unsqueeze(1)], dim=1)
        return X, self.sales[item_idx, t+self.seq_length], self.weights[item_idx]

# --- 3. Unified Model Class ---
class HierarchicalLSTMModel(BaseModel):
    @property
    def model_name(self): return "hierarchical_lstm"

    def preprocess(self):
        self.cat_mappings = {col: {val: i for i, val in enumerate(self.train_raw[col].unique())} 
                             for col in ['state_id', 'store_id', 'cat_id', 'dept_id']}
        weight_dict = self.item_weights.to_dict()
        self.train_processed = M5TimeSeriesDataset(self.train_raw, self.cat_mappings, weight_dict)
        self.val_processed = M5TimeSeriesDataset(self.val_raw, self.cat_mappings, weight_dict)
        self.test_processed = M5TimeSeriesDataset(self.test_raw, self.cat_mappings, weight_dict)

    def train(self, epochs=5, lr=0.001, batch_size=4096):
        train_loader = DataLoader(self.train_processed, batch_size=batch_size, shuffle=True)
        self.model = HMLSTM(input_dim=6, num_quantiles=len(self.QUANTILES)).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = WeightedPinballLoss(self.QUANTILES)
        for epoch in range(epochs):
            self.model.train()
            for batch_X, batch_y, batch_w in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                optimizer.zero_grad()
                loss = criterion(self.model(batch_X.to(self.device)), batch_y.to(self.device), batch_w.to(self.device))
                loss.backward()
                optimizer.step()
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, f"{self.model_name}.pth"))

    def predict(self) -> pd.DataFrame:
        # 1. Load weights and set to evaluation mode
        self.model = HMLSTM(input_dim=6, num_quantiles=len(self.QUANTILES)).to(self.device)
        weights_path = os.path.join(self.output_dir, f"{self.model_name}.pth")
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

        # 2. Generate raw predictions using the DataLoader
        test_loader = DataLoader(self.test_processed, batch_size=1024, shuffle=False)
        all_preds = []
        with torch.no_grad():
            for batch_X, _, _ in test_loader:
                # Move to device and convert to numpy
                all_preds.append(self.model(batch_X.to(self.device)).cpu().numpy())
        
        preds_array = np.concatenate(all_preds) # Shape: (N_series * 28, 9)
        
        # 3. Format into the standardized CSV structure
        q_cols = [f"q{q}" for q in self.QUANTILES]
        res_list = [
            pd.DataFrame(preds_array[i*28:(i+1)*28], columns=q_cols)
            .assign(id=name, day_ahead=np.arange(1, 29)) 
            for i, name in enumerate(self.test_processed.item_ids)
        ]
        preds_df = pd.concat(res_list).reset_index(drop=True)

        # 4. MANDATORY: Enforce non-decreasing quantiles and non-negativity
        # This prevents the "AssertionError: Quantiles non-monotonic"
        preds_df[q_cols] = (
            preds_df[q_cols]
            .clip(lower=0) # Ensure no negative sales predictions
            .apply(lambda row: np.sort(row.values), axis=1, result_type='expand')
            .set_axis(q_cols, axis=1)
        )

        # 5. Save the finalized predictions to disk
        out_path = os.path.join(self.output_dir, f"{self.model_name}_predictions.csv")
        preds_df.to_csv(out_path, index=False)
        print(f"✅ Predictions saved to {out_path}")
        
        return preds_df