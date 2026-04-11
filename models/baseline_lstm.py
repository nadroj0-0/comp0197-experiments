import os
import time
import gc
import pickle
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from .base_model import BaseModel

# --- Helper Classes (CRITICAL: These must be in the file) ---

class SlidingWindowDataset(Dataset):
    def __init__(self, df, feature_cols, seq_len):
        self.features = df[feature_cols].values.astype(np.float32)
        self.targets = df['sales'].values.astype(np.float32)
        self.seq_len = seq_len
        self.indices = np.arange(seq_len, len(df))

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        curr_idx = self.indices[idx]
        x = self.features[curr_idx - self.seq_len : curr_idx]
        y = self.targets[curr_idx]
        return torch.tensor(x), torch.tensor(y)

class QuantileLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, n_quantiles):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Updated to match the "head" keys found in your saved weights
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, n_quantiles)
        )

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Features)
        out, _ = self.lstm(x)
        # We take the output of the last time step
        last_step = out[:, -1, :] 
        return self.head(last_step)

class PinballLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles
    def forward(self, preds, target):
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))
        return torch.cat(losses, dim=1).mean()

# --- Main Class ---

class LSTMBaseline(BaseModel):

    model_name   = 'lstm_baseline'
    SEQ_LEN      = 28
    HIDDEN_SIZE  = 128
    NUM_LAYERS   = 2
    DROPOUT      = 0.2
    BATCH_SIZE   = 4096   
    EPOCHS       = 50
    LR           = 1e-3
    PATIENCE     = 10
    FEATURE_COLS = [
        'sell_price', 'is_available',
        'wday_sin', 'wday_cos', 'month_sin', 'month_cos',
        'is_event_day', 'snap',
        'sales_lag_7', 'sales_lag_28',
        'sales_roll_mean_7', 'sales_roll_mean_28',
    ] 

    # ── preprocess ──────────────────────────────────────────────

    def preprocess(self):
        print('\nPreprocessing...')

        train_df = self._add_features(self.train_raw)
        del self.train_raw; self.train_raw = None; gc.collect()

        val_df = self._add_features(self.val_raw)

        val_context = self.val_raw[
            self.val_raw['d_num'] > self.val_raw['d_num'].max() - self.SEQ_LEN
        ].copy()
        test_df = self._add_features(
            pd.concat([val_context, self.test_raw]).reset_index(drop=True)
        )
        del val_context; gc.collect()

        train_df, val_df, test_df = self._normalise(train_df, val_df, test_df)

        print('Building datasets...')
        self.train_processed = SlidingWindowDataset(train_df, self.FEATURE_COLS, self.SEQ_LEN)
        del train_df; gc.collect()

        self.val_processed = SlidingWindowDataset(val_df, self.FEATURE_COLS, self.SEQ_LEN)
        del val_df; gc.collect()

        self.test_processed = SlidingWindowDataset(test_df, self.FEATURE_COLS, self.SEQ_LEN)

        self._test_df = test_df
        proc_path = os.path.join(self.output_dir, f'{self.model_name}_test_processed.pkl')
        with open(proc_path, 'wb') as f:
            pickle.dump({'test_df': test_df, 'norm_stats': self._norm_stats}, f)
        print(f'Saved test_processed to {proc_path}')
        print('Preprocessing complete!')

    def _add_features(self, df):
        import polars as pl
        keep = ['id', 'd_num', 'store_id', 'wday', 'month',
                'event_name_1', 'snap_CA', 'snap_TX', 'snap_WI',
                'sell_price', 'is_available', 'sales']
        lf = pl.from_pandas(df[keep]).lazy().sort(['id', 'd_num'])
        lf = lf.with_columns([
            (2 * np.pi * pl.col('wday')  / 7).sin().cast(pl.Float32).alias('wday_sin'),
            (2 * np.pi * pl.col('wday')  / 7).cos().cast(pl.Float32).alias('wday_cos'),
            (2 * np.pi * pl.col('month') / 12).sin().cast(pl.Float32).alias('month_sin'),
            (2 * np.pi * pl.col('month') / 12).cos().cast(pl.Float32).alias('month_cos'),
            (pl.col('event_name_1') != 'none').cast(pl.Float32).alias('is_event_day'),
            pl.when(pl.col('store_id').str.starts_with('CA')).then(pl.col('snap_CA'))
              .when(pl.col('store_id').str.starts_with('TX')).then(pl.col('snap_TX'))
              .otherwise(pl.col('snap_WI')).cast(pl.Float32).alias('snap'),
            pl.col('sales').shift(7).over('id').alias('sales_lag_7'),
            pl.col('sales').shift(28).over('id').alias('sales_lag_28'),
            pl.col('sales').shift(1).rolling_mean(window_size=7,  min_samples=1).over('id').alias('sales_roll_mean_7'),
            pl.col('sales').shift(1).rolling_mean(window_size=28, min_samples=1).over('id').alias('sales_roll_mean_28'),
        ])
        return (lf
            .drop(['event_name_1', 'snap_CA', 'snap_TX', 'snap_WI'])
            .drop_nulls(subset=['sales_lag_7', 'sales_lag_28'])
            .collect()
            .to_pandas()
        )

    def _normalise(self, train_df, val_df, test_df):
        continuous_cols = [
            'sell_price', 'sales_lag_7', 'sales_lag_28',
            'sales_roll_mean_7', 'sales_roll_mean_28', 'sales',
        ]
        self._norm_stats = {}
        for col in continuous_cols:
            for df in (train_df, val_df, test_df):
                df[col] = np.log1p(df[col].clip(lower=0))
            mean = float(train_df[col].mean())
            std  = float(train_df[col].std()) + 1e-8
            self._norm_stats[col] = {'mean': mean, 'std': std}
            for df in (train_df, val_df, test_df):
                df[col] = (df[col] - mean) / std
        print('Normalisation complete!')
        return train_df, val_df, test_df

    def _denormalise(self, x):
        mean = self._norm_stats['sales']['mean']
        std  = self._norm_stats['sales']['std']
        return np.expm1(x * std + mean).clip(min=0)

    # ── train ────────────────────────────────────────────────────

    def train(self):
        print(f'\nTraining {self.model_name} (Sliding Window Lazy)...')

        train_loader = DataLoader(
            self.train_processed, batch_size=self.BATCH_SIZE,
            shuffle=True, num_workers=4, pin_memory=True,
        )
        val_loader = DataLoader(
            self.val_processed, batch_size=self.BATCH_SIZE,
            shuffle=False, num_workers=4, pin_memory=True,
        )

        self.model = QuantileLSTM(
            input_size=len(self.FEATURE_COLS), hidden_size=self.HIDDEN_SIZE,
            num_layers=self.NUM_LAYERS, dropout=self.DROPOUT,
            n_quantiles=len(self.QUANTILES),
        ).to(self.device)

        criterion = PinballLoss(self.QUANTILES).to(self.device)
        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.LR)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, patience=5, factor=0.5)

        best_val_loss, best_weights, patience_ctr = float('inf'), None, 0
        self._history = []

        for epoch in range(1, self.EPOCHS + 1):
            t0 = time.time()

            self.model.train()
            train_losses = []
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimiser.zero_grad()
                preds = self.model(x)          # (B, Q)
                loss  = criterion(preds, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimiser.step()
                train_losses.append(loss.item())

            self.model.eval()
            val_losses, med_preds, med_targs = [], [], []
            median_idx = self.QUANTILES.index(0.5)
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    preds = self.model(x)
                    val_losses.append(criterion(preds, y).item())
                    med_preds.append(preds[:, median_idx].cpu().numpy())
                    med_targs.append(y.cpu().numpy())

            train_loss = np.mean(train_losses)
            val_loss   = np.mean(val_losses)
            mp = np.concatenate(med_preds)
            mt = np.concatenate(med_targs)
            val_rmse = np.sqrt(np.mean((mp - mt) ** 2))
            val_mae  = np.mean(np.abs(mp - mt))

            scheduler.step(val_loss)
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_weights  = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_ctr  = 0
            else:
                patience_ctr += 1

            self._history.append({'epoch': epoch, 'train_loss': train_loss,
                                   'val_loss': val_loss, 'val_rmse': val_rmse, 'val_mae': val_mae})
            print(f'Epoch {epoch:02d} | train={train_loss:.4f} | val={val_loss:.4f} | '
                  f'rmse={val_rmse:.4f} | mae={val_mae:.4f} | {time.time()-t0:.1f}s')

            if patience_ctr >= self.PATIENCE:
                print(f'Early stopping at epoch {epoch}')
                break

        self.model.load_state_dict(best_weights)
        save_path = os.path.join(self.output_dir, f'{self.model_name}.pth')
        torch.save(self.model.state_dict(), save_path)
        print(f'\nBest val_loss: {best_val_loss:.4f}')
        print(f'Model saved to {save_path}')

    # ── predict ──────────────────────────────────────────────────

    def predict(self) -> pd.DataFrame:
        print(f'\nRunning Vectorized prediction for {self.model_name}...')
        
        # Load data and weights (same as before)
        proc_path = os.path.join(self.output_dir, f'{self.model_name}_test_processed.pkl')
        with open(proc_path, 'rb') as f:
            saved = pickle.load(f)
        test_df = saved['test_df']
        self._norm_stats = saved['norm_stats']

        weights_path = os.path.join(self.output_dir, f'{self.model_name}.pth')
        self.model = QuantileLSTM(
            input_size=len(self.FEATURE_COLS), hidden_size=self.HIDDEN_SIZE,
            num_layers=self.NUM_LAYERS, dropout=self.DROPOUT,
            n_quantiles=len(self.QUANTILES),
        ).to(self.device)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

        # 1. Prepare initial windows for ALL items
        print("Preparing initial state tensors...")
        # Get the context data (last 28 days before target)
        context_df = test_df[test_df['d_num'] < self.TARGET_START].sort_values(['id', 'd_num'])
        
        # We only keep items that have a full SEQ_LEN of history
        valid_ids = context_df.groupby('id').size()
        valid_ids = valid_ids[valid_ids >= self.SEQ_LEN].index.tolist()
        context_df = context_df[context_df['id'].isin(valid_ids)]
        
        # Extract features into a giant tensor (N_items, SEQ_LEN, N_features)
        init_features = context_df.groupby('id').tail(self.SEQ_LEN)[self.FEATURE_COLS].values
        x_window = torch.tensor(init_features, dtype=torch.float32).reshape(-1, self.SEQ_LEN, len(self.FEATURE_COLS)).to(self.device)
        
        # 2. Get static/future data for the 28 days
        target_days = test_df[test_df['d_num'] >= self.TARGET_START].sort_values(['id', 'd_num'])
        target_days = target_days[target_days['id'].isin(valid_ids)]
        
        # Index map for features
        fc = self.FEATURE_COLS
        idx = {col: fc.index(col) for col in fc}
        median_idx = self.QUANTILES.index(0.5)

        # Buffer to keep track of predictions to calculate lags/rolling means
        # Start with the history from context
        sales_history = init_features[:, idx['sales_lag_7']].reshape(-1, self.SEQ_LEN).tolist()
        all_step_preds = [] # To store (Step, N_items, Q)

        print(f"Starting recursive batch prediction for {len(valid_ids)} items...")
        with torch.no_grad():
            for step in range(self.PRED_LENGTH):
                # Predict next day for ALL items at once
                preds = self.model(x_window) # (N_items, Q)
                all_step_preds.append(preds.cpu().numpy())
                
                # Get median prediction to feed back as "actual" sales
                new_sales = preds[:, median_idx].cpu().numpy()
                
                # Update our sales history list for each item
                for i in range(len(valid_ids)):
                    sales_history[i].append(float(new_sales[i]))
                
                if step < self.PRED_LENGTH - 1:
                    # Construct the next input row for all items
                    # We start with a copy of the last time step
                    new_row = x_window[:, -1, :].clone()
                    
                    # Update dynamic features (Lags and Rolling)
                    for i in range(len(valid_ids)):
                        hist = sales_history[i]
                        new_row[i, idx['sales_lag_7']] = hist[-7]
                        new_row[i, idx['sales_lag_28']] = hist[-28] if len(hist) >= 28 else hist[0]
                        new_row[i, idx['sales_roll_mean_7']] = np.mean(hist[-7:])
                        new_row[i, idx['sales_roll_mean_28']] = np.mean(hist[-28:])
                    
                    # Update Calendar features for the specific day
                    # (This part is still a bit slow but much faster than before)
                    current_d = self.TARGET_START + step + 1
                    day_data = target_days[target_days['d_num'] == current_d][self.FEATURE_COLS].values
                    if len(day_data) == len(valid_ids):
                        # Update wday, month, event, snap from the actual test_df
                        cal_indices = [idx['wday_sin'], idx['wday_cos'], idx['month_sin'], idx['month_cos'], idx['is_event_day'], idx['snap']]
                        new_row[:, cal_indices] = torch.tensor(day_data[:, cal_indices], dtype=torch.float32).to(self.device)

                    # Slide window: drop oldest, add newest
                    x_window = torch.cat([x_window[:, 1:, :], new_row.unsqueeze(1)], dim=1)

        # 3. Post-process into DataFrame
        print("Finalizing results...")
        all_step_preds = np.array(all_step_preds) # (28, N_items, Q)
        q_cols = [f'q{q}' for q in self.QUANTILES]
        
        results = []
        for i, item_id in enumerate(valid_ids):
            item_preds = self._denormalise(all_step_preds[:, i, :])
            df = pd.DataFrame(item_preds, columns=q_cols)
            df['id'] = item_id
            df['day_ahead'] = np.arange(1, 29)
            results.append(df)

        preds_df = pd.concat(results).reset_index(drop=True)
        # Enforce monotonicity
        preds_df[q_cols] = preds_df[q_cols].clip(lower=0)
        preds_df[q_cols] = np.sort(preds_df[q_cols].values, axis=1)
        
        out_path = os.path.join(self.output_dir, f'{self.model_name}_predictions.csv')
        preds_df.to_csv(out_path, index=False)
        return preds_df