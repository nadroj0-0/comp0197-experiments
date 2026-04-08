# Field Guide — COMP0197 Group Training Infrastructure

---

## TL;DR — How to Add Your Model

1. Write `build_your_model(cfg)` in `utils/network/`
2. Write `your_step(model, inputs, labels, criterion, **kwargs)` in `utils/training/strategies.py` (or reuse `gru_step` if your model is standard)
3. Add your model to `configs/registry.yml`
4. Create `configs/models/your_model.yml`
5. If you want a BaseModel wrapper, add the wrapper class to `models/gru_models.py`
6. Add your model name to `configs/experiment.yml`
7. Run either `python legacy/legacy_train.py --experiment configs/experiment.yml` or `python train_gru_models.py --experiment configs/experiment.yml`

That's it. Core infrastructure modules under `utils/common/`, `utils/training/`, `utils/configs/`, `utils/data/`, and `utils/network/` implement the shared pipeline and should not be modified lightly.

## Setup

```bash
conda env create -f environment.yml
conda activate comp0197-group-pt
```

---

## Overview

This framework standardises training, evaluation, and comparison of time-series forecasting models on M5. It separates model architecture (`utils/network/`), training logic (`utils/training/strategies.py`), and experiment configuration (YAML files) so that only one variable changes between experiments — making ablation results directly comparable.

There are now two active entrypoint families:

- `legacy/legacy_train.py` / `search.py` = the original registry-driven training + search pipeline
- `train_gru_models.py` / `test_gru_models.py` = the BaseModel-facing wrapper train + evaluation pipeline

Both save into `runs/run_name/`. Configs are snapshotted at the start of each run, and search results are written back into the run copy, so runs stay reproducible even if the source YAML changes later.

---

## Project Structure

```
Group/
├── legacy/
│   └── legacy_train.py         ← original training entry point
├── search.py                   ← hyperparameter search runner
├── models/
│   ├── base_model.py           ← shared BaseModel abstraction
│   ├── gru_models.py           ← wrapper classes for GRU-family models
│   └── __init__.py             ← re-exports wrapper-facing imports
├── train_gru_models.py         ← BaseModel-facing training entry point
├── test_gru_models.py          ← wrapper-based evaluation runner
├── configs/
│   ├── experiment.yml          ← run-level defaults, search settings, model list
│   ├── registry.yml            ← maps model names to builder functions
│   └── models/                 ← per-model overrides and search spaces
└── utils/
    ├── configs/                ← config loading, registry resolution, run snapshots
    ├── data/                   ← data pipeline (do not modify)
    ├── network/                ← ADD YOUR MODEL CLASS + BUILDER HERE
    ├── training/               ← training sessions, strategies, hyperparameter search
    ├── common/                 ← training loop, metrics, losses, serialisation
    ├── eval/                   ← shared evaluation helpers
    ├── runners/                ← shared runner prep utilities
    └── ...                     ← everything else (do not modify)
```

---

## Running the Code

```bash
# Train all models listed in experiment.yml using the original pipeline
python legacy/legacy_train.py

# Train all models listed in experiment.yml using the BaseModel wrappers
python train_gru_models.py

# Override batch size / workers for GPU in the original pipeline
python legacy/legacy_train.py --batch_size 4096 --num_workers 8

# Point at a different experiment file
python legacy/legacy_train.py --experiment configs/my_experiment.yml
python train_gru_models.py --experiment configs/my_experiment.yml

# Run evaluation and generate plots
python test_gru_models.py
```

Search is controlled from `configs/experiment.yml`:

```yaml
search:
  enabled: true
```

If `search.enabled: false`, the model trains directly from the current config. If `search.enabled: true`, search runs first and the winning config is used for the final training run.

---

## BaseModel Wrapper Path

The wrapper path was added to make the GRU-family models usable through the abstract interface in `models/base_model.py`, while still reusing the original training/search infrastructure underneath.

In practice this means:

- `models/gru_models.py` contains one wrapper class per GRU-family model
- each wrapper still resolves the real builder and training step from `configs/registry.yml`
- final training still goes through `Experiment.train(...)`
- the wrapper path uses the BaseModel data split first, then builds the forecasting loaders on top of that split

Typical usage:

```bash
python train_gru_models.py --experiment configs/experiment.yml
python test_gru_models.py --run_name my_run
```

Or in a notebook:

```python
from models import HierarchicalGRUDet

m = HierarchicalGRUDet(run_name="my_run")
preds = m.predict()
metrics = m.evaluate(preds)
```

This is mainly intended for the GRU/baseline/hierarchical wrapper models. The original registry-based training/search path is still there if you want to use the full pipeline directly, but `test_gru_models.py` is the supported evaluation path for active runs.

---

## Search Behaviour

The two training paths handle search slightly differently:

- `legacy/legacy_train.py` stays fully inside the original registry/data pipeline
- `train_gru_models.py` uses the wrapper/BaseModel path for final training

For the wrapper path, search is treated as a config-selection step:

1. if `search.enabled: false`, the wrapper just trains normally from the BaseModel path
2. if `search.enabled: true`, it temporarily detours into the original search pipeline
3. search runs on the smaller stratified subset defined in `experiment.yml`
4. the winning config is written into the run snapshot
5. final training then returns to the normal wrapper/BaseModel path

So search is optional, but if it is enabled the final trained model still comes from the same wrapper flow your teammates use.

---

## Config Rules

Config values now come from two places:

- `configs/experiment.yml` = run-level defaults
- `configs/models/<model>.yml` = model-specific overrides and search space

Effective training config precedence is:

1. `experiment.yml -> train`
2. `model.yml -> train_config`
3. `model.yml -> best_config` (if search ran)
4. CLI/runtime overrides

This is why some common settings now live in `experiment.yml`, while architecture-specific settings still live in each model yml.

---

## Step 1 — Write Your Builder (`utils/network/`)

The builder takes a config dict and returns exactly four things:

```python
def build_my_model(cfg):
    model = MyModel(
        input_size  = int(cfg["n_features"]),  # set automatically — do not hardcode
        hidden_size = int(cfg["hidden"]),       # always cast to int
        num_layers  = int(cfg["layers"]),       # always cast to int
        dropout     = float(cfg["dropout"]),
        horizon     = int(cfg["horizon"]),
    ).to(device)

    criterion = nn.MSELoss()  # or your custom loss
    optimiser = OptimisationConfig.configure_optimiser(model, cfg)

    training_kwargs = OptimisationConfig.configure_training_kwargs(optimiser, cfg)
    training_kwargs["clip_grad_norm"] = 1.0
    training_kwargs["extra_metrics"]  = {"val_rmse": rmse, "val_mae": mae}

    return model, criterion, optimiser, training_kwargs
```

**Always cast `int()`** for hidden, layers, horizon etc — the search samples floats for all parameters and PyTorch will crash if you pass a float where it expects an int.

For probabilistic models that output `(mu, sigma)` or `(mu, alpha)`, look at `build_baseline_prob_gru`, `build_baseline_prob_gru_nb`, `build_hierarchical_prob_gru`, or `build_hierarchical_prob_gru_nb` in `utils/network/` as references.

---

## Step 2 — Write Your Training Step (`utils/training/strategies.py`)

Most models can reuse the existing `gru_step`. Only write a new one if your model has an unusual forward pass (e.g. multiple outputs, custom loss inputs):

```python
def my_step(model, inputs, labels, criterion, **kwargs):
    outputs = model(inputs)
    loss    = criterion(outputs, labels)
    return loss, outputs

my_step.valid_train_accuracy = False  # always False for regression
```

---

## Step 3 — Register Your Model (`configs/registry.yml`)

```yaml
my_model:
  builder:       build_my_model
  training_step: my_step       # or gru_step if standard
  is_prob:       false
  is_nb:         false
  is_quantile:   false
```

---

## Step 4 — Create Your Config (`configs/models/my_model.yml`)

```yaml
model_name: my_model
model_type: my_model
probabilistic: false

train_config:
  autoregressive: false         # false = seq2seq (recommended), true = AR rollout
  feature_set: sales_yen_hierarchy
  optimiser: adamw
  optimiser_params:
    lr: 0.001
    weight_decay: 0.0001
  scheduler: plateau
  scheduler_params:
    patience: 5
    factor: 0.5
  clip_grad_norm: 1.0
  hidden: 128
  layers: 2
  dropout: 0.1

best_config: {}

search_space:
  optimiser_params.lr: [5.0e-5, 0.005, log]
  hidden: [64, 256, uniform]
  layers: [1, 3, uniform]
  dropout: [0.05, 0.3, uniform]
```

---

## Step 5 — Add to Experiment (`configs/experiment.yml`)

```yaml
run_name: my_run

train:
  data_dir: ./data
  seq_len: 28
  horizon: 28
  use_normalise: false
  epochs: 60
  early_stopping_patience: 8
  early_stopping_min_delta: 0.0005
  sampling: all
  batch_size: 2048
  num_workers: 8
  seed: 42

search:
  enabled: false

eval:
  data_dir: ./data
  sampling: all

models:
  - my_model
```

Change `run_name` each time you start a new experiment — all outputs go into `runs/run_name/` and nothing gets overwritten.

---

## Training Outputs

Everything saves automatically to `runs/run_name/models/my_model/`:

```
my_model_model.pt              ← saved weights
my_model_train_history.json    ← per-epoch metrics
my_model_search.json           ← search results (if SEARCH=True)
```

---

## Important Notes

- `n_features` is set automatically from `feature_set` — do not hardcode it
- `vocab_sizes` is injected automatically for hierarchical models — ignore it for standard models
- Val/test loaders always return `(x, y)` — the weighted loader is handled automatically for `wquantile` models
- Early stopping is always active — it restores the best epoch weights automatically
- Outputs go to `runs/` not `models/` — the old path in earlier versions is now obsolete
- `feature_set` and some architecture choices still live in the model yml because they are genuinely model-specific in this repo

---

---

# Further Reading
### You don't need this to add a model. Read it if you want to understand what the pipeline is doing under the hood, or to verify your own data processing matches ours.

---

## Data Pipeline — What Actually Happens

When you call `build_dataloaders(...)`, the pipeline runs these steps in order:

**1. Download / load raw data**
The M5 CSV files (`sales_train_evaluation.csv`, `calendar.csv`, `sell_prices.csv`) are downloaded once from HuggingFace and cached in `./data/`. Subsequent runs use the local cache.

**2. Series selection**
The full dataset has 30,490 series. The `sampling` parameter controls which series are used:
- `"all"` — full dataset, used for final training
- `"stratified"` — 1,000 series from each of the top, middle, and bottom thirds of total sales volume. Used for hyperparameter search to ensure the search sees the full range of demand patterns, not just high-volume items
- `"top"` — top-N by volume only

**3. Melt wide → long**
The raw CSV is wide format (one row per item, one column per day). It's melted to long format: one row per (item, day).

**4. Feature encoding**
Depending on `feature_set`, static hierarchy columns are integer-encoded:

| `feature_set` | Features | Input size |
|---|---|---|
| `sales_only` | raw sales | 1 |
| `sales_hierarchy` | sales + state, store, cat, dept | 5 |
| `sales_hierarchy_dow` | sales + hierarchy + day of week | 6 |
| `sales_yen` | sales + price/calendar/SNAP/event features | 10 |
| `sales_yen_hierarchy` | yen-style features + hierarchy IDs | 14 |

Integer encoding uses pandas `cat.codes` which guarantees 0-indexed contiguous IDs. Choose the feature set that matches your experiment protocol. The current submission config uses `sales_yen_hierarchy`.

**5. Temporal split**
The split is purely date-based — no shuffling ever happens across the time axis. The legacy/data-loader path supports two protocols:

```
default:
  Train : d1    → d1885
  Val   : d1886 → d1913
  Test  : d1914 → d1941

yen_v1:
  Train : d1    → d1773
  Val   : d1774 → d1885
  Test  : d1886 → d1941
```

This is enforced by date boundaries, not index slicing, so it's robust to reordering. The current submission config uses `split_protocol: yen_v1`, which aligns the active training/evaluation workflow with the wrapper/BaseModel split regime.

**6. Normalisation (optional, off by default)**
If `use_normalise=True`, sales are log1p transformed and z-score normalised using stats fitted on the train split only. Stats are never computed on val or test data. Saved to `normalisation_stats.json` for use at evaluation time. **Most models in this project use `use_normalise=False` (raw counts).**

**7. Windowed dataset**
A sliding window is applied per series. Each window produces:
- `x` : sequence of length `seq_len` (default 28) — shape `(seq_len, n_features)`
- `y` : target — shape depends on `autoregressive` flag (see below)

Windows are assigned to train/val/test by the date of their target, not their input. This means a window whose input starts in the train period but whose target falls in the val period is correctly assigned to val — no leakage.

---

## Autoregressive vs Direct Multi-Step (Seq2Seq)

This is controlled by `autoregressive` in your model yml. It affects what `y` looks like during training and how inference works at test time.

**`autoregressive: true`**
- During training: `y` is a scalar — the single next step. The model predicts one step ahead.
- During test: the model predicts step 1, feeds that prediction back as input, predicts step 2, and so on for 28 steps. This is recursive rollout.
- Problem: errors compound. Each prediction error corrupts the next input, and mistakes accumulate over 28 steps. Particularly bad for volatile or intermittent series.

**`autoregressive: false` (seq2seq / direct multi-step)**
- During training: `y` is a vector of length `horizon=28` — all 28 future steps predicted at once from the same input window.
- During test: single forward pass produces all 28 predictions directly. No feedback loop, no error compounding.
- Your model's output head must produce `(B, 28)` not `(B, 1)`. Use `int(cfg["horizon"])` when building the output layer — the pipeline sets this automatically from your yml.

**Recommendation: use `autoregressive: false`** unless your architecture specifically requires step-by-step decoding. Seq2seq avoids error compounding and is simpler to implement correctly.

---

## Feature Sets — What Each Column Is

The current submission setup uses `feature_set: sales_yen_hierarchy`, so the active input tensor shape is `(batch, seq_len, 14)`:

| Column index | Feature | Type | Notes |
|---|---|---|---|
| 0 | `sales` | float | Raw unit sales count (or log1p normalised if `use_normalise=True`) |
| 1 | `sell_price` | float | Daily selling price after forward-fill within each series |
| 2 | `is_available` | float | 1 if a non-missing price exists for that day, else 0 |
| 3 | `wday` | float | M5 calendar weekday index |
| 4 | `month` | float | Calendar month |
| 5 | `year` | float | Calendar year |
| 6 | `snap_CA` | float | SNAP flag for California |
| 7 | `snap_TX` | float | SNAP flag for Texas |
| 8 | `snap_WI` | float | SNAP flag for Wisconsin |
| 9 | `has_event` | float | 1 if `event_name_1` is present, else 0 |
| 10 | `state_id_int` | int-as-float | 0-indexed state ID |
| 11 | `store_id_int` | int-as-float | 0-indexed store ID |
| 12 | `cat_id_int` | int-as-float | 0-indexed category ID |
| 13 | `dept_id_int` | int-as-float | 0-indexed department ID |

Hierarchy columns are stored as `float32` in the tensor but represent integer category codes. Baseline models treat them as raw numeric inputs. Hierarchical models cast them to `long` and pass them through embedding tables.

---

## Hyperparameter Search — How It Works

When `search.enabled: true`, the pipeline runs successive halving before final training:

1. Sample `init_models` (default 10) random configs from your `search_space`
2. Train all configs for the first stage's epoch count on a stratified subsample (3,000 series)
3. Keep the top performers, discard the rest
4. Train survivors for the next stage, repeat until one config remains
5. Write the winning config back into your model's run yml
6. Train the final model from scratch on the full dataset using the winning config

The search operates on a stratified subsample for speed — searching on 30,490 series would take days. The winning config is always used for a clean full-dataset training run afterwards.

Search space syntax in your yml:
```yaml
search_space:
  optimiser_params.lr: [5.0e-5, 0.005, log]     # log-uniform — good for learning rates
  hidden: [64, 256, uniform]                      # uniform — good for architecture params
  layers: [1, 3, uniform]
  dropout: [0.05, 0.3, uniform]
```

---

## Revenue Weighting

The `wquantile` models use a revenue-weighted pinball loss. Weights are computed as:

```
weight_i = (total_volume_i × avg_price_i) / sum(all revenues)
```

Weights sum to 1 across all series. High-revenue items get proportionally more influence on the loss. The weighted data loader returns `(x, y, weight)` tuples during training — the infrastructure handles this routing automatically. Val and test loaders always return `(x, y)` regardless.

---

## Config Snapshotting and Reproducibility

Every time you run `legacy/legacy_train.py` or `train_gru_models.py`, the current state of `experiment.yml` and all referenced model ymls are copied into `runs/run_name/configs/`. This means:

- Re-running `test_gru_models.py` always uses the exact config that trained the model, even if you've edited the configs since
- If hyperparameter search ran, the winning config is already merged into the run snapshot
- Different `run_name` values create completely independent output directories — nothing is ever overwritten

---

---

# File Reference

---

## `utils/data/`

The entire data pipeline lives here. You should not need to modify this.

| Function / Class | What it does |
|---|---|
| `build_dataloaders(...)` | Main entry point. Call this to get `(train_loader, val_loader, test_loader, stats, vocab_sizes, feature_index)` |
| `WindowedM5Dataset` | PyTorch Dataset that generates sliding windows per series. Handles both autoregressive (scalar y) and direct multi-step (vector y) modes |
| `get_feature_cols(feature_set)` | Returns the list of column names for a given feature set |
| `get_vocab_sizes(df)` | Returns a dict of `{col: n_unique}` for hierarchy embedding tables |
| `split_data(df)` | Applies the temporal train/val/test split. Returns three dataframes |
| `trim_data(df, n, sampling)` | Selects series by volume. `sampling="all"` returns full dataset |
| `encode_hierarchy(df)` | Adds integer-encoded hierarchy columns (`state_id_int` etc.) |
| `fit_normalisation_stats(train_df)` | Computes log1p + z-score stats from train data only |
| `apply_normalisation(df, stats)` | Applies precomputed stats to any split |
| `denormalise(preds, stats)` | Reverses normalisation for predictions at eval time |
| `load_or_download_m5(data_dir)` | Downloads M5 CSVs from HuggingFace if not cached locally |

---

## `utils/network/`

All model classes and their builder functions. **This is where you add your model.**

| Class | What it is |
|---|---|
| `BaselineGRU` | Vanilla single-layer GRU → linear head. Deterministic. |
| `BaselineProbGRU` | Vanilla GRU → (mu, sigma) heads. Gaussian NLL. |
| `BaselineProbGRU_NB` | Vanilla GRU → (mu, alpha) heads. Negative Binomial NLL. |
| `BaselineQuantileGRU` | Vanilla GRU → n_quantiles head. Pinball loss. |
| `HierarchicalGRU` + variants | All baseline GRU variants with learned hierarchy embeddings via `_HierarchyEmbedder`. |
| `_HierarchyEmbedder` | Shared embedding block for hierarchical models. Replaces raw integer hierarchy columns with dense learned vectors. |

Each class has a corresponding `build_*` function that constructs the model, loss, optimiser, and training kwargs from a config dict.

---

## `utils/training/strategies.py`

Training step functions — one per model type. Each takes `(model, inputs, labels, criterion, **kwargs)` and returns `(loss, outputs)`.

| Function | What it does |
|---|---|
| `gru_step` | Standard single-output regression step. Works for any deterministic model. |
| `prob_gru_step` | Unpacks `(mu, sigma)` from model output, calls Gaussian NLL loss. |
| `prob_nb_step` | Unpacks `(mu, alpha)` from model output, calls NB NLL loss. |
| `quantile_gru_step` | Calls pinball loss over all quantiles, returns median as point prediction for metrics. |
| `wquantile_gru_step` | Same as quantile but uses revenue-weighted pinball when `item_weights` present in kwargs. Falls back to unweighted for val/test. |

---

## `utils/common/`

Core training loop and shared utilities. Do not modify.

| Function | What it does |
|---|---|
| `train_model(...)` | Main epoch loop. Handles AMP, gradient clipping, scheduler stepping, early stopping, extra metrics logging. |
| `evaluate_model(...)` | Runs inference on a dataloader, returns loss + predictions + targets. |
| `full_train(...)` | Convenience wrapper: builds model via builder, creates TrainingSession, trains, saves. |
| `save_model(model, name, dir)` | Saves model weights as `.pt` file. |
| `save_history(...)` | Saves training history as JSON with full config, architecture string, and per-epoch metrics. |
| `load_history(path)` | Loads a saved history JSON. |
| `gaussian_nll_loss(mu, sigma, targets)` | Gaussian NLL with optional sigma regularisation to prevent variance collapse. |
| `nb_nll_loss(mu, alpha, targets)` | Negative Binomial NLL using the closed-form log-likelihood. |
| `pinball_loss(preds, targets, quantiles)` | Standard pinball loss summed over quantiles. |
| `weighted_pinball_loss(preds, targets, weights, quantiles)` | Revenue-weighted pinball loss. |
| `rmse / mae / mape / r2` | Metric functions used as `extra_metrics` in builders. |

---

## `utils/experiment.py`

The `Experiment` class orchestrates everything. Do not modify.

| Method | What it does |
|---|---|
| `__init__(name, cfg, model_dir)` | Creates experiment instance with config and output directory. |
| `prepare_data(data_fn, **kwargs)` | Loads data and stores loaders as instance attributes. Skipped when `preloaded=True`. |
| `search(search_space, builder, ...)` | Runs successive halving, updates `self.cfg` with the winning config in-place. |
| `train(builder, training_step)` | Trains the final model using `self.cfg`. Saves weights and history. |
| `run(...)` | Convenience: `prepare_data` → optional `search` → `train` in one call. |

---

## `utils/training/hyperparameter.py`

Successive halving search implementation. Do not modify.

| Function | What it does |
|---|---|
| `staged_search(...)` | Main search loop. Samples configs, trains in stages, prunes, returns best config. |
| `sample_config(base_config, search_space)` | Samples one random config from the search space. |
| `prune(sessions, keep)` | Keeps top-k sessions by best validation loss, discards the rest. |
| `Leaderboard` | Ranks sessions by validation loss. Used internally by `prune`. |

---

## `utils/training/optimisation.py`

Centralised optimiser and scheduler construction. Do not modify — use it in your builder.

| Method | What it does |
|---|---|
| `OptimisationConfig.configure_optimiser(model, cfg)` | Builds Adam / AdamW / SGD from cfg. Reads `optimiser` and `optimiser_params` keys. |
| `OptimisationConfig.configure_scheduler(optimiser, cfg)` | Builds ReduceLROnPlateau or CosineAnnealingLR from cfg. |
| `OptimisationConfig.configure_training_kwargs(optimiser, cfg)` | Returns the full `training_kwargs` dict with scheduler, clip norm, and optional extras. |

---

## `utils/configs/config_loader.py`

YAML config management. Do not modify.

| Function | What it does |
|---|---|
| `load_experiment(path)` | Loads `experiment.yml`. |
| `load_model_config(path)` | Loads a single model yml. |
| `load_registry(path)` | Loads `registry.yml` as raw dict (strings not yet resolved). |
| `resolve_registry_entry(entry)` | Converts string builder/step names to actual Python callables. |
| `create_run_dir(project_dir, run_name)` | Creates `runs/run_name/` directory. |
| `snapshot_configs(run_dir, ...)` | Copies all relevant configs into the run directory for reproducibility. |
| `write_best_config(model_yml, best_cfg)` | Merges winning search config back into the run yml after search completes. |
| `build_effective_train_config(...)` | Merges experiment defaults, model config, search winner, and runtime overrides. |
| `load_effective_train_config(...)` | Loads a model yml and returns the merged effective train config used by the active pipelines. |
| `load_search_space(path)` | Extracts the `search_space` block from a model yml. |

---

## `utils/training/early_stopping.py`

| Class | What it does |
|---|---|
| `EarlyStopping` | Monitors val loss each epoch. Stores best model state via `copy.deepcopy`. Triggers after `patience` epochs without improvement exceeding `min_delta`. Restores best weights on trigger. |

---

## `utils/training/session.py`

| Class / Function | What it does |
|---|---|
| `TrainingSession` | Stateful wrapper around model + optimiser + criterion. Tracks epoch count across multiple `.train()` calls — needed for successive halving where the same session is trained in multiple stages. |
