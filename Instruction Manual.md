# The Field Guide
### A Complete Instruction Manual for the COMP0197 Group Training Infrastructure

---

## What This Is

This codebase is a general-purpose deep learning training framework built for the M5 Walmart sales forecasting project. It handles data loading, model building, training, hyperparameter search, early stopping, and saving outputs — all from a single entry point: `train.py`.

The key design principle is that **you should never need to touch the infrastructure code to add a new model**. Everything model-specific lives in two small functions you write yourself. The rest is handled automatically.

---

## Project Structure

```
Group/
├── train.py                      ← entry point
├── utils/
│   ├── data.py                   ← data loading and preprocessing pipelines
│   ├── network.py                ← model classes + builder functions
│   ├── training_strategies.py    ← training step functions
│   ├── common.py                 ← training loop, evaluation, saving utilities
│   ├── experiment.py             ← Experiment class — orchestrates everything
│   ├── hyperparameter.py         ← successive halving search
│   ├── training_session.py       ← stateful training session (epoch tracking)
│   └── early_stopping.py        ← early stopping logic
└── models/                       ← all training outputs saved here
    ├── gru_deterministic/
    ├── lstm_deterministic/
    └── gru_probabilistic/        
```

---

## File Descriptions

**`train.py`**. Defines `TRAIN_CONFIG`, search spaces, the `experiments` dict, and runs everything.

**`utils/data.py`** — Two data pipelines. The batched pipeline (`build_dataloaders_from_batches`) loads preprocessed pickle files, filters to CA_3, applies rolling temporal split, log1p normalises, and returns DataLoaders. The raw CSV pipeline is a legacy fallback.

**`utils/network.py`** — Model class definitions (`SalesGRU`, `SalesLSTM`) and their builder functions (`build_gru`, `build_lstm`). Add new model classes and builder functions here.

**`utils/training_strategies.py`** — Training step functions. Currently contains `gru_step` for standard regression. Add new step functions here for models with custom forward pass logic.

**`utils/common.py`** — The core training loop (`train_model`), evaluation (`evaluate_model`), and utilities (`save_model`, `save_json`, `rmse`, `mae`, `full_train`). Do not modify unless you know what you're doing.

**`utils/experiment.py`** — The `Experiment` class. Orchestrates data loading, hyperparameter search, and training in one call. Do not modify.

**`utils/hyperparameter.py`** — Implements successive halving search. Do not modify.

**`utils/training_session.py`** — Tracks epoch state across multiple training calls (needed for successive halving). Do not modify.

**`utils/early_stopping.py`** — Monitors validation loss and stops training when it stops improving. Do not modify.

---

## Running the Code

```bash
# Full training run (uses TRAIN_CONFIG values)
python train.py

# Quick debug run (override any TRAIN_CONFIG value from CLI)
python train.py --max_series 10 --epochs 5 --num_workers 0

# On GPU
python train.py --num_workers 4
```

> **Note:** CLI argument overrides are a debug convenience. Remove `parse_args()` before final submission and replace with `cfg = TRAIN_CONFIG.copy()` as indicated by the comments in `train.py`.

---

## TRAIN_CONFIG

At the top of `train.py` is `TRAIN_CONFIG` — a dictionary of all hyperparameters and settings used for training:

```python
TRAIN_CONFIG = {
    "seed":                     42,
    "epochs":                   50,
    "lr":                       1e-3,
    "hidden":                   128,
    "layers":                   2,
    "dropout":                  0.2,
    "batch_size":               256,
    "seq_len":                  28,
    "horizon":                  28,
    "store_id":                 "CA_3",
    "data_dir":                 "./data",
    "max_series":               None,
    "num_workers":              2,
    "early_stopping_patience":  10,
    "early_stopping_min_delta": 0.001,
}
```

This config is passed through the entire pipeline. Your builder function receives it as `cfg` and reads whatever values it needs. When `SEARCH=False`, these values are used directly. When `SEARCH=True`, the search overwrites specific values with the best found configuration before final training.

Any parameter you want to be searchable must exist in `TRAIN_CONFIG` with a sensible default — the search will sample new values for it and override the default.

---

## How to Add a New Model

Adding a new model requires writing **two functions** and **two lines in `train.py`**. You do not touch anything else.

### Step 1 — Write your builder function in `utils/network.py`

The builder function creates your model, loss function, optimiser, and training kwargs from the config. It must return exactly four things in this order:

```python
def build_my_model(cfg):
    """
    Builder for MyModel.
    Returns (model, criterion, optimiser, training_kwargs).
    """
    model = MyModel(
        input_size  = N_BATCH_FEATURES,
        hidden_size = int(cfg["hidden"]),   # cast to int — search samples floats
        num_layers  = int(cfg["layers"]),   # cast to int
        dropout     = cfg["dropout"],
        horizon     = int(cfg["horizon"]),
    ).to(device)

    criterion = nn.MSELoss()   # or your custom loss
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    training_kwargs = {
        "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, patience=5, factor=0.5
        ),
        "clip_grad_norm": 1.0,
        "extra_metrics": {"val_rmse": rmse, "val_mae": mae},
    }

    return model, criterion, optimiser, training_kwargs
```

**Important:** Always cast `int(cfg["hidden"])`, `int(cfg["layers"])`, `int(cfg["horizon"])` etc. The hyperparameter search samples floats even for integer parameters — not casting will crash PyTorch.

**`training_kwargs` options:**

| Key | Type | Description |
|-----|------|-------------|
| `scheduler` | `torch.optim.lr_scheduler.*` | LR scheduler, stepped on val loss each epoch |
| `clip_grad_norm` | `float` | Max gradient norm for clipping. `1.0` recommended for RNNs |
| `extra_metrics` | `dict` of `{name: fn}` | Extra metrics logged each epoch. Function signature: `fn(preds, targets) -> float` |

### Step 2 — Write your training step function in `utils/training_strategies.py`

The training step defines how a single batch is processed. For most standard models `gru_step` will work as-is — only write a new one if your model has unusual forward pass logic (e.g. multiple outputs, custom loss computation):

```python
def my_model_step(model, inputs, labels, criterion, **kwargs):
    """
    Training step for MyModel.
    Must return (loss, outputs).
    """
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    return loss, outputs

my_model_step.valid_train_accuracy = False  # always False for regression
```

The `valid_train_accuracy = False` attribute tells the training loop not to compute classification accuracy for this step.

### Step 3 — Add imports to `train.py`

At the top of `train.py`, import your new functions:

```python
from utils.network import build_gru, build_lstm, build_my_model
from utils.training_strategies import gru_step, my_model_step
```

### Step 4 — Add your model to the `experiments` dict in `train.py`

```python
experiments = {
    "gru_deterministic": dict(
        builder        = build_gru,
        training_step  = gru_step,
        search_space   = GRU_SEARCH_SPACE if SEARCH else None,
    ),
    "my_model": dict(
        builder        = build_my_model,
        training_step  = my_model_step,
        search_space   = MY_MODEL_SEARCH_SPACE if SEARCH else None,
    ),
}
```

Define a search space for your model above `main()`:

```python
MY_MODEL_SEARCH_SPACE = {
    "lr":      (1e-4, 1e-2, "log"),
    "hidden":  (64, 256, "uniform"),
    "layers":  (1, 3, "uniform"),
    "dropout": (0.1, 0.4, "uniform"),
}
```

That's everything. The infrastructure handles the rest.

---

## Hyperparameter Search

### Turning search on and off

At the top of `train.py`:

```python
SEARCH = True   # runs hyperparameter search before final training
SEARCH = False  # trains directly using TRAIN_CONFIG values — no search
```

When `SEARCH=False` the model trains once using exactly the values in `TRAIN_CONFIG`. This is the mode you want for final training runs.

When `SEARCH=True` the system runs successive halving search before training the final model.

### Defining a search space

A search space is a dictionary mapping parameter names to a tuple of `(low, high, mode)`:

```python
GRU_SEARCH_SPACE = {
    "lr":      (1e-4, 1e-2, "log"),       # log-uniform sampling between 1e-4 and 1e-2
    "hidden":  (64, 256, "uniform"),       # uniform sampling between 64 and 256
    "layers":  (1, 3, "uniform"),          # uniform sampling between 1 and 3
    "dropout": (0.1, 0.4, "uniform"),      # uniform sampling between 0.1 and 0.4
}
```

`"log"` mode is appropriate for learning rates and other parameters that span orders of magnitude. `"uniform"` is for everything else.

**The parameter names must match keys in `TRAIN_CONFIG`.** The search samples a new value for each listed parameter and merges it into the config — overriding the `TRAIN_CONFIG` default for that run. Parameters not listed in the search space keep their `TRAIN_CONFIG` values throughout.

### How successive halving works

The search schedule defines how many epochs each stage trains for and how many models survive to the next stage:

```python
HYPER_PARAM_INIT_MODELS = 20
HYPER_PARAM_SEARCH_SCHEDULE = [
    {"epochs": 10, "keep": 10},   # train 20 models for 10 epochs, keep best 10
    {"epochs": 10, "keep": 5},    # train 10 models for 10 more epochs, keep best 5
    {"epochs": 20, "keep": 1},    # train 5 models for 20 more epochs, keep best 1
]
```

The process step by step:

1. `HYPER_PARAM_INIT_MODELS` random configs are sampled from the search space
2. All models train for the first stage's epoch count
3. Models are ranked by best validation loss — the bottom half are eliminated
4. Survivors train for the next stage's epoch count, continuing from where they left off
5. This repeats until one model remains
6. The winning config is saved to `{model_dir}/{experiment_name}_search.json`
7. **The winning config is then used to train the final full model from scratch**

The key insight is that cheap early training (10 epochs) is enough to distinguish bad configs from promising ones. Resources are concentrated on the best candidates progressively. The final model is always trained from scratch with the best config — not the partially trained search model.

### What the search saves

The search saves a summary JSON to the model directory:

```json
{
    "search_type": "successive_halving",
    "timestamp": "2025-03-19 14:32:01",
    "initial_models": 20,
    "schedule": [...],
    "search_space": {...},
    "best_config": {"lr": 0.00312, "hidden": 187, ...},
    "best_epoch_metrics": {"epoch": 23, "train_loss": 0.41, "validation_loss": 0.38, ...},
    "runs": [...]
}
```

---

## Training Outputs

Each experiment saves its outputs to `models/{experiment_name}/`:

```
models/
└── gru_deterministic/
    ├── gru_deterministic_model.pt           ← saved model weights
    ├── gru_deterministic_train_history.json ← per-epoch metrics for every epoch
    ├── gru_deterministic_search.json        ← search summary (only if SEARCH=True)
    └── normalisation_stats.json             ← mean/std used to normalise the data
```

`train_history.json` contains per-epoch records with `train_loss`, `validation_loss`, `val_rmse`, `val_mae` and early stopping information. Pass `normalisation_stats.json` to `denormalise()` in `data.py` when converting predictions back to raw sales units in `test.py`.

---

## Early Stopping

Early stopping is always active. It monitors validation loss and stops training if it hasn't improved by at least `early_stopping_min_delta` for `early_stopping_patience` consecutive epochs. When triggered, it restores the model weights from the best epoch automatically.

These are controlled via `TRAIN_CONFIG`:

```python
"early_stopping_patience":  10,    # stop after 10 epochs of no improvement
"early_stopping_min_delta": 0.001, # improvement must be at least this large to count
```

---

## Important Rules

**Do not modify** `common.py`, `experiment.py`, `hyperparameter.py`, `training_session.py`, or `early_stopping.py`. These files are the infrastructure. Breaking them breaks everything.

**Only add code to** `network.py` (new model classes and builders) and `training_strategies.py` (new training step functions).

**Only edit** `train.py` — add imports at the top, add entries to the `experiments` dict, add search spaces above `main()`.

**Always cast integer parameters** in your builder function: `int(cfg["hidden"])`, `int(cfg["layers"])` etc. The search samples floats for all parameters. Passing a float to PyTorch where it expects an int will crash.

**Remove `parse_args()` before submission** — replace `cfg = parse_args()` with `cfg = TRAIN_CONFIG.copy()` as indicated by the comment in `train.py`.