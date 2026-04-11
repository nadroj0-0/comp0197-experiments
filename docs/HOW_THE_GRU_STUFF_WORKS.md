# How the GRU Stuff Works

This note explains the GRU-side additions inside the final submission folder without changing the group's front-page README or the original group pipeline structure.

It is mainly intended as a practical guide to what was added, where the GRU artifacts live, and what happens when the standard group scripts are run.

## What Was Added

The GRU forecasting system is now integrated directly into the group codebase. The active GRU runtime lives at the root of the final submission folder:

- `configs/`
- `models/gru_models.py`
- `models/utils/gru_inference.py`
- `models/utils/gru_pipeline.py`
- `models/utils/gru_registry.py`
- `utils/`
- `search.py`
- `runs/`

The original group code is still the same apart from:

- a small registry hook in `models_config.py`
- a small export in `models/__init__.py`

Everything else on the GRU side is additive.

## How GRU Models Enter the Group Pipeline

The original group scripts still work the same way:

- `run_data.py`
- `run_train_all.py`
- `run_evaluate_all.py`
- `run_pipeline.py`

The only GRU-specific glue is:

1. `configs/experiment.yml` lists the active GRU models for the current run.
2. `models_config.py` appends those active GRU models into `MODEL_REGISTRY`.
3. The unchanged group scripts iterate through `MODEL_REGISTRY`.
4. When a GRU wrapper is instantiated, it routes into the GRU config-driven pipeline under the hood.

So from the outside, the GRUs behave like part of the group registry, but internally they still keep their own run-based experiment structure.

## Current Frozen Submission Run

At submission time, the GRU side is currently frozen to:

```text
runs/baseline_ablation_sales_yen_hierarchy_embedded_seq2seq_search/
```

with the following active GRU models:

- `hierarchical_gru_det`
- `hierarchical_gru_prob`
- `hierarchical_quantile_gru`

and with search disabled in the root `configs/experiment.yml`.

## Source of Truth: `runs/`

The canonical GRU artifacts live in:

```text
runs/<run_name>/models/<model_name>/
```

This is where the real GRU outputs are stored:

- `<model_name>_model.pt`
- `<model_name>_test_metrics.json`
- `<model_name>_basemodel_metrics.json`
- `<model_name>_training_curves.png`
- search summaries
- run snapshots in `runs/<run_name>/configs/...`

This means ablations and final runs stay reproducible and do not overwrite each other as long as they use different `run_name` values.

## Public Mirror: `outputs/`

To make the GRU models feel native inside the group pipeline, a public mirror is maintained in:

```text
outputs/<model_name>/
```

This is the group-facing view. It mirrors the currently active GRU run and contains things the group scripts expect to find, such as:

- `<model_name>.pth`
- `<model_name>_predictions.csv`
- `<model_name>_test_metrics.json`
- `<model_name>_basemodel_metrics.json`
- `<model_name>_training_curves.png`
- `source_run.json`

The real GRU artifacts still live in `runs/`; `outputs/` is only the public compatibility layer.

This also means `outputs/` should be thought of as the public view of the currently selected GRU run, not as the archival location for every GRU experiment.

## How Training Is Controlled

The main GRU control file is:

- `configs/experiment.yml`

At the root level, this controls:

- `run_name`
- the active GRU model list
- whether search is enabled
- shared train defaults

For a fresh run from scratch, the effective config for each model is built from:

1. `configs/experiment.yml`
2. `configs/models/<model>.yml`
3. snapped `best_config` values if search was previously run

For shipped final models, the important source of truth is the snapped run config inside:

```text
runs/<run_name>/configs/
```

So in practice:

- the root `configs/experiment.yml` chooses which GRU run is active
- the snapped config inside `runs/<run_name>/configs/` defines the actual shipped model settings for that run

## What Happens When the Group Runs `run_train_all.py`

For GRUs:

1. The wrapper reads the active `run_name` from `configs/experiment.yml`.
2. It checks whether a real checkpoint already exists in:
   - `runs/<run_name>/models/<model_name>/<model_name>_model.pt`
3. If it exists, the wrapper mirrors a public `.pth` file into `outputs/<model_name>/`.
4. The unchanged group skip logic sees that `.pth` and skips retraining.

So if the final GRU checkpoints are already shipped, the group trainer should skip them cleanly.

## What Happens When the Group Runs `run_evaluate_all.py`

For GRUs:

1. The wrapper points evaluation at the frozen `run_name`.
2. If predictions already exist in `outputs/<model_name>/`, they can be loaded directly.
3. Otherwise predictions are regenerated from the shipped checkpoint in `runs/...`.
4. Metrics are computed and the GRU results are included in the group-level:
   - `outputs/model_comparison.csv`

So the GRUs appear in the same comparison table as the original group models.

## Search Behavior

Search is controlled by:

```yaml
search:
  enabled: true | false
```

If `search.enabled: false`, the GRU wrapper does not launch search.

If weights already exist for the active run, the unchanged group trainer skips the GRU before training even begins, so search is also skipped in practice.

For submission/final use, the intended configuration is:

- point `run_name` at the frozen final run
- include only the final chosen GRU models
- set `search.enabled: false`

With that setup, the normal group training script should skip GRU retraining if the shipped checkpoints are already present.

## Final Submission Convention

For the final submission, the GRU side is intended to work like this:

- `runs/` stores the real frozen final GRU artifacts
- `outputs/` is populated automatically when the group scripts run
- the group front-page README stays untouched
- this file is only a side document for understanding the GRU additions

## If Someone Wants to Retrain Safely

They should use a new `run_name`.

That protects the frozen final run and keeps new experiments separate from the submitted results.

If they retrain against the same `run_name`, they are intentionally modifying the shipped run artifacts.

That is why the recommended safe workflow for any new GRU experiment is:

1. create a new `run_name`
2. run training/search there
3. leave the frozen submission run untouched
