# =============================================================================
# utils/config_loader.py — YAML config management for the experiment pipeline
# COMP0197 Applied Deep Learning
# GenAI Note: Scaffolded with Claude (Anthropic). Verified by authors.
#
# Responsibilities:
#   - Load experiment.yml, registry.yml, and per-model yml files
#   - Resolve string builder/step names to actual Python callables
#   - Create and manage the runs/ directory structure
#   - Snapshot configs into run dirs at experiment start
#   - Read and write best_config back into model yml files after search
# =============================================================================

import importlib
import shutil
import yaml
from pathlib import Path


# =============================================================================
# LOADING
# =============================================================================

def load_experiment(path: str | Path) -> dict:
    """
    Load experiment.yml.
    Returns the full experiment config dict.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Experiment config not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"Experiment config is empty: {path}")
    return cfg


def load_model_config(path: str | Path) -> dict:
    """
    Load a single model yml file.
    Returns dict with keys: train_config, search_space, and model metadata.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model config not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"Model config is empty: {path}")
    return cfg


def load_registry(path: str | Path) -> dict:
    """
    Load registry.yml.
    Returns raw dict — strings not yet resolved to callables.
    Use resolve_registry_entry() to get actual Python objects.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Registry not found: {path}")
    with open(path) as f:
        reg = yaml.safe_load(f)
    if reg is None:
        raise ValueError(f"Registry is empty: {path}")
    return reg


# =============================================================================
# CALLABLE RESOLUTION
# =============================================================================

# Maps string names to the module where they live.
# Add new builders/steps here when new architectures are added to network.py
# or training_strategies.py.
_BUILDER_MODULE   = "utils.network"
_STEP_MODULE      = "utils.training_strategies"

_BUILDER_REGISTRY = None   # lazy-loaded cache
_STEP_REGISTRY    = None   # lazy-loaded cache


def _get_builders() -> dict:
    """Lazy-load all builder functions from utils.network."""
    global _BUILDER_REGISTRY
    if _BUILDER_REGISTRY is None:
        mod = importlib.import_module(_BUILDER_MODULE)
        _BUILDER_REGISTRY = {
            name: getattr(mod, name)
            for name in dir(mod)
            if name.startswith("build_") and callable(getattr(mod, name))
        }
    return _BUILDER_REGISTRY


def _get_steps() -> dict:
    """Lazy-load all training step functions from utils.training_strategies."""
    global _STEP_REGISTRY
    if _STEP_REGISTRY is None:
        mod = importlib.import_module(_STEP_MODULE)
        _STEP_REGISTRY = {
            name: getattr(mod, name)
            for name in dir(mod)
            if callable(getattr(mod, name)) and not name.startswith("_")
        }
    return _STEP_REGISTRY


def resolve_registry_entry(entry: dict) -> dict:
    """
    Convert a raw registry entry (strings) into a resolved entry
    (actual Python callables).

    Input entry (from registry.yml):
        builder:       "build_baseline_gru"
        training_step: "gru_step"
        is_prob:       false
        is_nb:         false
        is_quantile:   false

    Returns:
        {
            "builder":       <function build_baseline_gru>,
            "training_step": <function gru_step>,
            "is_prob":       False,
            "is_nb":         False,
            "is_quantile":   False,
        }
    """
    builders = _get_builders()
    steps    = _get_steps()

    builder_name = entry["builder"]
    step_name    = entry["training_step"]

    if builder_name not in builders:
        raise ValueError(
            f"Builder '{builder_name}' not found in {_BUILDER_MODULE}. "
            f"Available: {sorted(builders.keys())}"
        )
    if step_name not in steps:
        raise ValueError(
            f"Training step '{step_name}' not found in {_STEP_MODULE}. "
            f"Available: {sorted(steps.keys())}"
        )

    return {
        "builder":       builders[builder_name],
        "training_step": steps[step_name],
        "is_prob":       bool(entry.get("is_prob", False)),
        "is_nb":         bool(entry.get("is_nb",   False)),
        "is_quantile":   bool(entry.get("is_quantile", False)),
    }


# =============================================================================
# RUN DIRECTORY MANAGEMENT
# =============================================================================

def get_run_dir(base_dir: str | Path, run_name: str) -> Path:
    """Return the path to a run directory without creating it."""
    return Path(base_dir) / "runs" / run_name


def create_run_dir(base_dir: str | Path, run_name: str) -> Path:
    """
    Create the full run directory structure:

        runs/{run_name}/
        ├── configs/
        │   └── models/
        └── models/

    Returns the run root Path.
    """
    run_dir = get_run_dir(base_dir, run_name)
    (run_dir / "configs" / "models").mkdir(parents=True, exist_ok=True)
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    print(f"[config] Run directory: {run_dir}")
    return run_dir


def get_model_run_dir(run_dir: str | Path, model_name: str) -> Path:
    """
    Return the artifact directory for a specific model within a run.
    Creates it if it doesn't exist.
    """
    d = Path(run_dir) / "models" / model_name
    d.mkdir(parents=True, exist_ok=True)
    return d


# =============================================================================
# CONFIG SNAPSHOTTING
# =============================================================================

def snapshot_configs(
    run_dir:        str | Path,
    experiment_yml: str | Path,
    model_names:    list[str],
    models_cfg_dir: str | Path = "configs/models",
) -> None:
    """
    At the start of a run, copy all relevant config files into the run dir.
    This creates an immutable record of exactly what config was used.

        runs/{run_name}/configs/experiment.yml
        runs/{run_name}/configs/models/{model_name}.yml  (one per model)

    Args:
        run_dir        : Root of this run  e.g. runs/sales_only_top200
        experiment_yml : Path to the master experiment.yml
        model_names    : List of model names in the experiment
        models_cfg_dir : Directory containing the master model ymls
    """
    run_dir        = Path(run_dir)
    experiment_yml = Path(experiment_yml)
    models_cfg_dir = Path(models_cfg_dir)

    # Copy experiment.yml
    dst_exp = run_dir / "configs" / "experiment.yml"
    shutil.copy2(experiment_yml, dst_exp)
    print(f"[config] Snapshotted experiment config → {dst_exp}")

    # Copy each model yml
    for model_name in model_names:
        src = models_cfg_dir / f"{model_name}.yml"
        dst = run_dir / "configs" / "models" / f"{model_name}.yml"
        if not src.exists():
            raise FileNotFoundError(
                f"Model config not found for '{model_name}': {src}"
            )
        shutil.copy2(src, dst)
        print(f"[config] Snapshotted model config  → {dst}")


# =============================================================================
# SEARCH RESULT PERSISTENCE
# =============================================================================

def write_best_config(
    run_model_yml_path: str | Path,
    best_config:        dict,
) -> None:
    """
    After hyperparameter search, write the winning config back into the
    run copy of the model yml file.

    Updates two sections:
      - best_config : raw search winner (for reference)
      - train_config: merged with best_config so train.py always reads
                      a single source of truth

    Args:
        run_model_yml_path : Path to the model yml inside the run dir
                             e.g. runs/sales_only/configs/models/baseline_gru_det.yml
        best_config        : Dict of winning hyperparameters from staged_search
    """
    path = Path(run_model_yml_path)
    if not path.exists():
        raise FileNotFoundError(f"Run model config not found: {path}")

    with open(path) as f:
        cfg = yaml.safe_load(f)

    # Store the raw best config for inspection
    cfg["best_config"] = best_config

    # Merge into train_config so train.py reads a single consistent source
    if "train_config" not in cfg or cfg["train_config"] is None:
        cfg["train_config"] = {}
    cfg["train_config"].update(best_config)

    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"[config] Best config written → {path}")


def load_best_config(run_model_yml_path: str | Path) -> dict | None:
    """
    Read the best_config section from a run model yml.
    Returns None if no best_config has been written yet
    (i.e. search hasn't run for this model).
    """
    path = Path(run_model_yml_path)
    if not path.exists():
        return None
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("best_config", None)


def load_train_config(run_model_yml_path: str | Path) -> dict:
    """
    Load the train_config section from a run model yml.
    This is the config train.py uses for full training.
    If search has run, best_config values will already be merged in.
    If search has not run, these are the user-specified defaults.
    """
    path = Path(run_model_yml_path)
    cfg = load_model_config(path)
    train_cfg = cfg.get("train_config")
    if train_cfg is None:
        raise ValueError(f"No 'train_config' section in {path}")
    return train_cfg


# =============================================================================
# SEARCH SPACE PARSING
# =============================================================================

def load_search_space(model_yml_path: str | Path) -> dict | None:
    """
    Load the search_space section from a model yml.
    Converts list format [low, high, mode] to tuple format (low, high, mode)
    as expected by staged_search / sample_config.

    Returns None if no search_space section exists.

    YAML format:
        search_space:
          lr: [0.0001, 0.01, log]
          hidden: [32, 128, uniform]

    Returns:
        {
            "lr":     (0.0001, 0.01, "log"),
            "hidden": (32, 128, "uniform"),
        }
    """
    cfg = load_model_config(model_yml_path)
    raw = cfg.get("search_space")
    if raw is None:
        return None
    # YAML loads lists — convert to tuples for hyperparameter.py
    return {
        key: (float(vals[0]), float(vals[1]), str(vals[2]))
        for key, vals in raw.items()
    }


# =============================================================================
# CONVENIENCE: BUILD FULL EXPERIMENT CFG FOR A SINGLE MODEL
# =============================================================================

def build_model_run_cfg(
    model_name:        str,
    run_dir:           str | Path,
    exp_cfg:           dict,
    phase:             str = "train",
) -> dict:
    """
    Assemble a complete flat config dict for a single model, merging:
      - experiment-level data settings (from exp_cfg[phase])
      - model-level train_config (from run model yml)

    Model-level settings take precedence over experiment-level settings.

    Args:
        model_name : e.g. "baseline_gru_det"
        run_dir    : root of this run
        exp_cfg    : loaded experiment.yml dict
        phase      : "train" or "eval" — which exp_cfg block to use

    Returns flat dict ready to pass to build_dataloaders and builders.
    """
    run_dir = Path(run_dir)
    model_yml_path = run_dir / "configs" / "models" / f"{model_name}.yml"
    train_cfg = load_train_config(model_yml_path)

    # Start with experiment-level data settings
    phase_cfg = exp_cfg.get(phase, {}).copy()

    # Merge — model yml wins on conflicts
    merged = {**phase_cfg, **train_cfg}
    merged["model_name"] = model_name

    return merged