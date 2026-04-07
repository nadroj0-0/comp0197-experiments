# =============================================================================
# utils/config_loader.py — YAML config management for the experiment pipeline
# COMP0197 Applied Deep Learning
#
# Small helpers for loading configs, resolving registry entries, and managing
# run snapshots.
# =============================================================================

import importlib
import shutil
import yaml
from copy import deepcopy
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
    Load one model yml file.
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
    Convert a raw registry entry into callables plus metadata flags.
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
    Copy the active experiment and model configs into the run directory.

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
        if not dst.exists():
            shutil.copy2(src, dst)
            print(f"[config] Snapshotted model config  → {dst}")
        else:
            print(f"[config] Skipping snapshot (already exists) → {dst}")


# =============================================================================
# SEARCH RESULT PERSISTENCE
# =============================================================================

def write_best_config(
    run_model_yml_path: str | Path,
    best_config:        dict,
) -> None:
    """
    Save the winning search config into the run copy of the model yml.

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
    cfg["train_config"] = deep_merge_dicts(cfg["train_config"], best_config)

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
    Load the raw train_config section from a model yml.
    """
    path = Path(run_model_yml_path)
    cfg = load_model_config(path)
    train_cfg = cfg.get("train_config")
    if train_cfg is None:
        raise ValueError(f"No 'train_config' section in {path}")
    return train_cfg


def deep_merge_dicts(base: dict | None, override: dict | None) -> dict:
    """
    Recursively merge two dictionaries without mutating either input.
    Values from override win on conflicts.
    """
    base = deepcopy(base or {})
    override = override or {}

    for key, value in override.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            base[key] = deep_merge_dicts(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def build_effective_train_config(
    exp_cfg: dict,
    model_cfg: dict,
    runtime_overrides: dict | None = None,
) -> dict:
    """
    Build the effective train config used at runtime.

    Order of precedence:
      1. experiment.yml train
      2. model train_config
      3. model best_config
      4. runtime overrides
    """
    merged = deep_merge_dicts(exp_cfg.get("train", {}), model_cfg.get("train_config", {}))
    merged = deep_merge_dicts(merged, model_cfg.get("best_config", {}))
    merged = deep_merge_dicts(merged, runtime_overrides or {})
    return merged


def load_effective_train_config(
    exp_cfg: dict,
    run_model_yml_path: str | Path,
    runtime_overrides: dict | None = None,
) -> dict:
    model_cfg = load_model_config(run_model_yml_path)
    return build_effective_train_config(exp_cfg, model_cfg, runtime_overrides=runtime_overrides)


# =============================================================================
# SEARCH SPACE PARSING
# =============================================================================

def load_search_space(model_yml_path: str | Path) -> dict | None:
    """
    Load the search_space section from a model yml.

    YAML lists are converted to tuples because the search helpers expect that
    format.
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
    Build a flat runtime config for one model.

    Args:
        model_name : e.g. "baseline_gru_det"
        run_dir    : root of this run
        exp_cfg    : loaded experiment.yml dict
        phase      : "train" or "eval" — which exp_cfg block to use

    Returns flat dict ready to pass to build_dataloaders and builders.
    """
    run_dir = Path(run_dir)
    model_yml_path = run_dir / "configs" / "models" / f"{model_name}.yml"
    if phase == "train":
        merged = load_effective_train_config(exp_cfg, model_yml_path)
    else:
        train_cfg = load_train_config(model_yml_path)
        phase_cfg = exp_cfg.get(phase, {}).copy()
        merged = deep_merge_dicts(phase_cfg, train_cfg)
    merged["model_name"] = model_name

    return merged
