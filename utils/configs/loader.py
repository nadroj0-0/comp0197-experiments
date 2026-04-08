from copy import deepcopy
from pathlib import Path

import yaml


def load_experiment(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Experiment config not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"Experiment config is empty: {path}")
    return cfg


def load_model_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model config not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"Model config is empty: {path}")
    return cfg


def load_best_config(run_model_yml_path: str | Path) -> dict | None:
    path = Path(run_model_yml_path)
    if not path.exists():
        return None
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("best_config", None)


def load_train_config(run_model_yml_path: str | Path) -> dict:
    path = Path(run_model_yml_path)
    cfg = load_model_config(path)
    train_cfg = cfg.get("train_config")
    if train_cfg is None:
        raise ValueError(f"No 'train_config' section in {path}")
    return train_cfg


def deep_merge_dicts(base: dict | None, override: dict | None) -> dict:
    base = deepcopy(base or {})
    override = override or {}

    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            base[key] = deep_merge_dicts(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def build_effective_train_config(
    exp_cfg: dict,
    model_cfg: dict,
    runtime_overrides: dict | None = None,
) -> dict:
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


def load_search_space(model_yml_path: str | Path) -> dict | None:
    cfg = load_model_config(model_yml_path)
    raw = cfg.get("search_space")
    if raw is None:
        return None
    out = {}
    for key, vals in raw.items():
        low, high, mode = vals
        if mode == "choice":
            out[key] = (list(low), high, str(mode))
        else:
            out[key] = (float(low), float(high), str(mode))
    return out


def build_model_run_cfg(
    model_name: str,
    run_dir: str | Path,
    exp_cfg: dict,
    phase: str = "train",
) -> dict:
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

