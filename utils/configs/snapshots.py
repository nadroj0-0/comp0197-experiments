import shutil
from pathlib import Path

import yaml

from .loader import deep_merge_dicts


def get_run_dir(base_dir: str | Path, run_name: str) -> Path:
    return Path(base_dir) / "runs" / run_name


def create_run_dir(base_dir: str | Path, run_name: str) -> Path:
    run_dir = get_run_dir(base_dir, run_name)
    (run_dir / "configs" / "models").mkdir(parents=True, exist_ok=True)
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    print(f"[config] Run directory: {run_dir}")
    return run_dir


def get_model_run_dir(run_dir: str | Path, model_name: str) -> Path:
    d = Path(run_dir) / "models" / model_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def snapshot_configs(
    run_dir: str | Path,
    experiment_yml: str | Path,
    model_names: list[str],
    models_cfg_dir: str | Path = "configs/models",
) -> None:
    run_dir = Path(run_dir)
    experiment_yml = Path(experiment_yml)
    models_cfg_dir = Path(models_cfg_dir)

    dst_exp = run_dir / "configs" / "experiment.yml"
    shutil.copy2(experiment_yml, dst_exp)
    print(f"[config] Snapshotted experiment config → {dst_exp}")

    for model_name in model_names:
        src = models_cfg_dir / f"{model_name}.yml"
        dst = run_dir / "configs" / "models" / f"{model_name}.yml"
        if not src.exists():
            raise FileNotFoundError(f"Model config not found for '{model_name}': {src}")
        if not dst.exists():
            shutil.copy2(src, dst)
            print(f"[config] Snapshotted model config  → {dst}")
        else:
            print(f"[config] Skipping snapshot (already exists) → {dst}")


def write_best_config(run_model_yml_path: str | Path, best_config: dict) -> None:
    path = Path(run_model_yml_path)
    if not path.exists():
        raise FileNotFoundError(f"Run model config not found: {path}")

    with open(path) as f:
        cfg = yaml.safe_load(f)

    cfg["best_config"] = best_config
    if "train_config" not in cfg or cfg["train_config"] is None:
        cfg["train_config"] = {}
    cfg["train_config"] = deep_merge_dicts(cfg["train_config"], best_config)

    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"[config] Best config written → {path}")

