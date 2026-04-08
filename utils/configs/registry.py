import importlib
from pathlib import Path

import yaml


_BUILDER_MODULE = "utils.network"
_STEP_MODULE = "utils.training.strategies"

_BUILDER_REGISTRY = None
_STEP_REGISTRY = None


def load_registry(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Registry not found: {path}")
    with open(path) as f:
        reg = yaml.safe_load(f)
    if reg is None:
        raise ValueError(f"Registry is empty: {path}")
    return reg


def _get_builders() -> dict:
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
    builders = _get_builders()
    steps = _get_steps()

    builder_name = entry["builder"]
    step_name = entry["training_step"]

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
        "builder": builders[builder_name],
        "training_step": steps[step_name],
        "is_prob": bool(entry.get("is_prob", False)),
        "is_nb": bool(entry.get("is_nb", False)),
        "is_quantile": bool(entry.get("is_quantile", False)),
    }
