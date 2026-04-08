"""
Public config-loader module for the experiment pipeline.

The actual implementation is split across loader, registry, and snapshots, but
most callers want one import location.
"""

from .loader import (
    build_effective_train_config,
    build_model_run_cfg,
    deep_merge_dicts,
    load_best_config,
    load_effective_train_config,
    load_experiment,
    load_model_config,
    load_search_space,
    load_train_config,
)
from .registry import load_registry, resolve_registry_entry
from .snapshots import (
    create_run_dir,
    get_model_run_dir,
    get_run_dir,
    snapshot_configs,
    write_best_config,
)

__all__ = [
    "build_effective_train_config",
    "build_model_run_cfg",
    "create_run_dir",
    "deep_merge_dicts",
    "get_model_run_dir",
    "get_run_dir",
    "load_best_config",
    "load_effective_train_config",
    "load_experiment",
    "load_model_config",
    "load_registry",
    "load_search_space",
    "load_train_config",
    "resolve_registry_entry",
    "snapshot_configs",
    "write_best_config",
]

