# =============================================================================
# legacy_train.py — M5 Sales Forecasting — V3
# COMP0197 Applied Deep Learning
#
# Registry-driven training entrypoint.
# Reads the experiment config, snapshots it into runs/, optionally searches,
# then trains every requested model with shared full-data loaders.
#
# Usage:
#   python legacy/legacy_train.py                          — uses configs/experiment.yml
#   python legacy/legacy_train.py --batch_size 4096        — override batch size for GPU
#   python legacy/legacy_train.py --num_workers 8          — override workers for GPU
#   python legacy/legacy_train.py --experiment configs/experiment.yml  — explicit path
# =============================================================================

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.experiment import Experiment
from utils.configs.config_loader import (
    load_experiment,
    load_registry,
    create_run_dir,
    snapshot_configs,
)
from utils.common import save_json
from utils.runners.runner_utils import (
    attach_model_metadata,
    build_loader_cache_key,
    build_preloaded_experiment,
    load_shared_data,
    prepare_model_context,
    select_model_loaders,
)

PROJECT_DIR     = Path(__file__).resolve().parents[1]
EXPERIMENT_PATH = PROJECT_DIR / "configs" / "experiment.yml"
REGISTRY_PATH   = PROJECT_DIR / "configs" / "registry.yml"
MODELS_CFG_DIR  = PROJECT_DIR / "configs" / "models"

# =============================================================================
# CLI — only GPU convenience overrides, everything else comes from ymls
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="M5 Train V3")
    p.add_argument("--batch_size",  type=int, default=None,
                   help="Override batch_size from experiment.yml")
    p.add_argument("--num_workers", type=int, default=None,
                   help="Override num_workers from experiment.yml")
    p.add_argument("--experiment",  type=str, default=str(EXPERIMENT_PATH),
                   help="Path to experiment.yml (default: configs/experiment.yml)")
    return p.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print('main started')
    args = parse_args()
    print('args parsed')

    # ------------------------------------------------------------------
    # 1. Load experiment and registry
    # ------------------------------------------------------------------
    exp_cfg  = load_experiment(args.experiment)
    registry = load_registry(REGISTRY_PATH)
    run_name = exp_cfg.get("run_name")
    if not run_name:
        raise ValueError("experiment.yml must have a 'run_name' field.")
    models = exp_cfg.get("models", [])
    if not models:
        raise ValueError("experiment.yml 'models' list is empty.")

    print(f"\n{'='*60}")
    print(f"  M5 TRAINING — {run_name}")
    print(f"  Models : {models}")
    do_search = bool(exp_cfg.get("search", {}).get("enabled", False))
    print(f"  Search : {do_search}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 2. Create run directory and snapshot all configs into it
    # This must happen before search so search writes into the run copies
    # ------------------------------------------------------------------
    run_dir = create_run_dir(PROJECT_DIR, run_name)
    snapshot_configs(
        run_dir        = run_dir,
        experiment_yml = args.experiment,
        model_names    = models,
        models_cfg_dir = MODELS_CFG_DIR,
    )

    # ------------------------------------------------------------------
    # 3. Hyperparameter search (if SEARCH = True)
    # search.py writes best configs back into run yml files so that
    # step 4 onwards reads the winning hyperparameters automatically
    # ------------------------------------------------------------------
    if do_search:
        from search import run_search
        run_search(exp_cfg, run_dir)
        print(f"\n[train] Search complete — proceeding to full training.\n")

    # ------------------------------------------------------------------
    # 4. Load full training data per effective model config, with caching
    # Models can now diverge on autoregressive / feature / split settings
    # without silently sharing the wrong loader bundle.
    # ------------------------------------------------------------------
    exp_train = exp_cfg.get("train", {})
    loader_cache = {}
    print("[train] Preparing per-model cached loaders.\n")

    # ------------------------------------------------------------------
    # 5. Train each model
    # ------------------------------------------------------------------
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"  TRAINING: {model_name}")
        print(f"{'='*60}")

        model_ctx = prepare_model_context(
            model_name=model_name,
            exp_cfg=exp_cfg,
            run_dir=run_dir,
            registry=registry,
            runtime_overrides={
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
            },
        )
        cache_key = build_loader_cache_key(
            mode="train",
            exp_section=exp_train,
            first_train_cfg=model_ctx["train_cfg"],
            include_test=True,
            batch_size_override=args.batch_size,
            num_workers_override=args.num_workers,
        )
        if cache_key not in loader_cache:
            loader_cache[cache_key] = load_shared_data(
                mode="train",
                exp_section=exp_train,
                first_train_cfg=model_ctx["train_cfg"],
                include_test=True,
                batch_size_override=args.batch_size,
                num_workers_override=args.num_workers,
            )
        loaders = loader_cache[cache_key]
        train_cfg = attach_model_metadata(model_ctx["train_cfg"], loaders)
        routed = select_model_loaders(
            model_ctx["model_type"],
            model_ctx["probabilistic"],
            loaders,
        )
        exp = build_preloaded_experiment(
            model_name,
            train_cfg,
            model_ctx["model_dir"],
            routed,
        )

        # train directly — search already handled above
        exp.train(model_ctx["builder"], model_ctx["training_step"])

        # save normalisation stats if applicable
        if exp.stats is not None:
            save_json(exp.stats, model_ctx["model_dir"] / "normalisation_stats.json")

        print(f"\n  [DONE] {model_name} — artifacts in {model_ctx['model_dir']}")

    print(f"\n{'='*60}")
    print(f"  ALL TRAINING COMPLETE")
    print(f"  Run folder: {run_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
