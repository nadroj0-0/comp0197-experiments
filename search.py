# =============================================================================
# search.py — Hyperparameter Search Runner
# COMP0197 Applied Deep Learning
#
# Runs successive halving search for the models listed in experiment.yml.
# The winning params are written into the run snapshot so final training can
# reuse them without changing the source configs.
#
# Can be called from legacy/legacy_train.py or run on its own:
#   python search.py
#   python search.py --experiment configs/experiment.yml
# =============================================================================

import argparse
import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[0]))

from utils.experiment import Experiment, get_model_dir
from utils.configs.config_loader import (
    load_experiment,
    load_registry,
    load_effective_train_config,
    load_search_space,
    create_run_dir,
    snapshot_configs,
    write_best_config,
)
from utils.runners.runner_utils import (
    attach_model_metadata,
    build_preloaded_experiment,
    load_shared_data,
    prepare_model_context,
    select_model_loaders,
)

PROJECT_DIR     = Path(__file__).resolve().parent
EXPERIMENT_PATH = PROJECT_DIR / "configs" / "experiment.yml"
REGISTRY_PATH   = PROJECT_DIR / "configs" / "registry.yml"
MODELS_CFG_DIR  = PROJECT_DIR / "configs" / "models"


# =============================================================================
# DATA LOADING — load once upfront, mirrors legacy/legacy_train.py exactly
# =============================================================================

def search_model(
    model_name:   str,
    run_dir:      Path,
    exp_cfg:      dict,
    exp_search:   dict,
    registry:     dict,
    train_loader,
    val_loader,
    stats,
    vocab_sizes:  dict,
) -> dict:
    """
    Run search for one model and save the winner into the run snapshot.
    """
    model_ctx = prepare_model_context(
        model_name=model_name,
        exp_cfg=exp_cfg,
        run_dir=run_dir,
        registry=registry,
    )
    run_model_yml = model_ctx["run_model_yml"]
    model_dir = model_ctx["model_dir"]

    print(f"\n{'='*60}")
    print(f"  SEARCH: {model_name}")
    print(f"{'='*60}")

    train_cfg = attach_model_metadata(model_ctx["train_cfg"], {"vocab_sizes": vocab_sizes})
    builder = model_ctx["builder"]
    step = model_ctx["training_step"]

    # --- load search space from run yml ---
    search_space = load_search_space(run_model_yml)
    if not search_space:
        print(f"  [SKIP] {model_name} — no search_space in yml, using train_config defaults")
        return {}

    # --- create Experiment and inject pre-loaded loaders ---
    exp = build_preloaded_experiment(
        model_name,
        train_cfg,
        model_dir,
        {"train_loader": train_loader, "val_loader": val_loader, "stats": stats},
    )

    # --- run search via Experiment.search ---
    # This calls staged_search internally and updates exp.cfg with the winner
    init_models = int(exp_search.get("init_models", 10))
    schedule    = exp_search.get("schedule", [
        {"epochs": 10, "keep": math.ceil(init_models / 2)},
        {"epochs": 10, "keep": math.ceil(init_models / 4)},
        {"epochs": 20, "keep": 1},
    ])

    exp.search(
        search_space   = search_space,
        builder        = builder,
        training_step  = step,
        schedule       = schedule,
        initial_models = init_models,
    )

    # Experiment.search updates self.cfg with the winner in-place
    best_cfg = exp.cfg.copy()

    # --- write best config back into run yml ---
    write_best_config(run_model_yml, best_cfg)

    print(f"\n  Best config for {model_name}:")
    for k, v in best_cfg.items():
        if not isinstance(v, dict):
            print(f"    {k}: {v}")
        else:
            for kk, vv in v.items():
                print(f"    {k}.{kk}: {vv}")

    return best_cfg


# =============================================================================
# MAIN SEARCH LOOP — called by train.py or standalone
# =============================================================================

def run_search(exp_cfg: dict, run_dir: Path) -> dict:
    """
    Run search for every model in the experiment.

    Data is loaded once, then routed per model just like in legacy/legacy_train.py.
    """
    registry   = load_registry(REGISTRY_PATH)
    exp_search = exp_cfg.get("search", {})
    exp_train  = exp_cfg.get("train",  {})
    models     = exp_cfg.get("models", [])

    if not models:
        print("[search] No models in experiment.yml — nothing to search.")
        return {}

    print(f"\n[search] Hyperparameter search — {len(models)} model(s)")
    print(f"[search] Sampling  : {exp_search.get('sampling', 'stratified')}")
    print(f"[search] Per-strat : {exp_search.get('top_k_series', 1000)}")
    print(f"[search] Init mods : {exp_search.get('init_models', 10)}")
    print(f"[search] Schedule  : {exp_search.get('schedule')}")

    # ------------------------------------------------------------------
    # Load data ONCE using the first model's config for data settings
    # All models in an experiment share the same data settings
    # (feature_set, seq_len, horizon, autoregressive, use_normalise)
    # ------------------------------------------------------------------
    first_model_yml = run_dir / "configs" / "models" / f"{models[0]}.yml"
    first_train_cfg = load_effective_train_config(exp_cfg, first_model_yml)
    loaders = load_shared_data(
        mode="search",
        exp_section=exp_search,
        first_train_cfg=first_train_cfg,
        include_test=False,
    )
    vocab_sizes = loaders.get("vocab_sizes", {})
    print(f"[search] Data loaded. Starting search loop.\n")

    # ------------------------------------------------------------------
    # Route correct loaders to each model — same logic as legacy/legacy_train.py
    # det models    → zscore loaders (loaders_det)
    # prob models   → raw count loaders (loaders_gauss)
    # NB models     → raw count loaders (loaders_nb)
    # ------------------------------------------------------------------
    all_best = {}
    for model_name in models:
        model_ctx = prepare_model_context(
            model_name=model_name,
            exp_cfg=exp_cfg,
            run_dir=run_dir,
            registry=registry,
        )
        routed = select_model_loaders(
            model_ctx["model_type"],
            model_ctx["probabilistic"],
            loaders,
        )

        best = search_model(
            model_name   = model_name,
            run_dir      = run_dir,
            exp_cfg      = exp_cfg,
            exp_search   = exp_search,
            registry     = registry,
            train_loader = routed["train_loader"],
            val_loader   = routed["val_loader"],
            stats        = routed["stats"],
            vocab_sizes=vocab_sizes,
        )
        all_best[model_name] = best

    print(f"\n[search] Done — {len(all_best)} model(s) searched.")
    return all_best


# =============================================================================
# STANDALONE ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="M5 Hyperparameter Search")
    parser.add_argument(
        "--experiment",
        type    = str,
        default = str(EXPERIMENT_PATH),
        help    = "Path to experiment.yml (default: configs/experiment.yml)",
    )
    args = parser.parse_args()

    exp_cfg  = load_experiment(args.experiment)
    run_name = exp_cfg.get("run_name")
    if not run_name:
        raise ValueError("experiment.yml must have a 'run_name' field.")

    # create run dir and snapshot configs
    # idempotent — safe if train.py already did this
    run_dir = create_run_dir(PROJECT_DIR, run_name)
    snapshot_configs(
        run_dir        = run_dir,
        experiment_yml = args.experiment,
        model_names    = exp_cfg.get("models", []),
        models_cfg_dir = MODELS_CFG_DIR,
    )

    run_search(exp_cfg, run_dir)


if __name__ == "__main__":
    main()
