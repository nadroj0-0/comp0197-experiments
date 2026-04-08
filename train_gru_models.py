# =============================================================================
# train_gru_models.py — M5 Sales Forecasting — V3 (BaseModel interface)
# COMP0197 Applied Deep Learning
#
# BaseModel-facing training entrypoint.
# Each wrapper model handles its own preprocessing, then trains through the
# same underlying Experiment code as the registry-based pipeline.
#
# Usage:
#   python train_gru_models.py                          — uses configs/experiment.yml
#   python train_gru_models.py --experiment configs/experiment.yml
#   python train_gru_models.py --run_name my_run        — override run name
# =============================================================================

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from utils.configs.config_loader import (
    load_experiment,
    load_registry,
    create_run_dir,
    snapshot_configs,
)
from models import get_model_class, get_available_model_names

PROJECT_DIR     = Path(__file__).resolve().parent
EXPERIMENT_PATH = PROJECT_DIR / "configs" / "experiment.yml"
REGISTRY_PATH   = PROJECT_DIR / "configs" / "registry.yml"
MODELS_CFG_DIR  = PROJECT_DIR / "configs" / "models"

# =============================================================================
# CLI — similar to train.py, but data loading happens inside each wrapper
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="M5 Train GRU Models (BaseModel interface)")
    p.add_argument("--experiment", type=str, default=str(EXPERIMENT_PATH),
                   help="Path to experiment.yml (default: configs/experiment.yml)")
    p.add_argument("--run_name",   type=str, default=None,
                   help="Override run_name from experiment.yml")
    p.add_argument("--num_workers", type=int, default=None,
                   help="Override num_workers from experiment.yml")
    return p.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("main started")
    args = parse_args()
    print("args parsed")

    # ------------------------------------------------------------------
    # 1. Load experiment config
    # ------------------------------------------------------------------
    exp_cfg  = load_experiment(args.experiment)
    registry = load_registry(REGISTRY_PATH)
    run_name = args.run_name or exp_cfg.get("run_name")
    if not run_name:
        raise ValueError("experiment.yml must have a 'run_name' field.")
    models = exp_cfg.get("models", [])
    if not models:
        raise ValueError("experiment.yml 'models' list is empty.")

    do_search = bool(exp_cfg.get("search", {}).get("enabled", False))

    print(f"\n{'='*60}")
    print(f"  M5 TRAINING (BaseModel) — {run_name}")
    print(f"  Models : {models}")
    print(f"  Search : {do_search}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 2. Create run directory and snapshot configs before training starts.
    # ------------------------------------------------------------------
    run_dir = create_run_dir(PROJECT_DIR, run_name)
    # Use the models directory next to the chosen experiment file
    experiment_path = Path(args.experiment)
    models_cfg_dir = experiment_path.parent / "models"
    snapshot_configs(
        run_dir        = run_dir,
        experiment_yml = args.experiment,
        model_names    = models,
        #models_cfg_dir = MODELS_CFG_DIR,
        models_cfg_dir = models_cfg_dir,
    )

    # ------------------------------------------------------------------
    # 3. Train each requested wrapper model.
    # ------------------------------------------------------------------
    data_dir = exp_cfg.get("train", {}).get("data_dir", "./data")

    results = []
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"  TRAINING: {model_name}")
        print(f"{'='*60}")

        if model_name not in registry:
            print(f"  [SKIP] '{model_name}' not in registry.yml — "
                  f"available: {sorted(registry.keys())}")
            continue

        try:
            cls = get_model_class(model_name)
        except KeyError:
            print(f"  [SKIP] '{model_name}' has no BaseModel wrapper — "
                  f"available wrappers: {get_available_model_names()}")
            continue

        try:
            m = cls(
                data_dir   = data_dir,
                output_dir = "./outputs",
                run_name   = run_name,
                do_search  = do_search,
                num_workers=args.num_workers,
            )
            m.run()
            results.append(model_name)
            print(f"\n  [DONE] {model_name}")

        except Exception as e:
            print(f"\n  [ERROR] {model_name} failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  ALL TRAINING COMPLETE")
    print(f"  Trained  : {len(results)}/{len(models)} models")
    print(f"  Run folder: {run_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
