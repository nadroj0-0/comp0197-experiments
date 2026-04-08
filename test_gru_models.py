# =============================================================================
# test_gru_models.py — Wrapper-based evaluation runner
# COMP0197 Applied Deep Learning
#
# Evaluates every requested model in a run through the BaseModel wrappers.
#
# Usage:
#   python test_gru_models.py
#   python test_gru_models.py --run_name sales_only_top200
#   python test_gru_models.py --experiment configs/experiment.yml
# =============================================================================

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from utils.eval.runner import run_batch_evaluation
from models import get_model_class, get_available_model_names

PROJECT_DIR     = Path(__file__).resolve().parent
EXPERIMENT_PATH = PROJECT_DIR / "configs" / "experiment.yml"
REGISTRY_PATH   = PROJECT_DIR / "configs" / "registry.yml"

# Default run name for quick local use. CLI can override it.
RUN_NAME = "baseline_ablation_sales_only"


print("Using wrapper-based evaluation runner")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="M5 Test V3")
    p.add_argument("--run_name",   type=str, default=None,
                   help="Override the default run name")
    p.add_argument("--experiment", type=str, default=str(EXPERIMENT_PATH),
                   help="Path to experiment.yml (default: configs/experiment.yml)")
    return p.parse_args()


# =============================================================================
# SINGLE MODEL EVALUATION
# =============================================================================

def evaluate_model(model_name: str, run_dir: Path,
                   exp_eval: dict, registry: dict) -> dict | None:
    """
    Evaluate a single trained wrapper model through the BaseModel inference API.
    """
    if model_name not in registry:
        print(f"  [SKIP] {model_name} — not in registry.yml")
        return None

    try:
        wrapper_cls = get_model_class(model_name)
    except KeyError:
        print(f"  [SKIP] {model_name} — no BaseModel wrapper — "
              f"available wrappers: {get_available_model_names()}")
        return None

    wrapper = wrapper_cls(
        data_dir   = str(exp_eval.get("data_dir", "./data")),
        output_dir = "./outputs",
        run_name   = run_dir.name,
        do_search  = False,
    )
    return wrapper.run_inference_pipeline()


# =============================================================================
# MAIN
# =============================================================================

def main():
    args     = parse_args()
    run_name = args.run_name or RUN_NAME
    run_batch_evaluation(run_name, PROJECT_DIR, REGISTRY_PATH, evaluate_model)


if __name__ == "__main__":
    main()
