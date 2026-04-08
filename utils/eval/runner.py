import csv
import json
from pathlib import Path

import pandas as pd

from utils.configs.config_loader import load_experiment, load_registry
from utils.eval.helpers import plot_model_comparison


def _save_comparison_plots(run_dir: Path, exp_eval: dict, all_results: list[dict]) -> None:
    if not all_results:
        return

    from models import get_model_class

    first_model = all_results[0]["model"]
    wrapper_cls = get_model_class(first_model)
    wrapper = wrapper_cls(
        data_dir=str(exp_eval.get("data_dir", "./data")),
        output_dir="./outputs",
        run_name=run_dir.name,
        do_search=False,
    )
    wrapper.load_and_split_data()

    actual_data = wrapper.test_raw[wrapper.test_raw["d_num"] >= wrapper.TARGET_START]
    top_items = wrapper.item_weights.sort_values(ascending=False).head(3).index.tolist()

    all_predictions = {}
    for result in all_results:
        model_name = result["model"]
        preds_path = run_dir / "models" / model_name / f"{model_name}_predictions.csv"
        if preds_path.exists():
            all_predictions[model_name] = pd.read_csv(preds_path)

    if not all_predictions:
        return

    plots_dir = run_dir / "comparison_plots"
    plots_dir.mkdir(exist_ok=True)

    for item_id in top_items:
        item_actuals = actual_data[actual_data["id"] == item_id].sort_values("d_num")["sales"]
        if item_actuals.empty:
            continue
        plot_model_comparison(
            item_id=item_id,
            actual_series=item_actuals,
            all_model_preds=all_predictions,
            save_path=plots_dir / f"comparison_{item_id}.png",
        )


def run_batch_evaluation(run_name: str, project_dir: Path, registry_path: Path, evaluate_model_fn):
    run_dir = project_dir / "runs" / run_name

    if not run_dir.exists():
        raise FileNotFoundError(
            f"Run directory not found: {run_dir}\n"
            f"Have you run train.py with run_name='{run_name}'?"
        )

    exp_cfg = load_experiment(run_dir / "configs" / "experiment.yml")
    registry = load_registry(registry_path)

    exp_eval = exp_cfg.get("eval", {})
    models = exp_cfg.get("models", [])

    if not models:
        raise ValueError(f"No models in {run_dir}/configs/experiment.yml")

    print(f"\n{'='*60}")
    print(f"  BATCH EVALUATION — {run_name}")
    print(f"  Models   : {models}")
    print(
        f"  Eval data: sampling={exp_eval.get('sampling','all')} | "
        f"top_k={exp_eval.get('top_k_series',30490)} | "
        f"feature_set={exp_eval.get('feature_set','sales_only')}"
    )
    print(f"{'='*60}")

    all_results = []

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"  Evaluating: {model_name}")
        print(f"{'='*60}")

        result = evaluate_model_fn(
            model_name=model_name,
            run_dir=run_dir,
            exp_eval=exp_eval,
            registry=registry,
        )
        if result is not None:
            all_results.append(result)

    if all_results:
        print(f"\n{'='*70}")
        print("  RESULTS SUMMARY")
        print(f"{'='*70}")
        print(f"{'Model':<25} {'RMSE':>8} {'W-RMSE':>8} {'MAE':>8} {'W-MAE':>8} {'R²':>8} {'CRPS':>8}")
        print("-" * 80)
        for result in all_results:
            print(
                f"{result['model']:<25} "
                f"{result['rmse']:>8.4f} "
                f"{result.get('w_rmse', float('nan')):>8.4f} "
                f"{result['mae']:>8.4f} "
                f"{result.get('w_mae', float('nan')):>8.4f} "
                f"{result['r2']:>8.4f} "
                f"{result.get('crps', float('nan')):>8.4f}"
            )
        print("=" * 70)

        out_path = run_dir / "all_test_metrics.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Combined metrics saved: {out_path}")

        csv_path = run_dir / "model_comparison.csv"
        fieldnames = list(all_results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"  Comparison CSV saved: {csv_path}")

        try:
            _save_comparison_plots(run_dir, exp_eval, all_results)
            print(f"  Comparison plots saved: {run_dir / 'comparison_plots'}")
        except Exception as exc:
            print(f"  [WARN] Comparison plots skipped: {exc}")

    print(f"\n  Done — evaluated {len(all_results)}/{len(models)} models")
    return all_results
