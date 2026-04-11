# =============================================================================
# gru_models.py — GRU/Hierarchical model subclasses of BaseModel
# =============================================================================

import json
import os
import shutil
from pathlib import Path

import torch

from .base_model import BaseModel
from .utils.gru_inference import evaluate_predictions, predict_from_trained_model
from .utils.gru_pipeline import (
    ensure_run_snapshot,
    preprocess_from_base_model,
    run_full_pipeline,
    run_search_from_snapshot,
    train_from_preprocess,
)
from .utils.gru_registry import get_available_model_names, get_model_class

from utils.configs.config_loader import load_experiment, load_registry, resolve_registry_entry

PROJECT_DIR = Path(__file__).resolve().parents[1]
EXPERIMENT_PATH = PROJECT_DIR / "configs" / "experiment.yml"
REGISTRY_PATH = PROJECT_DIR / "configs" / "registry.yml"


class _BaseGRUWrapper(BaseModel):
    model_name = None
    include_weights = False

    def __init__(self, data_dir=None, output_dir="./outputs",
                 run_name=None, do_search=None, num_workers=None):
        exp_cfg = load_experiment(EXPERIMENT_PATH)
        train_cfg = exp_cfg.get("train", {})
        resolved_data_dir = data_dir or train_cfg.get("data_dir", "./data")
        resolved_run_name = run_name or exp_cfg.get("run_name") or f"{self.model_name}_run"
        resolved_do_search = (
            bool(exp_cfg.get("search", {}).get("enabled", False))
            if do_search is None else bool(do_search)
        )

        super().__init__(data_dir=resolved_data_dir, output_dir=output_dir)
        self._run_name = resolved_run_name
        self._do_search = resolved_do_search
        self._num_workers = num_workers
        self._exp = None
        self._preprocessed = False
        self._public_output_dir = self._resolve_public_output_dir(output_dir)
        self._public_output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = str(self._public_output_dir)

        self._ensure_run_model_dir()
        self._refresh_public_artifacts()

    def _resolve_public_output_dir(self, output_dir) -> Path:
        base = Path(output_dir)
        if not base.is_absolute():
            base = PROJECT_DIR / base
        return base / self.model_name

    def _run_dir(self) -> Path:
        return PROJECT_DIR / "runs" / self._run_name

    def _model_dir(self) -> Path:
        return self._run_dir() / "models" / self.model_name

    def _actual_checkpoint_path(self) -> Path:
        return self._model_dir() / f"{self.model_name}_model.pt"

    def _public_checkpoint_path(self) -> Path:
        return self._public_output_dir / f"{self.model_name}.pth"

    def _public_predictions_path(self) -> Path:
        return self._public_output_dir / f"{self.model_name}_predictions.csv"

    def _public_source_metadata_path(self) -> Path:
        return self._public_output_dir / "source_run.json"

    def _ensure_run_model_dir(self):
        self._model_dir().mkdir(parents=True, exist_ok=True)

    def _ensure_run_snapshot(self):
        ensure_run_snapshot(self.model_name, self._run_name)
        self._ensure_run_model_dir()

    def _clear_public_artifacts(self):
        mirrored = [
            self._public_checkpoint_path(),
            self._public_predictions_path(),
            self._public_output_dir / f"{self.model_name}_test_metrics.json",
            self._public_output_dir / f"{self.model_name}_basemodel_metrics.json",
            self._public_output_dir / f"{self.model_name}_training_curves.png",
            self._public_source_metadata_path(),
        ]
        for path in mirrored:
            if path.exists():
                path.unlink()
        plots_dir = self._public_output_dir / "plots"
        if plots_dir.exists():
            shutil.rmtree(plots_dir)
        for plot_path in self._public_output_dir.glob("forecast_plot_*.png"):
            plot_path.unlink()

    def _copy_if_exists(self, src: Path, dst: Path):
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    def _sync_public_artifacts(self):
        model_dir = self._model_dir()
        actual_path = self._actual_checkpoint_path()
        if not actual_path.exists():
            self._clear_public_artifacts()
            return

        self._public_output_dir.mkdir(parents=True, exist_ok=True)
        self._copy_if_exists(actual_path, self._public_checkpoint_path())
        self._copy_if_exists(
            model_dir / f"{self.model_name}_predictions.csv",
            self._public_predictions_path(),
        )
        self._copy_if_exists(
            model_dir / f"{self.model_name}_test_metrics.json",
            self._public_output_dir / f"{self.model_name}_test_metrics.json",
        )
        self._copy_if_exists(
            model_dir / f"{self.model_name}_basemodel_metrics.json",
            self._public_output_dir / f"{self.model_name}_basemodel_metrics.json",
        )
        self._copy_if_exists(
            model_dir / f"{self.model_name}_training_curves.png",
            self._public_output_dir / f"{self.model_name}_training_curves.png",
        )

        run_plots = model_dir / "plots"
        public_plots = self._public_output_dir / "plots"
        if run_plots.exists():
            shutil.rmtree(run_plots)
        if public_plots.exists():
            shutil.rmtree(public_plots)

        with open(self._public_source_metadata_path(), "w") as f:
            json.dump(
                {
                    "run_name": self._run_name,
                    "model_name": self.model_name,
                    "source_dir": str(model_dir),
                },
                f,
                indent=2,
            )

    def _refresh_public_artifacts(self):
        self._sync_public_artifacts()

    def has_trained_artifacts(self):
        return self._actual_checkpoint_path().exists()

    def get_weights_path(self):
        return str(self._actual_checkpoint_path())

    def get_predictions_path(self):
        return str(self._public_predictions_path())

    def get_artifact_dir(self):
        return str(self._public_output_dir)

    def prepare_for_predict(self):
        return None

    def preprocess(self):
        self._ensure_run_snapshot()
        preprocess_from_base_model(
            self, self.model_name, self._run_name, include_weights=self.include_weights
        )
        self._preprocessed = True

    def train(self, epochs=None):
        self._ensure_run_snapshot()
        if self._do_search:
            run_search_from_snapshot(self.model_name, self._run_name)
            self.preprocess()
        elif not self._preprocessed:
            self.preprocess()

        registry = load_registry(REGISTRY_PATH)
        resolved = resolve_registry_entry(registry[self.model_name])
        train_from_preprocess(
            self, self.model_name, self._run_name,
            resolved["builder"], resolved["training_step"],
        )
        self._sync_public_artifacts()

    def predict(self):
        self._ensure_run_snapshot()
        preds_df = predict_from_trained_model(self, self.model_name, self._run_name)
        self._sync_public_artifacts()
        return preds_df

    def evaluate(self, preds_df):
        self._ensure_run_snapshot()
        metrics = evaluate_predictions(self, self.model_name, self._run_name, preds_df)
        self._sync_public_artifacts()
        return metrics

    def run(self):
        self._ensure_run_snapshot()
        self._exp = run_full_pipeline(
            self,
            model_name=self.model_name,
            run_name=self._run_name,
            do_search=self._do_search,
            include_weights=self.include_weights,
        )
        self._preprocessed = True
        self._sync_public_artifacts()


class BaselineGRUDet(_BaseGRUWrapper):
    """Vanilla GRU — MSE loss (deterministic)."""
    model_name = "baseline_gru_det"


class BaselineGRUProb(_BaseGRUWrapper):
    """Vanilla GRU — Gaussian NLL loss (probabilistic)."""
    model_name = "baseline_gru_prob"


class BaselineGRUNB(_BaseGRUWrapper):
    """Vanilla GRU — Negative Binomial NLL loss."""
    model_name = "baseline_gru_nb"


class BaselineQuantileGRU(_BaseGRUWrapper):
    """Vanilla GRU — unweighted pinball loss, 9 quantiles."""
    model_name = "baseline_quantile_gru"


class BaselineWQuantileGRU(_BaseGRUWrapper):
    """Vanilla GRU — revenue-weighted pinball loss, 9 quantiles."""
    model_name = "baseline_wquantile_gru"
    include_weights = True


class HierarchicalGRUDet(_BaseGRUWrapper):
    """GRU with learned hierarchy embeddings — MSE loss (deterministic)."""
    model_name = "hierarchical_gru_det"


class HierarchicalGRUProb(_BaseGRUWrapper):
    """GRU with learned hierarchy embeddings — Gaussian NLL loss."""
    model_name = "hierarchical_gru_prob"


class HierarchicalGRUNB(_BaseGRUWrapper):
    """GRU with learned hierarchy embeddings — Negative Binomial NLL loss."""
    model_name = "hierarchical_gru_nb"


class HierarchicalQuantileGRU(_BaseGRUWrapper):
    """GRU with learned hierarchy embeddings — unweighted pinball loss."""
    model_name = "hierarchical_quantile_gru"


class HierarchicalWQuantileGRU(_BaseGRUWrapper):
    """GRU with learned hierarchy embeddings — revenue-weighted pinball loss."""
    model_name = "hierarchical_wquantile_gru"
    include_weights = True
