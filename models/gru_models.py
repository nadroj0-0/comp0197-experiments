# =============================================================================
# gru_models.py — GRU/Hierarchical model subclasses of BaseModel
# =============================================================================

from .base_model import BaseModel
from .utils.gru_inference import evaluate_predictions, predict_from_trained_model
from .utils.gru_pipeline import preprocess_from_base_model, run_full_pipeline, train_from_preprocess
from .utils.gru_registry import get_available_model_names, get_model_class

from utils.configs.config_loader import load_registry, resolve_registry_entry

REGISTRY_PATH = __import__("pathlib").Path(__file__).resolve().parents[1] / "configs" / "registry.yml"


class _BaseGRUWrapper(BaseModel):
    model_name = None
    include_weights = False

    def __init__(self, data_dir="./data", output_dir="./outputs",
                 run_name=None, do_search=False, num_workers=None):
        super().__init__(data_dir=data_dir, output_dir=output_dir)
        self._run_name = run_name or self.model_name + "_run"
        self._do_search = do_search
        self._num_workers = num_workers
        self._exp = None

    def preprocess(self):
        preprocess_from_base_model(
            self, self.model_name, self._run_name, include_weights=self.include_weights
        )

    def train(self):
        registry = load_registry(REGISTRY_PATH)
        resolved = resolve_registry_entry(registry[self.model_name])
        train_from_preprocess(
            self, self.model_name, self._run_name,
            resolved["builder"], resolved["training_step"],
        )

    def predict(self):
        return predict_from_trained_model(self, self.model_name, self._run_name)

    def evaluate(self, preds_df):
        return evaluate_predictions(self, self.model_name, self._run_name, preds_df)

    def run(self):
        self._exp = run_full_pipeline(
            self,
            model_name=self.model_name,
            run_name=self._run_name,
            do_search=self._do_search,
            include_weights=self.include_weights,
        )


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
