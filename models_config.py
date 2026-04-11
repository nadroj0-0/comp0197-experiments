from models.h_lstm import HierarchicalLSTMModel
from models.baseline_lstm import LSTMBaseline
from models.linear_model import LinearModel 
from models.lightgbm_nn import LightGBM_NN
from models.tft_model import TFTModel  # <--- Add this

MODEL_REGISTRY = {
    "lgbm_baseline": LightGBM_NN,  # Reusing LSTM baseline for simplicity
    "h_lstm": HierarchicalLSTMModel,
    "lstm_baseline": LSTMBaseline,     # <--- Add this
    "linear": LinearModel, 
 #   "tft": TFTModel,                   # <--- Add this
}

try:
    from pathlib import Path
    import yaml
    from models.utils.gru_registry import get_model_class

    _root = Path(__file__).resolve().parent
    _exp_cfg = yaml.safe_load((_root / "configs" / "experiment.yml").read_text()) or {}
    _gru_registry = yaml.safe_load((_root / "configs" / "registry.yml").read_text()) or {}

    for _model_name in _exp_cfg.get("models", []):
        if _model_name in _gru_registry:
            MODEL_REGISTRY[_model_name] = get_model_class(_model_name)
except Exception:
    pass
