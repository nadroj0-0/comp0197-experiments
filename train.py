# =============================================================================
# train.py  —  M5 Sales Forecasting — V3
# COMP0197 Applied Deep Learning
# GenAI Note: Scaffolded with Claude (Anthropic). Verified by authors.
# =============================================================================

import sys
import math
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[0]))
from utils.experiment import *
from utils.network import (
    build_gru, build_lstm, build_prob_gru, build_prob_lstm,
    build_transformer, build_prob_transformer, build_prob_gru_nb,
    build_baseline_gru, build_baseline_prob_gru, build_baseline_prob_gru_nb,
)
from utils.training_strategies import gru_step, prob_gru_step, prob_nb_step
from utils.data import build_dataloaders

import argparse

PROJECT_DIR = Path(__file__).resolve().parent

# =============================================================================
# EXPERIMENT CONFIGS
# =============================================================================
configs = [
    {"model_type": "baseline_gru",      "probabilistic": False},
    {"model_type": "baseline_gru",      "probabilistic": True},
    # {"model_type": "gru",         "probabilistic": False},
    # {"model_type": "gru",         "probabilistic": True},
    # {"model_type": "lstm",        "probabilistic": False},
    # {"model_type": "lstm",        "probabilistic": True},
    # {"model_type": "transformer", "probabilistic": False},
    # {"model_type": "transformer", "probabilistic": True},
    {"model_type": "baseline_gru_nb", "probabilistic": True},
    # {"model_type": "gru_nb",      "probabilistic": True},
]

# =============================================================================
# BASE CONFIG
# =============================================================================
TRAIN_CONFIG = {
    "seed":   42,
    "epochs": 60,

    # model
    "model_type":    "gru",
    "probabilistic": False,
    "hidden":        256,
    "layers":        2,
    "dropout":       0.15,
    "horizon":       28,

    # transformer-specific
    "d_model": 256,
    "n_heads":  8,
    "ff_dim":  512,

    # data — V3 unified workflow
    "batch_size":      1024,
    "seq_len":         28,           # matches teammate
    "top_k_series":    200,          # matches teammate (top-200 by volume)
    "feature_set":     "sales_only", # "sales_only" | "sales_hierarchy" | "sales_hierarchy_dow"
    "autoregressive":  True,         # True = 1-step ahead + recursive rollout at test
    "use_normalise":   False,        # False = raw counts, matching teammate
    "data_dir":        "./data",
    "max_series":      None,
    "num_workers":     8,

    # optimisation — unchanged from V2
    "optimiser": "adamw",
    "optimiser_params": {
        "lr":           1e-3,
        "weight_decay": 1e-4,
    },
    "scheduler": "plateau",
    "scheduler_params": {
        "patience": 5,
        "factor":   0.5,
    },
    "clip_grad_norm": 1.0,

    # early stopping — unchanged from V2
    "early_stopping_patience":  8,
    "early_stopping_min_delta": 5e-4,

    # probabilistic
    "sigma_reg": 1e-3,
}

SEARCH = True

# =============================================================================
# SEARCH SPACES — unchanged from V2
# =============================================================================
BASELINE_DET_SEARCH_SPACE = {
    "optimiser_params.lr":           (1e-4, 1e-2,  "log"),
    "optimiser_params.weight_decay": (1e-7, 1e-3,  "log"),
    "hidden":                        (32,   128,    "uniform"),
    "layers":                        (1,    2,      "uniform"),
    "dropout":                       (0.0,  0.3,    "uniform"),
    "clip_grad_norm":                (0.5,  2.0,    "uniform"),
}

BASELINE_PROB_SEARCH_SPACE = {
    "optimiser_params.lr":           (5e-5, 5e-3,  "log"),   # tighter — NLL more sensitive
    "optimiser_params.weight_decay": (1e-7, 1e-3,  "log"),
    "hidden":                        (32,   128,    "uniform"),
    "layers":                        (1,    2,      "uniform"),
    "dropout":                       (0.0,  0.3,    "uniform"),
    "clip_grad_norm":                (0.25, 1.5,    "uniform"),  # tighter
    "sigma_reg":                     (1e-4, 5e-2,   "log"),   # prevent sigma collapse
}

BASELINE_NB_SEARCH_SPACE = {
    "optimiser_params.lr":           (5e-5, 1e-3,  "log"),   # tightest — NB most unstable
    "optimiser_params.weight_decay": (1e-6, 1e-2,  "log"),   # stronger to regularise alpha
    "hidden":                        (32,   128,    "uniform"),
    "layers":                        (1,    2,      "uniform"),
    "dropout":                       (0.0,  0.3,    "uniform"),
    "clip_grad_norm":                (0.25, 1.0,    "uniform"),  # tightest
}

GRU_SEARCH_SPACE = {
    "optimiser_params.lr":           (5e-5, 5e-3, "log"),
    "optimiser_params.weight_decay": (1e-7, 5e-3, "log"),
    "hidden":                        (128,  512,   "uniform"),
    "layers":                        (1,    3,     "uniform"),
    "dropout":                       (0.05, 0.35,  "uniform"),
    "clip_grad_norm":                (0.5,  2.0,   "uniform"),
}

LSTM_SEARCH_SPACE = {
    "optimiser_params.lr":           (5e-5, 5e-3, "log"),
    "optimiser_params.weight_decay": (1e-7, 5e-3, "log"),
    "hidden":                        (128,  512,   "uniform"),
    "layers":                        (1,    3,     "uniform"),
    "dropout":                       (0.05, 0.35,  "uniform"),
    "clip_grad_norm":                (0.5,  2.0,   "uniform"),
}

PROB_SEARCH_SPACE = {
    "optimiser_params.lr":           (5e-5, 1e-3,  "log"),
    "optimiser_params.weight_decay": (1e-7, 5e-3,  "log"),
    "hidden":                        (128,  512,    "uniform"),
    "layers":                        (1,    3,      "uniform"),
    "dropout":                       (0.05, 0.35,   "uniform"),
    "sigma_reg":                     (1e-4, 5e-2,   "log"),
    "clip_grad_norm":                (0.25, 1.5,    "uniform"),
}

TRANSFORMER_SEARCH_SPACE = {
    "optimiser_params.lr":           (1e-4, 3e-3,  "log"),
    "optimiser_params.weight_decay": (1e-7, 1e-2,  "log"),
    "d_model":                       (128,  512,    "uniform"),
    "layers":                        (2,    6,      "uniform"),
    "dropout":                       (0.05, 0.3,    "uniform"),
    "ff_dim":                        (256,  1024,   "uniform"),
    "clip_grad_norm":                (0.5,  2.0,    "uniform"),
}

HYPER_PARAM_INIT_MODELS = 10
HYPER_PARAM_SEARCH_SCHEDULE = [
    {"epochs": 10, "keep": math.ceil(HYPER_PARAM_INIT_MODELS / 2)},
    {"epochs": 10, "keep": math.ceil(HYPER_PARAM_INIT_MODELS / 4)},
    {"epochs": 20, "keep": 1},
]

# =============================================================================
# HELPERS
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Train M5 V3")
    for k, v in TRAIN_CONFIG.items():
        if isinstance(v, dict):
            continue
        elif isinstance(v, bool):
            p.add_argument(f"--{k}", action="store_true", default=v)
        elif v is None:
            p.add_argument(f"--{k}", type=str, default=v)
        else:
            p.add_argument(f"--{k}", type=type(v), default=v)
    return vars(p.parse_args())


def get_experiment_kwargs(cfg):
    from utils.data import get_feature_cols
    cfg["n_features"] = len(get_feature_cols(cfg["feature_set"]))
    model_type = cfg["model_type"]
    is_prob    = cfg["probabilistic"]

    if model_type == "baseline_gru":
        return dict(
            builder=build_baseline_prob_gru if is_prob else build_baseline_gru,
            training_step=prob_gru_step if is_prob else gru_step,
            search_space=(BASELINE_PROB_SEARCH_SPACE if is_prob else BASELINE_DET_SEARCH_SPACE)
            if SEARCH else None,
        )
    elif model_type == "gru":
        return dict(
            builder       = build_prob_gru    if is_prob else build_gru,
            training_step = prob_gru_step     if is_prob else gru_step,
            search_space  = (PROB_SEARCH_SPACE if is_prob else GRU_SEARCH_SPACE)
                            if SEARCH else None,
        )
    elif model_type == "lstm":
        return dict(
            builder       = build_prob_lstm   if is_prob else build_lstm,
            training_step = prob_gru_step     if is_prob else gru_step,
            search_space  = (PROB_SEARCH_SPACE if is_prob else LSTM_SEARCH_SPACE)
                            if SEARCH else None,
        )
    elif model_type == "transformer":
        return dict(
            builder       = build_prob_transformer if is_prob else build_transformer,
            training_step = prob_gru_step          if is_prob else gru_step,
            search_space  = TRANSFORMER_SEARCH_SPACE if SEARCH else None,
        )
    elif model_type == "baseline_gru_nb":
        return dict(
            builder=build_baseline_prob_gru_nb,
            training_step=prob_nb_step,
            search_space=BASELINE_NB_SEARCH_SPACE if SEARCH else None,
        )
    elif model_type == "gru_nb":
        return dict(
            builder       = build_prob_gru_nb,
            training_step = prob_nb_step,
            search_space  = PROB_SEARCH_SPACE if SEARCH else None,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    cli_args = parse_args()
    cfg      = TRAIN_CONFIG.copy()
    explicitly_passed = {
        arg.lstrip("-").replace("-", "_")
        for arg in sys.argv[1:]
        if arg.startswith("--")
    }
    for k in explicitly_passed:
        if k in cli_args:
            cfg[k] = cli_args[k]

    # ------------------------------------------------------------------
    # Base data kwargs — shared across all experiments
    # Note: zscore_target is NOT included here — it is injected per-model
    # below because det vs prob models require different normalisation
    # ------------------------------------------------------------------
    data_kwargs = dict(
        data_dir       = cfg["data_dir"],
        seq_len        = cfg["seq_len"],
        horizon        = cfg["horizon"],
        batch_size     = cfg["batch_size"],
        top_k_series   = cfg["top_k_series"],
        feature_set    = cfg["feature_set"],
        autoregressive = cfg["autoregressive"],
        use_normalise  = cfg["use_normalise"],
        max_series     = cfg["max_series"],
        num_workers    = cfg["num_workers"],
        seed           = cfg["seed"],
    )

    # ------------------------------------------------------------------
    # Load data — behaviour depends on use_normalise flag:
    #
    # use_normalise=False (default, matching teammate):
    #   Single loader, raw counts, all models share it.
    #
    # use_normalise=True:
    #   Three loaders — det (zscore=True), Gaussian prob (zscore=False),
    #   NB (zscore=False) — same as V2 pattern.
    # ------------------------------------------------------------------
    print("\n[main] Loading data once for all experiments...")

    if not cfg["use_normalise"]:
        train_loader, val_loader, test_loader, stats = build_dataloaders(**data_kwargs)
        train_loader_det   = train_loader_gauss = train_loader_nb = train_loader
        val_loader_det     = val_loader_gauss   = val_loader_nb   = val_loader
        test_loader_det    = test_loader_gauss  = test_loader_nb  = test_loader
        stats_det          = stats_gauss        = stats_nb        = stats
    else:
        train_loader_det,   val_loader_det,   test_loader_det,   stats_det   = build_dataloaders(**data_kwargs, zscore_target=True)
        train_loader_gauss, val_loader_gauss, test_loader_gauss, stats_gauss = build_dataloaders(**data_kwargs, zscore_target=False)
        train_loader_nb,    val_loader_nb,    test_loader_nb,    stats_nb    = build_dataloaders(**data_kwargs, zscore_target=False)

    print("[main] Data loaded. Starting experiments.\n")

    for override in configs:
        cfg = TRAIN_CONFIG.copy()
        cfg.update(override)

        kwargs   = get_experiment_kwargs(cfg)
        is_prob  = cfg["probabilistic"]
        exp_name = f"{cfg['model_type']}_{'prob' if is_prob else 'det'}"
        exp      = Experiment(exp_name, cfg, model_dir=get_model_dir(exp_name, PROJECT_DIR))

        # Inject correct loaders — same routing logic as V2
        if cfg["model_type"] in ("gru_nb", "baseline_gru_nb"):
            exp.train_loader = train_loader_nb
            exp.val_loader   = val_loader_nb
            exp.test_dataset = test_loader_nb
            exp.stats        = stats_nb
        elif is_prob:
            exp.train_loader = train_loader_gauss
            exp.val_loader   = val_loader_gauss
            exp.test_dataset = test_loader_gauss
            exp.stats        = stats_gauss
        else:
            exp.train_loader = train_loader_det
            exp.val_loader   = val_loader_det
            exp.test_dataset = test_loader_det
            exp.stats        = stats_det
        exp.preloaded = True

        exp.run(
            data_fn        = build_dataloaders,
            schedule       = HYPER_PARAM_SEARCH_SCHEDULE,
            initial_models = HYPER_PARAM_INIT_MODELS,
            zscore_target  = not is_prob,
            **data_kwargs,
            **kwargs,
        )

        if exp.stats is not None:
            save_json(exp.stats, exp.model_dir / "normalisation_stats.json")


if __name__ == "__main__":
    main()