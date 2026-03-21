# =============================================================================
# train.py  —  Deterministic GRU / LSTM baselines for M5 sales forecasting
# COMP0197 Applied Deep Learning
# =============================================================================


import sys
import math
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[0]))
from utils.experiment import *
from utils.network import (build_gru, build_lstm, build_prob_gru, build_prob_lstm, build_transformer,
                           build_prob_transformer)
from utils.training_strategies import gru_step, prob_gru_step
from utils.data import build_dataloaders

import argparse #for debuggin remove later before submission
from utils.optuna_search import optuna_search #DELETE before submission

PROJECT_DIR = Path(__file__).resolve().parent

# TRAIN_CONFIG = {
#     "seed":                     42,
#     "epochs":                   50,
#     "lr":                       1e-3,
#     "hidden":                   128,
#     "layers":                   2,
#     "dropout":                  0.2,
#     "batch_size":               256,
#     "seq_len":                  28,
#     "horizon":                  28,
#     "store_id":                 "CA_3",
#     "data_dir":                 "./data",
#     "max_series":               None,
#     "num_workers":              0,
#     "early_stopping_patience":  10,
#     "early_stopping_min_delta": 0.001,
#     "sigma_reg": 0.01,
# }
configs = [
    {"model_type": "gru", "probabilistic": False, "use_temporal_head": False},
    {"model_type": "gru", "probabilistic": False, "use_temporal_head": True},
    {"model_type": "lstm", "probabilistic": False, "use_temporal_head": True},
    {"model_type": "transformer", "probabilistic": False},
    {"model_type": "gru", "probabilistic": True, "use_temporal_head": True},
]

TRAIN_CONFIG = {
    "seed": 42,
    "epochs": 50,

    # model
    "model_type": "gru",              # "gru" | "lstm" | "transformer"
    "probabilistic": False,
    "hidden": 128,
    "layers": 2,
    "dropout": 0.2,
    "horizon": 28,
    "use_temporal_head": True,

    # transformer-specific
    "d_model": 128,
    "n_heads": 4,
    "ff_dim": 256,

    # data
    "batch_size": 256,
    "seq_len": 28,
    "store_id": "CA_3",
    "data_dir": "./data",
    "max_series": None,
    "num_workers": 0,

    # training / optimisation
    "optimiser": "adamw",
    "optimiser_params": {
        "lr": 1e-3,
        "weight_decay": 1e-4,
    },
    "scheduler": "plateau",
    "scheduler_params": {
        "patience": 5,
        "factor": 0.5,
    },
    "clip_grad_norm": 1.0,

    # early stopping
    "early_stopping_patience": 10,
    "early_stopping_min_delta": 0.001,

    # probabilistic
    "sigma_reg": 0.01,
}

SEARCH = True
USE_OPTUNA = True #DELETE before submission

GRU_SEARCH_SPACE = {
    "optimiser_params.lr": (1e-4, 1e-2, "log"),
    "optimiser_params.weight_decay": (1e-6, 1e-3, "log"),
    "hidden": (64, 256, "uniform"),
    "layers": (1, 4, "uniform"),
    "dropout": (0.1, 0.4, "uniform"),
    "clip_grad_norm": (0.5, 2.0, "uniform"),
}

LSTM_SEARCH_SPACE = {
    "optimiser_params.lr": (1e-4, 1e-2, "log"),
    "optimiser_params.weight_decay": (1e-6, 1e-3, "log"),
    "hidden": (64, 256, "uniform"),
    "layers": (1, 4, "uniform"),
    "dropout": (0.1, 0.4, "uniform"),
    "clip_grad_norm": (0.5, 2.0, "uniform"),
}

PROB_SEARCH_SPACE = {
    "optimiser_params.lr": (1e-4, 1e-2, "log"),
    "optimiser_params.weight_decay": (1e-6, 1e-3, "log"),
    "hidden": (64, 256, "uniform"),
    "layers": (1, 3, "uniform"),
    "dropout": (0.1, 0.4, "uniform"),
    "sigma_reg": (0.0, 0.05, "uniform"),
    "clip_grad_norm": (0.25, 1.5, "uniform"),
}

TRANSFORMER_SEARCH_SPACE = {
    "optimiser_params.lr": (1e-4, 5e-3, "log"),
    "optimiser_params.weight_decay": (1e-6, 1e-3, "log"),
    "d_model": (64, 256, "uniform"),
    "layers": (1, 4, "uniform"),
    "dropout": (0.1, 0.4, "uniform"),
    "ff_dim": (128, 512, "uniform"),
    "clip_grad_norm": (0.5, 2.0, "uniform"),
}

HYPER_PARAM_INIT_MODELS = 20
HYPER_PARAM_SEARCH_SCHEDULE = [
    {"epochs": 10,  "keep": math.ceil(HYPER_PARAM_INIT_MODELS / 2)},
    {"epochs": 10, "keep": math.ceil(HYPER_PARAM_INIT_MODELS / 4)},
    {"epochs": 20, "keep": 1},
]

# DEBUG — parse_args allows CLI overrides for quick testing
# e.g. python train.py --max_series 10 --epochs 2 --num_workers 0
# REMOVE parse_args() and uncomment comment out bit below before submission
def parse_args():
    p = argparse.ArgumentParser(description="Train GRU/LSTM on M5")
    for k, v in TRAIN_CONFIG.items():
        if isinstance(v, dict):
            continue  # skip nested dicts — not CLI overridable
        elif isinstance(v, bool):
            p.add_argument(f"--{k}", action="store_true", default=v)
        elif v is None:
            p.add_argument(f"--{k}", type=str, default=v)
        else:
            p.add_argument(f"--{k}", type=type(v), default=v)
    return vars(p.parse_args())


def get_experiment_kwargs(cfg):
    if cfg["model_type"] == "gru":
        if cfg["probabilistic"]:
            return dict(
                builder=build_prob_gru,
                training_step=prob_gru_step,
                search_space=PROB_SEARCH_SPACE if SEARCH else None,
            )
        return dict(
            builder=build_gru,
            training_step=gru_step,
            search_space=GRU_SEARCH_SPACE if SEARCH else None,
        )

    elif cfg["model_type"] == "lstm":
        if cfg["probabilistic"]:
            return dict(
                builder=build_prob_lstm,
                training_step=prob_gru_step,
                search_space=PROB_SEARCH_SPACE if SEARCH else None,
            )
        return dict(
            builder=build_lstm,
            training_step=gru_step,
            search_space=LSTM_SEARCH_SPACE if SEARCH else None,
        )

    elif cfg["model_type"] == "transformer":
        return dict(
            builder=build_prob_transformer if cfg["probabilistic"] else build_transformer,
            training_step=prob_gru_step if cfg["probabilistic"] else gru_step,
            search_space=TRANSFORMER_SEARCH_SPACE if SEARCH else None,
        )

    else:
        raise ValueError("Unknown model_type")


def main():
    # cfg = parse_args()  # REMOVE before submission and uncomment the commented out bit below
    cli_args = parse_args()  # use the existing function
    cfg = TRAIN_CONFIG.copy()
    import sys
    explicitly_passed = {arg.lstrip('-').replace('-', '_') for arg in sys.argv[1:] if arg.startswith('--')}
    for k in explicitly_passed:
        if k in cli_args:
            cfg[k] = cli_args[k]

    # try:
    #     cfg = TRAIN_CONFIG.copy()
    # except NameError:
    #     raise RuntimeError(
    #         "TRAIN_CONFIG must be defined before calling main(). "
    #         "It defines the experiment hyperparameters."
    #     )

    # shared data kwargs passed to build_dataloaders
    data_kwargs = dict(
        data_dir    = cfg["data_dir"],
        seq_len     = cfg["seq_len"],
        horizon     = cfg["horizon"],
        batch_size  = cfg["batch_size"],
        store_id    = cfg["store_id"],
        max_series  = cfg["max_series"],
        num_workers = cfg["num_workers"],
        seed        = cfg["seed"],
    )

    # experiments = {
    #     "gru_deterministic": dict(
    #         builder=build_gru,
    #         training_step=gru_step,
    #         search_space=GRU_SEARCH_SPACE if SEARCH else None,
    #     ),
    #     "lstm_deterministic": dict(
    #         builder=build_lstm,
    #         training_step=gru_step,
    #         search_space=LSTM_SEARCH_SPACE if SEARCH else None,
    #     ),
    #     "gru_probabilistic": dict(
    #         builder=build_prob_gru,
    #         training_step=prob_gru_step,
    #         search_space=PROB_SEARCH_SPACE if SEARCH else None,
    #     ),
    #     "lstm_probabilistic": dict(
    #         builder=build_prob_lstm,
    #         training_step=prob_gru_step,
    #         search_space=PROB_SEARCH_SPACE if SEARCH else None,
    #     ),
    #     "transformer_deterministic": dict(
    #         builder=build_transformer,
    #         training_step=gru_step,
    #         search_space=TRANSFORMER_SEARCH_SPACE if SEARCH else None,
    #     ),
    #     "transformer_probabilistic": dict(
    #         builder=build_prob_transformer,
    #         training_step=prob_gru_step,
    #         search_space=TRANSFORMER_SEARCH_SPACE if SEARCH else None,
    #     ),
    # }
    # for name, kwargs in experiments.items():
    #     is_prob = "prob" in name
    #     exp = Experiment(name, cfg, model_dir=get_model_dir(name, PROJECT_DIR))
    #     exp.run(
    #         data_fn=build_dataloaders,
    #         schedule=HYPER_PARAM_SEARCH_SCHEDULE,
    #         initial_models=HYPER_PARAM_INIT_MODELS,
    #         zscore_target=not is_prob,
    #         **data_kwargs,
    #         **kwargs,
    #     )
    #     save_json(exp.stats, exp.model_dir / "normalisation_stats.json")

    # load data ONCE before the loop
    # load data ONCE
    print("\n[main] Loading data once for all experiments...")
    train_loader_det, val_loader_det, test_loader_det, stats_det = build_dataloaders(**data_kwargs, zscore_target=True)
    train_loader_prob, val_loader_prob, test_loader_prob, stats_prob = build_dataloaders(**data_kwargs,
                                                                                         zscore_target=False)
    print("[main] Data loaded. Starting experiments.\n")

    for override in configs:
        cfg = TRAIN_CONFIG.copy()
        cfg.update(override)

        kwargs = get_experiment_kwargs(cfg)
        is_prob = cfg["probabilistic"]
        head = "temp" if cfg.get("use_temporal_head") else "direct"
        exp_name = f"{cfg['model_type']}_{head}_{'prob' if is_prob else 'det'}"
        exp = Experiment(exp_name, cfg, model_dir=get_model_dir(exp_name, PROJECT_DIR))

        # inject correct loaders
        if is_prob:
            exp.train_loader = train_loader_prob
            exp.val_loader = val_loader_prob
            exp.test_dataset = test_loader_prob
            exp.stats = stats_prob
        else:
            exp.train_loader = train_loader_det
            exp.val_loader = val_loader_det
            exp.test_dataset = test_loader_det
            exp.stats = stats_det
        exp.preloaded = True

        if not USE_OPTUNA:
            exp.run(
                data_fn=build_dataloaders,
                schedule=HYPER_PARAM_SEARCH_SCHEDULE,
                initial_models=HYPER_PARAM_INIT_MODELS,
                zscore_target=not is_prob,
                **data_kwargs,
                **kwargs,
            )
        else:
            best_cfg = optuna_search(
                search_space=kwargs["search_space"],
                train_loader=exp.train_loader,  # correct — already set above
                val_loader=exp.val_loader,
                builder=kwargs['builder'],
                model_dir=exp.model_dir,
                base_config=cfg,
                training_step=kwargs["training_step"],
                n_trials=30,
            )
            print(f"\nBest config from Optuna: {best_cfg}")
            # update config with best params
            exp.cfg = best_cfg.copy()
            # train final model with best config
            exp.train(kwargs['builder'], kwargs['training_step'])

        save_json(exp.stats, exp.model_dir / "normalisation_stats.json")


if __name__ == "__main__":
    main()


    #free(): invalid pointer