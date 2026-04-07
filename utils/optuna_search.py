import optuna
import copy

from utils.training_session import TrainingSession
from utils.common import save_json


def _sample_from_space(trial, search_space):
    """
    Converts your existing search_space format into Optuna suggestions.
    Supports nested keys like 'optimiser_params.lr'
    """
    cfg_updates = {}

    for key, (low, high, mode) in search_space.items():
        if mode == "uniform":
            value = trial.suggest_float(key, low, high)
        elif mode == "log":
            value = trial.suggest_float(key, low, high, log=True)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # --- handle nested keys ---
        parts = key.split(".")
        d = cfg_updates
        for p in parts[:-1]:
            if p not in d:
                d[p] = {}
            d = d[p]
        d[parts[-1]] = value

    return cfg_updates


def _merge_dict(base, updates):
    """
    Recursively merge dicts (for nested cfg updates)
    """
    for k, v in updates.items():
        if isinstance(v, dict) and k in base:
            _merge_dict(base[k], v)
        else:
            base[k] = v
    return base


def optuna_search(
    search_space,
    train_loader,
    val_loader,
    builder,
    model_dir,
    base_config,
    training_step,
    n_trials=30,
    timeout=None,
    study_name="optuna_search",
    direction="minimize",
):
    """
    Optuna-based hyperparameter search.

    Fully compatible with your builder + TrainingSession system.
    """

    def objective(trial):
        # --- sample config ---
        cfg = copy.deepcopy(base_config)
        updates = _sample_from_space(trial, search_space)
        cfg = _merge_dict(cfg, updates)

        # --- build model ---
        model, criterion, optimiser, training_kwargs = builder(cfg)

        session = TrainingSession(
            model=model,
            optimiser=optimiser,
            criterion=criterion,
            config=cfg,
            training_step=training_step,
            training_kwargs=training_kwargs,
        )

        # --- train ---
        session.train(cfg["epochs"], train_loader, val_loader)

        # --- get best val loss ---
        best_loss = min(
            m["validation_loss"] for m in session.history["epoch_metrics"]
        )

        # --- log for Optuna ---
        trial.set_user_attr("config", cfg)

        return best_loss

    study = optuna.create_study(
        direction=direction,
        study_name=study_name,
    )

    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    # --- extract best ---
    best_trial = study.best_trial
    best_cfg = best_trial.user_attrs["config"]

    # --- save results ---
    summary = {
        "best_value": best_trial.value,
        "best_params": best_trial.params,
        "best_config": best_cfg,
    }

    save_json(summary, model_dir / f"{study_name}.json")

    print("\nBest trial:")
    print(best_trial.value)
    print(best_trial.params)

    return best_cfg