"""
Hyperparameter search utilities.

Implements staged random search with progressive pruning.
"""

import random
import math
import json
import time

from utils.training_session import *
from utils.common import *

class Leaderboard:
    """
    Simple leaderboard tracking validation losses.
    """
    def __init__(self, sessions):
        self.entries = []
        for cfg, session in sessions:
            loss = min(m["validation_loss"] for m in session.history["epoch_metrics"])
            self.entries.append({
                "config": cfg,
                "session": session,
                "loss": loss
            })
    def add(self, cfg, session):
        loss = min(m["validation_loss"] for m in session.history["epoch_metrics"])
        self.entries.append({
            "config": cfg,
            "session": session,
            "loss": loss
        })
    def ranked(self):
        return sorted(self.entries, key=lambda x: x["loss"])
    def top(self, k):
        return self.ranked()[:k]

def sample_uniform(low, high):
    return random.uniform(low, high)


def sample_log_uniform(low, high):
    return 10 ** random.uniform(math.log10(low), math.log10(high))


def sample_parameter(low, high,  mode):
    if mode == "uniform":
        return sample_uniform(low, high)
    if mode == "log":
        return sample_log_uniform(low, high)
    raise ValueError(f"Unknown sampling mode: {mode}")

def set_nested(cfg: dict, key: str, value):
    parts = key.split(".")
    d = cfg
    for p in parts[:-1]:
        if p not in d or not isinstance(d[p], dict):
            d[p] = {}
        d = d[p]
    d[parts[-1]] = value

def sample_config(base_config, search_space):
    import copy
    cfg = {} if base_config is None else copy.deepcopy(base_config)
    for param, (low, high, mode) in search_space.items():
        set_nested(cfg, param, sample_parameter(low, high, mode))
    return cfg

def prune(sessions, keep):
    leaderboard = Leaderboard(sessions)
    best = leaderboard.top(keep)
    return [(e["config"], e["session"]) for e in best]

def select_best(sessions):
    best_loss = float("inf")
    best = None
    for cfg, session in sessions:
        loss = min(m["validation_loss"] for m in session.history["epoch_metrics"])
        if loss < best_loss:
            best_loss = loss
            best = (cfg, session)
    return best


def staged_search(search_space,train_loader,val_loader, builder, model_dir, base_config=None,
                  training_step=gru_step, save_outputs=False, schedule=None,
                  initial_models=10, search_name="hyperparameter_search",**kwargs):
    """
    Generic successive-halving hyperparameter search.
    """
    if schedule is None:
        schedule = [
            {"epochs": 10, "keep": math.ceil(initial_models / 2)},
            {"epochs": 10, "keep": math.ceil(initial_models / 4)},
            {"epochs": 20, "keep": 1}
        ]
    sessions = []
    run_records = {}
    # --- initialise configs ---
    for i in range(initial_models):
        cfg = sample_config(base_config, search_space)
        # session = create_training_session(images, labels, method, cfg.get("reg_dropout", dropout_prob), cfg,
        #                                   training_step, **cfg)
        model, criterion, optimiser, training_kwargs = builder(cfg)
        session = TrainingSession(model=model, optimiser=optimiser, criterion=criterion, config=cfg,
                                  training_step=training_step, training_kwargs=training_kwargs)
        session.id = f"model_{i}"
        sessions.append((cfg, session))

    for stage_idx, stage in enumerate(schedule):
        epochs, keep = stage["epochs"], stage["keep"]
        print(f"\nStage {stage_idx}: training {len(sessions)} models for {epochs} epochs")
        for i, (cfg, session) in enumerate(sessions):
            # full_train(name=f"search_stage{stage_idx}_{i}", images=images, labels=labels, train_loader=train_loader,
            #            val_loader=val_loader, method=method, epochs=epochs, model_dir=model_dir, config=cfg,
            #            dropout_prob=cfg.get("reg_dropout", dropout_prob), training_step=training_step,
            #            save_outputs=False, session=session)
            full_train(name=f"search_stage{stage_idx}_{i}", builder=builder, cfg=cfg, train_loader=train_loader,
                       val_loader=val_loader, epochs=epochs, model_dir=model_dir, training_step=training_step,
                       save_outputs=False, session=session)
            if session.id not in run_records:
                run_records[session.id] = {
                    "id": session.id,
                    "config": cfg.copy(),
                    "total_epochs": None,
                    "epoch_metrics": []
                }
            run_records[session.id]["epoch_metrics"] = session.history["epoch_metrics"]
            run_records[session.id]["total_epochs"] = session.epoch
        if keep is not None:
            sessions = prune(sessions, keep=keep)
            print(f"Pruned to {len(sessions)} models")
    best_cfg, best_session = select_best(sessions)
    best_metrics = min(best_session.history["epoch_metrics"], key=lambda m: m["validation_loss"])
    search_summary = {
        "search_type": "successive_halving",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "initial_models": initial_models,
        "schedule": schedule,
        "search_space": search_space,
        "best_config": best_cfg,
        "best_epoch_metrics": best_metrics,
        "runs": list(run_records.values())
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    summary_path = model_dir / f"{search_name}.json"
    save_json(search_summary, summary_path)
    print(f"\nHyperparameter search summary saved to: {summary_path}")
    return best_cfg