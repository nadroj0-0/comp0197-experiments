from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils.common import save_json
from utils.configs.config_loader import (
    create_run_dir,
    get_model_run_dir,
    load_effective_train_config,
    load_experiment,
    load_model_config,
    load_registry,
    load_search_space,
    resolve_registry_entry,
    snapshot_configs,
)
from utils.data import WindowedM5Dataset, encode_hierarchy, get_feature_cols, get_vocab_sizes, set_seed
from utils.experiment import Experiment
from utils.runners.runner_utils import load_shared_data, prepare_model_context, select_model_loaders

PROJECT_DIR = Path(__file__).resolve().parents[2]
MODELS_CFG_DIR = PROJECT_DIR / "configs" / "models"
REGISTRY_PATH = PROJECT_DIR / "configs" / "registry.yml"


def preprocess_from_base_model(self_model, model_name: str,
                               run_name: str, include_weights: bool = False):
    run_dir = PROJECT_DIR / "runs" / run_name
    run_model_yml = run_dir / "configs" / "models" / f"{model_name}.yml"
    run_experiment_yml = run_dir / "configs" / "experiment.yml"
    exp_yml = run_experiment_yml if run_experiment_yml.exists() else PROJECT_DIR / "configs" / "experiment.yml"
    cfg_yml = run_model_yml if run_model_yml.exists() else MODELS_CFG_DIR / f"{model_name}.yml"

    exp_cfg = load_experiment(exp_yml)
    train_cfg = load_effective_train_config(exp_cfg, cfg_yml)

    feature_set = str(train_cfg.get("feature_set", "sales_yen_hierarchy"))
    seq_len = int(train_cfg.get("seq_len", 28))
    horizon = int(train_cfg.get("horizon", 28))
    batch_size = int(train_cfg.get("batch_size", 1024))
    autoregressive = bool(train_cfg.get("autoregressive", True))
    num_workers = int(train_cfg.get("num_workers", 0))
    if self_model._num_workers is not None:
        num_workers = self_model._num_workers
    seed = int(train_cfg.get("seed", 42))

    generator, _ = set_seed(seed)
    self_model.load_and_split_data()

    featured = pd.concat([
        self_model.train_raw,
        self_model.val_raw,
        self_model.test_raw,
    ]).reset_index(drop=True)
    include_dow = feature_set == "sales_hierarchy_dow"
    featured = encode_hierarchy(featured, include_dow=include_dow)
    featured["has_event"] = (
        (~featured["event_name_1"].astype(str).isin(["none", "nan", "None"]))
        .astype(np.float32)
    )
    if not pd.api.types.is_datetime64_any_dtype(featured["date"]):
        featured["date"] = pd.to_datetime(featured["date"])

    val_start_date = pd.Timestamp(self_model.val_raw["date"].min())
    test_start_date = pd.Timestamp(self_model.test_raw["date"].min())
    feature_cols = get_feature_cols(feature_set)
    feature_index = {col: i for i, col in enumerate(feature_cols)}
    series_ids = featured["id"].drop_duplicates().tolist()

    train_item_weights = None
    if include_weights:
        train_item_weights = (
            self_model.item_weights.reindex(series_ids).fillna(0.0).values.astype(np.float32)
        )
        total = train_item_weights.sum()
        if total > 0:
            train_item_weights /= total

    shared = dict(
        feature_cols=feature_cols,
        seq_len=seq_len,
        horizon=horizon,
        val_start_date=val_start_date.to_datetime64(),
        test_start_date=test_start_date.to_datetime64(),
        autoregressive=autoregressive,
    )

    train_ds = WindowedM5Dataset(
        featured, split="train", item_weights=train_item_weights, series_ids=series_ids, **shared
    )
    val_ds = WindowedM5Dataset(featured, split="val", series_ids=series_ids, **shared)
    test_ds = WindowedM5Dataset(featured, split="test", series_ids=series_ids, **shared)

    _pin = torch.cuda.is_available()
    self_model.train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, generator=generator, pin_memory=_pin,
    )
    self_model.val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=_pin,
    )
    self_model.test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=_pin,
    )

    hierarchy_cols = ["state_id_int", "store_id_int", "cat_id_int", "dept_id_int"]
    self_model.vocab_sizes = get_vocab_sizes(featured) if any(c in feature_cols for c in hierarchy_cols) else {}
    self_model.feature_index = feature_index
    self_model.stats = None

    train_cfg["n_features"] = len(feature_cols)
    train_cfg["vocab_sizes"] = self_model.vocab_sizes
    train_cfg["feature_index"] = self_model.feature_index
    self_model._train_cfg = train_cfg

    print(f"[preprocess] {model_name} — "
          f"{len(train_ds):,} train / {len(val_ds):,} val / {len(test_ds):,} test windows")


def train_from_preprocess(self_model, model_name: str, run_name: str, builder, step):
    run_dir = PROJECT_DIR / "runs" / run_name
    model_dir = get_model_run_dir(run_dir, model_name)
    train_cfg = getattr(self_model, "_train_cfg", {})

    exp = Experiment(model_name, train_cfg, model_dir=model_dir)
    exp.train_loader = self_model.train_loader
    exp.val_loader = self_model.val_loader
    exp.test_dataset = self_model.test_loader
    exp.stats = self_model.stats
    exp.preloaded = True

    exp.train(builder, step)

    if exp.stats is not None:
        save_json(exp.stats, model_dir / "normalisation_stats.json")

    self_model.model = exp.model
    self_model.history = exp.history
    print(f"\n  [DONE] {model_name} — artefacts in {model_dir}")
    return exp


def run_full_pipeline(self_model, model_name: str, run_name: str = None,
                      do_search: bool = False, include_weights: bool = False):
    if run_name is None:
        run_name = model_name + "_run"

    run_dir = create_run_dir(PROJECT_DIR, run_name)
    run_experiment_yml = run_dir / "configs" / "experiment.yml"
    run_model_yml = run_dir / "configs" / "models" / f"{model_name}.yml"
    if not run_experiment_yml.exists() or not run_model_yml.exists():
        snapshot_configs(
            run_dir=run_dir,
            experiment_yml=str(PROJECT_DIR / "configs" / "experiment.yml"),
            model_names=[model_name],
            models_cfg_dir=MODELS_CFG_DIR,
        )

    if do_search:
        search_space = load_search_space(run_model_yml)
        if search_space:
            from search import search_model

            exp_cfg = load_experiment(run_experiment_yml)
            exp_search = exp_cfg.get("search", {})
            registry = load_registry(REGISTRY_PATH)
            first_train_cfg = load_effective_train_config(exp_cfg, run_model_yml)
            loaders = load_shared_data(
                mode="search",
                exp_section=exp_search,
                first_train_cfg=first_train_cfg,
                include_test=False,
            )
            model_ctx = prepare_model_context(
                model_name=model_name,
                exp_cfg=exp_cfg,
                run_dir=run_dir,
                registry=registry,
            )
            routed = select_model_loaders(
                model_ctx["model_type"],
                model_ctx["probabilistic"],
                loaders,
            )

            search_model(
                model_name=model_name,
                run_dir=run_dir,
                exp_cfg=exp_cfg,
                exp_search=exp_search,
                registry=registry,
                train_loader=routed["train_loader"],
                val_loader=routed["val_loader"],
                stats=routed["stats"],
                vocab_sizes=loaders.get("vocab_sizes", {}),
                feature_index=loaders.get("feature_index", {}),
            )
            print(f"\n[models] Search detour complete for {model_name}")

    preprocess_from_base_model(self_model, model_name, run_name, include_weights=include_weights)
    registry = load_registry(REGISTRY_PATH)
    resolved = resolve_registry_entry(registry[model_name])
    return train_from_preprocess(self_model, model_name, run_name, resolved["builder"], resolved["training_step"])
