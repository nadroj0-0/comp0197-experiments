# =============================================================================
# legacy_train.py — M5 Sales Forecasting — V3
# COMP0197 Applied Deep Learning
#
# Registry-driven training entrypoint.
# Reads the experiment config, snapshots it into runs/, optionally searches,
# then trains every requested model with shared full-data loaders.
#
# Usage:
#   python legacy/legacy_train.py                          — uses configs/experiment.yml
#   python legacy/legacy_train.py --batch_size 4096        — override batch size for GPU
#   python legacy/legacy_train.py --num_workers 8          — override workers for GPU
#   python legacy/legacy_train.py --experiment configs/experiment.yml  — explicit path
# =============================================================================

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.experiment import Experiment
from utils.config_loader import (
    load_experiment,
    load_registry,
    load_model_config,
    load_effective_train_config,
    resolve_registry_entry,
    create_run_dir,
    snapshot_configs,
    get_model_run_dir,
)
from utils.data import build_dataloaders, get_feature_cols
from utils.common import save_json

PROJECT_DIR     = Path(__file__).resolve().parents[1]
EXPERIMENT_PATH = PROJECT_DIR / "configs" / "experiment.yml"
REGISTRY_PATH   = PROJECT_DIR / "configs" / "registry.yml"
MODELS_CFG_DIR  = PROJECT_DIR / "configs" / "models"

# =============================================================================
# CLI — only GPU convenience overrides, everything else comes from ymls
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="M5 Train V3")
    p.add_argument("--batch_size",  type=int, default=None,
                   help="Override batch_size from experiment.yml")
    p.add_argument("--num_workers", type=int, default=None,
                   help="Override num_workers from experiment.yml")
    p.add_argument("--experiment",  type=str, default=str(EXPERIMENT_PATH),
                   help="Path to experiment.yml (default: configs/experiment.yml)")
    return p.parse_args()


# =============================================================================
# DATA LOADING — load once upfront, route per model
# =============================================================================

def _load_train_data(exp_train: dict, first_train_cfg: dict,
                     batch_size_override=None,
                     num_workers_override=None) -> dict:
    """
    Build the shared full-training loaders for this experiment.

    Deterministic, Gaussian, and NB models can route to different loader
    variants when normalisation is enabled. Weighted quantile models also get
    a separate weighted training loader.
    """
    use_norm = bool(first_train_cfg.get("use_normalise", False))
    split_protocol = exp_train.get("split_protocol", "default")
    weight_protocol = exp_train.get("weight_protocol", "default")

    data_kwargs = dict(
        data_dir       = exp_train.get("data_dir",        "./data"),
        seq_len        = int(first_train_cfg.get("seq_len",     28)),
        horizon        = int(first_train_cfg.get("horizon",     28)),
        batch_size     = int(batch_size_override  or exp_train.get("batch_size",   1024)),
        top_k_series   = int(exp_train.get("top_k_series",      30490)),
        feature_set    = str(first_train_cfg.get("feature_set", "sales_only")),
        autoregressive = bool(first_train_cfg.get("autoregressive", True)),
        use_normalise  = use_norm,
        sampling       = str(exp_train.get("sampling",          "all")),
        max_series     = first_train_cfg.get("max_series"),
        num_workers    = int(num_workers_override or exp_train.get("num_workers",  4)),
        seed           = int(first_train_cfg.get("seed",         42)),
        split_protocol=split_protocol,
        weight_protocol=weight_protocol,
    )

    print("\n[train] Loading data once for all experiments...")
    print(f"[train] sampling    : {data_kwargs['sampling']}")
    print(f"[train] top_k       : {data_kwargs['top_k_series']}")
    print(f"[train] feature_set : {data_kwargs['feature_set']}")
    print(f"[train] batch_size  : {data_kwargs['batch_size']}")

    if not use_norm:
        # single loader — all models share it (raw counts, no zscore)
        train_loader, val_loader, test_loader, stats, vocab_sizes, feature_index = build_dataloaders(**data_kwargs)
        # weighted train loader — same data, include_weights=True for wquantile model
        # val/test are shared with det — they always return (x, y)
        train_loader_w, _, _, _, _, _ = build_dataloaders(**data_kwargs, include_weights=True)
        return dict(
            train_loader_det       = train_loader,   val_loader_det   = val_loader,  test_loader_det   = test_loader,  stats_det   = stats,
            train_loader_gauss     = train_loader,   val_loader_gauss = val_loader,  test_loader_gauss = test_loader,  stats_gauss = stats,
            train_loader_nb        = train_loader,   val_loader_nb    = val_loader,  test_loader_nb    = test_loader,  stats_nb    = stats,
            train_loader_wquantile = train_loader_w, vocab_sizes      = vocab_sizes, feature_index = feature_index,
        )
    else:
        # three loaders — det gets zscore, prob and NB get raw log1p
        tl_det,   vl_det,   tel_det,   s_det, vocab_sizes, feature_index   = build_dataloaders(**data_kwargs, zscore_target=True)
        tl_gauss, vl_gauss, tel_gauss, s_gauss, _, _ = build_dataloaders(**data_kwargs, zscore_target=False)
        tl_nb,    vl_nb,    tel_nb,    s_nb, _, _    = build_dataloaders(**data_kwargs, zscore_target=False)
        tl_w,     _,        _,         _, _, _       = build_dataloaders(**data_kwargs, zscore_target=True, include_weights=True)
        return dict(
            train_loader_det       = tl_det,   val_loader_det   = vl_det,   test_loader_det   = tel_det,   stats_det   = s_det,
            train_loader_gauss     = tl_gauss, val_loader_gauss = vl_gauss, test_loader_gauss = tel_gauss, stats_gauss = s_gauss,
            train_loader_nb        = tl_nb,    val_loader_nb    = vl_nb,    test_loader_nb    = tel_nb,    stats_nb    = s_nb,
            train_loader_wquantile = tl_w,     vocab_sizes      = vocab_sizes, feature_index = feature_index,
        )


# =============================================================================
# MAIN
# =============================================================================

def main():
    print('main started')
    args = parse_args()
    print('args parsed')

    # ------------------------------------------------------------------
    # 1. Load experiment and registry
    # ------------------------------------------------------------------
    exp_cfg  = load_experiment(args.experiment)
    registry = load_registry(REGISTRY_PATH)
    run_name = exp_cfg.get("run_name")
    if not run_name:
        raise ValueError("experiment.yml must have a 'run_name' field.")
    models = exp_cfg.get("models", [])
    if not models:
        raise ValueError("experiment.yml 'models' list is empty.")

    print(f"\n{'='*60}")
    print(f"  M5 TRAINING — {run_name}")
    print(f"  Models : {models}")
    do_search = bool(exp_cfg.get("search", {}).get("enabled", False))
    print(f"  Search : {do_search}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 2. Create run directory and snapshot all configs into it
    # This must happen before search so search writes into the run copies
    # ------------------------------------------------------------------
    run_dir = create_run_dir(PROJECT_DIR, run_name)
    snapshot_configs(
        run_dir        = run_dir,
        experiment_yml = args.experiment,
        model_names    = models,
        models_cfg_dir = MODELS_CFG_DIR,
    )

    # ------------------------------------------------------------------
    # 3. Hyperparameter search (if SEARCH = True)
    # search.py writes best configs back into run yml files so that
    # step 4 onwards reads the winning hyperparameters automatically
    # ------------------------------------------------------------------
    if do_search:
        from search import run_search
        run_search(exp_cfg, run_dir)
        print(f"\n[train] Search complete — proceeding to full training.\n")

    # ------------------------------------------------------------------
    # 4. Load full training data once
    # All models in an experiment share the same data settings by design
    # Use the first model's effective config for data field values
    # ------------------------------------------------------------------
    exp_train       = exp_cfg.get("train", {})
    first_model_yml = run_dir / "configs" / "models" / f"{models[0]}.yml"
    first_train_cfg = load_effective_train_config(
        exp_cfg,
        first_model_yml,
        runtime_overrides={
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
        },
    )

    loaders = _load_train_data(
        exp_train            = exp_train,
        first_train_cfg      = first_train_cfg,
        batch_size_override  = args.batch_size,
        num_workers_override = args.num_workers,
    )
    print("[train] Data loaded. Starting training loop.\n")

    # ------------------------------------------------------------------
    # 5. Train each model
    # ------------------------------------------------------------------
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"  TRAINING: {model_name}")
        print(f"{'='*60}")

        run_model_yml = run_dir / "configs" / "models" / f"{model_name}.yml"
        model_cfg     = load_model_config(run_model_yml)
        train_cfg     = load_effective_train_config(
            exp_cfg,
            run_model_yml,
            runtime_overrides={
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
            },
        )
        model_type    = model_cfg.get("model_type", "")
        is_prob       = bool(model_cfg.get("probabilistic", False))
        is_nb         = model_type in ("baseline_gru_nb", "hierarchical_gru_nb")
        is_hierarchical = model_type.startswith("hierarchical")

        # set n_features so builder sizes input layer correctly
        feature_set             = str(train_cfg.get("feature_set", "sales_only"))
        train_cfg["n_features"] = len(get_feature_cols(feature_set))
        # inject vocab_sizes for hierarchical models — builders require this
        # baseline models ignore it safely (they don't read the key)
        train_cfg["vocab_sizes"] = loaders.get("vocab_sizes", {})
        train_cfg["feature_index"] = loaders.get("feature_index", {})

        # resolve builder and training step from registry
        if model_name not in registry:
            raise KeyError(
                f"'{model_name}' not in registry.yml. "
                f"Available: {sorted(registry.keys())}"
            )
        resolved = resolve_registry_entry(registry[model_name])
        builder  = resolved["builder"]
        step     = resolved["training_step"]

        # all artifacts go into the run folder
        model_dir = get_model_run_dir(run_dir, model_name)

        # create Experiment and inject correct pre-loaded loaders
        exp = Experiment(model_name, train_cfg, model_dir=model_dir)

        if is_nb:
            exp.train_loader = loaders["train_loader_nb"]
            exp.val_loader   = loaders["val_loader_nb"]
            exp.test_dataset = loaders["test_loader_nb"]
            exp.stats        = loaders["stats_nb"]
        elif model_type in ("baseline_wquantile_gru", "hierarchical_wquantile_gru"):
            # weighted pinball — needs (x, y, weight) batches for training
            # val/test share the det loaders (always return (x, y))
            exp.train_loader = loaders["train_loader_wquantile"]
            exp.val_loader   = loaders["val_loader_det"]
            exp.test_dataset = loaders["test_loader_det"]
            exp.stats        = loaders["stats_det"]
        elif is_prob:
            exp.train_loader = loaders["train_loader_gauss"]
            exp.val_loader   = loaders["val_loader_gauss"]
            exp.test_dataset = loaders["test_loader_gauss"]
            exp.stats        = loaders["stats_gauss"]
        else:
            exp.train_loader = loaders["train_loader_det"]
            exp.val_loader   = loaders["val_loader_det"]
            exp.test_dataset = loaders["test_loader_det"]
            exp.stats        = loaders["stats_det"]
        exp.preloaded = True

        # train directly — search already handled above
        exp.train(builder, step)

        # save normalisation stats if applicable
        if exp.stats is not None:
            save_json(exp.stats, model_dir / "normalisation_stats.json")

        print(f"\n  [DONE] {model_name} — artifacts in {model_dir}")

    print(f"\n{'='*60}")
    print(f"  ALL TRAINING COMPLETE")
    print(f"  Run folder: {run_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
