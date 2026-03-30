# =============================================================================
# search.py — Hyperparameter Search Runner
# COMP0197 Applied Deep Learning
# GenAI Note: Scaffolded with Claude (Anthropic). Verified by authors.
#
# Runs successive halving hyperparameter search for all models listed in
# experiment.yml, then writes the best config back into each model's yml
# inside the run directory.
#
# Uses Experiment.search() so all search behaviour is managed by the
# Experiment class — consistent with how train.py runs full training.
#
# Called by train.py when SEARCH = True, or run standalone:
#   python search.py
#   python search.py --experiment configs/experiment.yml
# =============================================================================

import argparse
import math
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[0]))

from utils.experiment import Experiment, get_model_dir
from utils.config_loader import (
    load_experiment,
    load_registry,
    load_model_config,
    load_search_space,
    resolve_registry_entry,
    create_run_dir,
    snapshot_configs,
    write_best_config,
    get_model_run_dir,
)
from utils.data import build_dataloaders, get_feature_cols

PROJECT_DIR     = Path(__file__).resolve().parent
EXPERIMENT_PATH = PROJECT_DIR / "configs" / "experiment.yml"
REGISTRY_PATH   = PROJECT_DIR / "configs" / "registry.yml"
MODELS_CFG_DIR  = PROJECT_DIR / "configs" / "models"


# =============================================================================
# DATA LOADING — load once upfront, mirrors train.py exactly
# =============================================================================

def _load_search_data(exp_search: dict, exp_train: dict, first_model_cfg: dict) -> dict:
    """
    Load all data loaders once before the search loop starts.
    Mirrors train.py's upfront loading pattern exactly.

    Uses search data settings (stratified, smaller dataset) not full training.

    Returns a dict with keys:
        train_loader_det, val_loader_det, stats_det
        train_loader_gauss, val_loader_gauss, stats_gauss
        train_loader_nb, val_loader_nb, stats_nb
    If use_normalise=False all three point to the same loader (same as train.py).
    """
    train_cfg = first_model_cfg.get("train_config", {})
    use_norm  = bool(train_cfg.get("use_normalise", False))

    data_kwargs = dict(
        data_dir       = exp_train.get("data_dir", "./data"),
        seq_len        = int(train_cfg.get("seq_len",  28)),
        horizon        = int(train_cfg.get("horizon",  28)),
        batch_size     = int(exp_train.get("batch_size", 1024)),
        top_k_series   = int(exp_search.get("top_k_series", 1000)),
        feature_set    = str(train_cfg.get("feature_set", "sales_only")),
        autoregressive = bool(train_cfg.get("autoregressive", True)),
        use_normalise  = use_norm,
        sampling       = str(exp_search.get("sampling", "stratified")),
        max_series     = train_cfg.get("max_series"),
        num_workers = int(exp_search.get("num_workers", exp_train.get("num_workers", 4))),
        seed           = int(train_cfg.get("seed", 42)),
    )

    print(f"\n[search] Loading data once for all models...")
    print(f"[search] Sampling  : {data_kwargs['sampling']} | "
          f"top_k={data_kwargs['top_k_series']} | "
          f"feature_set={data_kwargs['feature_set']}")

    if not use_norm:
        # single loader — all models share it (raw counts, no zscore)
        train_loader, val_loader, _, stats, vocab_sizes = build_dataloaders(**data_kwargs)
        train_loader_w, _, _, _, _            = build_dataloaders(**data_kwargs, include_weights=True)
        return dict(
            train_loader_det       = train_loader,   val_loader_det   = val_loader, stats_det   = stats,
            train_loader_gauss     = train_loader,   val_loader_gauss = val_loader, stats_gauss = stats,
            train_loader_nb        = train_loader,   val_loader_nb    = val_loader, stats_nb    = stats,
            train_loader_wquantile = train_loader_w, vocab_sizes = vocab_sizes,
        )
    else:
        # three loaders — det gets zscore, prob and NB get raw log1p
        tl_det,   vl_det,   _, s_det, vocab_sizes   = build_dataloaders(**data_kwargs, zscore_target=True)
        tl_gauss, vl_gauss, _, s_gauss, _ = build_dataloaders(**data_kwargs, zscore_target=False)
        tl_nb,    vl_nb,    _, s_nb, _    = build_dataloaders(**data_kwargs, zscore_target=False)
        tl_w,     _,        _, _, _       = build_dataloaders(**data_kwargs, zscore_target=True, include_weights=True)
        return dict(
            train_loader_det       = tl_det,   val_loader_det   = vl_det,   stats_det   = s_det,
            train_loader_gauss     = tl_gauss, val_loader_gauss = vl_gauss, stats_gauss = s_gauss,
            train_loader_nb        = tl_nb,    val_loader_nb    = vl_nb,    stats_nb    = s_nb,
            train_loader_wquantile = tl_w, vocab_sizes = vocab_sizes,
        )


# =============================================================================
# SINGLE MODEL SEARCH
# =============================================================================

def search_model(
    model_name:   str,
    run_dir:      Path,
    exp_search:   dict,
    registry:     dict,
    train_loader,
    val_loader,
    stats,
    vocab_sizes:  dict,
) -> dict:
    """
    Run hyperparameter search for a single model using Experiment.search().

    Data loaders are passed in pre-loaded — loading happens once in run_search,
    not once per model.

    Flow:
        1. Load model yml from run snapshot
        2. Resolve builder + step from registry
        3. Create Experiment, inject pre-loaded loaders, call exp.search()
        4. Read best config from exp.cfg (updated in-place by Experiment.search)
        5. Write best config back to run yml via config_loader
        6. Return best_config dict
    """
    run_model_yml = run_dir / "configs" / "models" / f"{model_name}.yml"
    model_dir     = get_model_run_dir(run_dir, model_name)

    print(f"\n{'='*60}")
    print(f"  SEARCH: {model_name}")
    print(f"{'='*60}")

    # --- load model config from run snapshot ---
    model_cfg = load_model_config(run_model_yml)
    train_cfg = model_cfg.get("train_config", {})

    # --- set n_features so builders size input layers correctly ---
    feature_set             = str(train_cfg.get("feature_set", "sales_only"))
    train_cfg["n_features"] = len(get_feature_cols(feature_set))
    # --- inject vocab_sizes — hierarchical builders require this ---
    # baseline builders ignore it safely (they don't read the key)
    train_cfg["vocab_sizes"] = vocab_sizes

    # --- resolve builder and step from registry ---
    if model_name not in registry:
        raise KeyError(
            f"'{model_name}' not in registry.yml. "
            f"Available: {sorted(registry.keys())}"
        )
    resolved = resolve_registry_entry(registry[model_name])
    builder  = resolved["builder"]
    step     = resolved["training_step"]

    # --- load search space from run yml ---
    search_space = load_search_space(run_model_yml)
    if not search_space:
        print(f"  [SKIP] {model_name} — no search_space in yml, using train_config defaults")
        return {}

    # --- create Experiment and inject pre-loaded loaders ---
    exp              = Experiment(model_name, train_cfg, model_dir=model_dir)
    exp.train_loader = train_loader
    exp.val_loader   = val_loader
    exp.stats        = stats
    exp.preloaded    = True

    # --- run search via Experiment.search ---
    # This calls staged_search internally and updates exp.cfg with the winner
    init_models = int(exp_search.get("init_models", 10))
    schedule    = exp_search.get("schedule", [
        {"epochs": 10, "keep": math.ceil(init_models / 2)},
        {"epochs": 10, "keep": math.ceil(init_models / 4)},
        {"epochs": 20, "keep": 1},
    ])

    exp.search(
        search_space   = search_space,
        builder        = builder,
        training_step  = step,
        schedule       = schedule,
        initial_models = init_models,
    )

    # --- extract best config from exp.cfg ---
    # Experiment.search updates self.cfg with the winner in-place
    best_cfg = exp.cfg.copy()

    # --- write best config back into run yml ---
    write_best_config(run_model_yml, best_cfg)

    print(f"\n  Best config for {model_name}:")
    for k, v in best_cfg.items():
        if not isinstance(v, dict):
            print(f"    {k}: {v}")
        else:
            for kk, vv in v.items():
                print(f"    {k}.{kk}: {vv}")

    return best_cfg


# =============================================================================
# MAIN SEARCH LOOP — called by train.py or standalone
# =============================================================================

def run_search(exp_cfg: dict, run_dir: Path) -> dict:
    """
    Run hyperparameter search for all models in the experiment.

    Data is loaded ONCE upfront then routed to each model — same pattern
    as train.py. Models are searched sequentially.

    Called by train.py when SEARCH = True.
    Can also be called standalone via main() below.
    """
    registry   = load_registry(REGISTRY_PATH)
    exp_search = exp_cfg.get("search", {})
    exp_train  = exp_cfg.get("train",  {})
    models     = exp_cfg.get("models", [])

    if not models:
        print("[search] No models in experiment.yml — nothing to search.")
        return {}

    print(f"\n[search] Hyperparameter search — {len(models)} model(s)")
    print(f"[search] Sampling  : {exp_search.get('sampling', 'stratified')}")
    print(f"[search] Per-strat : {exp_search.get('top_k_series', 1000)}")
    print(f"[search] Init mods : {exp_search.get('init_models', 10)}")
    print(f"[search] Schedule  : {exp_search.get('schedule')}")

    # ------------------------------------------------------------------
    # Load data ONCE using the first model's config for data settings
    # All models in an experiment share the same data settings
    # (feature_set, seq_len, horizon, autoregressive, use_normalise)
    # ------------------------------------------------------------------
    first_model_yml = run_dir / "configs" / "models" / f"{models[0]}.yml"
    first_model_cfg = load_model_config(first_model_yml)
    loaders = _load_search_data(exp_search, exp_train, first_model_cfg)
    vocab_sizes = loaders.get("vocab_sizes", {})
    print(f"[search] Data loaded. Starting search loop.\n")

    # ------------------------------------------------------------------
    # Route correct loaders to each model — same logic as train.py
    # det models    → zscore loaders (loaders_det)
    # prob models   → raw count loaders (loaders_gauss)
    # NB models     → raw count loaders (loaders_nb)
    # ------------------------------------------------------------------
    all_best = {}
    for model_name in models:
        model_yml = run_dir / "configs" / "models" / f"{model_name}.yml"
        model_cfg = load_model_config(model_yml)
        model_type = model_cfg.get("model_type", "")
        is_prob    = bool(model_cfg.get("probabilistic", False))
        is_nb      = model_type in ("baseline_gru_nb", "gru_nb", "hierarchical_gru_nb")

        if is_nb:
            tl, vl, st = loaders["train_loader_nb"], loaders["val_loader_nb"], loaders["stats_nb"]
        elif model_type in ("baseline_wquantile_gru", "hierarchical_wquantile_gru"):
            tl, vl, st = loaders["train_loader_wquantile"], loaders["val_loader_det"], loaders["stats_det"]
        elif is_prob:
            tl, vl, st = loaders["train_loader_gauss"], loaders["val_loader_gauss"], loaders["stats_gauss"]
        else:
            tl, vl, st = loaders["train_loader_det"], loaders["val_loader_det"], loaders["stats_det"]

        best = search_model(
            model_name   = model_name,
            run_dir      = run_dir,
            exp_search   = exp_search,
            registry     = registry,
            train_loader = tl,
            val_loader   = vl,
            stats        = st,
            vocab_sizes=vocab_sizes,
        )
        all_best[model_name] = best

    print(f"\n[search] Done — {len(all_best)} model(s) searched.")
    return all_best


# =============================================================================
# STANDALONE ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="M5 Hyperparameter Search")
    parser.add_argument(
        "--experiment",
        type    = str,
        default = str(EXPERIMENT_PATH),
        help    = "Path to experiment.yml (default: configs/experiment.yml)",
    )
    args = parser.parse_args()

    exp_cfg  = load_experiment(args.experiment)
    run_name = exp_cfg.get("run_name")
    if not run_name:
        raise ValueError("experiment.yml must have a 'run_name' field.")

    # create run dir and snapshot configs
    # idempotent — safe if train.py already did this
    run_dir = create_run_dir(PROJECT_DIR, run_name)
    snapshot_configs(
        run_dir        = run_dir,
        experiment_yml = args.experiment,
        model_names    = exp_cfg.get("models", []),
        models_cfg_dir = MODELS_CFG_DIR,
    )

    run_search(exp_cfg, run_dir)


if __name__ == "__main__":
    main()