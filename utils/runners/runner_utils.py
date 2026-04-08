from pathlib import Path

from utils.configs.config_loader import (
    get_model_run_dir,
    load_effective_train_config,
    load_model_config,
    resolve_registry_entry,
)
from utils.data import build_dataloaders, get_feature_cols
from utils.experiment import Experiment


def _first_non_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


def build_loader_cache_key(
    *,
    mode: str,
    exp_section: dict,
    first_train_cfg: dict,
    include_test: bool,
    batch_size_override=None,
    num_workers_override=None,
) -> tuple:
    if mode == "search":
        batch_size = int(_first_non_none(first_train_cfg.get("batch_size"), 1024))
        num_workers = int(_first_non_none(exp_section.get("num_workers"), first_train_cfg.get("num_workers"), 4))
    else:
        batch_size = int(_first_non_none(batch_size_override, first_train_cfg.get("batch_size"), exp_section.get("batch_size"), 1024))
        num_workers = int(_first_non_none(num_workers_override, first_train_cfg.get("num_workers"), exp_section.get("num_workers"), 4))

    return (
        mode,
        str(_first_non_none(first_train_cfg.get("data_dir"), exp_section.get("data_dir"), "./data")),
        int(first_train_cfg.get("seq_len", 28)),
        int(first_train_cfg.get("horizon", 28)),
        batch_size,
        int(exp_section.get("top_k_series", 1000 if mode == "search" else 30490)),
        str(first_train_cfg.get("feature_set", "sales_only")),
        bool(first_train_cfg.get("autoregressive", True)),
        bool(first_train_cfg.get("use_normalise", False)),
        str(exp_section.get("sampling", "stratified" if mode == "search" else "all")),
        first_train_cfg.get("max_series"),
        num_workers,
        int(first_train_cfg.get("seed", 42)),
        str(_first_non_none(first_train_cfg.get("split_protocol"), exp_section.get("split_protocol"), "default")),
        str(_first_non_none(first_train_cfg.get("weight_protocol"), exp_section.get("weight_protocol"), "default")),
        bool(include_test),
    )


def load_shared_data(
    *,
    mode: str,
    exp_section: dict,
    first_train_cfg: dict,
    include_test: bool,
    batch_size_override=None,
    num_workers_override=None,
) -> dict:
    use_norm = bool(first_train_cfg.get("use_normalise", False))
    if mode == "search":
        data_kwargs = dict(
            data_dir=_first_non_none(first_train_cfg.get("data_dir"), exp_section.get("data_dir"), "./data"),
            seq_len=int(first_train_cfg.get("seq_len", 28)),
            horizon=int(first_train_cfg.get("horizon", 28)),
            batch_size=int(_first_non_none(first_train_cfg.get("batch_size"), 1024)),
            top_k_series=int(exp_section.get("top_k_series", 1000)),
            feature_set=str(first_train_cfg.get("feature_set", "sales_only")),
            autoregressive=bool(first_train_cfg.get("autoregressive", True)),
            use_normalise=use_norm,
            sampling=str(exp_section.get("sampling", "stratified")),
            max_series=first_train_cfg.get("max_series"),
            num_workers=int(_first_non_none(exp_section.get("num_workers"), first_train_cfg.get("num_workers"), 4)),
            seed=int(first_train_cfg.get("seed", 42)),
            split_protocol=_first_non_none(first_train_cfg.get("split_protocol"), "default"),
            weight_protocol=_first_non_none(first_train_cfg.get("weight_protocol"), "default"),
        )
        print("\n[search] Loading data once for all models...")
        print(
            f"[search] Sampling  : {data_kwargs['sampling']} | "
            f"top_k={data_kwargs['top_k_series']} | "
            f"feature_set={data_kwargs['feature_set']}"
        )
    else:
        data_kwargs = dict(
            data_dir=_first_non_none(first_train_cfg.get("data_dir"), exp_section.get("data_dir"), "./data"),
            seq_len=int(first_train_cfg.get("seq_len", 28)),
            horizon=int(first_train_cfg.get("horizon", 28)),
            batch_size=int(_first_non_none(batch_size_override, first_train_cfg.get("batch_size"), exp_section.get("batch_size"), 1024)),
            top_k_series=int(exp_section.get("top_k_series", 30490)),
            feature_set=str(first_train_cfg.get("feature_set", "sales_only")),
            autoregressive=bool(first_train_cfg.get("autoregressive", True)),
            use_normalise=use_norm,
            sampling=str(exp_section.get("sampling", "all")),
            max_series=first_train_cfg.get("max_series"),
            num_workers=int(_first_non_none(num_workers_override, first_train_cfg.get("num_workers"), exp_section.get("num_workers"), 4)),
            seed=int(first_train_cfg.get("seed", 42)),
            split_protocol=_first_non_none(first_train_cfg.get("split_protocol"), exp_section.get("split_protocol"), "default"),
            weight_protocol=_first_non_none(first_train_cfg.get("weight_protocol"), exp_section.get("weight_protocol"), "default"),
        )
        print("\n[train] Loading data once for all experiments...")
        print(f"[train] sampling    : {data_kwargs['sampling']}")
        print(f"[train] top_k       : {data_kwargs['top_k_series']}")
        print(f"[train] feature_set : {data_kwargs['feature_set']}")
        print(f"[train] batch_size  : {data_kwargs['batch_size']}")

    if not use_norm:
        train_loader, val_loader, test_loader, stats, vocab_sizes, feature_index = build_dataloaders(
            **data_kwargs
        )
        train_loader_w, _, _, _, _, _ = build_dataloaders(**data_kwargs, include_weights=True)
        out = dict(
            train_loader_det=train_loader,
            val_loader_det=val_loader,
            stats_det=stats,
            train_loader_gauss=train_loader,
            val_loader_gauss=val_loader,
            stats_gauss=stats,
            train_loader_nb=train_loader,
            val_loader_nb=val_loader,
            stats_nb=stats,
            train_loader_wquantile=train_loader_w,
            vocab_sizes=vocab_sizes,
            feature_index=feature_index,
        )
        if include_test:
            out.update(
                test_loader_det=test_loader,
                test_loader_gauss=test_loader,
                test_loader_nb=test_loader,
            )
        return out

    tl_det, vl_det, tel_det, s_det, vocab_sizes, feature_index = build_dataloaders(
        **data_kwargs, zscore_target=True
    )
    tl_gauss, vl_gauss, tel_gauss, s_gauss, _, _ = build_dataloaders(
        **data_kwargs, zscore_target=False
    )
    tl_nb, vl_nb, tel_nb, s_nb, _, _ = build_dataloaders(
        **data_kwargs, zscore_target=False
    )
    tl_w, _, _, _, _, _ = build_dataloaders(
        **data_kwargs, zscore_target=True, include_weights=True
    )
    out = dict(
        train_loader_det=tl_det,
        val_loader_det=vl_det,
        stats_det=s_det,
        train_loader_gauss=tl_gauss,
        val_loader_gauss=vl_gauss,
        stats_gauss=s_gauss,
        train_loader_nb=tl_nb,
        val_loader_nb=vl_nb,
        stats_nb=s_nb,
        train_loader_wquantile=tl_w,
        vocab_sizes=vocab_sizes,
        feature_index=feature_index,
    )
    if include_test:
        out.update(
            test_loader_det=tel_det,
            test_loader_gauss=tel_gauss,
            test_loader_nb=tel_nb,
        )
    return out


def prepare_model_context(
    *,
    model_name: str,
    exp_cfg: dict,
    run_dir: Path,
    registry: dict,
    runtime_overrides=None,
) -> dict:
    run_model_yml = run_dir / "configs" / "models" / f"{model_name}.yml"
    model_cfg = load_model_config(run_model_yml)
    train_cfg = load_effective_train_config(
        exp_cfg,
        run_model_yml,
        runtime_overrides=runtime_overrides,
    )
    feature_set = str(train_cfg.get("feature_set", "sales_only"))
    train_cfg["n_features"] = len(get_feature_cols(feature_set))

    if model_name not in registry:
        raise KeyError(
            f"'{model_name}' not in registry.yml. Available: {sorted(registry.keys())}"
        )
    resolved = resolve_registry_entry(registry[model_name])
    return {
        "run_model_yml": run_model_yml,
        "model_cfg": model_cfg,
        "train_cfg": train_cfg,
        "model_type": model_cfg.get("model_type", ""),
        "probabilistic": bool(model_cfg.get("probabilistic", False)),
        "builder": resolved["builder"],
        "training_step": resolved["training_step"],
        "model_dir": get_model_run_dir(run_dir, model_name),
    }


def attach_model_metadata(train_cfg: dict, loaders: dict) -> dict:
    train_cfg["vocab_sizes"] = loaders.get("vocab_sizes", {})
    train_cfg["feature_index"] = loaders.get("feature_index", {})
    return train_cfg


def select_model_loaders(model_type: str, probabilistic: bool, loaders: dict) -> dict:
    is_nb = model_type in ("baseline_gru_nb", "hierarchical_gru_nb")
    is_wquantile = model_type in ("baseline_wquantile_gru", "hierarchical_wquantile_gru")

    if is_nb:
        return {
            "train_loader": loaders["train_loader_nb"],
            "val_loader": loaders["val_loader_nb"],
            "stats": loaders["stats_nb"],
            "test_loader": loaders.get("test_loader_nb"),
        }
    if is_wquantile:
        return {
            "train_loader": loaders["train_loader_wquantile"],
            "val_loader": loaders["val_loader_det"],
            "stats": loaders["stats_det"],
            "test_loader": loaders.get("test_loader_det"),
        }
    if probabilistic:
        return {
            "train_loader": loaders["train_loader_gauss"],
            "val_loader": loaders["val_loader_gauss"],
            "stats": loaders["stats_gauss"],
            "test_loader": loaders.get("test_loader_gauss"),
        }
    return {
        "train_loader": loaders["train_loader_det"],
        "val_loader": loaders["val_loader_det"],
        "stats": loaders["stats_det"],
        "test_loader": loaders.get("test_loader_det"),
    }


def build_preloaded_experiment(model_name: str, train_cfg: dict, model_dir: Path, routed: dict) -> Experiment:
    exp = Experiment(model_name, train_cfg, model_dir=model_dir)
    exp.train_loader = routed["train_loader"]
    exp.val_loader = routed["val_loader"]
    exp.stats = routed["stats"]
    if routed.get("test_loader") is not None:
        exp.test_dataset = routed["test_loader"]
    exp.preloaded = True
    return exp
