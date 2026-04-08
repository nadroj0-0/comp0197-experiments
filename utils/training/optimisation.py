import inspect

import torch


class OptimisationConfig:
    """
    Centralised construction of optimiser, scheduler, and training kwargs.

    Design goals:
    -------------
    - Fully config-driven (no hidden defaults in builders)
    - Safe (invalid params caught early)
    - Extensible (add new optimisers/schedulers via registry)
    - Supports param groups (advanced research setups)
    """

    OPTIMISERS = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
    }

    SCHEDULERS = {
        "plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
    }

    @staticmethod
    def _filter_valid_kwargs(cls, params: dict):
        sig = inspect.signature(cls)
        valid_keys = set(sig.parameters.keys())

        filtered = {k: v for k, v in params.items() if k in valid_keys}
        invalid = set(params.keys()) - valid_keys

        if invalid:
            raise ValueError(
                f"Invalid parameters for {cls.__name__}: {invalid}\n"
                f"Valid parameters: {valid_keys}"
            )

        return filtered

    @staticmethod
    def _build_param_groups(model, cfg):
        if "param_groups" not in cfg:
            return model.parameters()

        groups = []
        named_params = dict(model.named_parameters())

        for group_cfg in cfg["param_groups"]:
            group = group_cfg.copy()
            name_filter = group.pop("params")

            selected_params = [p for n, p in named_params.items() if name_filter in n]

            if not selected_params:
                raise ValueError(f"No parameters matched: {name_filter}")

            group["params"] = selected_params
            groups.append(group)

        return groups

    @staticmethod
    def configure_optimiser(model, cfg):
        name = cfg.get("optimiser", "adam").lower()

        if name not in OptimisationConfig.OPTIMISERS:
            raise ValueError(f"Unknown optimiser: {name}")

        optimiser_class = OptimisationConfig.OPTIMISERS[name]
        params = cfg.get("optimiser_params", {}).copy()

        params.setdefault("lr", cfg.get("lr", 1e-3))
        if "weight_decay" in cfg:
            params.setdefault("weight_decay", cfg["weight_decay"])
        if "momentum" in cfg:
            params.setdefault("momentum", cfg["momentum"])

        params = OptimisationConfig._filter_valid_kwargs(optimiser_class, params)
        param_groups = OptimisationConfig._build_param_groups(model, cfg)
        return optimiser_class(param_groups, **params)

    @staticmethod
    def configure_scheduler(optimiser, cfg):
        name = cfg.get("scheduler", "plateau")

        if name is None:
            return None

        name = name.lower()

        if name not in OptimisationConfig.SCHEDULERS:
            raise ValueError(f"Unknown scheduler: {name}")

        scheduler_class = OptimisationConfig.SCHEDULERS[name]
        params = cfg.get("scheduler_params", {}).copy()

        if name == "plateau":
            params.setdefault("patience", 5)
            params.setdefault("factor", 0.5)

        if name == "cosine":
            params.setdefault("T_max", cfg.get("epochs", 50))

        params = OptimisationConfig._filter_valid_kwargs(scheduler_class, params)
        return scheduler_class(optimiser, **params)

    @staticmethod
    def configure_training_kwargs(optimiser, cfg):
        scheduler = OptimisationConfig.configure_scheduler(optimiser, cfg)

        kwargs = {
            "scheduler": scheduler,
            "clip_grad_norm": cfg.get("clip_grad_norm", 1.0),
            "extra_metrics": cfg.get("extra_metrics", None),
        }

        optional_keys = [
            "sigma_reg",
            "label_smoothing",
            "mixup_alpha",
            "grad_accum_steps",
        ]

        for key in optional_keys:
            if key in cfg:
                kwargs[key] = cfg[key]

        return kwargs

