def n_features(cfg: dict) -> int:
    """Resolve input feature size from config."""
    return int(cfg.get("n_features", 1))


def output_size(cfg: dict) -> int:
    """
    Resolve model output size from config.
    autoregressive=True  -> 1  (1-step ahead, recursive rollout at test)
    autoregressive=False -> horizon  (direct multi-step)
    """
    if cfg.get("autoregressive", True):
        return 1
    return int(cfg["horizon"])


def rounded_hidden_size(raw_hidden: int) -> int:
    return max(8, (int(raw_hidden) // 8) * 8)
