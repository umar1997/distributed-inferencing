"""Model catalog and one small adapter per model family."""


def adapter_for(config):
    """Pick the adapter from the config's `model_type` (e.g. 'granitemoe')."""
    from moe_dist.models.granite import GraniteAdapter

    adapters = {"granitemoe": GraniteAdapter}
    if config.model_type not in adapters:
        raise ValueError(f"no adapter for model type {config.model_type!r}; supported: {', '.join(adapters)}")
    return adapters[config.model_type](config)
