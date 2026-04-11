from ..base_model import BaseModel


def _iter_descendant_subclasses(base_cls):
    """
    Yield every subclass in the wrapper hierarchy, not just direct children.
    """
    for cls in base_cls.__subclasses__():
        yield cls
        yield from _iter_descendant_subclasses(cls)


def _iter_wrapper_classes():
    for cls in _iter_descendant_subclasses(BaseModel):
        if getattr(cls, "model_name", None):
            yield cls


def get_model_class(model_name: str):
    """
    Resolve a BaseModel wrapper class by its model_name attribute.
    """
    for cls in _iter_wrapper_classes():
        if getattr(cls, "model_name", None) == model_name:
            return cls
    raise KeyError(
        f"No BaseModel wrapper class found for '{model_name}'. "
        f"Available wrappers: {sorted(get_available_model_names())}"
    )


def get_available_model_names():
    return sorted(
        cls.model_name
        for cls in _iter_wrapper_classes()
    )
