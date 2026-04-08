from ..base_model import BaseModel


def get_model_class(model_name: str):
    """
    Resolve a BaseModel wrapper class by its model_name attribute.
    """
    for cls in BaseModel.__subclasses__():
        if getattr(cls, "model_name", None) == model_name:
            return cls
    raise KeyError(
        f"No BaseModel wrapper class found for '{model_name}'. "
        f"Available wrappers: {sorted(get_available_model_names())}"
    )


def get_available_model_names():
    return sorted(
        cls.model_name
        for cls in BaseModel.__subclasses__()
        if getattr(cls, "model_name", None)
    )
