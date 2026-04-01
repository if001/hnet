from importlib import import_module

__all__ = ["HNetForCausalLM"]


def __getattr__(name: str):
    if name == "HNetForCausalLM":
        return import_module(".models.mixer_seq", __name__).HNetForCausalLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
