from importlib import import_module

from .config import DatasetSource, TrainingConfig

__all__ = ["DatasetSource", "TrainingConfig", "train"]


def __getattr__(name: str):
    if name == "train":
        return import_module(".trainer", __name__).train
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
