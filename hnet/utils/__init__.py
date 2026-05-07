from importlib import import_module

__all__ = ["ByteTokenizer", "group_params", "load_balancing_loss"]


def __getattr__(name: str):
    if name == "ByteTokenizer":
        return import_module(".tokenizers", __name__).ByteTokenizer
    if name in {"group_params", "load_balancing_loss"}:
        train_mod = import_module(".train", __name__)
        return getattr(train_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
