"""Diffusion math can be imported without the optional Transformer dependencies."""


__all__ = ["PixArt", "PixArt_XL_2"]

def __getattr__(name):
    from importlib import import_module
    nets = import_module(".nets", __name__)
    try:
        return getattr(nets, name)
    except AttributeError:
        raise AttributeError(name) from None
