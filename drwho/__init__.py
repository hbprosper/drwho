"""drwho -- shared utilities for Harrison B. Prosper's physics and ML repositories.

Submodules are imported lazily, so ``import drwho`` stays fast: the heavy
dependencies (torch, matplotlib, ...) are pulled in only when you first touch
the submodule that needs them.

    import drwho
    cfg = drwho.nn.Config(...)      # torch is imported at this point

Explicit imports work as usual:

    from drwho import nn, data
"""
import importlib as _importlib

# Single source of truth for the version: pyproject.toml reads it from here
# via [tool.setuptools.dynamic]. Bump it here on every release.
__version__ = "0.1.0"

__all__ = ["data", "kdtree", "monitor", "nn"]


def __getattr__(name):
    """Import drwho.<name> on first access (PEP 562)."""
    if name in __all__:
        return _importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(__all__) + ["__version__"])
