from importlib.metadata import PackageNotFoundError, version

from .cmodule import CModule


try:
    __version__ = version("tinycwrap")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["CModule", "__version__"]
