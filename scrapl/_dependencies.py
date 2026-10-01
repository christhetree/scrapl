"""Optional dependency and backend import utilities for scrapl."""

from importlib import import_module
from types import ModuleType


def require_backend(name: str) -> ModuleType:
    """
    Import and return an optional backend module with helpful installation instructions
    on failure.

    Parameters
    ----------
    name : str
        The top-level module name of the backend to import (e.g., 'torch' or 'jax').

    Returns
    -------
    types.ModuleType
        The imported backend module.

    Raises
    ------
    ModuleNotFoundError
        If the target backend is not installed, raising an error with guidance on
        how to install the required optional extra. Transitive import errors
        (e.g., a missing sub-dependency within the backend) are re-raised as-is.
    """
    try:
        module: ModuleType = import_module(name)
        # Ensure the imported module is a concrete package/module, not an empty
        # namespace package.
        if getattr(module, "__file__", None) is None:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return module
    except ModuleNotFoundError as error:
        # If a secondary/transitive dependency caused the error, propagate it directly.
        if error.name != name:
            raise
        # Raise an informative error directing the user to the correct optional extra.
        raise ModuleNotFoundError(
            f"The {name} backend requires the optional '{name}' extra. "
            f"Install it with pip install 'scrapl-loss[{name}]', "
            f"or 'uv sync --extra {name}' from a source checkout.",
            name=name,
        ) from error
