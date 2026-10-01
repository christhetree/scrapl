import os
import sys

# This adds 'scrapl/kymatio' & 'scrapl/pytorch_hessian_eigenthings' to the Python path.
# This allows the internal code to import them successfully.
_submodule_paths = [
    os.path.join(os.path.dirname(__file__), "kymatio"),
    os.path.join(os.path.dirname(__file__), "pytorch_hessian_eigenthings"),
]
for _submodule_path in _submodule_paths:
    sys.path.append(_submodule_path)

__all__ = ["SCRAPLLoss"]


def __getattr__(name: str):
    """Lazily load PyTorch-dependent attributes on first access (PEP 562).

    This prevents `import scrapl` from unconditionally importing PyTorch, allowing
    JAX-only environments to import the top-level package or `scrapl.jax` without
    requiring PyTorch to be installed.
    """
    if name != "SCRAPLLoss":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    # Validate that PyTorch backend is installed before loading PyTorch SCRAPLLoss
    from ._dependencies import require_backend
    require_backend("torch")

    from .scrapl_loss import SCRAPLLoss

    # Cache in globals so subsequent lookups bypass __getattr__
    globals()[name] = SCRAPLLoss
    return SCRAPLLoss


def __dir__():
    """
    Ensure lazily-loaded exports in __all__ appear in dir() and IDE autocompletion
    (PEP 562).
    """
    return sorted(set(globals()) | set(__all__))
