import os
import sys

# This adds 'scrapl/kymatio' & 'scrapl/pytorch_hessian_eigenthings' to the Python path.
# This allows the internal code import them successfully.
_submodule_paths = [
    os.path.join(os.path.dirname(__file__), "kymatio"),
    os.path.join(os.path.dirname(__file__), "pytorch_hessian_eigenthings"),
]
for _submodule_path in _submodule_paths:
    sys.path.append(_submodule_path)

__all__ = ["SCRAPLLoss"]


def __getattr__(name):
    if name != "SCRAPLLoss":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from ._dependencies import require_backend

    require_backend("torch")
    from .scrapl_loss import SCRAPLLoss

    globals()[name] = SCRAPLLoss
    return SCRAPLLoss


def __dir__():
    return sorted(set(globals()) | set(__all__))
