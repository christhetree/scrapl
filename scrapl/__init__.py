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

from .scrapl_loss import SCRAPLLoss

__all__ = ["SCRAPLLoss"]
