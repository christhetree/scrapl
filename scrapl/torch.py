"""PyTorch backend exports for scrapl."""

from ._dependencies import require_backend

require_backend("torch")

from .scrapl_loss import SCRAPLLoss

__all__ = ["SCRAPLLoss"]
