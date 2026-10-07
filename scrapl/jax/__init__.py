from .._dependencies import require_backend

require_backend("jax")

from .loss import SCRAPLLoss
from .optim import (
    PAdamState,
    PSAGAState,
    adam_grad_norm_cont,
    p_adam,
    p_saga,
    scale_by_gradient_multiplier,
)
from .util import safe_lp_norm
from .warmup import ThetaISResult, theta_importance_probs, warmup_lc_hvp

__all__ = [
    "SCRAPLLoss",
    "PAdamState",
    "PSAGAState",
    "ThetaISResult",
    "adam_grad_norm_cont",
    "p_adam",
    "p_saga",
    "safe_lp_norm",
    "scale_by_gradient_multiplier",
    "theta_importance_probs",
    "warmup_lc_hvp",
]
