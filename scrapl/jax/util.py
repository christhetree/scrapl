import logging
import os

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, PyTree

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def safe_lp_norm(
    x: Float[Array, "..."],
    p: float = 2,
    axis: int = -1,
) -> Float[Array, "..."]:
    """Compute the Lp norm safely with well-defined, finite gradients at exact zero.

    For p == 1 (L1 norm), computes the sum of absolute values with exact
    zero handling.

    For p > 1 (e.g. L2 norm), d/dx ||x||_p divides by ||x||_p^(p-1), which produces
    0/0 = NaN gradients when x is zero (e.g. target == prediction).
    This function applies JAX's "safe norm" (double-where) pattern:
    1. Identify non-zero samples along the specified axis.
    2. Substitute a safe dummy vector (all ones) for all-zero samples so the norm
       and its backward pass evaluate with finite, non-zero gradients.
    3. Evaluate Lp norm safely.
    4. Zero out the result for samples that were genuinely zero.

    Parameters
    ----------
    x : Float[Array, "..."]
        Input tensor.
    p : float, default=2
        Order of the Lp norm (1, 2, ... or float('inf')).
    axis : int, default=-1
        Axis along which the norm is computed.

    Returns
    -------
    Float[Array, "..."]
        Lp norm tensor along the specified axis.
    """
    if p == 1:
        magnitude = jnp.where(x == 0, 0, jnp.abs(x))
        return magnitude.sum(axis=axis)

    nonzero = jnp.any(x != 0, axis=axis, keepdims=True)
    safe_x = jnp.where(nonzero, x, jnp.ones_like(x))
    distance = jnp.linalg.norm(safe_x, ord=p, axis=axis)
    return jnp.where(jnp.squeeze(nonzero, axis=axis), distance, 0)


def tree_l2_normalize(
    tree: PyTree[Float[Array, "..."]],
    eps: float,
) -> PyTree[Float[Array, "..."]]:
    """Divides every leaf of a PyTree by the global L2 norm of the PyTree.

    Parameters
    ----------
    tree : PyTree[Float[Array, "..."]]
        PyTree of floating arrays.
    eps : float
        Small epsilon added to the norm for numerical stability.

    Returns
    -------
    PyTree[Float[Array, "..."]]
        Normalized PyTree with the same structure and dtypes.
    """
    norm = optax.tree_utils.tree_norm(tree, ord=2)
    # Cast back since the norm of mixed-precision leaves is promoted.
    norm_tree = jax.tree.map(lambda x: (x / (norm + eps)).astype(x.dtype), tree)
    return norm_tree
