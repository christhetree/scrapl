import jax
import jax.numpy as jnp


def safe_lp_norm(
    difference: jax.Array,
    p: float = 2,
    axis: int = -1,
) -> jax.Array:
    """Compute the Lp norm safely with well-defined, finite gradients at exact zero.

    For p == 1 (L1 norm), computes the sum of absolute differences with exact
    zero handling.

    For p > 1 (e.g. L2 norm), d/dx ||x||_p divides by ||x||_p^(p-1), which produces
    0/0 = NaN gradients when difference is zero (i.e. target == prediction).
    This function applies JAX's "safe norm" (double-where) pattern:
    1. Identify non-zero sample differences along the specified axis.
    2. Substitute a safe dummy vector (all ones) for all-zero samples so the norm
       and its backward pass evaluate with finite, non-zero gradients.
    3. Evaluate Lp norm safely.
    4. Zero out the result for samples that were genuinely zero.

    Parameters
    ----------
    difference : jax.Array
        Input difference tensor.
    p : float, default=2
        Order of the Lp norm (1, 2, ... or float('inf')).
    axis : int, default=-1
        Axis along which the norm is computed.

    Returns
    -------
    jax.Array
        Lp norm tensor along the specified axis.
    """
    if p == 1:
        magnitude = jnp.where(difference == 0, 0, jnp.abs(difference))
        return magnitude.sum(axis=axis)

    nonzero = jnp.any(difference != 0, axis=axis, keepdims=True)
    safe_difference = jnp.where(nonzero, difference, jnp.ones_like(difference))
    distance = jnp.linalg.norm(safe_difference, ord=p, axis=axis)
    return jnp.where(jnp.squeeze(nonzero, axis=axis), distance, 0)
