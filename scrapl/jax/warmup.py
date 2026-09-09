import logging
import math
from functools import partial
from numbers import Integral
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from .loss import SCRAPLLoss

log = logging.getLogger(__name__)


class ThetaISResult(NamedTuple):
    """Warmup arrays: (paths, theta) estimates/residuals and (paths,) probabilities."""

    curvatures: jax.Array
    relative_residuals: jax.Array
    probs: jax.Array


def _validate_probability_settings(min_prob_frac, eps):
    if not 0 <= min_prob_frac < 1:
        raise ValueError("min_prob_frac must be in [0, 1)")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")


def theta_importance_probs(
    curvatures: jax.Array, *, min_prob_frac: float = 0.0, eps: float = 1e-12
) -> jax.Array:
    """Normalise over paths for each theta, then average and mix with uniform.

    All-zero estimates give uniform probabilities. Negative or non-finite
    estimates give NaN probabilities, including when called inside JIT.
    """
    _validate_probability_settings(min_prob_frac, eps)
    curvatures = jnp.asarray(curvatures)
    if curvatures.ndim != 2 or 0 in curvatures.shape:
        raise ValueError("curvatures must have nonempty (paths, theta) shape")
    if curvatures.dtype not in (jnp.float32, jnp.float64):
        raise TypeError("curvatures must use float32 or float64")
    if not 0 < np.asarray(eps, dtype=curvatures.dtype) < np.inf:
        raise ValueError("eps must be representable in the curvature dtype")
    valid = jnp.all(jnp.isfinite(curvatures) & (curvatures >= 0))
    log_values = jnp.log(jnp.maximum(curvatures, eps))
    per_theta = jax.nn.softmax(log_values, axis=0)
    probs = (1 - min_prob_frac) * per_theta.mean(axis=1)
    probs = probs + min_prob_frac / curvatures.shape[0]
    return jnp.where(valid, probs, jnp.nan)


def _theta_param_grad(
    weights, theta_idx, x, *, unravel, theta_fn, synth_fn, loss, path_idx
):
    theta, pullback = jax.vjp(lambda w: theta_fn(unravel(w), x), weights)
    sensitivity = jax.grad(lambda t: loss(x, synth_fn(t), path_idx=path_idx))(theta)
    cotangent = jnp.zeros_like(theta).at[:, theta_idx].set(sensitivity[:, theta_idx])
    return pullback(cotangent)[0]


def _theta_curvature_product(weights, tangent, theta_idx, xs, **kwargs):
    """Sum transposed Jacobian products, as in PyTorch's grad-of-grad warmup."""

    def accumulate(total, x):
        gradient_fn = partial(_theta_param_grad, theta_idx=theta_idx, x=x, **kwargs)
        _, pullback = jax.vjp(gradient_fn, weights)
        return total + pullback(tangent)[0], None

    return jax.lax.scan(accumulate, jnp.zeros_like(weights), xs)[0]


def _power_iteration(operator, vector, n_iter, eps):
    def iterate(vector, _):
        norm = jnp.linalg.norm(vector)
        unit = vector / jnp.where(norm > 0, norm, 1)
        product = operator(unit)
        eigenvalue = jnp.vdot(unit, product).real
        residual = jnp.linalg.norm(product - eigenvalue * unit)
        relative_residual = residual / jnp.maximum(jnp.linalg.norm(product), eps)
        return product, (jnp.abs(eigenvalue), relative_residual)

    _, (estimates, residuals) = jax.lax.scan(iterate, vector, None, length=n_iter)
    return estimates[-1], residuals[-1]


def warmup_lc_hvp(
    loss: SCRAPLLoss,
    params,
    theta_fn: Callable,
    synth_fn: Callable,
    xs: jax.Array,
    *,
    key: jax.Array,
    n_iter: int = 20,
    min_prob_frac: float = 0.0,
    eps: float = 1e-12,
) -> ThetaISResult:
    """Estimate theta-IS curvature and sampling probabilities for every path.

    ``theta_fn(params, x)`` maps a waveform batch to (batch, theta), and
    ``synth_fn(theta)`` reconstructs that batch. Both must be deterministic.
    ``params`` is a PyTree of floating encoder weights, treated as one group.
    ``xs`` has shape (n_batches, batch, channels, samples). Batch curvature
    products are summed, matching the PyTorch implementation.

    Call outside JIT. Each path is compiled separately, with sequential theta
    and batch evaluation to limit memory use. Returns estimates and final
    relative eigenvector residuals; a finite estimate need not have converged.
    This function does not modify params, the loss, or sampling state.
    """
    if not isinstance(n_iter, Integral) or isinstance(n_iter, bool) or n_iter < 1:
        raise ValueError("n_iter must be a positive integer")
    _validate_probability_settings(min_prob_frac, eps)
    params = jax.tree_util.tree_map(jnp.asarray, params)
    leaves = jax.tree_util.tree_leaves(params)
    if not leaves or not any(leaf.size for leaf in leaves):
        raise ValueError("params must contain trainable arrays")
    if any(leaf.dtype not in (jnp.float32, jnp.float64) for leaf in leaves):
        raise TypeError("Parameter leaves must use float32 or float64")
    if any(not np.isfinite(np.asarray(leaf)).all() for leaf in leaves):
        raise ValueError("params must be finite")
    weights, unravel = ravel_pytree(params)
    if not 0 < np.asarray(eps, dtype=weights.dtype) < np.inf:
        raise ValueError("eps must be representable in the parameter dtype")
    xs = jnp.asarray(xs)
    if xs.ndim != 4 or 0 in xs.shape or xs.shape[-1] != loss.shape:
        raise ValueError(
            "xs must have nonempty (n_batches, batch, channels, samples) shape"
        )
    if xs.dtype not in (jnp.float32, jnp.float64):
        raise TypeError("xs must use float32 or float64")
    if not np.isfinite(np.asarray(xs)).all():
        raise ValueError("xs must be finite")
    theta = jnp.asarray(theta_fn(params, xs[0]))
    if theta.ndim != 2 or theta.shape[0] != xs.shape[1] or theta.shape[1] == 0:
        raise ValueError("theta_fn must return (batch, theta) with at least one theta")
    if theta.dtype not in (jnp.float32, jnp.float64):
        raise TypeError("theta_fn must return floating arrays")
    prediction = jnp.asarray(synth_fn(theta))
    if prediction.shape != xs.shape[1:]:
        raise ValueError("synth_fn must reconstruct the waveform batch shape")
    if (
        not np.isfinite(np.asarray(theta)).all()
        or not np.isfinite(np.asarray(prediction)).all()
    ):
        raise ValueError("The encoder and synthesiser must return finite arrays")
    if not np.allclose(theta, theta_fn(params, xs[0])) or not np.allclose(
        prediction, synth_fn(theta)
    ):
        raise ValueError(
            "The encoder and synthesiser must be deterministic during warmup"
        )

    def path_curvatures(weights, xs, key, *, path_idx):
        def estimate(theta_idx):
            theta_key = jax.random.fold_in(key, theta_idx)
            vector = jax.random.normal(theta_key, weights.shape, dtype=weights.dtype)
            operator = partial(
                _theta_curvature_product,
                weights,
                theta_idx=theta_idx,
                xs=xs,
                unravel=unravel,
                theta_fn=theta_fn,
                synth_fn=synth_fn,
                loss=loss,
                path_idx=path_idx,
            )
            return _power_iteration(operator, vector, n_iter, eps)

        return jax.lax.map(estimate, jnp.arange(theta.shape[1]))

    compiled = jax.jit(path_curvatures, static_argnames=("path_idx",))
    curvatures, residuals = [], []
    for path_idx in range(loss.n_paths):
        path_key = jax.random.fold_in(key, path_idx)
        values, errors = compiled(weights, xs, path_key, path_idx=path_idx)
        if (
            not np.isfinite(np.asarray(values)).all()
            or not np.isfinite(np.asarray(errors)).all()
        ):
            raise FloatingPointError(
                f"Non-finite curvature estimates for path {path_idx}"
            )
        curvatures.append(values)
        residuals.append(errors)
        log.info("Theta-IS warmup: path %s/%s complete", path_idx + 1, loss.n_paths)
    curvatures = jnp.stack(curvatures)
    probs = theta_importance_probs(curvatures, min_prob_frac=min_prob_frac, eps=eps)
    return ThetaISResult(curvatures, jnp.stack(residuals), probs)
