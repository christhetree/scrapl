import logging
import math
import os
from functools import partial
from numbers import Integral
from typing import Callable, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, Bool, Float, Int, Key, PyTree
from tqdm import tqdm

from .loss import SCRAPLLoss
from .util import tree_l2_normalize

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class ThetaISResult(NamedTuple):
    """Per-path curvature estimates, final relative residuals, and probabilities.

    `curvatures` and `relative_residuals` have shape `(n_paths, n_theta)` and
    `probs` has shape `(n_paths,)`.
    """

    curvatures: Float[Array, "n_paths n_theta"]
    relative_residuals: Float[Array, "n_paths n_theta"]
    probs: Float[Array, "n_paths"]


def theta_importance_probs(
    curvatures: Float[Array, "n_paths n_theta"],
    *,
    min_prob_frac: float = 0.0,
    eps: float = 1e-12,
) -> Float[Array, "n_paths"]:
    """Computes path sampling probabilities from per-theta curvature estimates.

    Curvatures are normalized over paths for each theta, averaged over theta, and
    mixed with the uniform distribution. All-zero estimates give uniform
    probabilities. Negative or non-finite estimates give NaN probabilities,
    including when called inside JIT.

    Args:
        curvatures (Float[Array, "n_paths n_theta"]): Curvature estimates of shape `(n_paths, n_theta)`.
        min_prob_frac (float, optional): Fraction of the probability mass that is
            distributed uniformly over all paths, in [0, 1).
            Defaults to 0.0.
        eps (float, optional): Lower bound applied to curvatures before taking
            their log.
            Defaults to 1e-12.

    Returns:
        Float[Array, "n_paths"]: Path sampling probabilities of shape `(n_paths,)`.

    Raises:
        AssertionError: If `min_prob_frac` is not in [0, 1).
        AssertionError: If `eps` is not finite and positive or not representable in
            the curvature dtype.
        AssertionError: If `curvatures` is not a nonempty 2D float32 or float64
            array.
    """
    assert 0 <= min_prob_frac < 1, "min_prob_frac must be in [0, 1)"
    assert math.isfinite(eps) and eps > 0, "eps must be finite and positive"
    curvatures = jnp.asarray(curvatures)
    assert (
        curvatures.ndim == 2 and 0 not in curvatures.shape
    ), "curvatures must have nonempty (n_paths, n_theta) shape"
    assert curvatures.dtype in (
        jnp.float32,
        jnp.float64,
    ), "curvatures must use float32 or float64"
    assert (
        0 < np.asarray(eps, dtype=curvatures.dtype) < np.inf
    ), "eps must be representable in the curvature dtype"
    is_valid = jnp.all(jnp.isfinite(curvatures) & (curvatures >= 0))
    max_curvatures = jnp.maximum(curvatures, eps)
    log_vals = jnp.log(max_curvatures)
    theta_probs = jax.nn.softmax(log_vals, axis=0)
    avg_probs = theta_probs.mean(axis=1)
    unif_prob = 1.0 / curvatures.shape[0]
    probs = (1.0 - min_prob_frac) * avg_probs + min_prob_frac * unif_prob
    valid_probs = jnp.where(is_valid, probs, jnp.nan)
    return valid_probs


def calc_loss(
    theta: Float[Array, "bs n_theta"],
    x_target: Float[Array, "bs n_ch n_samples"],
    *,
    dec_fn: Callable[
        [Float[Array, "bs n_theta"]], Float[Array, "bs n_ch n_samples"]
    ],
    loss_fn: Callable[
        [Float[Array, "bs n_ch n_samples"], Float[Array, "bs n_ch n_samples"]],
        Float[Array, ""],
    ],
) -> Float[Array, ""]:
    """Decodes theta and computes the loss against the target waveform.

    Args:
        theta (Float[Array, "bs n_theta"]): Theta of shape `(batch, n_theta)`.
        x_target (Float[Array, "bs n_ch n_samples"]): Target waveform batch of shape `(batch, channels, samples)`.
        dec_fn (Callable[[Float[Array, "bs n_theta"]], Float[Array, "bs n_ch n_samples"]]): Decoder `dec_fn(theta)`.
        loss_fn (Callable[[Float[Array, "bs n_ch n_samples"], Float[Array, "bs n_ch n_samples"]], Float[Array, ""]]): Loss function
            `loss_fn(x, x_target)` returning a scalar.

    Returns:
        Float[Array, ""]: The scalar loss value.
    """
    x = dec_fn(theta)
    loss_val = loss_fn(x, x_target)
    return loss_val


def calc_sensitivity(
    theta: Float[Array, "bs n_theta"],
    x_target: Float[Array, "bs n_ch n_samples"],
    *,
    theta_loss_fn: Callable[
        [Float[Array, "bs n_theta"], Float[Array, "bs n_ch n_samples"]],
        Float[Array, ""],
    ],
) -> Float[Array, "bs n_theta"]:
    """Computes the gradient of the loss w.r.t. theta.

    Args:
        theta (Float[Array, "bs n_theta"]): Theta of shape `(batch, n_theta)`.
        x_target (Float[Array, "bs n_ch n_samples"]): Target waveform batch of shape `(batch, channels, samples)`.
        theta_loss_fn (Callable[[Float[Array, "bs n_theta"], Float[Array, "bs n_ch n_samples"]], Float[Array, ""]]): Loss function
            `theta_loss_fn(theta, x_target)` returning a scalar.

    Returns:
        Float[Array, "bs n_theta"]: dL/dtheta of shape `(batch, n_theta)`.
    """
    # jax.grad uses jax.vjp under the hood to pull back the gradient of the loss
    # function w.r.t. theta (the first argument).
    dL_dtheta = jax.grad(theta_loss_fn)(theta, x_target)
    return dL_dtheta


def calc_s_dE_dw_at_theta(
    enc_params: PyTree[Float[Array, "..."]],
    x_target: Float[Array, "bs n_ch n_samples"],
    *,
    theta_idx: Int[Array, ""] | int,
    enc_fn: Callable[
        [PyTree[Float[Array, "..."]], Float[Array, "bs n_ch n_samples"]],
        Float[Array, "bs n_theta"],
    ],
    theta_loss_fn: Callable[
        [Float[Array, "bs n_theta"], Float[Array, "bs n_ch n_samples"]],
        Float[Array, ""],
    ],
) -> PyTree[Float[Array, "..."]]:
    """Computes the parameter gradient contributed by a single theta.

    Args:
        enc_params (PyTree[Float[Array, "..."]]): Floating encoder weights.
        x_target (Float[Array, "bs n_ch n_samples"]): Waveform batch of shape
            `(batch, channels, samples)`.
        theta_idx (Int[Array, ""] | int): Theta index in [0, n_theta).
        enc_fn (Callable[[PyTree[Float[Array, "..."]], Float[Array, "bs n_ch n_samples"]], Float[Array, "bs n_theta"]]):
            Encoder function `enc_fn(enc_params, x)`.
        theta_loss_fn (Callable[[Float[Array, "bs n_theta"], Float[Array, "bs n_ch n_samples"]], Float[Array, ""]]): Loss function
            `theta_loss_fn(theta, x_target)` returning a scalar.

    Returns:
        PyTree[Float[Array, "..."]]: Gradient PyTree with the same structure as `enc_params`.
    """
    # Use jax.vjp instead of jax.jvp since we want the gradient of the encoder
    # w.r.t. the weights (n_theta, n_params) at theta_idx (1, n_params). Because dim_out << dim_in,
    # VJP is significantly more efficient than JVP.
    theta, vjp_fn = jax.vjp(lambda p: enc_fn(p, x_target), enc_params)

    # Sensitivity must be computed inside the HVP each time since it depends on enc_params.
    sensitivity = calc_sensitivity(theta, x_target, theta_loss_fn=theta_loss_fn)
    s_at_theta = sensitivity[:, theta_idx]

    s_theta_idx_one_hot = jnp.zeros_like(theta).at[:, theta_idx].set(s_at_theta)
    (s_dE_dw_at_theta,) = vjp_fn(s_theta_idx_one_hot)
    return s_dE_dw_at_theta


# noinspection PyTypeHints
def calc_hvp_at_theta(
    carry: PyTree[Float[Array, "..."]],
    x_target: Float[Array, "bs n_ch n_samples"],
    *,
    enc_params: PyTree[Float[Array, "..."]],
    theta_idx: Int[Array, ""] | int,
    tangent: PyTree[Float[Array, "..."]],
    enc_fn: Callable[
        [PyTree[Float[Array, "..."]], Float[Array, "bs n_ch n_samples"]],
        Float[Array, "bs n_theta"],
    ],
    theta_loss_fn: Callable[
        [Float[Array, "bs n_theta"], Float[Array, "bs n_ch n_samples"]],
        Float[Array, ""],
    ],
) -> tuple[PyTree[Float[Array, "..."]], None]:
    """Adds the curvature-vector product of a single batch to a running sum.

    Used as a `jax.lax.scan` step over batches.

    Args:
        carry (PyTree[Float[Array, "..."]]): Running sum of curvature-vector products.
        x_target (Float[Array, "bs n_ch n_samples"]): Waveform batch of shape
            `(batch, channels, samples)`.
        enc_params (PyTree[Float[Array, "..."]]): Floating encoder weights.
        theta_idx (Int[Array, ""] | int): Theta index in [0, n_theta).
        tangent (PyTree[Float[Array, "..."]]): Tangent vector with the same structure as `enc_params`.
        enc_fn (Callable[[PyTree[Float[Array, "..."]], Float[Array, "bs n_ch n_samples"]], Float[Array, "bs n_theta"]]):
            Encoder function `enc_fn(enc_params, x)`.
        theta_loss_fn (Callable[[Float[Array, "bs n_theta"], Float[Array, "bs n_ch n_samples"]], Float[Array, ""]]): Loss function
            `theta_loss_fn(theta, x_target)` returning a scalar.

    Returns:
        tuple[PyTree[Float[Array, "..."]], None]: Updated running sum and no stacked output.
    """
    s_dE_dw_fn = partial(
        calc_s_dE_dw_at_theta,
        theta_idx=theta_idx,
        x_target=x_target,
        enc_fn=enc_fn,
        theta_loss_fn=theta_loss_fn,
    )
    # Forward-over-reverse computes J @ tangent. PyTorch's grad-of-grad computes
    # J.T @ tangent instead, but J and J.T share the same eigenvalues.
    _, tangent_out = jax.jvp(s_dE_dw_fn, (enc_params,), (tangent,))
    new_carry = jax.tree.map(jnp.add, carry, tangent_out)
    # Accumulate the running sum without stacking per-batch products.
    return new_carry, None


def calc_hvp_at_theta_multibatch(
    tangent: PyTree[Float[Array, "..."]],
    xs: Float[Array, "n_batches bs n_ch n_samples"],
    *,
    enc_params: PyTree[Float[Array, "..."]],
    theta_idx: Int[Array, ""] | int,
    enc_fn: Callable[
        [PyTree[Float[Array, "..."]], Float[Array, "bs n_ch n_samples"]],
        Float[Array, "bs n_theta"],
    ],
    theta_loss_fn: Callable[
        [Float[Array, "bs n_theta"], Float[Array, "bs n_ch n_samples"]],
        Float[Array, ""],
    ],
) -> PyTree[Float[Array, "..."]]:
    """Sums the curvature-vector products over batches, matching PyTorch.

    Args:
        tangent (PyTree[Float[Array, "..."]]): Tangent vector with the same structure as `enc_params`.
        xs (Float[Array, "n_batches bs n_ch n_samples"]): Waveform batches of shape
            `(n_batches, batch, channels, samples)`.
        enc_params (PyTree[Float[Array, "..."]]): Floating encoder weights.
        theta_idx (Int[Array, ""] | int): Theta index in [0, n_theta).
        enc_fn (Callable[[PyTree[Float[Array, "..."]], Float[Array, "bs n_ch n_samples"]], Float[Array, "bs n_theta"]]):
            Encoder function `enc_fn(enc_params, x)`.
        theta_loss_fn (Callable[[Float[Array, "bs n_theta"], Float[Array, "bs n_ch n_samples"]], Float[Array, ""]]): Loss function
            `theta_loss_fn(theta, x_target)` returning a scalar.

    Returns:
        PyTree[Float[Array, "..."]]: Summed curvature-vector product with the same structure as `enc_params`.
    """
    hvp_fn = partial(
        calc_hvp_at_theta,
        enc_params=enc_params,
        theta_idx=theta_idx,
        tangent=tangent,
        enc_fn=enc_fn,
        theta_loss_fn=theta_loss_fn,
    )
    Hv_init = jax.tree.map(jnp.zeros_like, tangent)
    # Use sequential scan over batches instead of associative scan or vmap to avoid out-of-memory errors.
    Hv, _ = jax.lax.scan(f=hvp_fn, init=Hv_init, xs=xs)
    return Hv


def generate_random_tangent(
    key: Key[Array, ""], enc_params: PyTree[Float[Array, "..."]]
) -> PyTree[Float[Array, "..."]]:
    """Generates a standard normal tangent vector with the structure of `enc_params`.

    Args:
        key (Key[Array, ""]): JAX PRNG key.
        enc_params (PyTree[Float[Array, "..."]]): Floating encoder weights.

    Returns:
        PyTree[Float[Array, "..."]]: Random tangent with the same structure and dtypes as `enc_params`.
    """
    # Split PRNG keys for every leaf in the parameter PyTree.
    leaves, treedef = jax.tree.flatten(enc_params)
    key_leaves = jax.random.split(key, len(leaves))
    key_tree = jax.tree.unflatten(treedef, list(key_leaves))
    # Initialize a random tangent vector with matching structure and dtypes.
    tangent = jax.tree.map(
        lambda k, p: jax.random.normal(k, p.shape, dtype=p.dtype), key_tree, enc_params
    )
    return tangent


def calc_largest_eig_power_iter(
    hvp_fn: Callable[[PyTree[Float[Array, "..."]]], PyTree[Float[Array, "..."]]],
    tangent: PyTree[Float[Array, "..."]],
    *,
    n_iter: int,
    eps: float,
    tol: float = 0.0,
) -> tuple[Float[Array, ""], Float[Array, "n_iter"]]:
    """Estimates the dominant eigenvalue magnitude of a linear operator.

    Args:
        hvp_fn (Callable[[PyTree[Float[Array, "..."]]], PyTree[Float[Array, "..."]]]): Linear operator applied to
            PyTrees with the structure of `tangent`.
        tangent (PyTree[Float[Array, "..."]]): Initial vector for the power iteration.
        n_iter (int): Maximum number of power iteration steps.
        eps (float): Small epsilon for numerical stability.
        tol (float, optional): Stops early once the relative residual drops below
            `tol`. 0.0 always runs `n_iter` steps.
            Defaults to 0.0.

    Returns:
        tuple[Float[Array, ""], Float[Array, "n_iter"]]: (eig, errors) where eig is the scalar
        dominant eigenvalue magnitude and errors are the relative residuals
        `norm(Hv - eig * v) / norm(Hv)` of shape `(n_iter,)`. Entries after an
        early stop repeat the final residual, so `errors[-1]` is always the final
        residual.
    """

    # noinspection PyTypeHints
    def power_iter_step(
        tangent: PyTree[Float[Array, "..."]],
    ) -> tuple[
        PyTree[Float[Array, "..."]], tuple[Float[Array, ""], Float[Array, ""]]
    ]:
        # Normalize the tangent vector to unit norm.
        tangent_norm = tree_l2_normalize(tangent, eps=eps)
        Hv = hvp_fn(tangent_norm)
        # Estimate the current dominant eigenvalue using the Rayleigh quotient.
        eig = optax.tree_utils.tree_vdot(tangent_norm, Hv).real
        # Compute the residual PyTree: Hv - eig * v.
        residual = jax.tree.map(lambda hv, v: hv - eig * v, Hv, tangent_norm)
        # Relative residual normalized by ||Hv|| for comparability across theta and paths.
        residual_norm = optax.tree_utils.tree_norm(residual, ord=2)
        Hv_norm = optax.tree_utils.tree_norm(Hv, ord=2)
        error = residual_norm / jnp.maximum(Hv_norm, eps)
        return Hv, (eig, error)

    # Run power iteration for at most n_iter steps, stopping once the relative
    # residual drops below tol. A while loop is used instead of a scan since warmup
    # is never differentiated through.
    def cond_fn(state: tuple) -> Bool[Array, ""]:
        step_idx, _, _, error, _ = state
        return (step_idx < n_iter) & ~(error < tol)

    def body_fn(state: tuple) -> tuple:
        step_idx, tangent, _, _, errors = state
        Hv, (eig, error) = power_iter_step(tangent)
        errors = errors.at[step_idx].set(error)
        return step_idx + 1, Hv, eig, error, errors

    _, (eig_shape, error_shape) = jax.eval_shape(power_iter_step, tangent)
    init_state = (
        jnp.asarray(0),
        tangent,
        jnp.zeros((), eig_shape.dtype),
        jnp.asarray(jnp.inf, error_shape.dtype),
        jnp.zeros((n_iter,), error_shape.dtype),
    )
    n_steps, _, eig, error, errors = jax.lax.while_loop(cond_fn, body_fn, init_state)
    # Repeat the final residual after an early stop.
    errors = jnp.where(jnp.arange(n_iter) < n_steps, errors, error)
    eig = jnp.abs(eig)  # Return the eigenvalue magnitude.
    return eig, errors


def warmup_lc_hvp(
    loss: SCRAPLLoss,
    enc_fn: Callable[
        [PyTree[Float[Array, "..."]], Float[Array, "bs n_ch n_samples"]],
        Float[Array, "bs n_theta"],
    ],
    dec_fn: Callable[
        [Float[Array, "bs n_theta"]], Float[Array, "bs n_ch n_samples"]
    ],
    xs: Float[Array, "n_batches bs n_ch n_samples"],
    enc_params: PyTree[Float[Array, "..."]],
    *,
    key: Key[Array, ""],
    n_iter: int = 20,
    tol: float = 0.01,
    min_prob_frac: float = 0.0,
    eps: float = 1e-12,
    vmap_theta: bool = False,
) -> ThetaISResult:
    """Estimates theta-IS curvatures and sampling probabilities for every path.

    Must be called outside JIT. Each path is compiled separately and batches are
    evaluated sequentially to limit memory use. Batch curvature products are
    summed, matching the PyTorch implementation. A finite estimate need not have
    converged, so check `relative_residuals`. Does not modify `enc_params`, `loss`,
    or any sampling state.

    Args:
        loss (SCRAPLLoss): SCRAPL loss module.
        enc_fn (Callable[[PyTree[Float[Array, "..."]], Float[Array, "bs n_ch n_samples"]], Float[Array, "bs n_theta"]]):
            Deterministic encoder `enc_fn(enc_params, x)` mapping a waveform batch to `(batch, n_theta)`.
        dec_fn (Callable[[Float[Array, "bs n_theta"]], Float[Array, "bs n_ch n_samples"]]): Deterministic decoder
            `dec_fn(theta)` reconstructing the waveform batch.
        xs (Float[Array, "n_batches bs n_ch n_samples"]): Waveform batches of shape
            `(n_batches, batch, channels, samples)`.
        enc_params (PyTree): Encoder PyTree, such as a dict of arrays or an
            `eqx.Module`. Its floating-point arrays are treated as one group of
            weights and all other leaves are held fixed.
        key (Key[Array, ""]): JAX PRNG key for the initial power iteration vectors.
        n_iter (int, optional): Maximum number of power iteration steps.
            Defaults to 20.
        tol (float, optional): Stops each power iteration early once its relative
            residual drops below `tol`. 0.0 always runs `n_iter` steps. Early
            stopping saves no time with `vmap_theta=True`, since the vmapped loop
            runs until every theta has stopped.
            Defaults to 0.01.
        min_prob_frac (float, optional): Fraction of the probability mass that is
            distributed uniformly over all paths, in [0, 1).
            Defaults to 0.0.
        eps (float, optional): Small epsilon for numerical stability.
            Defaults to 1e-12.
        vmap_theta (bool, optional): If True, evaluates theta indices in parallel,
            which is faster but multiplies peak memory by `n_theta`. Otherwise
            evaluates them sequentially.
            Defaults to False.

    Returns:
        ThetaISResult: Curvature estimates, final relative residuals, and path
        sampling probabilities.

    Raises:
        AssertionError: If `n_iter` is not a positive integer.
        AssertionError: If `tol` is not finite and non-negative.
        AssertionError: If `min_prob_frac` is not in [0, 1).
        AssertionError: If `eps` is not finite and positive or not representable in
            the parameter dtypes.
        AssertionError: If `enc_params` contains no nonempty floating-point arrays,
            or any of them are not finite float32 or float64.
        AssertionError: If `xs` is not a nonempty finite float32 or float64 array of
            shape `(n_batches, batch, channels, samples)`.
        AssertionError: If `enc_fn` or `dec_fn` return invalid shapes, dtypes,
            or non-finite values, or are not deterministic.
        AssertionError: If any curvature estimate is not finite.
    """
    assert (
        isinstance(n_iter, Integral) and not isinstance(n_iter, bool) and n_iter > 0
    ), "n_iter must be a positive integer"
    assert math.isfinite(tol) and tol >= 0, "tol must be finite and non-negative"
    assert 0 <= min_prob_frac < 1, "min_prob_frac must be in [0, 1)"
    assert math.isfinite(eps) and eps > 0, "eps must be finite and positive"

    # Only floating-point arrays are differentiated; all other leaves are held fixed.
    diff_params, static_params = eqx.partition(enc_params, eqx.is_inexact_array)
    diff_params = jax.tree.map(jnp.asarray, diff_params)

    def diff_enc_fn(
        diff_params: PyTree[Float[Array, "..."]], x: Float[Array, "bs n_ch n_samples"]
    ) -> Float[Array, "bs n_theta"]:
        return enc_fn(eqx.combine(diff_params, static_params), x)

    leaves = jax.tree.leaves(diff_params)
    assert any(leaf.size for leaf in leaves), "enc_params must contain trainable arrays"
    assert all(
        leaf.dtype in (jnp.float32, jnp.float64) for leaf in leaves
    ), "enc_params must use float32 or float64"
    assert all(
        np.isfinite(np.asarray(leaf)).all() for leaf in leaves
    ), "enc_params must be finite"
    assert all(
        0 < np.asarray(eps, dtype=leaf.dtype) < np.inf for leaf in leaves
    ), "eps must be representable in the parameter dtypes"

    xs = jnp.asarray(xs)
    assert (
        xs.ndim == 4 and 0 not in xs.shape and xs.shape[-1] == loss.shape
    ), "xs must have nonempty (n_batches, batch, channels, samples) shape"
    assert xs.dtype in (jnp.float32, jnp.float64), "xs must use float32 or float64"
    assert np.isfinite(np.asarray(xs)).all(), "xs must be finite"

    theta = jnp.asarray(diff_enc_fn(diff_params, xs[0]))
    assert (
        theta.ndim == 2 and theta.shape[0] == xs.shape[1] and theta.shape[1] > 0
    ), "enc_fn must return (batch, n_theta) with at least one theta"
    assert theta.dtype in (
        jnp.float32,
        jnp.float64,
    ), "enc_fn must return float32 or float64"
    x = jnp.asarray(dec_fn(theta))
    assert (
        x.shape == xs.shape[1:]
    ), "dec_fn must reconstruct the waveform batch shape"
    assert (
        np.isfinite(np.asarray(theta)).all() and np.isfinite(np.asarray(x)).all()
    ), "enc_fn and dec_fn must return finite arrays"
    assert np.allclose(theta, diff_enc_fn(diff_params, xs[0])) and np.allclose(
        x, dec_fn(theta)
    ), "enc_fn and dec_fn must be deterministic during warmup"
    # Derive n_theta from the encoder output since out-of-range indices could be silently clamped.
    n_theta = theta.shape[1]
    log.info(
        f"Starting warmup_lc_hvp for {loss.n_paths} path(s), {n_theta} theta, "
        f"{xs.shape[0]} batch(es), {n_iter} iter, tol = {tol} (vmap_theta = {vmap_theta})"
    )

    def calc_path_eigs(
        enc_params: PyTree[Float[Array, "..."]],
        xs: Float[Array, "n_batches bs n_ch n_samples"],
        key: Key[Array, ""],
        *,
        path_idx: int,
    ) -> tuple[Float[Array, "n_theta"], Float[Array, "n_theta"]]:
        signal_loss_fn = partial(loss, path_idx=path_idx)
        theta_loss_fn = partial(calc_loss, dec_fn=dec_fn, loss_fn=signal_loss_fn)

        def calc_theta_eig(
            theta_idx: Int[Array, ""],
        ) -> tuple[Float[Array, ""], Float[Array, ""]]:
            key_theta = jax.random.fold_in(key, theta_idx)
            tangent = generate_random_tangent(key_theta, enc_params)
            hvp_fn = partial(
                calc_hvp_at_theta_multibatch,
                enc_params=enc_params,
                theta_idx=theta_idx,
                xs=xs,
                enc_fn=diff_enc_fn,
                theta_loss_fn=theta_loss_fn,
            )
            eig, errors = calc_largest_eig_power_iter(
                hvp_fn, tangent, n_iter=n_iter, eps=eps, tol=tol
            )
            final_error = errors[-1]
            return eig, final_error

        theta_indices = jnp.arange(n_theta)
        if vmap_theta:
            path_eigs, path_errors = jax.vmap(calc_theta_eig)(theta_indices)
        else:
            path_eigs, path_errors = jax.lax.map(calc_theta_eig, theta_indices)
        return path_eigs, path_errors

    compiled = jax.jit(calc_path_eigs, static_argnames=("path_idx",))
    curvatures, residuals = [], []
    for path_idx in tqdm(range(loss.n_paths), desc="Theta-IS warmup", unit="path"):
        key_path = jax.random.fold_in(key, path_idx)
        eigs, errors = compiled(diff_params, xs, key_path, path_idx=path_idx)
        assert (
            np.isfinite(np.asarray(eigs)).all()
            and np.isfinite(np.asarray(errors)).all()
        ), f"Non-finite curvature estimates for path {path_idx}"
        curvatures.append(eigs)
        residuals.append(errors)
        log.debug(f"Theta-IS warmup: path {path_idx + 1}/{loss.n_paths} complete")
    curvatures = jnp.stack(curvatures)
    relative_residuals = jnp.stack(residuals)
    probs = theta_importance_probs(curvatures, min_prob_frac=min_prob_frac, eps=eps)
    result = ThetaISResult(
        curvatures=curvatures,
        relative_residuals=relative_residuals,
        probs=probs,
    )
    return result
