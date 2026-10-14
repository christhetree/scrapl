import logging
import math
import os
import re
from functools import partial
from numbers import Integral
from typing import Any, Callable, NamedTuple

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
        curvatures (Float[Array, "n_paths n_theta"]): Curvature estimates of shape
            `(n_paths, n_theta)`.
        min_prob_frac (float, optional): Fraction of the probability mass that is
            distributed uniformly over all paths, in [0, 1).
            Defaults to 0.0.
        eps (float, optional): Lower bound applied to curvatures before
            normalizing them.
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
    # A single scalar so that any invalid entry invalidates every probability.
    is_valid = jnp.all(jnp.isfinite(curvatures) & (curvatures >= 0))
    # Clamp so that zero curvatures still get a small nonzero probability.
    max_curvatures = jnp.maximum(curvatures, eps)
    # Normalize over paths for each theta.
    theta_probs = max_curvatures / max_curvatures.sum(axis=0)
    # Average over theta.
    avg_probs = theta_probs.mean(axis=1)
    # Mix with the uniform distribution to enforce min_prob_frac.
    unif_prob = 1.0 / curvatures.shape[0]
    probs = (1.0 - min_prob_frac) * avg_probs + min_prob_frac * unif_prob
    # Use NaN instead of raising since this may be called inside JIT.
    valid_probs = jnp.where(is_valid, probs, jnp.nan)
    return valid_probs


def load_warmup_from_dir(
    loss: SCRAPLLoss,
    warmup_dir: str,
    *,
    min_prob_frac: float = 0.0,
    eps: float = 1e-12,
) -> ThetaISResult:
    """Loads a theta-IS warmup saved by one or more `warmup_lc_hvp` runs.

    Combines the per-path results written to `warmup_dir` by runs over disjoint
    [`start_path_idx`, `end_path_idx`) ranges and recomputes the path sampling
    probabilities from them.

    Args:
        loss (SCRAPLLoss): SCRAPL loss module the warmup was run with.
        warmup_dir (str): Directory passed as `save_dir` to `warmup_lc_hvp`.
        min_prob_frac (float, optional): Fraction of the probability mass that is
            distributed uniformly over all paths, in [0, 1).
            Defaults to 0.0.
        eps (float, optional): Lower bound applied to curvatures before
            normalizing them.
            Defaults to 1e-12.

    Returns:
        ThetaISResult: Curvature estimates, final relative residuals, and path
        sampling probabilities for every path.

    Raises:
        AssertionError: If `warmup_dir` does not exist, is missing results for
            any path, or has results for path indices >= `loss.n_paths`.
        AssertionError: If any saved curvature estimate is not finite.
        AssertionError: If any saved curvature estimate is negative.
        AssertionError: If the saved results do not all have the same shape.
    """
    assert os.path.isdir(warmup_dir), f"{warmup_dir} is not a directory"
    vals_paths = [
        os.path.join(warmup_dir, f"vals_{path_idx}.npz")
        for path_idx in range(loss.n_paths)
    ]
    missing = [
        path_idx
        for path_idx, vals_path in enumerate(vals_paths)
        if not os.path.isfile(vals_path)
    ]
    assert not missing, f"Warmup results are missing for path indices {missing}"
    # Extra results mean the warmup was run with a loss that has more paths.
    extra = sorted(
        int(match.group(1))
        for name in os.listdir(warmup_dir)
        if (match := re.fullmatch(r"vals_(\d+)\.npz", name))
        and int(match.group(1)) >= loss.n_paths
    )
    assert not extra, (
        f"Warmup results exist for path indices {extra} outside "
        f"[0, {loss.n_paths}); the warmup was run with a different loss"
    )
    curvatures, relative_residuals = [], []
    for path_idx, vals_path in enumerate(vals_paths):
        with np.load(vals_path) as vals:
            eigs, errors = vals["curvatures"], vals["relative_residuals"]
        assert (
            np.isfinite(eigs).all() and np.isfinite(errors).all()
        ), f"Non-finite curvature estimates for path {path_idx}"
        assert (eigs >= 0).all(), f"Negative curvature estimates for path {path_idx}"
        curvatures.append(eigs)
        relative_residuals.append(errors)
    assert (
        len({c.shape for c in curvatures + relative_residuals}) == 1
    ), "Saved warmup results must all have the same (n_theta,) shape"
    curvatures = jnp.asarray(np.stack(curvatures))
    relative_residuals = jnp.asarray(np.stack(relative_residuals))
    probs = theta_importance_probs(curvatures, min_prob_frac=min_prob_frac, eps=eps)
    log.info(
        f"Loaded theta-IS warmup from directory:\n\t{os.path.abspath(warmup_dir)}\n\t"
        f"min prob = {float(probs.min()):.6f}, max prob = {float(probs.max()):.6f}"
    )
    result = ThetaISResult(
        curvatures=curvatures,
        relative_residuals=relative_residuals,
        probs=probs,
    )
    return result


def calc_loss(
    theta: Float[Array, "bs n_theta"],
    x_target: Float[Array, "bs n_ch n_samples"],
    *,
    dec_fn: Callable[..., Float[Array, "bs n_ch n_samples"]],
    loss_fn: Callable[
        [Float[Array, "bs n_ch n_samples"], Float[Array, "bs n_ch n_samples"]],
        Float[Array, ""],
    ],
    dec_kwargs: dict[str, Any] | None = None,
) -> Float[Array, ""]:
    """Decodes theta and computes the loss against the target waveform.

    Args:
        theta (Float[Array, "bs n_theta"]): Theta of shape `(batch, n_theta)`.
        x_target (Float[Array, "bs n_ch n_samples"]): Target waveform batch of shape
            `(batch, channels, samples)`.
        dec_fn (Callable[..., Float[Array, "bs n_ch n_samples"]]): Decoder
            `dec_fn(theta, **dec_kwargs)`.
        loss_fn (Callable[[Float[Array, "bs n_ch n_samples"],
            Float[Array, "bs n_ch n_samples"]], Float[Array, ""]]): Loss function
            `loss_fn(x, x_target)` returning a scalar.
        dec_kwargs (dict[str, Any] | None, optional): Extra keyword arguments for
            `dec_fn`.
            Defaults to None.

    Returns:
        Float[Array, ""]: The scalar loss value.
    """
    if dec_kwargs is None:
        dec_kwargs = {}
    x = dec_fn(theta, **dec_kwargs)
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
        x_target (Float[Array, "bs n_ch n_samples"]): Target waveform batch of shape
            `(batch, channels, samples)`.
        theta_loss_fn (Callable[[Float[Array, "bs n_theta"],
            Float[Array, "bs n_ch n_samples"]], Float[Array, ""]]): Loss function
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
        enc_fn (Callable[[PyTree[Float[Array, "..."]],
            Float[Array, "bs n_ch n_samples"]], Float[Array, "bs n_theta"]]):
            Encoder function `enc_fn(enc_params, x)`.
        theta_loss_fn (Callable[[Float[Array, "bs n_theta"],
            Float[Array, "bs n_ch n_samples"]], Float[Array, ""]]): Loss function
            `theta_loss_fn(theta, x_target)` returning a scalar.

    Returns:
        PyTree[Float[Array, "..."]]: Gradient PyTree with the same structure as
        `enc_params`.
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
    batch: tuple[Float[Array, "bs n_ch n_samples"], dict[str, Any], dict[str, Any]],
    *,
    enc_params: PyTree[Float[Array, "..."]],
    theta_idx: Int[Array, ""] | int,
    tangent: PyTree[Float[Array, "..."]],
    enc_fn: Callable[..., Float[Array, "bs n_theta"]],
    theta_loss_fn: Callable[..., Float[Array, ""]],
) -> tuple[PyTree[Float[Array, "..."]], None]:
    """Adds the curvature-vector product of a single batch to a running sum.

    Used as a `jax.lax.scan` step over batches.

    Args:
        carry (PyTree[Float[Array, "..."]]): Running sum of curvature-vector products.
        batch (tuple[Float[Array, "bs n_ch n_samples"], dict[str, Any],
            dict[str, Any]]):
            `(x_target, enc_kwargs, dec_kwargs)` for one batch, where `x_target`
            has shape `(batch, channels, samples)`.
        enc_params (PyTree[Float[Array, "..."]]): Floating encoder weights.
        theta_idx (Int[Array, ""] | int): Theta index in [0, n_theta).
        tangent (PyTree[Float[Array, "..."]]): Tangent vector with the same structure
            as `enc_params`.
        enc_fn (Callable[..., Float[Array, "bs n_theta"]]): Encoder function
            `enc_fn(enc_params, x, **enc_kwargs)`.
        theta_loss_fn (Callable[..., Float[Array, ""]]): Loss function
            `theta_loss_fn(theta, x_target, dec_kwargs=dec_kwargs)` returning a
            scalar.

    Returns:
        tuple[PyTree[Float[Array, "..."]], None]: Updated running sum and no stacked
        output.
    """
    x_target, enc_kwargs, dec_kwargs = batch
    # Bind this batch's extra arguments so the encoder and loss take two arguments.
    s_dE_dw_fn = partial(
        calc_s_dE_dw_at_theta,
        theta_idx=theta_idx,
        x_target=x_target,
        enc_fn=lambda p, x: enc_fn(p, x, **enc_kwargs),
        theta_loss_fn=partial(theta_loss_fn, dec_kwargs=dec_kwargs),
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
    enc_fn: Callable[..., Float[Array, "bs n_theta"]],
    theta_loss_fn: Callable[..., Float[Array, ""]],
    enc_kwargs: dict[str, Any] | None = None,
    dec_kwargs: dict[str, Any] | None = None,
) -> PyTree[Float[Array, "..."]]:
    """Sums the curvature-vector products over batches, matching PyTorch.

    Args:
        tangent (PyTree[Float[Array, "..."]]): Tangent vector with the same structure
            as `enc_params`.
        xs (Float[Array, "n_batches bs n_ch n_samples"]): Waveform batches of shape
            `(n_batches, batch, channels, samples)`.
        enc_params (PyTree[Float[Array, "..."]]): Floating encoder weights.
        theta_idx (Int[Array, ""] | int): Theta index in [0, n_theta).
        enc_fn (Callable[..., Float[Array, "bs n_theta"]]): Encoder function
            `enc_fn(enc_params, x, **enc_kwargs)`.
        theta_loss_fn (Callable[..., Float[Array, ""]]): Loss function
            `theta_loss_fn(theta, x_target, dec_kwargs=dec_kwargs)` returning a
            scalar, such as `calc_loss` with `dec_fn` and `loss_fn` bound.
        enc_kwargs (dict[str, Any] | None, optional): Extra keyword arguments for
            `enc_fn` whose array leaves have a leading `n_batches` axis.
            Defaults to None.
        dec_kwargs (dict[str, Any] | None, optional): Extra keyword arguments for
            the decoder whose array leaves have a leading `n_batches` axis.
            Defaults to None.

    Returns:
        PyTree[Float[Array, "..."]]: Summed curvature-vector product with the same
        structure as `enc_params`.
    """
    if enc_kwargs is None:
        enc_kwargs = {}
    if dec_kwargs is None:
        dec_kwargs = {}
    hvp_fn = partial(
        calc_hvp_at_theta,
        enc_params=enc_params,
        theta_idx=theta_idx,
        tangent=tangent,
        enc_fn=enc_fn,
        theta_loss_fn=theta_loss_fn,
    )
    Hv_init = jax.tree.map(jnp.zeros_like, tangent)
    # Use sequential scan over batches instead of associative scan or vmap to avoid
    # out-of-memory errors.
    Hv, _ = jax.lax.scan(f=hvp_fn, init=Hv_init, xs=(xs, enc_kwargs, dec_kwargs))
    return Hv


def generate_random_tangent(
    key: Key[Array, ""], enc_params: PyTree[Float[Array, "..."]]
) -> PyTree[Float[Array, "..."]]:
    """Generates a standard normal tangent vector with the structure of `enc_params`.

    Args:
        key (Key[Array, ""]): JAX PRNG key.
        enc_params (PyTree[Float[Array, "..."]]): Floating encoder weights.

    Returns:
        PyTree[Float[Array, "..."]]: Random tangent with the same structure and
        dtypes as `enc_params`.
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


# noinspection PyTypeHints
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
        hvp_fn (Callable[[PyTree[Float[Array, "..."]]], PyTree[Float[Array, "..."]]]):
            Linear operator applied to PyTrees with the structure of `tangent`.
        tangent (PyTree[Float[Array, "..."]]): Initial vector for the power iteration.
        n_iter (int): Maximum number of power iteration steps.
        eps (float): Small epsilon for numerical stability.
        tol (float, optional): Stops early once the relative residual drops below
            `tol`. 0.0 always runs `n_iter` steps.
            Defaults to 0.0.

    Returns:
        tuple[Float[Array, ""], Float[Array, "n_iter"]]: (eig, errors) where eig is
        the scalar dominant eigenvalue magnitude and errors are the relative
        residuals `norm(Hv - eig * v) / norm(Hv)` of shape `(n_iter,)`. Entries
        after an early stop repeat the final residual, so `errors[-1]` is always
        the final residual.
    """

    # noinspection PyTypeHints
    def power_iter_step(
        tangent: PyTree[Float[Array, "..."]],
    ) -> tuple[PyTree[Float[Array, "..."]], tuple[Float[Array, ""], Float[Array, ""]]]:
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

    # The eigenvalue and residual use the dtype promoted across all tangent leaves.
    dtype = jnp.result_type(*jax.tree.leaves(tangent))
    init_state = (
        jnp.asarray(0),
        tangent,
        jnp.zeros((), dtype),
        jnp.asarray(jnp.inf, dtype),
        jnp.zeros((n_iter,), dtype),
    )
    n_steps, _, eig, error, errors = jax.lax.while_loop(cond_fn, body_fn, init_state)
    # Repeat the final residual after an early stop.
    errors = jnp.where(jnp.arange(n_iter) < n_steps, errors, error)
    eig = jnp.abs(eig)  # Return the eigenvalue magnitude.
    return eig, errors


def warmup_lc_hvp(
    loss: SCRAPLLoss,
    enc_fn: Callable[..., Float[Array, "bs n_theta"]],
    dec_fn: Callable[..., Float[Array, "bs n_ch n_samples"]],
    xs: Float[Array, "n_batches bs n_ch n_samples"],
    enc_params: PyTree[Float[Array, "..."]],
    enc_kwargs: dict[str, Any] | None = None,
    dec_kwargs: dict[str, Any] | None = None,
    *,
    key: Key[Array, ""],
    start_path_idx: int = 0,
    end_path_idx: int | None = None,
    save_dir: str | None = None,
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

    Arguments that are the same for every batch, such as a sample rate, should be
    bound into `enc_fn` or `dec_fn` with a closure or `functools.partial`.
    Arguments that change per batch are passed through `enc_kwargs` and
    `dec_kwargs` as arrays stacked along a leading `n_batches` axis, mirroring
    `theta_fn_kwargs` and `synth_fn_kwargs` in the PyTorch implementation. A list of
    per-batch dicts can be stacked with
    `jax.tree.map(lambda *a: jnp.stack(a), *kwargs_list)`.

    **Parallelization:** Warmup can be split across processes or devices by
    assigning each one a disjoint [`start_path_idx`, `end_path_idx`) range and the
    same `key` and `save_dir`. Each path's PRNG key is derived from `key` and the
    path index, so split runs give the same estimates as a single run. After all
    runs finish, load the combined result with `load_warmup_from_dir`.

    Args:
        loss (SCRAPLLoss): SCRAPL loss module.
        enc_fn (Callable[..., Float[Array, "bs n_theta"]]): Deterministic encoder
            `enc_fn(enc_params, x, **enc_kwargs)` mapping a waveform batch to
            `(batch, n_theta)`. Any non-trainable encoder state, such as the static
            part of a partitioned `eqx.Module`, should be closed over.
        dec_fn (Callable[..., Float[Array, "bs n_ch n_samples"]]): Deterministic
            decoder `dec_fn(theta, **dec_kwargs)` reconstructing the waveform batch.
        xs (Float[Array, "n_batches bs n_ch n_samples"]): Waveform batches of shape
            `(n_batches, batch, channels, samples)`.
        enc_params (PyTree[Float[Array, "..."]]): Trainable encoder weights whose
            leaves are all floating-point arrays, treated as one group of weights.
        enc_kwargs (dict[str, Any] | None, optional): Per-batch keyword arguments
            for `enc_fn`. Every leaf is an array with a leading `n_batches` axis,
            and batch `b` receives the slices `[b]`.
            Defaults to None.
        dec_kwargs (dict[str, Any] | None, optional): Per-batch keyword arguments
            for `dec_fn`, stacked the same way as `enc_kwargs`.
            Defaults to None.
        key (Key[Array, ""]): JAX PRNG key for the initial power iteration vectors.
        start_path_idx (int, optional): First path index to compute.
            Defaults to 0.
        end_path_idx (int | None, optional): Path index to stop at (exclusive). If
            None, defaults to `loss.n_paths`.
            Defaults to None.
        save_dir (str | None, optional): Directory where each path's curvatures and
            relative residuals are saved as `vals_{path_idx}.npz`, and the final
            probabilities as `probs.npy` when every path was computed. If None,
            nothing is saved.
            Defaults to None.
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
        sampling probabilities. Rows for paths outside [`start_path_idx`,
        `end_path_idx`) are NaN, so the probabilities are NaN unless every path was
        computed.

    Raises:
        AssertionError: If `n_iter` is not a positive integer.
        AssertionError: If `tol` is not finite and non-negative.
        AssertionError: If [`start_path_idx`, `end_path_idx`) is not a nonempty
            range within [0, `loss.n_paths`].
        AssertionError: If `save_dir` already contains results for a path being
            computed.
        AssertionError: If `min_prob_frac` is not in [0, 1).
        AssertionError: If `eps` is not finite and positive or not representable in
            the parameter dtypes.
        AssertionError: If `enc_params` contains no nonempty arrays, or any of its
            leaves are not finite float32 or float64 arrays.
        AssertionError: If `xs` is not a nonempty finite float32 or float64 array of
            shape `(n_batches, batch, channels, samples)`.
        AssertionError: If `enc_kwargs` or `dec_kwargs` has a leaf without a leading
            `n_batches` axis.
        AssertionError: If `enc_fn` or `dec_fn` return invalid shapes, dtypes,
            or non-finite values, or are not deterministic.
        AssertionError: If any curvature estimate is not finite.
    """
    assert (
        isinstance(n_iter, Integral) and not isinstance(n_iter, bool) and n_iter > 0
    ), "n_iter must be a positive integer"
    assert math.isfinite(tol) and tol >= 0, "tol must be finite and non-negative"
    if end_path_idx is None:
        end_path_idx = loss.n_paths
    assert (
        0 <= start_path_idx < end_path_idx <= loss.n_paths
    ), (f"Path range [{start_path_idx}, {end_path_idx}) must be nonempty and within "
        f"[0, {loss.n_paths}]")
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        for path_idx in range(start_path_idx, end_path_idx):
            save_path = os.path.join(save_dir, f"vals_{path_idx}.npz")
            assert not os.path.isfile(save_path), f"{save_path} already exists"
    assert 0 <= min_prob_frac < 1, "min_prob_frac must be in [0, 1)"
    assert math.isfinite(eps) and eps > 0, "eps must be finite and positive"

    leaves = jax.tree.leaves(enc_params)
    assert all(
        isinstance(leaf, (jax.Array, np.ndarray)) for leaf in leaves
    ), "enc_params leaves must all be arrays; close over static state in enc_fn"
    enc_params = jax.tree.map(jnp.asarray, enc_params)
    leaves = jax.tree.leaves(enc_params)
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

    n_batches = xs.shape[0]
    enc_kwargs = {} if enc_kwargs is None else jax.tree.map(jnp.asarray, enc_kwargs)
    dec_kwargs = {} if dec_kwargs is None else jax.tree.map(jnp.asarray, dec_kwargs)
    for name, kwargs in (("enc_kwargs", enc_kwargs), ("dec_kwargs", dec_kwargs)):
        assert all(
            leaf.ndim > 0 and leaf.shape[0] == n_batches
            for leaf in jax.tree.leaves(kwargs)
        ), f"{name} leaves must have a leading n_batches = {n_batches} axis"
    enc_kwargs_0 = jax.tree.map(lambda leaf: leaf[0], enc_kwargs)
    dec_kwargs_0 = jax.tree.map(lambda leaf: leaf[0], dec_kwargs)

    theta = jnp.asarray(enc_fn(enc_params, xs[0], **enc_kwargs_0))
    assert (
        theta.ndim == 2 and theta.shape[0] == xs.shape[1] and theta.shape[1] > 0
    ), "enc_fn must return (batch, n_theta) with at least one theta"
    assert theta.dtype in (
        jnp.float32,
        jnp.float64,
    ), "enc_fn must return float32 or float64"
    x = jnp.asarray(dec_fn(theta, **dec_kwargs_0))
    assert x.shape == xs.shape[1:], "dec_fn must reconstruct the waveform batch shape"
    assert (
        np.isfinite(np.asarray(theta)).all() and np.isfinite(np.asarray(x)).all()
    ), "enc_fn and dec_fn must return finite arrays"
    assert np.allclose(
        theta, enc_fn(enc_params, xs[0], **enc_kwargs_0)
    ) and np.allclose(
        x, dec_fn(theta, **dec_kwargs_0)
    ), "enc_fn and dec_fn must be deterministic during warmup"
    # Derive n_theta from the encoder output since out-of-range indices could be
    # silently clamped.
    n_theta = theta.shape[1]
    log.info(
        f"Starting warmup_lc_hvp for paths [{start_path_idx}, {end_path_idx}) of "
        f"{loss.n_paths}, {n_theta} theta, "
        f"{xs.shape[0]} batch(es), {n_iter} iter, tol = {tol} (vmap_theta = {vmap_theta})"
    )

    def calc_path_eigs(
        enc_params: PyTree[Float[Array, "..."]],
        xs: Float[Array, "n_batches bs n_ch n_samples"],
        enc_kwargs: dict[str, Any],
        dec_kwargs: dict[str, Any],
        key: Key[Array, ""],
        *,
        path_idx: int,
    ) -> tuple[Float[Array, "n_theta"], Float[Array, "n_theta"]]:
        audio_loss_fn = partial(loss, path_idx=path_idx)
        theta_loss_fn = partial(calc_loss, dec_fn=dec_fn, loss_fn=audio_loss_fn)

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
                enc_fn=enc_fn,
                theta_loss_fn=theta_loss_fn,
                enc_kwargs=enc_kwargs,
                dec_kwargs=dec_kwargs,
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
    # Paths outside [start_path_idx, end_path_idx) stay NaN.
    curvatures = np.full((loss.n_paths, n_theta), np.nan, dtype=theta.dtype)
    relative_residuals = np.full((loss.n_paths, n_theta), np.nan, dtype=theta.dtype)
    path_indices = range(start_path_idx, end_path_idx)
    for path_idx in tqdm(path_indices, desc="Theta-IS warmup", unit="path"):
        key_path = jax.random.fold_in(key, path_idx)
        eigs, errors = compiled(
            enc_params, xs, enc_kwargs, dec_kwargs, key_path, path_idx=path_idx
        )
        eigs, errors = np.asarray(eigs), np.asarray(errors)
        assert (
            np.isfinite(eigs).all() and np.isfinite(errors).all()
        ), f"Non-finite curvature estimates for path {path_idx}"
        curvatures[path_idx] = eigs
        relative_residuals[path_idx] = errors
        # Save intermediate results for parallel computation
        if save_dir is not None:
            # Write to a temporary file and rename it so that an interrupted run
            # never leaves a partial file behind.
            save_path = os.path.join(save_dir, f"vals_{path_idx}.npz")
            tmp_path = f"{save_path}.tmp"
            with open(tmp_path, "wb") as f:
                np.savez(f, curvatures=eigs, relative_residuals=errors)
            os.replace(tmp_path, save_path)
        log.debug(f"Theta-IS warmup: path {path_idx + 1}/{loss.n_paths} complete")
    curvatures = jnp.asarray(curvatures)
    relative_residuals = jnp.asarray(relative_residuals)
    probs = theta_importance_probs(curvatures, min_prob_frac=min_prob_frac, eps=eps)
    # Only save the final probs if every path was computed
    if save_dir is not None and start_path_idx == 0 and end_path_idx == loss.n_paths:
        probs_path = os.path.join(save_dir, "probs.npy")
        log.info(
            f"Saving theta-IS warmup probabilities to {os.path.abspath(probs_path)}"
        )
        np.save(probs_path, np.asarray(probs))
    result = ThetaISResult(
        curvatures=curvatures,
        relative_residuals=relative_residuals,
        probs=probs,
    )
    return result
