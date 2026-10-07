import logging
import math
import os
from typing import Any, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import optax

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def scale_by_gradient_multiplier(
    grad_mult: float = 1.0,
) -> optax.GradientTransformationExtraArgs:
    """Scales gradients by a constant multiplier factor.

    Applied to gradients before variance normalization or optimizer updates to prevent
    underflow and numerical precision issues when squaring gradient values in JTFS.

    Args:
        grad_mult (float, optional): Gradient multiplier factor.
            Defaults to 1.0.

    Returns:
        optax.GradientTransformationExtraArgs: An Optax gradient transformation.

    Raises:
        AssertionError: If `grad_mult` is not finite and positive.
    """
    assert (
        math.isfinite(grad_mult) and grad_mult > 0
    ), "grad_mult must be finite and positive"

    def init_fn(params: optax.Params) -> optax.EmptyState:
        return optax.EmptyState()

    def update_fn(
        updates: optax.Updates,
        state: optax.EmptyState,
        params: optax.Params | None = None,
        **extra_args: Any,
    ) -> tuple[optax.Updates, optax.EmptyState]:
        scaled_grads = jax.tree.map(lambda g: g * grad_mult, updates)
        return scaled_grads, state

    return optax.GradientTransformationExtraArgs(init=init_fn, update=update_fn)


def adam_grad_norm_cont(
    grad: jax.Array,
    prev_m: jax.Array,
    prev_v: jax.Array,
    t: jax.Array | float,
    prev_t: jax.Array | float,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Computes continuous time-step Adam gradient normalization per path.

    Args:
        grad (jax.Array): Current path gradient tensor.
        prev_m (jax.Array): Previous first moment estimate for the path.
        prev_v (jax.Array): Previous second moment estimate for the path.
        t (jax.Array | float): Normalized current time-step (curr_t / n_paths).
        prev_t (jax.Array | float): Normalized previous time-step (prev_t / n_paths).
        b1 (float, optional): Beta 1 decay factor. Defaults to 0.9.
        b2 (float, optional): Beta 2 decay factor. Defaults to 0.999.
        eps (float, optional): Small epsilon for numerical stability. Defaults to 1e-8.

    Returns:
        tuple[jax.Array, jax.Array, jax.Array]: (grad_hat, m, v) where grad_hat is
        normalized gradient, and m, v are updated moments.
    """
    delta_t = t - prev_t
    eff_b1 = b1**delta_t
    eff_b2 = b2**delta_t
    m = eff_b1 * prev_m + (1.0 - eff_b1) * grad
    v = eff_b2 * prev_v + (1.0 - eff_b2) * (grad**2)
    m_hat = m / (1.0 - b1**t)
    v_hat = v / (1.0 - b2**t)
    grad_hat = m_hat / (jnp.sqrt(v_hat) + eps)
    return grad_hat, m, v


class PAdamState(NamedTuple):
    """Completed update count, path timestamps, and per-path moment PyTrees."""

    scrapl_t: jax.Array
    prev_t_s: jax.Array
    prev_m_s: optax.Updates
    prev_v_s: optax.Updates


def p_adam(
    n_paths: int,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
) -> optax.GradientTransformationExtraArgs:
    """Pathwise Adam (P-Adam) gradient normalization transformation.

    Computes running first and second moments of gradients per scattering path
    to normalize gradients across stochastic path choices.

    Args:
        n_paths (int): Total number of scattering paths.
        b1 (float, optional): Beta 1 decay hyperparameter in [0, 1).
            Defaults to 0.9.
        b2 (float, optional): Beta 2 decay hyperparameter in [0, 1).
            Defaults to 0.999.
        eps (float, optional): Small epsilon for numerical stability.
            Defaults to 1e-8.

    Returns:
        optax.GradientTransformationExtraArgs: An Optax gradient transformation with
        custom path_idx argument support.

    Raises:
        AssertionError: If `n_paths` is not a positive integer.
        AssertionError: If `b1` or `b2` is not in [0, 1).
        AssertionError: If `eps` is not finite and positive.
    """
    assert 0 < n_paths, "n_paths must be a positive integer"
    assert 0 <= b1 < 1 and 0 <= b2 < 1, "b1 and b2 must be in [0, 1)"
    assert math.isfinite(eps) and eps > 0, "eps must be finite and positive"

    def init_fn(params: optax.Params) -> PAdamState:
        """Initializes the PAdamState with zeroed moment arrays.

        Args:
            params (optax.Params): Model parameter PyTree.

        Returns:
            PAdamState: Initial state with moments allocated for all paths.
        """
        params = eqx.filter(params, eqx.is_inexact_array)
        scrapl_t = jnp.asarray(0, dtype=jnp.int32)
        prev_t_s = jnp.zeros(n_paths, dtype=jnp.int32)
        prev_m_s = jax.tree.map(
            lambda p: jnp.zeros((n_paths, *p.shape), dtype=p.dtype), params
        )
        prev_v_s = jax.tree.map(jnp.zeros_like, prev_m_s)
        return PAdamState(
            scrapl_t=scrapl_t,
            prev_t_s=prev_t_s,
            prev_m_s=prev_m_s,
            prev_v_s=prev_v_s,
        )

    def update_fn(
        updates: optax.Updates,
        state: PAdamState,
        params: optax.Params | None = None,
        *,
        path_idx: int,
    ) -> tuple[optax.Updates, PAdamState]:
        """Normalizes path gradients and returns updated P-Adam state.

        Args:
            updates (optax.Updates): Incoming gradients PyTree.
            state (PAdamState): Current P-Adam optimizer state.
            params (optax.Params | None, optional): Model parameters.
                Defaults to None.
            path_idx (int): Path index used for the current step.

        Returns:
            tuple[optax.Updates, PAdamState]: Normalized gradients and updated state.

        Raises:
            AssertionError: If `path_idx` is out of range [0, n_paths).
        """
        assert 0 <= path_idx < n_paths, f"path_idx must be in [0, {n_paths})"
        path_idx = jnp.asarray(path_idx, dtype=jnp.int32)
        # TODO(cm): Look into this
        new_scrapl_t = state.scrapl_t + 1  # Incremented in forward() in Torch
        curr_t = new_scrapl_t + 1
        prev_t = state.prev_t_s[path_idx]
        t_norm = curr_t.astype(jnp.float32) / n_paths
        prev_t_norm = prev_t.astype(jnp.float32) / n_paths

        def _update_leaf(
            grad: jax.Array, prev_m_s: jax.Array, prev_v_s: jax.Array
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            prev_m = prev_m_s[path_idx]
            prev_v = prev_v_s[path_idx]
            grad_hat, m, v = adam_grad_norm_cont(
                grad,
                prev_m,
                prev_v,
                t_norm,
                prev_t_norm,
                b1=b1,
                b2=b2,
                eps=eps,
            )
            return grad_hat, prev_m_s.at[path_idx].set(m), prev_v_s.at[path_idx].set(v)

        # Map _update_leaf over parameter leaves to obtain a PyTree where each leaf
        # is a 3-tuple (grad_hat, new_prev_m_s_leaf, new_prev_v_s_leaf).
        results = jax.tree.map(_update_leaf, updates, state.prev_m_s, state.prev_v_s)

        # Transpose Tree[Tuple[grad, m_s, v_s]] -> Tuple[Tree[grad], Tree[m_s], Tree[v_s]]
        # outer_def defines the container structure of the model updates, and inner_def
        # defines the 3-element tuple structure at each leaf.
        outer_def = jax.tree.structure(updates)
        inner_def = jax.tree.structure((0, 0, 0))
        normalized_grads, new_prev_m_s, new_prev_v_s = jax.tree.transpose(
            outer_def, inner_def, results
        )

        new_prev_t_s = state.prev_t_s.at[path_idx].set(curr_t)

        new_state = PAdamState(
            scrapl_t=new_scrapl_t,
            prev_t_s=new_prev_t_s,
            prev_m_s=new_prev_m_s,
            prev_v_s=new_prev_v_s,
        )
        return normalized_grads, new_state

    return optax.GradientTransformationExtraArgs(init=init_fn, update=update_fn)


class PSAGAState(NamedTuple):
    """Path visit counts and the most recent input gradient for every path."""

    path_counts: jax.Array
    prev_path_grads: optax.Updates


def p_saga(n_paths: int) -> optax.GradientTransformationExtraArgs:
    """Pathwise SAGA (P-SAGA) gradient correction transformation.

    Maintains historical gradients across visited paths and computes variance-reduced
    gradient corrections.

    Args:
        n_paths (int): Total number of scattering paths.

    Returns:
        optax.GradientTransformationExtraArgs: An Optax gradient transformation with
        custom path_idx argument support.

    Raises:
        AssertionError: If `n_paths` is not a positive integer.
    """
    assert 0 < n_paths, "n_paths must be a positive integer"

    def init_fn(params: optax.Params) -> PSAGAState:
        """Initializes the PSAGAState with zeroed visit counts and zeroed gradients.

        Args:
            params (optax.Params): Model parameter PyTree.

        Returns:
            PSAGAState: Initial state with gradient history allocated for all paths.
        """
        params = eqx.filter(params, eqx.is_inexact_array)
        path_counts = jnp.zeros(n_paths, dtype=jnp.int32)
        prev_path_grads = jax.tree.map(
            lambda p: jnp.zeros((n_paths, *p.shape), dtype=p.dtype), params
        )
        return PSAGAState(path_counts=path_counts, prev_path_grads=prev_path_grads)

    def update_fn(
        updates: optax.Updates,
        state: PSAGAState,
        params: optax.Params | None = None,
        *,
        path_idx: int,
    ) -> tuple[optax.Updates, PSAGAState]:
        """Corrects gradients using historical path gradients and returns new state.

        Args:
            updates (optax.Updates): Incoming gradients PyTree.
            state (PSAGAState): Current P-SAGA optimizer state.
            params (optax.Params | None, optional): Model parameters.
                Defaults to None.
            path_idx (int): Path index used for the current step.

        Returns:
            tuple[optax.Updates, PSAGAState]: Corrected gradients and updated state.

        Raises:
            AssertionError: If `path_idx` is out of range [0, n_paths).
        """
        assert 0 <= path_idx < n_paths, f"path_idx must be in [0, {n_paths})"
        path_idx = jnp.asarray(path_idx, dtype=jnp.int32)
        path_counts = state.path_counts.at[path_idx].add(1)
        n_paths_seen = jnp.sum(path_counts > 0, dtype=jnp.int32)
        denominator = jnp.maximum(1, n_paths_seen - 1)

        def _update_leaf(
            grad: jax.Array, prev_path_grads: jax.Array
        ) -> tuple[jax.Array, jax.Array]:
            prev_avg_grad = prev_path_grads.sum(axis=0) / denominator.astype(grad.dtype)
            prev_path_grad = prev_path_grads[path_idx]
            saga_grad = grad - prev_path_grad + prev_avg_grad
            prev_path_grads = prev_path_grads.at[path_idx].set(grad)
            return saga_grad, prev_path_grads

        # Map _update_leaf over parameter leaves to obtain a PyTree where each leaf
        # is a 2-tuple (saga_grad, new_prev_path_grads_leaf).
        results = jax.tree.map(_update_leaf, updates, state.prev_path_grads)

        # Transpose Tree[Tuple[saga_grad, prev_path_grads]] -> Tuple[Tree[saga_grad],
        # Tree[prev_path_grads]] outer_def defines the container structure of the model
        # updates, and inner_def defines the 2-element tuple structure at each leaf.
        outer_def = jax.tree.structure(updates)
        inner_def = jax.tree.structure((0, 0))
        saga_grads, new_prev_path_grads = jax.tree.transpose(
            outer_def, inner_def, results
        )

        new_state = PSAGAState(
            path_counts=path_counts, prev_path_grads=new_prev_path_grads
        )
        return saga_grads, new_state

    return optax.GradientTransformationExtraArgs(init=init_fn, update=update_fn)
