import math
from dataclasses import dataclass
from numbers import Integral
from typing import NamedTuple

import jax
import jax.numpy as jnp


def _validate_n_paths(n_paths):
    if not isinstance(n_paths, Integral) or isinstance(n_paths, bool) or n_paths < 1:
        raise ValueError("n_paths must be a positive integer")


def _gradient_arrays(tree):
    tree = jax.tree_util.tree_map(jnp.asarray, tree)
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves or not any(leaf.size for leaf in leaves):
        raise ValueError("The parameter or gradient tree must contain arrays")
    if any(leaf.dtype not in (jnp.float32, jnp.float64) for leaf in leaves):
        raise TypeError("Parameter and gradient leaves must use float32 or float64")
    return tree


def _path_index(path_idx, n_paths):
    if isinstance(path_idx, Integral) and not isinstance(path_idx, bool):
        if not 0 <= path_idx < n_paths:
            raise ValueError(f"path_idx must be in [0, {n_paths})")
    path_idx = jnp.asarray(path_idx)
    if path_idx.ndim != 0 or not jnp.issubdtype(path_idx.dtype, jnp.integer):
        raise TypeError("path_idx must be a scalar integer")
    index = path_idx.astype(jnp.int32)
    valid = (
        (index >= 0) & (index < n_paths) & (index.astype(path_idx.dtype) == path_idx)
    )
    return jnp.clip(index, 0, n_paths - 1), valid


class PAdamState(NamedTuple):
    """Completed update count, path timestamps, and per-path moment PyTrees."""

    count: jax.Array
    last_steps: jax.Array
    m: object
    v: object


@dataclass(frozen=True)
class PAdam:
    """Pathwise Adam gradient normalisation with explicit, checkpointable state.

    ``update`` returns normalised gradients, not signed parameter updates.
    Apply plain SGD afterwards: ``params - learning_rate * gradients``.
    The first timestamp is 2, matching the PyTorch forward-then-hook convention.
    Every update processes the entire gradient PyTree using one shared path.
    """

    n_paths: int
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8
    grad_mult: float = 1.0

    def __post_init__(self):
        _validate_n_paths(self.n_paths)
        if not 0 <= self.b1 < 1 or not 0 <= self.b2 < 1:
            raise ValueError("b1 and b2 must be in [0, 1)")
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("eps must be finite and positive")
        if not math.isfinite(self.grad_mult) or self.grad_mult <= 0:
            raise ValueError("grad_mult must be finite and positive")

    def init(self, params) -> PAdamState:
        """Allocate two moment arrays of shape (n_paths, *leaf.shape) per leaf."""
        params = _gradient_arrays(params)
        moments = jax.tree_util.tree_map(
            lambda p: jnp.zeros((self.n_paths, *p.shape), dtype=p.dtype), params
        )
        return PAdamState(
            jnp.asarray(0, dtype=jnp.int32),
            jnp.zeros(self.n_paths, dtype=jnp.int32),
            moments,
            jax.tree_util.tree_map(jnp.zeros_like, moments),
        )

    @staticmethod
    def _decay(beta, time, dtype):
        if beta == 0:
            return jnp.asarray(0, dtype=dtype), jnp.asarray(1, dtype=dtype)
        exponent = jnp.asarray(math.log(beta), dtype=dtype) * time
        return jnp.exp(exponent), -jnp.expm1(exponent)

    def _normalise(self, gradient, m, v, current_step, previous_step):
        gradient = gradient * self.grad_mult
        elapsed = (current_step - previous_step).astype(gradient.dtype) / self.n_paths
        time = current_step.astype(gradient.dtype) / self.n_paths
        decay1, weight1 = self._decay(self.b1, elapsed, gradient.dtype)
        decay2, weight2 = self._decay(self.b2, elapsed, gradient.dtype)
        _, correction1 = self._decay(self.b1, time, gradient.dtype)
        _, correction2 = self._decay(self.b2, time, gradient.dtype)
        m = decay1 * m + weight1 * gradient
        v = decay2 * v + weight2 * jnp.square(gradient)
        normalised = (m / correction1) / (jnp.sqrt(v / correction2) + self.eps)
        return normalised, m, v

    def update(self, grads, state: PAdamState, *, path_idx: int | jax.Array):
        """Normalise one path's gradients and return a new state.

        Invalid scalar array indices, non-finite gradients or numerical failures
        return NaN gradients and unchanged state. Invalid Python indices raise.
        Shapes/dtypes must match init; counters are scalar/vector int32 arrays.
        """
        grads = _gradient_arrays(grads)
        if not isinstance(state, PAdamState):
            raise TypeError("state must be a PAdamState from init")
        state = jax.tree_util.tree_map(jnp.asarray, state)
        structure = jax.tree_util.tree_structure(grads)
        if structure != jax.tree_util.tree_structure(
            state.m
        ) or structure != jax.tree_util.tree_structure(state.v):
            raise ValueError(
                "Gradient and moment PyTrees must have matching structures"
            )
        grad_leaves = jax.tree_util.tree_leaves(grads)
        m_leaves, v_leaves = map(jax.tree_util.tree_leaves, (state.m, state.v))
        for g, m, v in zip(grad_leaves, m_leaves, v_leaves):
            if m.shape != (self.n_paths, *g.shape) or v.shape != m.shape:
                raise ValueError("Moment shapes must match n_paths and gradient shapes")
            if m.dtype != g.dtype or v.dtype != g.dtype:
                raise TypeError("Gradient and moment dtypes must match")
        if state.count.shape != () or state.last_steps.shape != (self.n_paths,):
            raise ValueError("Invalid PAdam counter shapes")
        if state.count.dtype != jnp.int32 or state.last_steps.dtype != jnp.int32:
            raise TypeError("PAdam counters must use int32")
        index, valid = _path_index(path_idx, self.n_paths)
        current_step = state.count + 2
        previous_step = state.last_steps[index]
        valid &= (state.count >= 0) & (state.count <= jnp.iinfo(jnp.int32).max - 2)
        valid &= (previous_step >= 0) & (previous_step < current_step)
        results = [
            self._normalise(g, m[index], v[index], current_step, previous_step)
            for g, m, v in zip(grad_leaves, m_leaves, v_leaves)
        ]
        for result, old_v in zip(results, v_leaves):
            valid &= jnp.all(old_v[index] >= 0)
            for value in result:
                valid &= jnp.all(jnp.isfinite(value))

        def accept(_):
            normalised = jax.tree_util.tree_unflatten(
                structure, [r[0] for r in results]
            )
            m = jax.tree_util.tree_unflatten(
                structure,
                [old.at[index].set(r[1]) for old, r in zip(m_leaves, results)],
            )
            v = jax.tree_util.tree_unflatten(
                structure,
                [old.at[index].set(r[2]) for old, r in zip(v_leaves, results)],
            )
            return normalised, PAdamState(
                state.count + 1, state.last_steps.at[index].set(current_step), m, v
            )

        def reject(_):
            return (
                jax.tree_util.tree_map(lambda g: jnp.full_like(g, jnp.nan), grads),
                state,
            )

        return jax.lax.cond(valid, accept, reject, operand=None)


class PSAGAState(NamedTuple):
    """Visited-path mask and the most recent input gradient for every path."""

    seen: jax.Array
    path_grads: object


@dataclass(frozen=True)
class PSAGA:
    """Pathwise SAGA gradient correction matching this repo's PyTorch hook.

    Apply after optional gradient scaling and P-Adam, then use plain SGD.
    The history stores incoming gradients, before the SAGA correction.
    The average divides by max(1, paths seen including this update minus 1),
    preserving the reference behaviour on both new and repeated path visits.
    """

    n_paths: int

    def __post_init__(self):
        _validate_n_paths(self.n_paths)

    def init(self, params) -> PSAGAState:
        """Allocate one gradient array of shape (n_paths, *leaf.shape) per leaf."""
        params = _gradient_arrays(params)
        return PSAGAState(
            jnp.zeros(self.n_paths, dtype=jnp.bool_),
            jax.tree_util.tree_map(
                lambda p: jnp.zeros((self.n_paths, *p.shape), dtype=p.dtype), params
            ),
        )

    def update(self, grads, state: PSAGAState, *, path_idx: int | jax.Array):
        """Return corrected gradients and a new history for one shared path.

        Invalid scalar array indices, non-finite gradients or numerical failures
        return NaN gradients and unchanged state. Invalid Python indices raise.
        Parameter/gradient trees and history must have matching shapes and dtypes.
        """
        grads = _gradient_arrays(grads)
        if not isinstance(state, PSAGAState):
            raise TypeError("state must be a PSAGAState from init")
        state = jax.tree_util.tree_map(jnp.asarray, state)
        structure = jax.tree_util.tree_structure(grads)
        if structure != jax.tree_util.tree_structure(state.path_grads):
            raise ValueError(
                "Gradient and history PyTrees must have matching structures"
            )
        grad_leaves = jax.tree_util.tree_leaves(grads)
        history_leaves = jax.tree_util.tree_leaves(state.path_grads)
        for gradient, history in zip(grad_leaves, history_leaves):
            if history.shape != (self.n_paths, *gradient.shape):
                raise ValueError(
                    "History shapes must match n_paths and gradient shapes"
                )
            if history.dtype != gradient.dtype:
                raise TypeError("Gradient and history dtypes must match")
        if state.seen.shape != (self.n_paths,):
            raise ValueError("PSAGA seen mask must have shape (n_paths,)")
        if state.seen.dtype != jnp.bool_:
            raise TypeError("PSAGA seen mask must use bool")
        index, valid = _path_index(path_idx, self.n_paths)
        seen = state.seen.at[index].set(True)
        denominator = jnp.maximum(1, jnp.sum(seen, dtype=jnp.int32) - 1)
        directions = [
            gradient
            - history[index]
            + history.sum(axis=0) / denominator.astype(gradient.dtype)
            for gradient, history in zip(grad_leaves, history_leaves)
        ]
        for gradient, direction in zip(grad_leaves, directions):
            valid &= jnp.all(jnp.isfinite(gradient)) & jnp.all(jnp.isfinite(direction))

        def accept(_):
            history = jax.tree_util.tree_unflatten(
                structure,
                [old.at[index].set(g) for old, g in zip(history_leaves, grad_leaves)],
            )
            return jax.tree_util.tree_unflatten(structure, directions), PSAGAState(
                seen, history
            )

        def reject(_):
            return (
                jax.tree_util.tree_map(lambda g: jnp.full_like(g, jnp.nan), grads),
                state,
            )

        return jax.lax.cond(valid, accept, reject, operand=None)
