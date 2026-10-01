import math
from dataclasses import dataclass, field
from functools import partial
from numbers import Integral

import jax
import jax.numpy as jnp

from ..single_path_jtfs.jax import TimeFrequencyScrapl


@dataclass(frozen=True, eq=False)
class SCRAPLLoss:
    """Single-path scattering loss with explicit JAX keys and probabilities.

    Construct once outside JIT. Call with either ``key`` for a random path or
    ``path_idx`` for a specific path. Inputs have shape (batch, channels, samples).
    The scalar loss averages the path's Lp distances over batch and channels.

    Sampling uses one shared path for the entire batch. JIT traces all paths
    for a dynamic index, then executes the selected branch. A Python integer
    index traces only that path. Invalid Python indices raise ValueError;
    invalid scalar array indices produce NaN, including inside JIT.

    Configuration is frozen. No keys, path counts or gradient histories are
    stored or updated. Split keys in the training loop for fresh samples.
    """

    shape: int
    J: int
    Q1: int
    Q2: int
    J_fr: int
    Q_fr: int
    T: int | str | None = None
    F: int | str | None = None
    p: float = 2
    use_rho_log1p: bool = False
    log1p_eps: float = 1e-3
    jtfs: TimeFrequencyScrapl = field(init=False, repr=False)
    scrapl_keys: tuple[tuple[int, int], ...] = field(init=False)

    def __post_init__(self):
        if not isinstance(self.shape, Integral) or self.shape <= 0:
            raise ValueError("shape must be a positive integer sample count")
        if math.isnan(self.p) or self.p < 1:
            raise ValueError("p must be at least 1 (positive infinity is supported)")
        if not math.isfinite(self.log1p_eps) or self.log1p_eps <= 0:
            raise ValueError("log1p_eps must be finite and positive")

        jtfs = TimeFrequencyScrapl(
            shape=(self.shape,),
            J=self.J,
            Q=(self.Q1, self.Q2),
            J_fr=self.J_fr,
            Q_fr=self.Q_fr,
            T=self.T,
            F=self.F,
        )
        keys = tuple(key for key in jtfs.meta()["key"] if len(key) == 2)
        if not keys:
            raise ValueError("The filter bank contains no second-order SCRAPL paths")
        object.__setattr__(self, "jtfs", jtfs)
        object.__setattr__(self, "scrapl_keys", keys)

    @property
    def n_paths(self) -> int:
        return len(self.scrapl_keys)

    @property
    def unif_prob(self) -> float:
        return 1.0 / self.n_paths

    def sample_path(
        self, key: jax.Array, *, probs: jax.Array | None = None
    ) -> jax.Array:
        """Sample a path, uniformly by default; invalid probabilities return -1."""
        if probs is None:
            return jax.random.randint(key, (), 0, self.n_paths)
        probs = jnp.asarray(probs)
        if probs.shape != (self.n_paths,):
            raise ValueError(f"probs must have shape ({self.n_paths},)")
        if probs.dtype not in (jnp.float32, jnp.float64):
            raise TypeError("probs must use float32 or float64")
        valid = jnp.all(jnp.isfinite(probs) & (probs >= 0)) & jnp.isclose(
            probs.sum(), 1.0, rtol=1e-5, atol=1e-6
        )
        logits = jnp.log(jnp.where(valid, probs, jnp.ones_like(probs)))
        index = jax.random.categorical(key, logits)
        return jnp.where(valid, index, -1)

    def _path_loss(self, x, x_target, *, path_idx):
        n2, n_fr = self.scrapl_keys[path_idx]
        coef = self.jtfs.scattering_singlepath(x, n2, n_fr)["coef"]
        target_coef = self.jtfs.scattering_singlepath(x_target, n2, n_fr)["coef"]
        if self.use_rho_log1p:
            coef = jnp.log1p(coef / self.log1p_eps)
            target_coef = jnp.log1p(target_coef / self.log1p_eps)
        difference = (target_coef - coef).reshape((coef.shape[0], -1))
        if self.p == 1:
            magnitude = jnp.where(difference == 0, 0, jnp.abs(difference))
            return magnitude.sum(axis=-1).mean()
        nonzero = jnp.any(difference != 0, axis=-1)
        safe_difference = jnp.where(
            nonzero[:, None], difference, jnp.ones_like(difference)
        )
        distance = jnp.linalg.norm(safe_difference, ord=self.p, axis=-1)
        return jnp.where(nonzero, distance, 0).mean()

    def __call__(
        self,
        x: jax.Array,
        x_target: jax.Array,
        *,
        key: jax.Array | None = None,
        path_idx: int | jax.Array | None = None,
        probs: jax.Array | None = None,
    ) -> jax.Array:
        if (key is None) == (path_idx is None):
            raise ValueError("Provide exactly one of key or path_idx")
        if probs is not None and key is None:
            raise ValueError("probs applies only when sampling with key")
        x, x_target = jnp.asarray(x), jnp.asarray(x_target)
        if x.ndim != 3 or x.shape != x_target.shape:
            raise ValueError(
                "Inputs must have matching (batch, channels, samples) shapes"
            )
        if x.shape[-1] != self.shape or x.shape[0] == 0 or x.shape[1] == 0:
            raise ValueError(
                "Inputs must be nonempty and match the configured sample count"
            )
        if x.dtype not in (jnp.float32, jnp.float64) or x_target.dtype not in (
            jnp.float32,
            jnp.float64,
        ):
            raise TypeError("Inputs must use float32 or float64")

        if key is not None:
            path_idx = self.sample_path(key, probs=probs)
        if isinstance(path_idx, Integral) and not isinstance(path_idx, bool):
            if not 0 <= path_idx < self.n_paths:
                raise ValueError(f"path_idx must be in [0, {self.n_paths})")
            return self._path_loss(x, x_target, path_idx=int(path_idx))

        path_idx = jnp.asarray(path_idx)
        if path_idx.ndim != 0 or not jnp.issubdtype(path_idx.dtype, jnp.integer):
            raise TypeError("path_idx must be a scalar integer")
        branches = tuple(
            partial(self._path_loss, path_idx=i) for i in range(self.n_paths)
        )
        switch_index = path_idx.astype(jnp.int32)
        valid = (
            (switch_index >= 0)
            & (switch_index < self.n_paths)
            & (switch_index.astype(path_idx.dtype) == path_idx)
        )
        value = jax.lax.switch(switch_index, branches, x, x_target)
        return jnp.where(valid, value, jnp.nan)
