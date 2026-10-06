import logging
import math
import os
from numbers import Integral

import equinox as eqx
import jax
import jax.numpy as jnp

from ..single_path_jtfs.jax import TimeFrequencyScrapl
from .util import safe_lp_norm

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class SCRAPLLoss(eqx.Module):
    """Initializes a `SCRAPLLoss` module which contains an implementation of the
    "Scattering Transform with Random Paths for Machine Learning" (SCRAPL) algorithm
    for the joint time-frequency scattering transform (JTFS) in JAX.
    For documentation, examples, hyperparameters, and best practices, please visit:
    https://github.com/christhetree/scrapl
    For more information about the JTFS and the `J`, `Q1`, `Q2`, `J_fr`, `Q_fr`,
    `T`, and `F` hyperparameters, please visit:
    https://www.kymat.io/ismir23-tutorial/intro.html

    Configuration is frozen and registered as an Equinox PyTree module.
    """

    shape: int = eqx.field(static=True)
    J: int = eqx.field(static=True)
    Q1: int = eqx.field(static=True)
    Q2: int = eqx.field(static=True)
    J_fr: int = eqx.field(static=True)
    Q_fr: int = eqx.field(static=True)
    T: str | int | None = eqx.field(static=True, default=None)
    F: str | int | None = eqx.field(static=True, default=None)
    p: float = eqx.field(static=True, default=2)
    use_rho_log1p: bool = eqx.field(static=True, default=False)
    log1p_eps: float = eqx.field(static=True, default=1e-3)
    jtfs: TimeFrequencyScrapl = eqx.field(static=True, init=False, repr=False)
    scrapl_keys: tuple[tuple[int, int], ...] = eqx.field(static=True, init=False)

    def __init__(
        self,
        shape: int,
        J: int,
        Q1: int,
        Q2: int,
        J_fr: int,
        Q_fr: int,
        T: str | int | None = None,
        F: str | int | None = None,
        p: float = 2,
        use_rho_log1p: bool = False,
        log1p_eps: float = 1e-3,
    ) -> None:
        """Initializes a `SCRAPLLoss` module in JAX.

        Args:
            shape (int): The length of the input signal (number of samples).
            J (int): Number of octaves in the JTFS (1st and 2nd order temporal filters).
            Q1 (int): Wavelets per octave in the JTFS (1st order temporal filters).
            Q2 (int): Wavelets per octave in the JTFS (2nd order temporal filters).
            J_fr (int): Number of octaves in the JTFS (2nd order frequential filters).
            Q_fr (int): Wavelets per octave in the JTFS (2nd order frequential filters).
            T (str | int | None, optional): Temporal averaging size in samples
                of the JTFS. If 'global', averages over the entire signal. If None,
                averages over 2**J samples.
                Defaults to None.
            F (str | int | None, optional): Frequential averaging size in frequency
                bins of the JTFS. If 'global', averages over all bins. If None,
                averages over 2**J_fr bins.
                Defaults to None.
            p (float, optional): The order of the norm used for the distance calculation.
                Defaults to 2 (Euclidean norm).
            use_rho_log1p (bool, optional): If True, applies log1p scaling to the
                scattering coefficients (log(1 + x / log1p_eps)) before computing the
                distance.
                Defaults to False.
            log1p_eps (float, optional): The epsilon value used in the log1p scaling.
                Defaults to 1e-3.

        Raises:
            AssertionError: If `shape` is not a positive integer.
            AssertionError: If `p < 1` or `p` is NaN.
            AssertionError: If `log1p_eps <= 0` or not finite.
            AssertionError: If the filter bank contains no second-order paths.
        """
        assert (
            isinstance(shape, Integral) and shape > 0
        ), "shape must be a positive integer sample count"
        assert (
            not math.isnan(p) and p >= 1
        ), "p must be at least 1 (positive infinity is supported)"
        assert (
            math.isfinite(log1p_eps) and log1p_eps > 0
        ), "log1p_eps must be finite and positive"

        self.shape = shape
        self.J = J
        self.Q1 = Q1
        self.Q2 = Q2
        self.J_fr = J_fr
        self.Q_fr = Q_fr
        self.T = T
        self.F = F
        self.p = p
        self.use_rho_log1p = use_rho_log1p
        self.log1p_eps = log1p_eps

        jtfs = TimeFrequencyScrapl(
            shape=(shape,),
            J=J,
            Q=(Q1, Q2),
            J_fr=J_fr,
            Q_fr=Q_fr,
            T=T,
            F=F,
        )
        keys = tuple(key for key in jtfs.meta()["key"] if len(key) == 2)
        assert keys, "The filter bank contains no second-order SCRAPL paths"
        self.jtfs = jtfs
        self.scrapl_keys = keys

        log.info(
            f"SCRAPLLoss:\n"
            f"J={J}, Q1={Q1}, Q2={Q2}, Jfr={J_fr}, Qfr={Q_fr}, T={T}, F={F}\n"
            f"use_rho_log1p          = {use_rho_log1p}, eps = {log1p_eps}\n"
            f"number of SCRAPL paths = {self.n_paths}\n"
            f"unif_prob              = {self.unif_prob:.8f}\n"
        )

    @property
    def n_paths(self) -> int:
        """int: The total number of second-order SCRAPL paths."""
        return len(self.scrapl_keys)

    @property
    def unif_prob(self) -> float:
        """float: The uniform sampling probability for each path (1 / n_paths)."""
        return 1.0 / self.n_paths

    def sample_path(
        self, key: jax.Array, *, probs: jax.Array | None = None
    ) -> int:
        """Samples a single path index using a JAX PRNG key and optional probabilities.

        Args:
            key (jax.Array): JAX PRNG key (e.g. from `jax.random.key(seed)`).
            probs (jax.Array | None, optional): 1D array of shape `(n_paths,)`
                specifying sampling probabilities for each path. Must sum to 1.0.
                If None, paths are sampled uniformly at random.
                Defaults to None.

        Returns:
            int: The sampled path index in `[0, n_paths)`.

        Raises:
            AssertionError: If `probs` has invalid shape, dtype, non-finite/negative
                values, or does not sum to 1.0.
        """
        if probs is None:
            path_idx = int(jax.random.randint(key, (), 0, self.n_paths))
        else:
            probs = jnp.asarray(probs)
            assert probs.shape == (
                self.n_paths,
            ), f"probs must have shape ({self.n_paths},)"
            assert probs.dtype in (
                jnp.float32,
                jnp.float64,
            ), "probs must use float32 or float64"
            assert jnp.all(jnp.isfinite(probs)), "probs must be finite"
            assert jnp.all(probs >= 0), "probs must be non-negative"
            assert jnp.isclose(
                probs.sum(), 1.0, rtol=1e-5, atol=1e-6
            ), "probs must sum to 1.0"
            logits = jnp.log(probs)
            path_idx = int(jax.random.categorical(key, logits))
        return path_idx

    def _path_loss(
        self, x: jax.Array, x_target: jax.Array, *, path_idx: int
    ) -> jax.Array:
        """Computes the Lp distance between x and x_target for a single scattering path."""
        n2, n_fr = self.scrapl_keys[path_idx]
        coef = self.jtfs.scattering_singlepath(x, n2, n_fr)["coef"]
        target_coef = self.jtfs.scattering_singlepath(x_target, n2, n_fr)["coef"]
        if self.use_rho_log1p:
            coef = jnp.log1p(coef / self.log1p_eps)
            target_coef = jnp.log1p(target_coef / self.log1p_eps)
        difference = (target_coef - coef).reshape((coef.shape[0], -1))
        distance = safe_lp_norm(difference, p=self.p, axis=-1)
        loss = distance.mean()
        return loss

    def __call__(
        self,
        x: jax.Array,
        x_target: jax.Array,
        *,
        key: jax.Array | None = None,
        path_idx: int | jax.Array | None = None,
        probs: jax.Array | None = None,
    ) -> jax.Array:
        """Computes the SCRAPL distance loss between input and target signals.

        Args:
            x (jax.Array): Input signal batch of shape `(batch, channels, samples)`.
            x_target (jax.Array): Target signal batch of shape `(batch, channels, samples)`.
            key (jax.Array | None, optional): JAX PRNG key for random path sampling.
                Must be provided if `path_idx` is None.
                Defaults to None.
            path_idx (int | jax.Array | None, optional): Specific path index in
                `[0, n_paths)`. If provided, `key` is ignored.
                Defaults to None.
            probs (jax.Array | None, optional): Path sampling probabilities used when
                sampling with `key`.
                Defaults to None.

        Returns:
            jax.Array: The scalar SCRAPL loss value.

        Raises:
            AssertionError: If `x` and `x_target` shapes/sample lengths do not match or are empty.
            AssertionError: If `x` or `x_target` dtypes are not float32 or float64.
            AssertionError: If neither `key` nor `path_idx` is provided.
            AssertionError: If `path_idx` is out of the valid range `[0, n_paths)`.
            TypeError: If `path_idx` is not a scalar integer.
        """
        x, x_target = jnp.asarray(x), jnp.asarray(x_target)
        assert (
            x.ndim == 3 and x.shape == x_target.shape
        ), "Inputs must have matching (batch, channels, samples) shapes"
        assert (
            x.shape[-1] == self.shape and x.shape[0] > 0 and x.shape[1] > 0
        ), "Inputs must be nonempty and match the configured sample count"
        assert (
            x.dtype in (jnp.float32, jnp.float64)
            and x_target.dtype in (jnp.float32, jnp.float64)
        ), "Inputs must use float32 or float64"

        if path_idx is None:
            assert key is not None, "key must be provided when path_idx is None"
            path_idx = self.sample_path(key, probs=probs)
        if isinstance(path_idx, jax.Array):
            assert (
                path_idx.ndim == 0
                and jnp.issubdtype(path_idx.dtype, jnp.integer)
                and path_idx.dtype != jnp.bool_
            ), "path_idx must be a scalar integer"
            path_idx = int(path_idx)

        assert (
            0 <= path_idx < self.n_paths
        ), f"path_idx {path_idx} is out of range [0, {self.n_paths})"

        loss = self._path_loss(x, x_target, path_idx=path_idx)
        return loss
