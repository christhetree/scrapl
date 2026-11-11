"""
JAX equivalents of the JTFSTLoss and Scat1DLoss classes in losses.py, used by
benchmark_jax.py. Kept separate from losses.py so that they can be imported without
PyTorch.
"""

import logging
import os
from typing import Optional, Union

import jax.numpy as jnp
from jaxtyping import Array, Float

# This adds 'scrapl/kymatio' & 'scrapl/pytorch_hessian_eigenthings' to the Python path.
import scrapl

# Imported from the 1D frontend instead of kymatio.jax, since kymatio.jax also
# imports the 3D scattering frontend, which needs scipy.special.sph_harm (removed in
# scipy 1.17). These are the same classes as kymatio.jax.Scattering1D and
# kymatio.jax.TimeFrequencyScattering.
from kymatio.scattering1d.frontend.jax_frontend import (
    ScatteringJax1D as Scattering1D,
    TimeFrequencyScatteringJax as TimeFrequencyScattering,
)
from scrapl.jax import safe_lp_norm

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def _flat_lp_norm(x: Float[Array, "..."], p: float, n_dims: int) -> Float[Array, "..."]:
    """
    Lp norm over the last `n_dims` axes treated as one flattened vector (like
    torch.linalg.vector_norm with a tuple of dims). jnp.linalg.norm with a tuple of
    axes computes a matrix norm instead, so it cannot be used here.
    """
    x = x.reshape(x.shape[:-n_dims] + (-1,))
    return safe_lp_norm(x, p=p, axis=-1)


class JTFSTLoss:
    def __init__(
        self,
        shape: int,
        J: int,
        Q1: int,
        Q2: int,
        J_fr: int,
        Q_fr: int,
        T: Optional[Union[str, int]] = None,
        F: Optional[Union[str, int]] = None,
        format_: str = "joint",
        p: int = 2,
        use_rho_log1p: bool = False,
        log1p_eps: float = 1e-3,
    ):
        assert format_ in ["time", "joint"]
        self.format = format_
        self.p = p
        self.use_rho_log1p = use_rho_log1p
        self.log1p_eps = log1p_eps
        self.jtfs = TimeFrequencyScattering(
            shape=(shape,),
            J=J,
            Q=(Q1, Q2),
            Q_fr=Q_fr,
            J_fr=J_fr,
            T=T,
            F=F,
            format=format_,
        )
        jtfs_meta = self.jtfs.meta()
        jtfs_keys = [key for key in jtfs_meta["key"] if len(key) == 2]
        log.info(f"number of JTFS keys = {len(jtfs_keys)}")

    def __call__(
        self,
        x: Float[Array, "bs n_ch n_samples"],
        x_target: Float[Array, "bs n_ch n_samples"],
    ) -> Float[Array, ""]:
        assert x.ndim == x_target.ndim == 3
        bs, n_ch, n_samples = x.shape
        x = x.reshape(-1, 1, n_samples)
        x_target = x_target.reshape(-1, 1, n_samples)
        Sx = self.jtfs(x)
        Sx_target = self.jtfs(x_target)
        if self.use_rho_log1p:
            Sx = jnp.log1p(Sx / self.log1p_eps)
            Sx_target = jnp.log1p(Sx_target / self.log1p_eps)
        if self.format == "time":
            Sx = Sx[:, :, 1:, :]  # Remove the 0th order coefficients
            Sx_target = Sx_target[:, :, 1:, :]  # Remove the 0th order coefficients
            dist = _flat_lp_norm(Sx_target - Sx, p=self.p, n_dims=1)
        else:
            dist = _flat_lp_norm(Sx_target - Sx, p=self.p, n_dims=2)
        dist = jnp.mean(dist)
        return dist


class Scat1DLoss:
    def __init__(
        self,
        shape: int,
        J: int,
        Q1: int,
        Q2: int = 1,
        T: Optional[Union[str, int]] = None,
        max_order: int = 1,
        p: int = 2,
    ):
        self.max_order = max_order
        self.p = p
        self.scat_1d = Scattering1D(
            shape=(shape,),
            J=J,
            Q=(Q1, Q2),
            T=T,
            max_order=max_order,
        )

    def __call__(
        self,
        x: Float[Array, "bs n_ch n_samples"],
        x_target: Float[Array, "bs n_ch n_samples"],
    ) -> Float[Array, ""]:
        assert x.ndim == x_target.ndim == 3
        bs, n_ch, n_samples = x.shape
        x = x.reshape(-1, 1, n_samples)
        x_target = x_target.reshape(-1, 1, n_samples)
        Sx = self.scat_1d(x)
        Sx_target = self.scat_1d(x_target)
        Sx = Sx[:, :, 1:, :]  # Remove the 0th order coefficients
        Sx_target = Sx_target[:, :, 1:, :]  # Remove the 0th order coefficients

        if self.max_order == 1:
            dist = _flat_lp_norm(Sx_target - Sx, p=self.p, n_dims=2)
        else:
            dist = _flat_lp_norm(Sx_target - Sx, p=self.p, n_dims=1)

        dist = jnp.mean(dist)
        return dist
