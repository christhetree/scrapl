"""
JAX equivalents of the JTFSTLoss, Scat1DLoss, and LogMSSLoss classes in losses.py and
of auraloss.freq.MultiResolutionSTFTLoss, used by benchmark_jax.py. The MSS losses use
korvax for the STFT. Kept separate from losses.py so that they can be imported without
PyTorch.
"""

import logging
import os
from typing import List, Optional, Sequence, Union

import jax.numpy as jnp
import korvax
from jaxtyping import Array, Complex, Float
from korvax.loss import elementwise_loss

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


def _stft(
    x: Float[Array, "... n_samples"],
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: str,
) -> Complex[Array, "... n_bins n_frames"]:
    """
    STFT computed with korvax and centered like torch.stft(center=True), i.e. reflect
    padded by n_fft // 2 on both sides. korvax's own centering pads with zeros instead,
    and with one more sample on the right when n_fft is odd. Windows are periodic.
    """
    pad = n_fft // 2
    x = jnp.pad(x, [(0, 0)] * (x.ndim - 1) + [(pad, pad)], mode="reflect")
    return korvax.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=False,
    )


class MultiResolutionSTFTLoss:
    """
    auraloss.freq.MultiResolutionSTFTLoss in JAX, using korvax. Only the log and
    linear magnitude terms are implemented (no spectral convergence, phase,
    perceptual weighting, or frequency scale), which is all the mss_meso_log config
    uses. Like auraloss, the magnitudes are sqrt(max(|STFT|^2, eps)), and the loss is
    averaged over the resolutions. korvax.loss.mrstft_loss is not used since it
    computes magnitudes and pads the signal differently.
    """

    def __init__(
        self,
        fft_sizes: Sequence[int],
        hop_sizes: Sequence[int],
        win_lengths: Sequence[int],
        window: str = "hann_window",
        w_sc: float = 0.0,
        w_log_mag: float = 1.0,
        w_lin_mag: float = 0.0,
        w_phs: float = 0.0,
        eps: float = 1e-8,
        mag_distance: str = "L1",
    ):
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        assert w_sc == 0.0, "The spectral convergence term is not implemented"
        assert w_phs == 0.0, "The phase term is not implemented"
        assert window == "hann_window", "Only the hann_window window is implemented"
        assert mag_distance in ["L1", "L2"]
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.w_log_mag = w_log_mag
        self.w_lin_mag = w_lin_mag
        self.eps = eps
        self.mag_distance = mag_distance

    def __call__(
        self,
        x: Float[Array, "bs n_ch n_samples"],
        x_target: Float[Array, "bs n_ch n_samples"],
    ) -> Float[Array, ""]:
        assert x.ndim == x_target.ndim == 3
        loss = 0.0
        for fft_size, hop_size, win_length in zip(
            self.fft_sizes, self.hop_sizes, self.win_lengths
        ):
            mags = []
            for audio in [x, x_target]:
                X = _stft(audio, fft_size, hop_size, win_length, "hann")
                power = jnp.real(X) ** 2 + jnp.imag(X) ** 2
                mags.append(jnp.sqrt(jnp.maximum(power, self.eps)))
            x_mag, x_target_mag = mags
            if self.w_log_mag:
                log_mag_loss = elementwise_loss(
                    jnp.log(x_mag), jnp.log(x_target_mag), metric=self.mag_distance
                )
                loss += self.w_log_mag * log_mag_loss
            if self.w_lin_mag:
                lin_mag_loss = elementwise_loss(
                    x_mag, x_target_mag, metric=self.mag_distance
                )
                loss += self.w_lin_mag * lin_mag_loss
        loss /= len(self.fft_sizes)
        return loss


class LogMSSLoss:
    """
    LogMSSLoss of losses.py in JAX, using korvax for the STFTs. korvax's
    smooth_mrstft_loss is not used since it computes the mean squared error
    averaged over resolutions (instead of the Lp norm summed over resolutions), and
    its first hop size is 32 instead of 33.
    """

    # The scipy.signal.get_window names of the window types of losses.py
    WINDOWS = {"rect": "boxcar", "hann": "hann", "flat_top": "flattop"}

    def __init__(
        self,
        fft_sizes: Optional[List[int]] = None,
        hop_sizes: Optional[List[int]] = None,
        win_lengths: Optional[List[int]] = None,
        window: str = "flat_top",
        log_mag_eps: float = 1.0,
        gamma: float = 1.0,
        p: int = 2,
    ):
        assert window in self.WINDOWS, f"Unknown window type: {window}"
        if win_lengths is None:
            win_lengths = [67, 127, 257, 509, 1021, 2053]
            log.info(f"win_lengths = {win_lengths}")
        if fft_sizes is None:
            fft_sizes = win_lengths
            log.info(f"fft_sizes = {fft_sizes}")
        if hop_sizes is None:
            hop_sizes = [w // 2 for w in win_lengths]
            log.info(f"hop_sizes = {hop_sizes}")
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.window = self.WINDOWS[window]
        self.log_mag_eps = log_mag_eps
        self.gamma = gamma
        self.p = p

    def __call__(
        self,
        x: Float[Array, "bs n_ch n_samples"],
        x_target: Float[Array, "bs n_ch n_samples"],
    ) -> Float[Array, ""]:
        assert x.ndim == x_target.ndim == 3
        assert x.shape[1] == x_target.shape[1] == 1
        x = x[:, 0, :]
        x_target = x_target[:, 0, :]
        dists = []
        for fft_size, hop_size, win_length in zip(
            self.fft_sizes, self.hop_sizes, self.win_lengths
        ):
            log_mags = []
            for audio in [x, x_target]:
                S = jnp.abs(_stft(audio, fft_size, hop_size, win_length, self.window))
                if self.log_mag_eps == 1.0:
                    log_mags.append(jnp.log1p(self.gamma * S))
                else:
                    log_mags.append(jnp.log(self.gamma * S + self.log_mag_eps))
            log_Sx, log_Sx_target = log_mags
            dist = _flat_lp_norm(log_Sx_target - log_Sx, p=self.p, n_dims=2)
            dists.append(dist)
        dist = jnp.stack(dists, axis=1).sum(axis=1)
        dist = dist.mean()  # Aggregate the batch dimension
        return dist
