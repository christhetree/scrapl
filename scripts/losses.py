"""
Subset of experiments/src/losses.py containing only the losses needed by
benchmark_torch.py, so it can run in the root scrapl-loss environment.
"""

import logging
import os
from typing import Union, Optional, List

import scipy.signal
import torch as tr
import torch.nn as nn
from torch import Tensor as T

# This adds 'scrapl/kymatio' & 'scrapl/pytorch_hessian_eigenthings' to the Python path.
import scrapl

# Imported from the 1D frontend instead of kymatio.torch, since kymatio.torch also
# imports the 3D scattering frontend, which needs scipy.special.sph_harm (removed in
# scipy 1.17). These are the same classes as kymatio.torch.Scattering1D and
# kymatio.torch.TimeFrequencyScattering.
from kymatio.scattering1d.frontend.torch_frontend import (
    ScatteringTorch1D as Scattering1D,
    TimeFrequencyScatteringTorch as TimeFrequencyScattering,
)
from scrapl.util import ReadOnlyTensorDict

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class JTFSTLoss(nn.Module):
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
        log1p_eps: float = 1e-3,  # TODO: what's a good default here?
    ):
        super().__init__()
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

    def forward(self, x: T, x_target: T) -> T:
        assert x.ndim == x_target.ndim == 3
        bs, n_ch, n_samples = x.size()
        x = x.view(-1, 1, n_samples)
        x_target = x_target.view(-1, 1, n_samples)
        assert x.size(1) == x_target.size(1) == 1
        Sx = self.jtfs(x)
        x_target = x_target.contiguous()
        Sx_target = self.jtfs(x_target)
        if self.use_rho_log1p:
            Sx = tr.log1p(Sx / self.log1p_eps)
            Sx_target = tr.log1p(Sx_target / self.log1p_eps)
        if self.format == "time":
            Sx = Sx[:, :, 1:, :]  # Remove the 0th order coefficients
            Sx_target = Sx_target[:, :, 1:, :]  # Remove the 0th order coefficients
            dist = tr.linalg.vector_norm(Sx_target - Sx, ord=self.p, dim=-1)
        else:
            dist = tr.linalg.vector_norm(Sx_target - Sx, ord=self.p, dim=(-2, -1))
        dist = tr.mean(dist)
        return dist


class Scat1DLoss(nn.Module):
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
        super().__init__()
        self.max_order = max_order
        self.p = p
        self.scat_1d = Scattering1D(
            shape=(shape,),
            J=J,
            Q=(Q1, Q2),
            T=T,
            max_order=max_order,
        )

    def forward(self, x: T, x_target: T) -> T:
        assert x.ndim == x_target.ndim == 3
        bs, n_ch, n_samples = x.size()
        x = x.view(-1, 1, n_samples)
        x_target = x_target.view(-1, 1, n_samples)
        assert x.size(1) == x_target.size(1) == 1
        Sx = self.scat_1d(x)
        Sx_target = self.scat_1d(x_target)
        Sx = Sx[:, :, 1:, :]  # Remove the 0th order coefficients
        Sx_target = Sx_target[:, :, 1:, :]  # Remove the 0th order coefficients

        if self.max_order == 1:
            dist = tr.linalg.vector_norm(Sx_target - Sx, ord=self.p, dim=(-2, -1))
        else:
            dist = tr.linalg.vector_norm(Sx_target - Sx, ord=self.p, dim=-1)

        dist = tr.mean(dist)
        return dist


class LogMSSLoss(nn.Module):
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
        super().__init__()
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
        self.window = window
        self.log_mag_eps = log_mag_eps
        self.gamma = gamma
        self.p = p
        # Create windows
        windows = {}
        for win_length in win_lengths:
            win = self.make_window(window, win_length)
            windows[win_length] = win
        self.windows = ReadOnlyTensorDict(windows)

    def forward(self, x: T, x_target: T) -> T:
        assert x.ndim == x_target.ndim == 3
        assert x.size(1) == x_target.size(1) == 1
        x = x.squeeze(1)
        x_target = x_target.squeeze(1)
        dists = []
        for fft_size, hop_size, win_length in zip(
            self.fft_sizes, self.hop_sizes, self.win_lengths
        ):
            win = self.windows[win_length]
            Sx = tr.stft(
                x,
                n_fft=fft_size,
                hop_length=hop_size,
                win_length=win_length,
                window=win,
                return_complex=True,
            ).abs()
            Sx_target = tr.stft(
                x_target,
                n_fft=fft_size,
                hop_length=hop_size,
                win_length=win_length,
                window=win,
                return_complex=True,
            ).abs()
            if self.log_mag_eps == 1.0:
                log_Sx = tr.log1p(self.gamma * Sx)
                log_Sx_target = tr.log1p(self.gamma * Sx_target)
            else:
                log_Sx = tr.log(self.gamma * Sx + self.log_mag_eps)
                log_Sx_target = tr.log(self.gamma * Sx_target + self.log_mag_eps)
            dist = tr.linalg.vector_norm(
                log_Sx_target - log_Sx, ord=self.p, dim=(-2, -1)
            )
            dists.append(dist)
        dist = tr.stack(dists, dim=1).sum(dim=1)
        dist = dist.mean()  # Aggregate the batch dimension
        return dist

    @staticmethod
    def make_window(window: str, n: int) -> T:
        if window == "rect":
            return tr.ones(n)
        elif window == "hann":
            return tr.hann_window(n)
        elif window == "flat_top":
            window = scipy.signal.windows.flattop(n, sym=False)
            window = tr.from_numpy(window).float()
            return window
        else:
            raise ValueError(f"Unknown window type: {window}")
