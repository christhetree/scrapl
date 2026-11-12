"""
Benchmarks the runtime and peak memory of the SCRAPL θ-IS warmup
(SCRAPLLoss.warmup_lc_hvp) per path. A minimal encoder and decoder are used (see
util.make_theta_is_arrays), so that the cost is almost entirely the SCRAPL loss's.

Run from the repository root with: uv run python scripts/benchmark_warmup_torch.py
"""

import functools
import logging
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from typing import Any, Callable, List

import pandas as pd
import torch as tr
import torch.nn as nn
from torch import Tensor as T

import util

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

n_batches = 1
batch_size = 4
n_samples = 44100
num_threads = 1
# min_run_time = 15.0
min_run_time = 5.0
device = "cpu"
# device = "cuda"
n_theta = 1
n_iter = 20


def encoder(x: T, w: nn.Parameter) -> T:
    """
    The minimal encoder (theta_fn) used to benchmark the θ-IS warmup: the waveform x
    multiplied by the random weight matrix w of shape (n_samples, n_theta), the only
    weights (see util.make_theta_is_arrays).
    """
    return x.view(x.size(0), -1) @ w


def decoder(theta: T, basis: T) -> T:
    """
    The minimal decoder (synth_fn) used to benchmark the θ-IS warmup: a fixed linear
    synth without weights that mixes the n_theta sinusoids in `basis` with theta as
    their amplitudes (see util.make_theta_is_arrays).
    """
    return (theta @ basis).unsqueeze(1)


def benchmark_theta_is_path(
    loss_func: Any,
    path_idx: int,
    theta_fn: Callable[..., T],
    synth_fn: Callable[[T], T],
    xs: List[T],
    params: List[nn.Parameter],
) -> None:
    """
    The θ-IS warmup (SCRAPLLoss.warmup_lc_hvp) of one path with the batches `xs`,
    which updates its sampling probability and saves nothing: for each theta, the
    largest curvature eigenvalue w.r.t. the encoder weights is estimated with n_iter
    steps of power iteration. Early stopping is disabled (power_iter_err_threshold=0.0),
    so every path computes n_theta * n_iter * n_batches curvature-vector products, the
    same as benchmark_warmup_jax.py.
    """
    loss_func.warmup_lc_hvp(
        theta_fn=theta_fn,
        synth_fn=synth_fn,
        theta_fn_kwargs=[{"x": x} for x in xs],
        params=params,
        start_path_idx=path_idx,
        end_path_idx=path_idx + 1,
        save_dir=None,
        n_iter=n_iter,
        power_iter_err_threshold=0.0,
    )


def main() -> None:
    log.info(
        f"Benchmarking with device={device}, num_threads={num_threads}, "
        f"min_run_time={min_run_time}, n_batches={n_batches}, batch_size={batch_size}, "
        f"n_samples={n_samples}, n_theta={n_theta}, n_iter={n_iter}"
    )
    tr.manual_seed(42)

    shape = util.load_config("scrapl")["init_args"]["shape"]
    assert shape == n_samples, f"scrapl config shape {shape} != {n_samples}"
    # Override the config's n_theta so that the loss matches the encoder
    scrapl_loss = util.load_class_from_yaml("scrapl", n_theta=n_theta).to(device)
    n_paths = scrapl_loss.n_paths

    # The same warmup batches as benchmark_warmup_jax.py
    xs, _ = util.make_inputs(batch_size, n_samples, n_batches)
    xs = [tr.from_numpy(x).to(device) for x in xs]

    enc_w, basis = util.make_theta_is_arrays(n_theta, n_samples)
    enc_w = nn.Parameter(tr.from_numpy(enc_w).to(device))
    basis = tr.from_numpy(basis).to(device)
    theta_fn = functools.partial(encoder, w=enc_w)
    synth_fn = functools.partial(decoder, basis=basis)
    params = [enc_w]
    globals = {
        "fn": benchmark_theta_is_path,
        "loss_func": scrapl_loss,
        "theta_fn": theta_fn,
        "synth_fn": synth_fn,
        "xs": xs,
        "params": params,
    }
    stmt = "fn(loss_func, path_idx, theta_fn, synth_fn, xs, params)"
    log.info(
        f"Benchmarking theta-IS with n_theta={n_theta}, n_iter={n_iter}, "
        f"n_batches={len(xs)}"
    )
    theta_is_rows = []
    for path_idx in range(n_paths):
        log.info(f"Benchmarking theta-IS path {path_idx} / {n_paths}...")
        globals["path_idx"] = path_idx
        row = util.benchmark_torch(
            f"theta_is_path_{path_idx}",
            globals,
            device,
            num_threads=num_threads,
            min_run_time=min_run_time,
            stmt=stmt,
        )
        theta_is_rows.append(row)

    theta_is_df = pd.DataFrame(theta_is_rows, columns=util.TORCH_COLS)
    util.save_df(theta_is_df, f"theta_is_benchmark_torch_{device}.tsv")
    # Aggregated over the paths (see util.PATH_AGGS). The warmup runs every path once,
    # so its cost is scrapl_total.
    df = pd.DataFrame(util.summarize_paths(theta_is_df), columns=util.TORCH_COLS)
    util.save_df(df, f"benchmark_warmup_torch_{device}.tsv")


if __name__ == "__main__":
    main()
