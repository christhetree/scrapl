"""
Benchmarks the runtime and peak memory of the SCRAPL θ-IS warmup
(SCRAPLLoss.warmup_lc_hvp) per path. A minimal encoder and decoder are used (see
util.make_theta_is_arrays), so that the cost is almost entirely the SCRAPL loss's, and
the total cost is also logged in mean SCRAPL training steps (see util.log_warmup_cost),
which are benchmarked as well.

Run from the repository root with: uv run python scripts/benchmark_warmup_torch.py
"""

import functools
import logging
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from typing import Any, Callable, List, Tuple

import pandas as pd
import torch as tr
import torch.nn as nn
from torch import Tensor as T

import util

# This adds 'scrapl/kymatio' & 'scrapl/pytorch_hessian_eigenthings' to the Python path.
import scrapl
import hessian_eigenthings

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

batch_size = 4
n_samples = 32768
num_threads = 1
# min_run_time = 15.0
min_run_time = 5.0
device = "cpu"
# device = "cuda"
# Power iteration steps per theta and path. All of them are always run (no early
# stopping), so that both warmup benchmarks do the same amount of work.
n_iter = 20


def make_theta_is_fns(
    n_theta: int, device: str
) -> Tuple[Callable[..., T], Callable[[T], T], List[nn.Parameter]]:
    """
    The minimal encoder (theta_fn) and decoder (synth_fn) used to benchmark the θ-IS
    warmup, and the encoder weights (see util.make_theta_is_arrays).
    """
    enc_w, enc_b, basis = util.make_theta_is_arrays(n_theta, n_samples)
    enc_w = nn.Parameter(tr.from_numpy(enc_w).to(device))
    enc_b = nn.Parameter(tr.from_numpy(enc_b).to(device))
    basis = tr.from_numpy(basis).to(device)

    def theta_fn(x: T) -> T:
        pooled = x.view(x.size(0), util.THETA_IS_N_POOL, -1).mean(dim=-1)
        return pooled @ enc_w + enc_b

    def synth_fn(theta: T) -> T:
        return (theta @ basis).unsqueeze(1)

    return theta_fn, synth_fn, [enc_w, enc_b]


def benchmark_theta_is_path(
    loss_func: Any,
    path_idx: int,
    theta_fn: Callable[..., T],
    synth_fn: Callable[[T], T],
    x: T,
    params: List[nn.Parameter],
) -> None:
    """
    The θ-IS warmup (SCRAPLLoss.warmup_lc_hvp) of one path with a single batch, which
    updates its sampling probability and saves nothing: for each theta, the largest
    curvature eigenvalue w.r.t. the encoder weights is estimated with n_iter steps of
    power iteration. Early stopping is disabled in main, so every path computes
    n_theta * n_iter curvature-vector products, the same as benchmark_warmup_jax.py.
    """
    loss_func.warmup_lc_hvp(
        theta_fn=theta_fn,
        synth_fn=synth_fn,
        theta_fn_kwargs=[{"x": x}],
        params=params,
        start_path_idx=path_idx,
        end_path_idx=path_idx + 1,
        save_dir=None,
        n_iter=n_iter,
    )


def main() -> None:
    log.info(
        f"Benchmarking with device={device}, num_threads={num_threads}, "
        f"min_run_time={min_run_time}, batch_size={batch_size}, "
        f"n_samples={n_samples}, n_iter={n_iter}"
    )
    tr.manual_seed(42)

    scrapl_loss = util.load_class_from_yaml("scrapl").to(device)
    n_paths = scrapl_loss.n_paths

    # Create directly on device so x_hat is a leaf (no GPU -> CPU grad copy)
    x = tr.rand((batch_size, 1, n_samples), device=device)
    x_hat = tr.rand((batch_size, 1, n_samples), device=device, requires_grad=True)

    # The mean runtime of a SCRAPL training step over the paths, the unit of the
    # warmup cost (see util.log_warmup_cost)
    scrapl_df = util.benchmark_scrapl_paths_torch(
        scrapl_loss,
        x,
        x_hat,
        device,
        num_threads=num_threads,
        min_run_time=min_run_time,
    )
    step_time = scrapl_df["median_time"].mean()

    theta_fn, synth_fn, params = make_theta_is_fns(scrapl_loss.n_theta, device)
    globals = {
        "fn": benchmark_theta_is_path,
        "loss_func": scrapl_loss,
        "theta_fn": theta_fn,
        "synth_fn": synth_fn,
        "x": x,
        "params": params,
    }
    stmt = "fn(loss_func, path_idx, theta_fn, synth_fn, x, params)"
    log.info(
        f"Benchmarking theta-IS with n_theta={scrapl_loss.n_theta}, n_iter={n_iter}, "
        f"n_batches=1"
    )
    # Run all n_iter power iteration steps instead of stopping early once the
    # eigenvalue estimate changes by less than 1e-4 (which SCRAPLLoss does not
    # expose), so that both warmup benchmarks do the same amount of work
    deflated_power_iteration = hessian_eigenthings.deflated_power_iteration
    hessian_eigenthings.deflated_power_iteration = functools.partial(
        deflated_power_iteration, power_iter_err_threshold=0.0
    )
    try:
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
    finally:
        hessian_eigenthings.deflated_power_iteration = deflated_power_iteration

    theta_is_df = pd.DataFrame(theta_is_rows, columns=util.TORCH_COLS)
    util.save_df(theta_is_df, f"theta_is_benchmark_torch_{device}.tsv")
    # Aggregated over the paths (see util.PATH_AGGS). The warmup runs every path once,
    # so its cost is scrapl_total.
    df = pd.DataFrame(util.summarize_paths(theta_is_df), columns=util.TORCH_COLS)
    util.save_df(df, f"benchmark_warmup_torch_{device}.tsv")
    util.log_warmup_cost(df, step_time)


if __name__ == "__main__":
    main()
