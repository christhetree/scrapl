"""
Benchmarks the runtime, peak memory, and compilation time of the SCRAPL θ-IS warmup
(scrapl.jax.warmup_lc_hvp) per path in JAX. The JAX equivalent of
benchmark_warmup_torch.py, using the same loss config and minimal encoder and decoder
(see util.make_theta_is_arrays), so that the cost is almost entirely the SCRAPL
loss's. The total cost is also logged in mean SCRAPL training steps (see
util.log_warmup_cost), which are benchmarked as well.

Run from the repository root with: uv run python scripts/benchmark_warmup_jax.py
"""

import os

import util

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

util.setup_jax_env(device, num_threads)  # Before JAX is imported

import logging
from functools import partial
from typing import Any, Callable, Dict

import jax
import jax.numpy as jnp
import pandas as pd
from jaxtyping import Array, Float

from scrapl.jax import SCRAPLLoss
from scrapl.jax.warmup import calc_path_eigs

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def enc_fn(
    enc_params: Dict[str, Float[Array, "..."]], x: Float[Array, "bs n_ch n_samples"]
) -> Float[Array, "bs n_theta"]:
    """The minimal encoder (see util.make_theta_is_arrays)."""
    pooled = x.reshape(x.shape[0], util.THETA_IS_N_POOL, -1).mean(axis=-1)
    return pooled @ enc_params["w"] + enc_params["b"]


def make_path_fn(
    loss: SCRAPLLoss,
    dec_fn: Callable[..., Float[Array, "bs n_ch n_samples"]],
    path_idx: int,
    n_theta: int,
) -> Callable[..., Any]:
    """
    One path of scrapl.jax.warmup_lc_hvp: the same function that it JIT compiles for
    each path (calc_path_eigs with its default eps and vmap_theta=False), which is
    used directly so that it can be compiled once and timed, since warmup_lc_hvp
    recompiles every path in each call. For each theta, the largest curvature
    eigenvalue w.r.t. the encoder weights is estimated with n_iter steps of power
    iteration. Early stopping is disabled (tol=0.0), so every path computes
    n_theta * n_iter * n_batches curvature-vector products, the same as
    benchmark_warmup_torch.py. Takes (enc_params, xs, enc_kwargs, dec_kwargs, key).
    """
    return partial(
        calc_path_eigs,
        loss=loss,
        enc_fn=enc_fn,
        dec_fn=dec_fn,
        path_idx=path_idx,
        n_theta=n_theta,
        n_iter=n_iter,
        tol=0.0,
        eps=1e-12,
        vmap_theta=False,
    )


def main() -> None:
    log.info(
        f"Benchmarking with device={device}, num_threads={num_threads}, "
        f"min_run_time={min_run_time}, batch_size={batch_size}, "
        f"n_samples={n_samples}, n_iter={n_iter}"
    )
    log.info(f"JAX devices: {jax.devices()}")

    scrapl_loss = util.load_class_from_yaml("scrapl", SCRAPLLoss)
    n_paths = scrapl_loss.n_paths
    # n_theta is not an init arg of the JAX SCRAPLLoss (warmup_lc_hvp infers it from
    # the encoder output), so it is read from the config directly
    n_theta = util.load_config("scrapl")["init_args"]["n_theta"]

    # Created on the default device, which is set by JAX_PLATFORMS
    key_x, key_x_hat = jax.random.split(jax.random.key(42))
    x = jax.random.uniform(key_x, (batch_size, 1, n_samples))
    x_hat = jax.random.uniform(key_x_hat, (batch_size, 1, n_samples))

    # The mean runtime of a SCRAPL training step over the paths, the unit of the
    # warmup cost (see util.log_warmup_cost)
    scrapl_df = util.benchmark_scrapl_paths_jax(
        scrapl_loss, x, x_hat, min_run_time=min_run_time, with_flops=False
    )
    step_time = scrapl_df["median_time"].mean()

    enc_w, enc_b, basis = util.make_theta_is_arrays(n_theta, n_samples)
    enc_params = {"w": jnp.asarray(enc_w), "b": jnp.asarray(enc_b)}
    basis = jnp.asarray(basis)

    def dec_fn(theta: Float[Array, "bs n_theta"]) -> Float[Array, "bs 1 n_samples"]:
        """The minimal decoder (see util.make_theta_is_arrays)."""
        return (theta @ basis)[:, None, :]

    xs = x[None]  # A single batch
    key = jax.random.key(0)
    log.info(
        f"Benchmarking theta-IS with n_theta={n_theta}, n_iter={n_iter}, "
        f"n_batches={xs.shape[0]}"
    )
    theta_is_rows = []
    for path_idx in range(n_paths):
        log.info(f"Benchmarking theta-IS path {path_idx} / {n_paths}...")
        fn = make_path_fn(scrapl_loss, dec_fn, path_idx, n_theta)
        # No encoder or decoder kwargs, and the same PRNG key per path as
        # warmup_lc_hvp
        args = (enc_params, xs, {}, {}, jax.random.fold_in(key, path_idx))
        row = util.benchmark_jax(
            f"theta_is_path_{path_idx}", fn, args, min_run_time, with_flops=False
        )
        theta_is_rows.append(row)

    theta_is_df = pd.DataFrame(theta_is_rows, columns=util.JAX_WARMUP_COLS)
    util.save_df(theta_is_df, f"theta_is_benchmark_jax_{device}.tsv")
    # Aggregated over the paths (see util.PATH_AGGS). The warmup runs every path once,
    # so its cost is scrapl_total.
    df = pd.DataFrame(util.summarize_paths(theta_is_df), columns=util.JAX_WARMUP_COLS)
    util.save_df(df, f"benchmark_warmup_jax_{device}.tsv")
    util.log_warmup_cost(df, step_time)


if __name__ == "__main__":
    main()
