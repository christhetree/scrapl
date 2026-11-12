"""
Benchmarks the runtime, peak memory, FLOPs, and compilation time of a JIT compiled
forward and backward pass of the mss_log, mss_rev, scat_1d, jtfs, and scrapl (per
path) losses in JAX. The JAX equivalent of benchmark_torch.py (the MSS losses use
korvax for the STFT), using the same loss configs, and only needs the root
scrapl-loss environment with the dmrn extra (no PyTorch).

Run from the repository root with: uv run python scripts/benchmark_jax.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import util

batch_size = 4
n_samples = 44100
num_threads = 1
# min_run_time = 15.0
min_run_time = 5.0
device = "cpu"
# device = "cuda"
run_scrapl = True
compilation_cache_dir = None
# compilation_cache_dir = os.path.join(util.ROOT_DIR, "jax_cache")

# Before JAX is imported
util.setup_jax_env(device, num_threads, compilation_cache_dir)

import logging

import jax
import jax.numpy as jnp
import pandas as pd

from losses_jax import JTFSTLoss, LogMSSLoss, MultiResolutionSTFTLoss, Scat1DLoss
from scrapl.jax import SCRAPLLoss

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


def main() -> None:
    log.info(
        f"Benchmarking with device={device}, num_threads={num_threads}, "
        f"min_run_time={min_run_time}, batch_size={batch_size}, "
        f"n_samples={n_samples}, run_scrapl={run_scrapl}"
    )
    log.info(f"JAX devices: {jax.devices()}")
    for config_name in ["scat_1d", "jtfs", "scrapl"]:
        shape = util.load_config(config_name)["init_args"]["shape"]
        assert shape == n_samples, f"{config_name} config shape {shape} != {n_samples}"

    mss_log_loss = util.load_class_from_yaml("mss_meso_log", MultiResolutionSTFTLoss)
    mss_rev_loss = util.load_class_from_yaml("mss_revisited", LogMSSLoss)
    scat_1d_loss = util.load_class_from_yaml("scat_1d", Scat1DLoss)
    jtfs_loss = util.load_class_from_yaml("jtfs", JTFSTLoss)

    # The same inputs as benchmark_torch.py, on the default device, which is set by
    # JAX_PLATFORMS
    xs, x_hat = util.make_inputs(batch_size, n_samples)
    x, x_hat = jnp.asarray(xs[0]), jnp.asarray(x_hat)

    df_rows = []
    loss_funcs = [
        ("mss_log", mss_log_loss),
        ("mss_rev", mss_rev_loss),
        ("scat_1d", scat_1d_loss),
        ("jtfs", jtfs_loss),
    ]

    for name, loss_func in loss_funcs:
        log.info(f"Benchmarking {name}...")
        fn = util.make_loss_grad_fn(loss_func)
        df_rows.append(util.benchmark_jax(name, fn, (x_hat, x), min_run_time))

    if run_scrapl:
        scrapl_loss = util.load_class_from_yaml("scrapl", SCRAPLLoss)
        scrapl_df = util.benchmark_scrapl_paths_jax(
            scrapl_loss, x, x_hat, min_run_time=min_run_time
        )
        util.save_df(scrapl_df, f"scrapl_benchmark_jax_{device}.tsv")
        # Aggregated over the paths (see util.PATH_AGGS)
        df_rows.extend(util.summarize_paths(scrapl_df))

    df = pd.DataFrame(df_rows, columns=util.JAX_COLS)
    util.save_df(df, f"benchmark_jax_{device}.tsv")


if __name__ == "__main__":
    main()
