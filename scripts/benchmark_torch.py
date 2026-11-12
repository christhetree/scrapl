"""
Benchmarks the runtime and peak memory of a forward and backward pass of the mss_log,
mss_rev, scat_1d, jtfs, and scrapl (per path) losses. Adapted from
experiments/scripts/benchmark.py, but only needs the root scrapl-loss environment.

Run from the repository root with: uv run python scripts/benchmark_torch.py
"""

import logging
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pandas as pd
import torch as tr

import util

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

batch_size = 4
n_samples = 44100
num_threads = 1
# min_run_time = 15.0
min_run_time = 5.0
device = "cpu"
# device = "cuda"
run_scrapl = True
use_torch_compile = False
# use_torch_compile = True


def main() -> None:
    log.info(
        f"Benchmarking with device={device}, num_threads={num_threads}, "
        f"min_run_time={min_run_time}, batch_size={batch_size}, "
        f"n_samples={n_samples}, run_scrapl={run_scrapl}, "
        f"use_torch_compile={use_torch_compile}"
    )
    cols = util.TORCH_COMPILE_COLS if use_torch_compile else util.TORCH_COLS
    suffix = "_compile" if use_torch_compile else ""
    if use_torch_compile:
        # torch.compile compiles a function again for each new value of its int
        # arguments (e.g. path_idx) and of its input shapes, which the scattering
        # losses exceed recompile_limit (8) with. It would then silently fall back to
        # eager mode for that function, so raise the limits, and fail if they are hit.
        tr._dynamo.config.recompile_limit = 1024
        tr._dynamo.config.accumulated_recompile_limit = 8192
        tr._dynamo.config.fail_on_recompile_limit_hit = True
        # Otherwise torch.compile loads the compiled code that earlier runs cached on
        # disk, and compile_time measures loading it. The config is set too, since it
        # is read from the environment when torch is imported.
        os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
        tr.compiler.config.force_disable_caches = True

    for config_name in ["scat_1d", "jtfs", "scrapl"]:
        shape = util.load_config(config_name)["init_args"]["shape"]
        assert shape == n_samples, f"{config_name} config shape {shape} != {n_samples}"

    mss_log_loss = util.load_class_from_yaml("mss_meso_log")
    mss_rev_loss = util.load_class_from_yaml("mss_revisited")
    scat_1d_loss = util.load_class_from_yaml("scat_1d")
    jtfs_loss = util.load_class_from_yaml("jtfs")

    # The same inputs as benchmark_jax.py. requires_grad is set after moving x_hat to
    # the device so that it is a leaf (no GPU -> CPU grad copy).
    xs, x_hat = util.make_inputs(batch_size, n_samples)
    x = tr.from_numpy(xs[0]).to(device)
    x_hat = tr.from_numpy(x_hat).to(device).requires_grad_()

    df_rows = []
    globals = {
        "x": x,
        "x_hat": x_hat,
        "fn": util.benchmark_loss_func,
        "path_idx": None,
    }
    loss_funcs = [
        ("mss_log", mss_log_loss),
        ("mss_rev", mss_rev_loss),
        ("scat_1d", scat_1d_loss),
        ("jtfs", jtfs_loss),
    ]

    for name, loss_func in loss_funcs:
        log.info(f"Benchmarking {name}...")
        loss_func = loss_func.to(device)
        if use_torch_compile:
            loss_func = tr.compile(loss_func)
        globals["loss_func"] = loss_func
        row = util.benchmark_torch(
            name,
            globals,
            device,
            num_threads=num_threads,
            min_run_time=min_run_time,
            time_first_call=use_torch_compile,
        )
        df_rows.append(row)

    if run_scrapl:
        scrapl_loss = util.load_class_from_yaml("scrapl").to(device)
        if use_torch_compile:
            scrapl_loss = tr.compile(scrapl_loss)
        scrapl_df = util.benchmark_scrapl_paths_torch(
            scrapl_loss,
            x,
            x_hat,
            device,
            num_threads=num_threads,
            min_run_time=min_run_time,
            time_first_call=use_torch_compile,
        )
        util.save_df(scrapl_df, f"scrapl_benchmark_torch{suffix}_{device}.tsv")
        # Aggregated over the paths (see util.PATH_AGGS)
        df_rows.extend(util.summarize_paths(scrapl_df))

    df = pd.DataFrame(df_rows, columns=cols)
    util.save_df(df, f"benchmark_torch{suffix}_{device}.tsv")


if __name__ == "__main__":
    main()
