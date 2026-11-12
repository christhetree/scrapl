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
n_samples = 32768
num_threads = 1
# min_run_time = 15.0
min_run_time = 5.0
device = "cpu"
# device = "cuda"
run_scrapl = True


def main() -> None:
    log.info(
        f"Benchmarking with device={device}, num_threads={num_threads}, "
        f"min_run_time={min_run_time}, batch_size={batch_size}, "
        f"n_samples={n_samples}, run_scrapl={run_scrapl}"
    )
    tr.manual_seed(42)

    mss_log_loss = util.load_class_from_yaml("mss_meso_log")
    mss_rev_loss = util.load_class_from_yaml("mss_revisited")
    scat_1d_loss = util.load_class_from_yaml("scat_1d")
    jtfs_loss = util.load_class_from_yaml("jtfs")

    # Create directly on device so x_hat is a leaf (no GPU -> CPU grad copy)
    x = tr.rand((batch_size, 1, n_samples), device=device)
    x_hat = tr.rand((batch_size, 1, n_samples), device=device, requires_grad=True)

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
        globals["loss_func"] = loss_func.to(device)
        row = util.benchmark_torch(
            name,
            globals,
            device,
            num_threads=num_threads,
            min_run_time=min_run_time,
        )
        df_rows.append(row)

    if run_scrapl:
        scrapl_loss = util.load_class_from_yaml("scrapl").to(device)
        scrapl_df = util.benchmark_scrapl_paths_torch(
            scrapl_loss,
            x,
            x_hat,
            device,
            num_threads=num_threads,
            min_run_time=min_run_time,
        )
        util.save_df(scrapl_df, f"scrapl_benchmark_torch_{device}.tsv")
        # Aggregated over the paths (see util.PATH_AGGS)
        df_rows.extend(util.summarize_paths(scrapl_df))

    df = pd.DataFrame(df_rows, columns=util.TORCH_COLS)
    util.save_df(df, f"benchmark_torch_{device}.tsv")


if __name__ == "__main__":
    main()
