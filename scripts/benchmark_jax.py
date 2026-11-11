"""
Benchmarks the runtime, peak memory, and FLOPs of a JIT compiled forward and backward
pass of the mss_log, mss_rev, scat_1d, jtfs, and scrapl (per path) losses in JAX. The
JAX equivalent of benchmark_torch.py (the MSS losses use korvax for the STFT), using
the same loss configs, and only needs the root scrapl-loss environment with the dmrn
extra (no PyTorch).

Run from the repository root with: uv run python scripts/benchmark_jax.py
"""

import logging
import os

batch_size = 4
n_samples = 32768
num_threads = 1
# min_run_time = 15.0
min_run_time = 5.0
device = "cpu"
# device = "cuda"
run_scrapl = True

# These must be set before JAX is imported
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["JAX_PLATFORMS"] = device
if device == "cpu":
    # Size of the XLA CPU thread pool (the equivalent of torch.set_num_threads)
    os.environ["PJRT_NPROC"] = str(num_threads)
# Otherwise JAX preallocates 75% of the GPU memory on startup
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import gc
import inspect
import time
from typing import Any, Callable, List, Optional, Tuple

import jax
import numpy as np
import pandas as pd
import yaml
from jax.stages import Compiled
from jaxtyping import Array, Float

from losses_jax import JTFSTLoss, LogMSSLoss, MultiResolutionSTFTLoss, Scat1DLoss
from scrapl.jax import SCRAPLLoss

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPTS_DIR)
# The same configs as benchmark_torch.py; only their init_args are used
CONFIGS_DIR = os.path.join(SCRIPTS_DIR, "configs")
OUT_DIR = os.path.join(ROOT_DIR, "out")


def load_class_from_yaml(config_path: str, cls: Callable[..., Any]) -> Any:
    """
    Instantiates `cls` with the init_args of a loss config, so that both benchmarks
    use identical hyperparameters. Init args that `cls` does not accept are dropped
    (e.g. the SCRAPL optimizer args, which are part of scrapl_optimizer in JAX and
    are not used by the forward and backward pass benchmarked here).
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    init_args = config["init_args"]
    params = inspect.signature(cls).parameters
    ignored = sorted(k for k in init_args if k not in params)
    if ignored:
        log.info(f"Ignoring init args not used by JAX {cls.__name__}: {ignored}")
    cls_instantiated = cls(**{k: v for k, v in init_args.items() if k in params})
    return cls_instantiated


def compile_benchmark_fn(
    loss_func: Callable[..., Float[Array, ""]],
    x: Float[Array, "bs n_ch n_samples"],
    x_hat: Float[Array, "bs n_ch n_samples"],
    path_idx: Optional[int] = None,
) -> Tuple[Compiled, float, float]:
    """
    Ahead-of-time JIT compiles a forward and backward pass (the gradient w.r.t.
    x_hat) of `loss_func`. `path_idx` must be a Python int, so each SCRAPL path is
    compiled separately. Returns the compiled function, the compilation time, and
    the FLOPs estimated by XLA's HloCostAnalysis. FLOPs are estimated before XLA's
    optimizations (e.g. fusion), so they do not depend on the device (transcendental
    ops such as sqrt and log are counted separately by XLA, so they are not included).
    """

    def fn(x_hat: Float[Array, "..."], x: Float[Array, "..."]) -> Float[Array, ""]:
        if path_idx is None:
            return loss_func(x_hat, x)
        return loss_func(x_hat, x, path_idx=path_idx)

    start = time.perf_counter()
    lowered = jax.jit(jax.value_and_grad(fn)).lower(x_hat, x)
    compiled = lowered.compile()
    compile_time = time.perf_counter() - start
    flops = lowered.cost_analysis().get("flops", 0.0)  # XLA omits zero counts
    return compiled, compile_time, flops


def time_n_calls(fn: Callable[..., Any], args: Tuple[Any, ...], number: int) -> float:
    """
    Times `number` calls of `fn`, like timeit.Timer.timeit as used by
    torch.utils.benchmark: garbage collection is disabled while timing, and since JAX
    dispatches asynchronously, the device is only synchronized once at the end (the
    same as torch.utils.benchmark on CUDA). Calls execute in order on the device, so
    waiting for the last output waits for all of them.
    """
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        out = None
        start = time.perf_counter()
        for _ in range(number):
            out = fn(*args)
        jax.block_until_ready(out)
        return time.perf_counter() - start
    finally:
        if gc_was_enabled:
            gc.enable()


def blocked_autorange(
    fn: Callable[..., Any], args: Tuple[Any, ...], min_run_time: float
) -> Tuple[List[float], int]:
    """
    The same algorithm as torch.utils.benchmark.Timer.blocked_autorange: estimating
    the block size (calls per measurement) also serves as an untimed warmup, then
    blocks are timed until `min_run_time` has elapsed. Returns the per call time of
    each block and the number of calls per block.
    """
    overhead = float(np.median([time_n_calls(fn, args, 0) for _ in range(5)]))
    number = 1
    while True:
        time_taken = time_n_calls(fn, args, number)
        if overhead / time_taken <= 1e-4 and time_taken >= min_run_time / 1000:
            break
        if time_taken > min_run_time:
            break
        number *= 10
    raw_times = []
    while sum(raw_times) < min_run_time:
        raw_times.append(time_n_calls(fn, args, number))
    times = [t / number for t in raw_times]
    return times, number


def calc_peak_mem_mb(compiled: Compiled) -> float:
    """
    Peak memory (MB) allocated while running `compiled` once, relative to what was
    already allocated beforehand (the inputs and the loss's filters), as planned by
    XLA's buffer assignment: temporary buffers (reused based on liveness) plus the
    newly allocated outputs. Computed the same way on CPU and CUDA. Unlike the
    runtime measurements in benchmark_torch.py, this is a compile-time plan, so it
    may not include allocations made outside of XLA's buffers (e.g. FFT library
    workspaces).
    """
    stats = compiled.memory_analysis()
    peak_bytes = (
        stats.temp_size_in_bytes
        + stats.output_size_in_bytes
        - stats.alias_size_in_bytes
    )
    return peak_bytes / (1024**2)


def calc_benchmark_stats(
    loss_func: Callable[..., Float[Array, ""]],
    x: Float[Array, "bs n_ch n_samples"],
    x_hat: Float[Array, "bs n_ch n_samples"],
    min_run_time: float = 1.0,
    path_idx: Optional[int] = None,
) -> Tuple[float, float, int, float, float]:
    compiled, compile_time, flops = compile_benchmark_fn(loss_func, x, x_hat, path_idx)
    log.info(f"Compiled in {compile_time:.2f} s")
    times, number = blocked_autorange(compiled, (x_hat, x), min_run_time)
    # Linear interpolation, the same as torch.quantile in torch.utils.benchmark
    q1, median, q3 = np.quantile(times, [0.25, 0.5, 0.75])
    n_runs = len(times) * number
    max_mem = calc_peak_mem_mb(compiled)
    return float(median), float(q3 - q1), n_runs, max_mem, flops


def main() -> None:
    log.info(
        f"Benchmarking with device={device}, num_threads={num_threads}, "
        f"min_run_time={min_run_time}, batch_size={batch_size}, "
        f"n_samples={n_samples}, run_scrapl={run_scrapl}"
    )
    log.info(f"JAX devices: {jax.devices()}")
    os.makedirs(OUT_DIR, exist_ok=True)

    config_path = os.path.join(CONFIGS_DIR, "losses/mss_meso_log.yml")
    mss_log_loss = load_class_from_yaml(config_path, MultiResolutionSTFTLoss)
    config_path = os.path.join(CONFIGS_DIR, "losses/mss_revisited.yml")
    mss_rev_loss = load_class_from_yaml(config_path, LogMSSLoss)
    config_path = os.path.join(CONFIGS_DIR, "losses/scat_1d.yml")
    scat_1d_loss = load_class_from_yaml(config_path, Scat1DLoss)
    config_path = os.path.join(CONFIGS_DIR, "losses/jtfs.yml")
    jtfs_loss = load_class_from_yaml(config_path, JTFSTLoss)

    # Created on the default device, which is set by JAX_PLATFORMS
    key_x, key_x_hat = jax.random.split(jax.random.key(42))
    x = jax.random.uniform(key_x, (batch_size, 1, n_samples))
    x_hat = jax.random.uniform(key_x_hat, (batch_size, 1, n_samples))

    df_cols = ["name", "median_time", "iqr", "n_runs", "max_mem_MB", "flops"]
    df_rows = []
    loss_funcs = [
        ("mss_log", mss_log_loss),
        ("mss_rev", mss_rev_loss),
        ("scat_1d", scat_1d_loss),
        ("jtfs", jtfs_loss),
    ]

    for name, loss_func in loss_funcs:
        log.info(f"Benchmarking {name}...")
        row = calc_benchmark_stats(
            loss_func=loss_func,
            x=x,
            x_hat=x_hat,
            min_run_time=min_run_time,
        )
        df_rows.append((name,) + row)

    if run_scrapl:
        config_path = os.path.join(CONFIGS_DIR, "losses/scrapl.yml")
        scrapl_loss = load_class_from_yaml(config_path, SCRAPLLoss)
        n_paths = scrapl_loss.n_paths
        scrapl_rows = []
        for path_idx in range(n_paths):
            log.info(f"Benchmarking scrapl path {path_idx} / {n_paths}...")
            row = calc_benchmark_stats(
                loss_func=scrapl_loss,
                x=x,
                x_hat=x_hat,
                min_run_time=min_run_time,
                path_idx=path_idx,
            )
            scrapl_rows.append((f"scrapl_path_{path_idx}",) + row)

        scrapl_df = pd.DataFrame(scrapl_rows, columns=df_cols)
        print(scrapl_df.to_string(index=False))
        save_path = os.path.join(OUT_DIR, f"scrapl_benchmark_jax_{device}.tsv")
        scrapl_df.to_csv(save_path, index=False, sep="\t")

        # Aggregated over the paths with the median (scrapl) and the mean (scrapl_mean,
        # the expected cost of a step when paths are sampled uniformly). Peak memory
        # is the max over the paths for both.
        for name, agg in [("scrapl", "median"), ("scrapl_mean", "mean")]:
            df_rows.append(
                (
                    name,
                    scrapl_df["median_time"].agg(agg),
                    scrapl_df["iqr"].agg(agg),
                    scrapl_df["n_runs"].agg(agg),
                    scrapl_df["max_mem_MB"].max(),
                    scrapl_df["flops"].agg(agg),
                )
            )
    df = pd.DataFrame(df_rows, columns=df_cols)
    print(df.to_string(index=False))
    save_path = os.path.join(OUT_DIR, f"benchmark_jax_{device}.tsv")
    df.to_csv(save_path, index=False, sep="\t")


if __name__ == "__main__":
    main()
