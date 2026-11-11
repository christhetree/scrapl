"""
Benchmarks the runtime and peak memory of a forward and backward pass of the mss_log,
mss_rev, scat_1d, jtfs, and scrapl (per path) losses. Adapted from
experiments/scripts/benchmark.py, but only needs the root scrapl-loss environment.

Run from the repository root with: uv run python scripts/benchmark_torch.py
"""

import logging
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import importlib
from typing import Callable, Any, Dict, Optional, Tuple

import pandas as pd
import yaml
import torch as tr
from torch import Tensor as T
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile
from torch.utils import benchmark
from torch.utils.benchmark import Measurement

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPTS_DIR)
# Copies of experiments/configs/losses/*.yml, with src.losses -> losses
CONFIGS_DIR = os.path.join(SCRIPTS_DIR, "configs")
OUT_DIR = os.path.join(ROOT_DIR, "out")

batch_size = 4
n_samples = 32768
num_threads = 1
# min_run_time = 15.0
min_run_time = 5.0
device = "cpu"
# device = "cuda"
run_scrapl = True


def load_class_from_yaml(config_path: str) -> Any:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    class_path = config["class_path"]
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    cls_instantiated = cls(**config["init_args"])
    return cls_instantiated


def benchmark_loss_func(
    x: T, x_hat: T, loss_func: Callable[..., T], path_idx: Optional[int] = None
) -> None:
    if path_idx is None:
        loss = loss_func(x_hat, x)
    else:
        loss = loss_func(x_hat, x, path_idx=path_idx)
    loss.backward()


def calc_cpu_peak_mem_mb(stmt: str, globals: Dict[str, Any], num_threads: int) -> float:
    """
    Peak CPU tensor memory (MB) allocated while running `stmt` once, relative to
    what was already allocated beforehand (the CPU equivalent of the CUDA
    measurement). Computed from the profiler's allocation / deallocation events.
    Frees of tensors allocated before profiling started are not reported (PyTorch
    warns when this happens), which can only overestimate the peak.
    """
    prev_num_threads = tr.get_num_threads()
    tr.set_num_threads(num_threads)
    try:
        with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
            eval(stmt, dict(globals))
    finally:
        tr.set_num_threads(prev_num_threads)
    mem_events = [
        e
        for e in prof.profiler.kineto_results.events()
        if e.name() == "[memory]" and e.device_type() == DeviceType.CPU
    ]
    mem_events.sort(key=lambda e: e.start_ns())
    curr_bytes = 0
    peak_bytes = 0
    for e in mem_events:
        curr_bytes += e.nbytes()  # Negative for deallocations
        peak_bytes = max(peak_bytes, curr_bytes)
    return peak_bytes / (1024**2)


def calc_cuda_peak_mem_mb(stmt: str, globals: Dict[str, Any], device: str) -> float:
    """
    Peak CUDA memory (MB) allocated while running `stmt` once, relative to what was
    already allocated beforehand (other loss modules, persistent .grad buffers) so
    rows are not cumulative.
    """
    tr.cuda.synchronize(device)
    base_mem = tr.cuda.memory_allocated(device)
    tr.cuda.reset_peak_memory_stats(device)
    eval(stmt, dict(globals))
    tr.cuda.synchronize(device)
    peak_bytes = tr.cuda.max_memory_allocated(device) - base_mem
    return peak_bytes / (1024**2)


def calc_benchmark_stats(
    globals: Dict[str, Any],
    device: str,
    num_threads: int = 1,
    min_run_time: float = 1.0,
    stmt: str = "fn(x, x_hat, loss_func, path_idx)",
) -> Tuple[float, float, int, float]:
    result: Measurement = benchmark.Timer(
        stmt=stmt,
        globals=globals,
        num_threads=num_threads,
    ).blocked_autorange(min_run_time=min_run_time)
    # Peak memory from a separate untimed steady-state pass after the Timer's
    # warmup, so first-call allocations are excluded and the CPU profiler
    # overhead does not affect the timings
    if device.startswith("cuda"):
        max_mem = calc_cuda_peak_mem_mb(stmt, globals, device)
    else:
        max_mem = calc_cpu_peak_mem_mb(stmt, globals, num_threads)
    # result.times has one entry per measurement block of number_per_run calls
    n_runs = len(result.raw_times) * result.number_per_run
    return result.median, result.iqr, n_runs, max_mem


def main() -> None:
    log.info(
        f"Benchmarking with device={device}, num_threads={num_threads}, "
        f"min_run_time={min_run_time}, batch_size={batch_size}, "
        f"n_samples={n_samples}, run_scrapl={run_scrapl}"
    )
    tr.manual_seed(42)
    os.makedirs(OUT_DIR, exist_ok=True)

    config_path = os.path.join(CONFIGS_DIR, "losses/mss_meso_log.yml")
    mss_log_loss = load_class_from_yaml(config_path)
    config_path = os.path.join(CONFIGS_DIR, "losses/mss_revisited.yml")
    mss_rev_loss = load_class_from_yaml(config_path)
    config_path = os.path.join(CONFIGS_DIR, "losses/scat_1d.yml")
    scat_1d_loss = load_class_from_yaml(config_path)
    config_path = os.path.join(CONFIGS_DIR, "losses/jtfs.yml")
    jtfs_loss = load_class_from_yaml(config_path)

    # Create directly on device so x_hat is a leaf (no GPU -> CPU grad copy)
    x = tr.rand((batch_size, 1, n_samples), device=device)
    x_hat = tr.rand((batch_size, 1, n_samples), device=device, requires_grad=True)

    df_cols = ["name", "median_time", "iqr", "n_runs", "max_mem_MB"]
    df_rows = []
    globals = {
        "x": x,
        "x_hat": x_hat,
        "fn": benchmark_loss_func,
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
        globals["loss_func"] = loss_func
        row = calc_benchmark_stats(
            globals=globals,
            device=device,
            num_threads=num_threads,
            min_run_time=min_run_time,
        )
        df_rows.append((name,) + row)

    if run_scrapl:
        config_path = os.path.join(CONFIGS_DIR, "losses/scrapl.yml")
        scrapl_loss = load_class_from_yaml(config_path)
        n_paths = scrapl_loss.n_paths
        scrapl_rows = []
        scrapl_loss = scrapl_loss.to(device)
        globals["loss_func"] = scrapl_loss
        for path_idx in range(n_paths):
            log.info(f"Benchmarking scrapl path {path_idx} / {n_paths}...")
            globals["path_idx"] = path_idx
            row = calc_benchmark_stats(
                globals=globals,
                device=device,
                num_threads=num_threads,
                min_run_time=min_run_time,
            )
            scrapl_rows.append((f"scrapl_path_{path_idx}",) + row)

        scrapl_df = pd.DataFrame(scrapl_rows, columns=df_cols)
        print(scrapl_df.to_string(index=False))
        save_path = os.path.join(OUT_DIR, f"scrapl_benchmark_torch_{device}.tsv")
        scrapl_df.to_csv(save_path, index=False, sep="\t")

        df_rows.append(
            (
                "scrapl",
                scrapl_df["median_time"].median(),
                scrapl_df["iqr"].median(),
                scrapl_df["n_runs"].median(),
                scrapl_df["max_mem_MB"].max(),
            )
        )
    df = pd.DataFrame(df_rows, columns=df_cols)
    print(df.to_string(index=False))
    save_path = os.path.join(OUT_DIR, f"benchmark_torch_{device}.tsv")
    df.to_csv(save_path, index=False, sep="\t")


if __name__ == "__main__":
    main()
