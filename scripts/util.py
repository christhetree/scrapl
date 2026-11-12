"""
Logic shared by the benchmark scripts: benchmark_torch.py, benchmark_jax.py,
benchmark_warmup_torch.py, and benchmark_warmup_jax.py. PyTorch and JAX are only
imported inside the functions that use them, so that the JAX scripts do not import
PyTorch, and can configure JAX with setup_jax_env before it is imported.
"""

import gc
import importlib
import inspect
import logging
import os
import sys
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

if TYPE_CHECKING:
    from jax.stages import Compiled
    from torch import Tensor as T

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPTS_DIR)
CONFIGS_DIR = os.path.join(SCRIPTS_DIR, "configs")
OUT_DIR = os.path.join(ROOT_DIR, "out")

# The columns of the benchmark results. The JAX benchmarks also report the FLOPs and
# the compilation wall-clock and CPU times (see compile_fn), except for the warmup
# benchmark, which omits the FLOPs.
TORCH_COLS = ["name", "median_time", "iqr", "n_runs", "max_mem_MB"]
JAX_COLS = TORCH_COLS + ["flops", "compile_time", "compile_cpu_time"]
JAX_WARMUP_COLS = TORCH_COLS + ["compile_time", "compile_cpu_time"]

# The summary rows of per-path results: their names and the aggregation over the paths
# (see summarize_paths). scrapl_mean is the expected cost when paths are sampled
# uniformly, and scrapl_total the cost of running every path once.
PATH_AGGS = [
    ("scrapl_median", "median"),
    ("scrapl_mean", "mean"),
    ("scrapl_total", "sum"),
]


# Shared =========================================================================
def load_config(config_name: str) -> Dict[str, Any]:
    """The loss config configs/losses/{config_name}.yml."""
    config_path = os.path.join(CONFIGS_DIR, "losses", f"{config_name}.yml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_class_from_yaml(
    config_name: str, cls: Optional[Callable[..., Any]] = None, **overrides: Any
) -> Any:
    """
    Instantiates the class_path class of the loss config
    configs/losses/{config_name}.yml with its init_args, updated with `overrides`
    (e.g. n_theta=2). If `cls` is given (e.g. the
    JAX equivalent of the class), it is instantiated instead, so that all benchmarks
    use identical hyperparameters, and init args that `cls` does not accept are
    dropped (e.g. the SCRAPL optimizer args, which are part of scrapl_optimizer in JAX
    and are not used by the benchmarks).
    """
    config = load_config(config_name)
    init_args = {**config["init_args"], **overrides}
    if cls is None:
        module_name, class_name = config["class_path"].rsplit(".", 1)
        cls = getattr(importlib.import_module(module_name), class_name)
        return cls(**init_args)
    params = inspect.signature(cls).parameters
    ignored = sorted(k for k in init_args if k not in params)
    if ignored:
        log.info(f"Ignoring init args not used by {cls.__name__}: {ignored}")
    return cls(**{k: v for k, v in init_args.items() if k in params})


def format_sci(val: float) -> str:
    """Formats `val` in scientific notation with 3 significant digits, e.g. 3.24e6."""
    if not np.isfinite(val):
        return str(val)
    mantissa, exponent = f"{val:.2e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def save_df(df: pd.DataFrame, file_name: str) -> None:
    """
    Prints `df`, with the FLOPs in scientific notation (see format_sci), and saves it
    as a TSV file in OUT_DIR with full precision.
    """
    formatters = {"flops": format_sci} if "flops" in df.columns else None
    print(df.to_string(index=False, formatters=formatters))
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, file_name), index=False, sep="\t")


def summarize_paths(df: pd.DataFrame) -> List[Tuple[Any, ...]]:
    """
    The summary rows of the per-path rows of `df`: one row per (name, agg) in
    PATH_AGGS, which aggregates every column over the paths with `agg` ("median",
    "mean", or "sum"), except for peak memory, which is the max over the paths. For
    "sum", the IQR is the sum of the IQRs, which is the IQR of the total runtime only
    if the runtimes of the paths are perfectly correlated (otherwise it usually
    overestimates it).
    """
    rows = []
    for name, agg in PATH_AGGS:
        row = [name]
        for col in df.columns[1:]:
            row.append(df[col].max() if col == "max_mem_MB" else df[col].agg(agg))
        rows.append(tuple(row))
    return rows


def make_inputs(
    batch_size: int, n_samples: int, n_batches: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Random inputs, uniform on [0, 1), shared by the PyTorch and JAX benchmarks so that
    both compute on identical data: the target batches xs of shape
    (n_batches, batch_size, 1, n_samples), the first of which is the target x of the
    training step benchmarks, and the prediction x_hat of shape
    (batch_size, 1, n_samples). x_hat and xs[0] do not depend on n_batches.
    """
    rng = np.random.default_rng(42)
    x_hat = rng.random((batch_size, 1, n_samples), dtype=np.float32)
    xs = rng.random((n_batches, batch_size, 1, n_samples), dtype=np.float32)
    return xs, x_hat


def make_theta_is_arrays(n_theta: int, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Weights of the minimal encoder and decoder (synth) used to benchmark the θ-IS
    warmup (see encoder and decoder in benchmark_warmup_torch.py and
    benchmark_warmup_jax.py), so that its cost is almost entirely the SCRAPL loss's,
    regardless of the encoder and decoder used in practice. The encoder is a random
    weight matrix (enc_w) that the waveform is multiplied by, scaled so that theta is
    O(1), and the decoder is a fixed linear synth without weights that mixes n_theta
    sinusoids (basis) with theta as their amplitudes.
    """
    rng = np.random.default_rng(0)
    enc_w = rng.standard_normal((n_samples, n_theta)) / np.sqrt(n_samples)
    freqs = 0.01 * np.arange(1, n_theta + 1)  # Cycles per sample
    basis = np.sin(2 * np.pi * freqs[:, None] * np.arange(n_samples))
    return enc_w.astype(np.float32), basis.astype(np.float32)


# PyTorch ========================================================================
def benchmark_loss_func(
    x_hat: "T",
    x: "T",
    loss_func: Callable[..., "T"],
    path_idx: Optional[int] = None,
) -> None:
    """
    A forward and backward pass of `loss_func`. The gradient w.r.t. x_hat is reset
    first, so that each call allocates a new one instead of accumulating into the
    previous one, the same as the JAX benchmarks (see make_loss_grad_fn).
    """
    x_hat.grad = None
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
    import torch as tr
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

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
    import torch as tr

    tr.cuda.synchronize(device)
    base_mem = tr.cuda.memory_allocated(device)
    tr.cuda.reset_peak_memory_stats(device)
    eval(stmt, dict(globals))
    tr.cuda.synchronize(device)
    peak_bytes = tr.cuda.max_memory_allocated(device) - base_mem
    return peak_bytes / (1024**2)


def benchmark_torch(
    name: str,
    globals: Dict[str, Any],
    device: str,
    num_threads: int = 1,
    min_run_time: float = 1.0,
    stmt: str = "fn(x_hat, x, loss_func, path_idx)",
) -> Tuple[Any, ...]:
    """
    A row of TORCH_COLS for running `stmt` with `globals`: its median runtime, the
    IQR of the runtime, the number of runs, and peak memory (MB).
    """
    from torch.utils import benchmark

    result = benchmark.Timer(
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
    return name, result.median, result.iqr, n_runs, max_mem


def benchmark_scrapl_paths_torch(
    scrapl_loss: Any,
    x: "T",
    x_hat: "T",
    device: str,
    num_threads: int = 1,
    min_run_time: float = 1.0,
) -> pd.DataFrame:
    """Benchmarks a forward and backward pass of every path of `scrapl_loss`."""
    globals = {
        "x": x,
        "x_hat": x_hat,
        "fn": benchmark_loss_func,
        "loss_func": scrapl_loss,
    }
    rows = []
    for path_idx in range(scrapl_loss.n_paths):
        log.info(f"Benchmarking scrapl path {path_idx} / {scrapl_loss.n_paths}...")
        globals["path_idx"] = path_idx
        row = benchmark_torch(
            f"scrapl_path_{path_idx}",
            globals,
            device,
            num_threads=num_threads,
            min_run_time=min_run_time,
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=TORCH_COLS)


# JAX ============================================================================
def setup_jax_env(
    device: str, num_threads: int, compilation_cache_dir: Optional[str] = None
) -> None:
    """
    Configures JAX, which must be done before it is imported. If
    `compilation_cache_dir` is given, compiled functions are cached on disk there and
    loaded in later runs (only those that took at least 1 s to compile). The
    compile_time and compile_cpu_time of a cached function then measure loading it,
    so delete the directory first to measure the compilation.
    """
    if "jax" in sys.modules:
        log.warning(
            "JAX was imported before setup_jax_env, so the device and number of "
            "threads may not be applied"
        )
    # On GPU, the CPU backend is also kept available (the GPU is still the default
    # device), since calc_flops needs it
    os.environ["JAX_PLATFORMS"] = device if device == "cpu" else f"{device},cpu"
    if device == "cpu":
        # The number of CPUs that the XLA CPU client uses, which sizes its thread pool
        # (the equivalent of torch.set_num_threads). With 1, a SCRAPL path used 1.00
        # CPU seconds per wall-clock second (4.45 without it). XLA_FLAGS cannot do this:
        # --xla_cpu_multi_thread_eigen=false only affects Eigen ops (1.83), and
        # intra_op_parallelism_threads is not an XLA flag.
        os.environ["PJRT_NPROC"] = str(num_threads)
    # Otherwise JAX preallocates 75% of the GPU memory on startup
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if compilation_cache_dir is not None:
        cache_dir = os.path.abspath(os.path.expanduser(compilation_cache_dir))
        os.environ["JAX_COMPILATION_CACHE_DIR"] = cache_dir
        log.info(f"Using the JAX compilation cache in {cache_dir}")


def make_loss_grad_fn(
    loss_func: Callable[..., Any], path_idx: Optional[int] = None
) -> Callable[..., Any]:
    """
    A forward and backward pass of `loss_func`, returning the loss and its gradient
    w.r.t. x_hat. `path_idx` must be a Python int, so each SCRAPL path is compiled
    separately.
    """
    import jax

    def fn(x_hat: Any, x: Any) -> Any:
        if path_idx is None:
            return loss_func(x_hat, x)
        return loss_func(x_hat, x, path_idx=path_idx)

    return jax.value_and_grad(fn)


def calc_flops(fn: Callable[..., Any], args: Tuple[Any, ...]) -> float:
    """
    FLOPs of `fn` estimated by XLA's HloCostAnalysis before XLA's optimizations (e.g.
    fusion), so they do not depend on the device. Transcendental ops such as sqrt and
    log are counted separately by XLA, so they are not included. XLA counts the body
    of a loop (e.g. lax.scan, lax.map, or lax.while_loop) only once. `fn` is always
    lowered for the CPU, from the shapes and dtypes of `args` only, since JAX only
    supports this cost analysis on its built-in backends, such as the CPU (for CUDA
    it returns None). Returns NaN if the CPU backend is unavailable.
    """
    import jax

    try:
        cpu = jax.devices("cpu")[0]
    except RuntimeError:
        log.warning("The JAX CPU backend is unavailable, so FLOPs are not calculated")
        return float("nan")
    sharding = jax.sharding.SingleDeviceSharding(cpu)
    cpu_args = jax.tree.map(
        lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype, sharding=sharding), args
    )
    cost = jax.jit(fn).lower(*cpu_args).cost_analysis()
    if cost is None:
        log.warning("XLA's cost analysis is unavailable, so FLOPs are not calculated")
        return float("nan")
    return cost.get("flops", 0.0)  # 0 is omitted


def compile_fn(
    fn: Callable[..., Any], args: Tuple[Any, ...]
) -> Tuple["Compiled", float, float]:
    """
    Ahead-of-time JIT compiles `fn` for `args`. Returns the compiled function, the
    compilation time (wall-clock), and the compilation CPU time (user + system time
    summed over all threads of the process). Unlike running the compiled function,
    compiling is not limited to num_threads (PJRT_NPROC), so the CPU time is roughly
    what compiling would take on a single core.
    """
    import jax

    start, cpu_start = time.perf_counter(), time.process_time()
    compiled = jax.jit(fn).lower(*args).compile()
    compile_time = time.perf_counter() - start
    compile_cpu_time = time.process_time() - cpu_start
    return compiled, compile_time, compile_cpu_time


def time_n_calls(fn: Callable[..., Any], args: Tuple[Any, ...], number: int) -> float:
    """
    Times `number` calls of `fn`, like timeit.Timer.timeit as used by
    torch.utils.benchmark: garbage collection is disabled while timing, and since JAX
    dispatches asynchronously, the device is only synchronized once at the end (the
    same as torch.utils.benchmark on CUDA). Calls execute in order on the device, so
    waiting for the last output waits for all of them.
    """
    import jax

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
    # At least one block is timed, even if min_run_time is 0, the same as PyTorch
    while not raw_times or sum(raw_times) < min_run_time:
        raw_times.append(time_n_calls(fn, args, number))
    times = [t / number for t in raw_times]
    return times, number


def calc_compiled_peak_mem_mb(compiled: "Compiled") -> float:
    """
    Peak memory (MB) allocated while running `compiled` once, relative to what was
    already allocated beforehand (the inputs and the loss's filters), as planned by
    XLA's buffer assignment: temporary buffers (reused based on liveness) plus the
    newly allocated outputs. Computed the same way on CPU and CUDA. Unlike the
    runtime measurements of the PyTorch benchmarks, this is a compile-time plan, so it
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


def benchmark_jax(
    name: str,
    fn: Callable[..., Any],
    args: Tuple[Any, ...],
    min_run_time: float = 1.0,
    with_flops: bool = True,
) -> Tuple[Any, ...]:
    """
    A row of JAX_COLS (JAX_WARMUP_COLS without `with_flops`) for `fn` with `args`:
    after compiling `fn`, its median runtime, the IQR of the runtime, the number of
    runs, peak memory (MB), FLOPs (see calc_flops), and the compilation wall-clock
    and CPU times (see compile_fn).
    """
    compiled, compile_time, compile_cpu_time = compile_fn(fn, args)
    log.info(f"Compiled in {compile_time:.2f} s ({compile_cpu_time:.2f} s CPU time)")
    times, number = blocked_autorange(compiled, args, min_run_time)
    # Linear interpolation, the same as torch.quantile in torch.utils.benchmark
    q1, median, q3 = np.quantile(times, [0.25, 0.5, 0.75])
    n_runs = len(times) * number
    max_mem = calc_compiled_peak_mem_mb(compiled)
    row = [name, float(median), float(q3 - q1), n_runs, max_mem]
    if with_flops:
        row.append(calc_flops(fn, args))
    return tuple(row + [compile_time, compile_cpu_time])


def benchmark_scrapl_paths_jax(
    scrapl_loss: Any,
    x: Any,
    x_hat: Any,
    min_run_time: float = 1.0,
) -> pd.DataFrame:
    """
    Benchmarks a forward and backward pass of every path of `scrapl_loss` (see
    benchmark_jax).
    """
    rows = []
    for path_idx in range(scrapl_loss.n_paths):
        log.info(f"Benchmarking scrapl path {path_idx} / {scrapl_loss.n_paths}...")
        fn = make_loss_grad_fn(scrapl_loss, path_idx)
        rows.append(
            benchmark_jax(f"scrapl_path_{path_idx}", fn, (x_hat, x), min_run_time)
        )
    return pd.DataFrame(rows, columns=JAX_COLS)
