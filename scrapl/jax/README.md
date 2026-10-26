<div align="center">
<h1><b>SCRAPL</b>: <b>Sc</b>attering Transform with <b>Ra</b>ndom <b>P</b>aths for Machine <b>L</b>earning</h1>

[![Release](https://img.shields.io/badge/PyPI-v0.1.0-green)](https://pypi.org/project/scrapl-loss/)
[![PyTorch](https://img.shields.io/badge/PyTorch-supported-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![JAX](https://img.shields.io/badge/JAX-supported-5E97F6)](https://github.com/jax-ml/jax)
[![OpenReview](https://img.shields.io/badge/OpenReview-ICLR_2026_Paper-b31b1b.svg)](https://openreview.net/forum?id=RuYwbd5xYa)
[![Video](https://img.shields.io/badge/%F0%9F%8E%A5-Video-blue)](https://iclr.cc/virtual/2026/poster/10009456)
[![Poster](https://img.shields.io/badge/%F0%9F%AA%A7-Poster-blue)](https://christhetree.github.io/scrapl/docs/mitcheltree2026scrapl_poster.pdf)
[![Listening Samples](https://img.shields.io/badge/%F0%9F%94%8A%F0%9F%8E%B6-Listening_Samples-blue)](https://christhetree.github.io/scrapl/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](https://opensource.org/license/mit)
</div>

`scrapl-loss` is a Python package for efficient evaluation of multivariable scattering transforms, specifically designed for use as a differentiable loss function in machine learning applications and perceptual quality assessment of deep inverse problems in computer vision, speech, and audio processing. 
It implements the "Scattering Transform with Random Paths for Machine Learning" (SCRAPL) algorithm, which accelerates gradient-based optimization of neural networks with scattering transforms by stochastically sampling scattering paths to approximate the full transform's gradient.
This reduces the computational and memory costs by one to three orders of magnitude. 
SCRAPL also leverages specialized stochastic optimization techniques (𝒫-Adam, 𝒫-SAGA) and an architecture-informed importance sampling heuristic (θ-IS) to stabilize gradients and improve convergence.
For more details, please see the [ICLR 2026 paper](https://openreview.net/forum?id=RuYwbd5xYa).

`scrapl-loss` contains a PyTorch and JAX implementation of the SCRAPL algorithm for the [joint time–frequency scattering transform (JTFS)](https://www.kymat.io/ismir23-tutorial/ch1_intro/why_wavelets.html), which demodulates spectrotemporal patterns at multiple scales and rates, and has been shown to correlate with human perception ([Lostanlen et al., 2021](https://link.springer.com/article/10.1186/s13636-020-00187-z); [Tian et al., 2025](https://doi.org/10.48550/arXiv.2507.07764)).
In our experiments, we find that SCRAPL accomplishes a favorable tradeoff between goodness of fit and computational efficiency on unsupervised sound matching, i.e., a nonlinear inverse problem in which the forward operator implements an audio synthesizer.
In the context of differentiable digital signal processing (DDSP), the state-of-the-art perceptual loss function for this task is multiscale spectral loss (MSS, [Yamamoto et al., 2020](https://doi.org/10.48550/arXiv.1910.11480); [Engel et al., 2020](https://openreview.net/forum?id=B1x1ma4tDr)). 
However, the gradient of MSS is uninformative when input and reconstruction are misaligned or when the synthesizer controls involve spectrotemporal modulations ([Vahidi et al., 2023](https://doi.org/10.48550/arXiv.2301.10183)). 
Taking advantage of the stability guarantees of JTFS, SCRAPL expands the class of synthesizers which can be effectively decoded via DDSP.

Overall, SCRAPL with the JTFS is best suited for comparing audio signals that meet one or more of the following criteria:
- Contain spectrotemporal modulations
- Are non-stationary
- Benefit from multi-resolution analysis like percussive sounds (which consist of both transients and sustained components)
- Are misaligned in time or frequency and therefore benefit from temporal and frequential shift invariance
- Cannot be effectively compared with time-domain (e.g. mean squared error (MSE) / error-to-signal ratio (ESR) loss), frequency-domain (e.g. multiscale spectral (MSS) loss), or neural embedding-based (e.g. CLAP / PANNs / EnCodec distances) losses 

> [!NOTE]
> This documentation README is for JAX, [please click here for the PyTorch documentation README](https://github.com/christhetree/scrapl/blob/main/README.md).


![image](https://raw.githubusercontent.com/christhetree/scrapl/main/docs/figs/relative_param_error_vs_computation_colored.svg)


_Figure 1: Mean average synthesizer parameter error (y-axis) versus computational cost (x-axis) of unsupervised sound matching models for the granular synthesis task. 
Both axes are rescaled by the performance of a supervised model with the same number of parameters. 
Whiskers denote 95% CI, estimated over 20 random seeds. Due to computational limitations, 
JTFS-based sound matching is evaluated only once._


## Table of Contents

- [Installation](#installation)
- [Examples (JAX)](#examples-jax)
- [Hyperparameters](#hyperparameters)
- [Best Practices](#best-practices)
- [Algorithm](#algorithm)
- [Implementation Details (JAX)](#implementation-details-jax)
- [Known Issues (JAX)](#known-issues-jax)
- [Citation](#citation)


## Installation


**1. JAX implementation:**

```sh
uv add 'scrapl-loss[jax]'
```

or

```sh
pip install 'scrapl-loss[jax]'
```

Then: 

`from scrapl.jax import SCRAPLLoss`


**2. PyTorch implementation:**

```sh
uv add 'scrapl-loss[torch]'
```

or

```sh
pip install 'scrapl-loss[torch]'
```

Then: 

`from scrapl.torch import SCRAPLLoss` or `from scrapl import SCRAPLLoss`


**3. All extras:**

```sh
uv add 'scrapl-loss[torch,jax,test]'
```

or

```sh
pip install 'scrapl-loss[torch,jax,test]'
```


**4. From source checkout:**

```sh
git submodule update --init scrapl/kymatio scrapl/pytorch_hessian_eigenthings
uv sync --extra torch --extra jax --extra test
```

To run the test suite with `uv`: 

```sh
uv run pytest
```

**Requirements:**

The package requires Python 3.13 or higher, `numpy>=2.1.0`, and `scipy>=1.14.1`.
The Torch backend requires `torch>=2.6.0` and the JAX backend requires
`jax>=0.5.0`, `equinox>=0.11.5`, `optax>=0.2.5`, `jaxtyping>=0.2.20`, and `tqdm>=4.66.0`.
A bare installation with no extras includes neither framework, so JAX users
should select `[jax]`.
The JAX backend does not require PyTorch.
Trying to access an uninstalled backend will report the required installation command.


## Examples (JAX)

> [!NOTE]
> This documentation README is for JAX, [please click here for the PyTorch documentation README](https://github.com/christhetree/scrapl/blob/main/README.md).


Importing and initializing `SCRAPLLoss` with the minimum required arguments:

```python
# Import SCRAPLLoss from the scrapl.jax Python module
from scrapl.jax import SCRAPLLoss

# Initialize SCRAPLLoss with the minimum required arguments
scrapl_loss = SCRAPLLoss(
    shape=48000,  # Length of x and x_target in samples
    J=12,         # Number of octaves (1st and 2nd order temporal filters)
    Q1=8,         # Wavelets per octave (1st order temporal filters)
    Q2=2,         # Wavelets per octave (2nd order temporal filters)
    J_fr=3,       # Number of octaves (2nd order frequential filters)
    Q_fr=2,       # Wavelets per octave (2nd order frequential filters)
)
```

Console output:

```text
INFO:scrapl.jax.loss:SCRAPLLoss:
J=12, Q1=8, Q2=2, Jfr=3, Qfr=2, T=None, F=None
use_rho_log1p          = False, eps = 0.001
number of SCRAPL paths = 315
unif_prob              = 0.00317460
```

Calculating the loss for two signals: 

```python 
import jax

# Create two random arrays of shape (batch_size, num_channels, signal_length)
key = jax.random.key(0)
key_x, key_target, key_loss = jax.random.split(key, 3)
x = jax.random.normal(key_x, (4, 1, 48000))  # Example input array
x_target = jax.random.normal(key_target, (4, 1, 48000))  # Example target array
# Compute the SCRAPL loss between x and x_target. Since SCRAPL is stochastic,
# a different PRNG key will usually sample a different scattering path.
loss = scrapl_loss(x, x_target, key=key_loss)
print(f"SCRAPL loss: {float(loss)}")
```

> [!NOTE]
> Unlike the PyTorch version which is stateful (tracking `last_sampled_path_idx`, `path_counts`, `state_dict`, and `clear()`), the JAX `SCRAPLLoss` is a pure, immutable Equinox module. Path sampling is explicit via `scrapl_loss.sample_path(key)` and all state lives in caller-managed PyTrees or Optax optimizer states.

`SCRAPLLoss` utility attributes and methods:

```python
print(f"Number of scattering paths: {scrapl_loss.n_paths}")
print(f"Uniform path sampling probability: {scrapl_loss.unif_prob:.6f}")

# Sample a path index explicitly with a JAX PRNG key
# (the same key will sample the same path index every time)
path_idx = scrapl_loss.sample_path(jax.random.key(42))
print(f"Sampled path index (PRNG key): {path_idx}")

# Calculate the loss for a specific path index
loss = scrapl_loss(x, x_target, path_idx=path_idx)
print(f"SCRAPL loss (path {path_idx}): {float(loss)}")
```

Console output:

```text
Number of scattering paths: 315
Uniform path sampling probability: 0.003175
Sampled path index (PRNG key): 99
SCRAPL loss (path 99): 0.0009700539521872997
```

### Using 𝒫-Adam and 𝒫-SAGA

𝒫-Adam and 𝒫-SAGA are enabled by default when creating an [Optax](https://github.com/google-deepmind/optax) optimizer using `scrapl_optimizer`.
The Adam hyperparameters β1, β2, and ε can be set using the `p_adam_b1`, `p_adam_b2`, and `p_adam_eps` arguments of `scrapl_optimizer`.
They can also be chained manually using the `scale_by_gradient_multiplier`, `p_adam`, and `p_saga` Optax gradient transformations.

> [!IMPORTANT]
> The same sampled `path_idx` must be passed to both `SCRAPLLoss` and the optimizer's `update()` for 𝒫-Adam and 𝒫-SAGA to update the correct path.
> `path_idx` must be a Python integer, so it should be sampled using `sample_path()` outside of JIT-compiled functions.

For example:

```python
import equinox as eqx
from scrapl.jax import scrapl_optimizer

# Toy example model
model = eqx.nn.MLP(
    in_size=48000,
    out_size=48000,
    width_size=8,
    depth=1,
    activation=jax.nn.relu,
    final_activation=jax.nn.tanh,
    key=jax.random.key(1),
)
params = eqx.filter(model, eqx.is_inexact_array)

# Create an Optax optimizer with P-Adam and P-SAGA (and the gradient multiplier,
# weight decay, and learning rate) chained together
optimizer = scrapl_optimizer(
    learning_rate=1e-4, weight_decay=0.01, n_paths=scrapl_loss.n_paths
)
opt_state = optimizer.init(params)

# Example training step with P-Adam and P-SAGA. `path_idx` must be a Python int,
# so it is a static argument of the JIT-compiled function (each new path index
# triggers a compilation the first time it is used)
@eqx.filter_jit
def train_step(model, opt_state, x, x_target, path_idx):
    def loss_fn(m):
        x_hat = jax.vmap(m)(x.squeeze(1))[:, None, :]
        return scrapl_loss(x_hat, x_target, path_idx=path_idx)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    params = eqx.filter(model, eqx.is_inexact_array)
    # The same path index must be passed to the optimizer for P-Adam and P-SAGA
    updates, opt_state = optimizer.update(
        grads, opt_state, params=params, path_idx=path_idx
    )
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

# Sample the path index outside of the JIT-compiled function
path_idx = scrapl_loss.sample_path(key_loss)
model, opt_state, loss = train_step(model, opt_state, x, x_target, path_idx)
print(f"SCRAPL loss: {float(loss)}")
```

### Importance Sampling Warmup

The SCRAPL algorithm includes an importance sampling heuristic (θ-IS) that learns a non-uniform sampling distribution over scattering paths given an encoder and decoder (synth) self-supervised training architecture. 
This is done by measuring the curvature of the loss landscape with respect to the encoder output / decoder (synth) input parameters θ<sub>synth</sub> for each path, see Sections 3.4, 4.3, and 5.2 in the [paper](https://openreview.net/forum?id=RuYwbd5xYa) for more details.
This warmup step is done once before training on a subset of the training data (typically 32-256 samples) and can be parallelized across paths and θ<sub>synth</sub> (using `vmap_theta=True`).

> [!IMPORTANT]
> The encoder and decoder (synth) must be deterministic during warmup, but can be non-deterministic otherwise.
> `warmup_lc_hvp()` verifies this before warmup and raises an error if they are not.

```python
import equinox as eqx
import jax
from scrapl.jax import SCRAPLLoss, load_warmup_from_dir, warmup_lc_hvp

# Setup
bs = 4
n_ch = 1
n_samples = 8096
# Number of parameters output by the encoder and input to the decoder (synth)
n_theta = 3
# Number of batches to use for warmup; when possible, one large batch filling all
# available GPU memory is more efficient than many smaller batches
n_batches = 1
key_enc, key_dec, key_data, key_warmup = jax.random.split(jax.random.key(42), 4)

# Provide a neural network encoder that outputs `n_theta` parameters
encoder = eqx.nn.MLP(
    in_size=n_samples,
    out_size=n_theta,
    width_size=n_theta,
    depth=1,
    activation=jax.nn.relu,
    final_activation=jax.nn.sigmoid,
    key=key_enc,
)
# Split the encoder into its trainable weights and its static parts
enc_params, enc_static = eqx.partition(encoder, eqx.is_inexact_array)
# Make the encoder forward call functional (stateless)
def enc_fn(params, x):
    enc = eqx.combine(params, enc_static)
    return jax.vmap(enc)(x.squeeze(1))

# Provide a differentiable decoder (synth) that takes as input `n_theta` parameters
decoder = eqx.nn.MLP(
    in_size=n_theta,
    out_size=n_samples,
    width_size=n_theta,
    depth=1,
    activation=jax.nn.relu,
    final_activation=jax.nn.tanh,
    key=key_dec,
)
# Make the decoder forward call functional (stateless) and ensure it outputs the
# correct shape for the SCRAPLLoss (bs, n_ch, n_samples)
dec_fn = lambda theta: jax.vmap(decoder)(theta)[:, None, :]

# Create an array with batches of input data for `enc_fn` of shape
# (n_batches, bs, n_ch, n_samples); typically this would be gathered from a dataloader
xs = jax.random.uniform(key_data, (n_batches, bs, n_ch, n_samples))

# Initialize SCRAPLLoss with the minimum required arguments
scrapl_loss = SCRAPLLoss(
    shape=n_samples,
    J=3,
    Q1=1,
    Q2=1,
    J_fr=2,
    Q_fr=1,
)
# Before warmup, all path sampling probabilities are uniform
print(f"Uniform path sampling probability: {scrapl_loss.unif_prob:.6f}")

# Run the warmup. This can be parallelized across paths by specifying different
# `start_path_idx` and `end_path_idx` subsets on multiple devices.
warmup = warmup_lc_hvp(
    scrapl_loss,
    enc_fn,
    dec_fn,
    xs,
    enc_params,
    key=key_warmup,
    save_dir="scrapl_warmup",
)
# We see that after warmup, the path sampling probabilities are no longer uniform
print(
    f"[min, max] path sampling probabilities (after warmup): "
    f"[{float(warmup.probs.min()):.6f}, {float(warmup.probs.max()):.6f}]"
)

# The path sampling probabilities are passed explicitly when sampling paths
path_idx = scrapl_loss.sample_path(jax.random.key(0), probs=warmup.probs)

# If warmup was conducted in parallel across multiple devices, the path sampling
# probabilities can be loaded from a directory
warmup = load_warmup_from_dir(scrapl_loss, "scrapl_warmup")
```

Console output:

```text
INFO:scrapl.jax.loss:SCRAPLLoss:
J=3, Q1=1, Q2=1, Jfr=2, Qfr=1, T=None, F=None
use_rho_log1p          = False, eps = 0.001
number of SCRAPL paths = 7
unif_prob              = 0.14285714

Uniform path sampling probability: 0.142857

INFO:scrapl.jax.warmup:Starting warmup_lc_hvp for paths [0, 7) of 7, 3 theta, 1 batch(es), 20 iter, tol = 0.01 (vmap_theta = False)
Theta-IS warmup: 100%|██████████| 7/7 [00:06<00:00,  1.13path/s]
INFO:scrapl.jax.warmup:Saving theta-IS warmup probabilities to /path/to/scrapl_warmup/probs.npy

[min, max] path sampling probabilities (after warmup): [0.112551, 0.194698]

INFO:scrapl.jax.warmup:Loaded theta-IS warmup from directory:
	/path/to/scrapl_warmup
	min prob = 0.112551, max prob = 0.194698
```

> [!NOTE]
> `warmup_lc_hvp` returns a `ThetaISResult` containing `curvatures` and `relative_residuals` of shape `(n_paths, n_theta)`, and `probs` of shape `(n_paths,)`. It leaves the loss and parameters unchanged, so the result should be kept in your training / checkpoint state and its probabilities passed explicitly to `sample_path(key, probs=...)` or `scrapl_loss(..., key=key, probs=...)`.


## Hyperparameters

### `SCRAPLLoss` (SCRAPL Algorithm Implementation for the JTFS in JAX)

While the only required hyperparameters to initialize `SCRAPLLoss` are those necessary to define the JTFS, there are several additional hyperparameters that can be used to customize the behavior of the SCRAPL algorithm.
However, the default values of these additional hyperparameters have been chosen to work well for most situations.
For more information about the JTFS and the `J`, `Q1`, `Q2`, `J_fr`, `Q_fr`, `T`, and `F` hyperparameters, please visit: https://www.kymat.io/ismir23-tutorial/intro.html

- `shape` (int)
  - The length of the input signal (number of samples).
- `J` (int)
  - Number of octaves in the JTFS (1st and 2nd order temporal filters)
- `Q1` (int)
  - Wavelets per octave in the JTFS (1st order temporal filters)
- `Q2` (int)
  - Wavelets per octave in the JTFS (2nd order temporal filters)
- `J_fr` (int)
  - Number of octaves in the JTFS (2nd order frequential filters)
- `Q_fr` (int)
  - Wavelets per octave in the JTFS (2nd order frequential filters)
- `T` (str | int | None)
  - Temporal averaging size in samples of the JTFS. If 'global', averages over the entire signal. If 'None', averages over 2**J samples. 
  - Defaults to None.
- `F` (str | int | None)
  - Frequential averaging size in frequency bins of the JTFS. If 'global', averages over all bins. If 'None', averages over J_fr * Q_fr bins. 
  - Defaults to None.
- `p` (float, optional)
  - The order of the norm used for the distance calculation. Must be at least 1; positive infinity is supported.
  - Defaults to 2 (Euclidean norm).
- `use_rho_log1p` (bool, optional)
  - If True, applies log1p scaling to the scattering coefficients (log(1 + x / `log1p_eps`)) before computing the distance. If True, `grad_mult` is no longer necessary and can be set to a value of 1.
  - Defaults to False.
- `log1p_eps` (float, optional)
  - The epsilon value used in the log1p scaling.
  - Defaults to 1e-3.


### `scrapl_optimizer` (𝒫-Adam and 𝒫-SAGA Optax Optimizer)

`scrapl_optimizer` chains gradient multiplier scaling (skipped if `grad_mult` is 1), 𝒫-Adam (if `use_p_adam`), 𝒫-SAGA (if `use_p_saga`), weight decay (if `weight_decay` > 0), and learning rate scaling into a single Optax gradient transformation whose `update()` accepts `path_idx`.

- `learning_rate` (float, optional)
  - Base learning rate for parameter updates.
  - Defaults to 1e-3.
- `weight_decay` (float, optional)
  - L2 weight decay factor added to updates before learning rate scaling.
  - Defaults to 0.01.
- `n_paths` (int)
  - Total number of scattering paths (i.e. `SCRAPLLoss.n_paths`).
- `grad_mult` (float, optional)
  - A scalar multiplier applied to gradients to prevent JTFS precision errors when squaring gradient values in commonly used optimizers like Adam. 
  - See https://hal.science/hal-05124224v1 for more information.
  - When `use_rho_log1p` is True, `grad_mult` is no longer necessary and can be set to a value of 1.
  - Defaults to 1e8.
- `use_p_adam` (bool, optional)
  - If True, enables the 𝒫-Adam algorithm. 
  - Defaults to True.
- `use_p_saga` (bool, optional)
  - If True, enables the 𝒫-SAGA algorithm. 
  - Defaults to True.
- `p_adam_b1` (float, optional)
  - β1 Adam hyperparameter for the internal 𝒫-Adam algorithm. 
  - Defaults to 0.9.
- `p_adam_b2` (float, optional)
  - β2 Adam hyperparameter for the internal 𝒫-Adam algorithm. 
  - Defaults to 0.999.
- `p_adam_eps` (float, optional)
  - ε Adam hyperparameter for the internal 𝒫-Adam algorithm. 
  - Defaults to 1e-8.


### `warmup_lc_hvp` (Importance Sampling (θ-IS) Warmup)

**Parallelization:** To speed up warmup, this function can be run on multiple GPUs simultaneously by assigning disjoint [`start_path_idx`, `end_path_idx`) ranges to each process and providing the same `key` and `save_dir`. Each path's PRNG key is derived from `key` and the path index, so split runs give the same estimates as a single run. After all processes finish, use `load_warmup_from_dir` to load the aggregated path sampling probability distribution.

- `loss` (SCRAPLLoss)
  - The `SCRAPLLoss` module.
- `enc_fn` (Callable[..., Array])
  - The encoder function `enc_fn(enc_params, x, **enc_kwargs)`. It must return an array θ<sub>synth</sub> of shape `(batch_size, n_theta)`. 
  - Any non-trainable encoder state, such as the static part of a partitioned `eqx.Module`, should be closed over.
  - This function should be deterministic during warmup, but can be non-deterministic otherwise.
- `dec_fn` (Callable[..., Array])
  - The decoder (synthesizer) function `dec_fn(theta, **dec_kwargs)`. It must accept the θ<sub>synth</sub> array output by `enc_fn` and return a signal array `x_hat` of shape `(batch_size, n_ch, n_samples)`. 
  - This function should be deterministic during warmup, but can be non-deterministic otherwise.
- `xs` (Array)
  - Batches of input signals of shape `(n_batches, batch_size, n_ch, n_samples)`. 
  - The number of batches is a tradeoff between computational cost and curvature estimation accuracy.
- `enc_params` (PyTree)
  - The encoder parameters (learnable weights) involved in the computation of θ<sub>synth</sub>, treated as one group of weights. 
  - All leaves must be floating-point arrays.
- `enc_kwargs` (dict[str, Any] | None, optional)
  - Per-batch keyword arguments for `enc_fn`, mirroring `theta_fn_kwargs` in PyTorch. Every leaf is an array with a leading `n_batches` axis. 
  - Arguments that are the same for every batch should instead be bound into `enc_fn` with a closure or `functools.partial`.
  - Defaults to None.
- `dec_kwargs` (dict[str, Any] | None, optional)
  - Per-batch keyword arguments for `dec_fn`, stacked the same way as `enc_kwargs` and mirroring `synth_fn_kwargs` in PyTorch. 
  - Defaults to None.
- `key` (Key)
  - JAX PRNG key for the initial power iteration vectors.
- `start_path_idx` (int, optional)
  - The starting index of the scattering paths to compute. Used for parallelizing the warmup across multiple devices. 
  - Defaults to 0.
- `end_path_idx` (int | None, optional)
  - The ending index (exclusive) of the scattering paths to compute. If None, defaults to `loss.n_paths`. 
  - Defaults to None.
- `save_dir` (str | None, optional)
  - Directory path where intermediate eigenvalues (`vals_{path_idx}.npz`) and the final path sampling probability distribution (`probs.npy`) will be saved. If None, nothing is saved. 
  - Defaults to None.
- `n_iter` (int, optional)
  - The maximum number of power iteration steps used to approximate the largest eigenvalue of the Hessian. 
  - Higher values increase precision at the cost of computation time. 
  - Defaults to 20.
- `tol` (float, optional)
  - Stops each power iteration early once its relative residual drops below `tol`. 0.0 always runs `n_iter` steps. 
  - Defaults to 0.01.
- `min_prob_frac` (float, optional)
  - The minimum fraction of the uniform sampling probability assigned to any path, ensuring no path has zero probability of being sampled. 
  - Defaults to 0.0.
- `eps` (float, optional)
  - A small value for numerical stability in power iteration and probability calculations. 
  - Defaults to 1e-12.
- `vmap_theta` (bool, optional)
  - If True, evaluates θ<sub>synth</sub> in parallel, which is faster but multiplies peak memory by `n_theta`. Otherwise evaluates them sequentially. 
  - Defaults to False.


## Best Practices

- SCRAPL with the JTFS is best suited for comparing audio signals that meet one or more of the following criteria:
  - Contain spectrotemporal modulations
  - Are non-stationary
  - Benefit from multi-resolution analysis like percussive sounds (which consist of both transients and sustained components)
  - Are misaligned in time or frequency and therefore benefit from temporal and frequential shift invariance
  - Cannot be effectively compared with time-domain (e.g. mean squared error (MSE) / error-to-signal ratio (ESR) loss), frequency-domain (e.g. multiscale spectral (MSS) loss), or neural embedding-based (e.g. CLAP / PANNs / EnCodec distances) losses 
- Choosing the best JTFS hyperparameters for a given task is very important and requires some understanding of how wavelet scattering transforms work. For an introduction to the JTFS for audio signal processing, check out our ISMIR 2023 tutorial: [Kymatio: Deep Learning meets Wavelet Theory for Music Signal Processing](https://www.kymat.io/ismir23-tutorial/intro.html)
- If GPU memory is becoming a bottleneck, try reducing the number of scattering paths by decreasing the required JTFS arguments or disabling 𝒫-SAGA and then 𝒫-Adam.
- If the SCRAPL loss is not converging and 𝒫-Adam and 𝒫-SAGA are enabled, try reducing the learning rate of `scrapl_optimizer`.
- When using 𝒫-Adam and / or 𝒫-SAGA, do not chain another Adam or momentum-based optimizer after them; `scrapl_optimizer` already applies optional weight decay and the learning rate.
- Always pass the same `path_idx` to both `SCRAPLLoss` and the optimizer's `update()`, and sample it using `sample_path()` outside of JIT-compiled functions.
- When using θ-IS, ensure that the encoder and decoder (synth) are deterministic during the warmup phase, but they can be non-deterministic otherwise.
- The `warmup_lc_hvp` function can be parallelized across paths by assigning disjoint [`start_path_idx`, `end_path_idx`) ranges to each process and providing the same `key` and `save_dir`. After all processes finish, use `load_warmup_from_dir` to load the aggregated path sampling probability distribution.
- The `warmup_lc_hvp` function is most efficient when used with a single large batch filling all available GPU memory rather than many smaller batches.
- Save the optimizer state (`PAdamState`, `PSAGAState`) and the θ-IS warmup result alongside the model parameters and PRNG keys when checkpointing.


## Algorithm

![image](https://raw.githubusercontent.com/christhetree/scrapl/main/docs/figs/scrapl_algorithm.png)


## Implementation Details (JAX)

### Path Sampling

Each call selects one path for the entire batch and averages its distances over batch and channels.
Inputs must have matching shapes `(batch, channels, samples)` and use float32 or float64.
`scrapl_loss.scrapl_keys` maps path indices to `(n2, n_fr)`.
Keys belong to the caller: the loss stores no random state, counters, or gradient history, and reusing a key reproduces its path.
PyTorch and JAX do not produce the same path sequence from the same numeric seed.

With uniform sampling, the sampled loss's expectation is the arithmetic mean of the individual path losses.
Weighted sampling retains the PyTorch implementation's scaling: the chosen path's loss and gradient are unchanged.
Its expectation is `sum(probs * path_losses)`, so it is not an importance-corrected estimator of the uniform path mean.

### θ-IS Warmup

For curvature estimates `c[path, theta]`, the completed-warmup distribution is:

```text
scores = maximum(c, eps)
per_theta = scores / sum(scores, axis=paths)
probs = (1 - min_prob_frac) * mean(per_theta, axis=theta)
        + min_prob_frac / n_paths
```

This matches the PyTorch warmup formula.
A zero-curvature theta contributes a uniform distribution; if all curvatures are zero, sampling stays uniform.
`theta_importance_probs(curvatures, min_prob_frac=..., eps=...)` exposes this conversion separately.

Each theta contributes a parameter-gradient vector whose Jacobian need not be symmetric.
The implementation applies the Jacobian itself with forward-over-reverse differentiation, which is cheaper than PyTorch's grad-of-grad calculation of its transpose; both have the same eigenvalues.
Power iteration estimates the magnitude of a dominant eigenvalue; it is a heuristic, not a certified Lipschitz bound.
Inspect `relative_residuals`: they measure `norm(A v - lambda v) / max(norm(A v), eps)` for the last unit iterate.
A large residual indicates that the estimate has not settled.
Curvature products are summed over batches, as in PyTorch.

### 𝒫-Adam

`p_adam` keeps separate first and second moments for each path and parameter leaf.
Its output is a normalized gradient direction, so it should be followed by `optax.scale_by_learning_rate` rather than another Adam optimizer.
With `curr_t = state.scrapl_t + 1`, `prev_t = state.prev_t_s[path_idx]` and `N = n_paths`, 𝒫-Adam uses `t = curr_t / N` and `delta_t = t - prev_t / N`.
For each selected moment row:

```text
m = b1**delta_t * prev_m + (1 - b1**delta_t) * grad
v = b2**delta_t * prev_v + (1 - b2**delta_t) * grad**2
m_hat = m / (1 - b1**t)
v_hat = v / (1 - b2**t)
grad_hat = m_hat / (sqrt(v_hat) + eps)
```

Elapsed time is measured in global steps, including steps spent on other paths.
Only the selected path's moments and timestamp are updated.
`PAdamState` is a NamedTuple containing `scrapl_t`, `prev_t_s`, `prev_m_s` and `prev_v_s`.
Moment memory contains two copies of the entire parameter tree per path, which can be substantial for large models and filter banks.

### 𝒫-SAGA

`p_saga` keeps the most recent incoming gradient for each path and parameter leaf.
Use it on its own or chain it after `p_adam` to match the combined PyTorch implementation.
For path `i`, incoming gradient `g`, and the stored gradients `h` before the update, the rule is:

```text
path_counts[i] += 1
n_paths_seen = sum(path_counts > 0)
denominator = max(1, n_paths_seen - 1)
direction = g - h[i] + sum(h, axis=paths) / denominator
h[i] = g
```

The initial history is zero, so the first direction equals the incoming gradient.
Only the selected history row changes; it stores the input to 𝒫-SAGA, including 𝒫-Adam normalization if enabled, rather than the corrected output.
𝒫-SAGA does not use the sampling probabilities in its correction, which matches PyTorch.
`PSAGAState` is a NamedTuple with `path_counts` and `prev_path_grads`.
𝒫-SAGA stores one copy of the entire parameter tree per path, in addition to 𝒫-Adam's two copies when used together.

### Validation

The JAX implementation is tested against the PyTorch implementation (see [Installation](#installation) to run the test suite).
The tests compare loss values and input gradients for randomly selected paths of the 315-path JTFS used in the examples above, θ-IS curvature estimates and probabilities (including split warmups loaded from a directory), and 𝒫-Adam / 𝒫-SAGA optimizer states over multiple training steps, as well as resuming training from a saved optimizer state.
Validation currently uses float32 on CPU with Python 3.13, JAX 0.12.0.1, and PyTorch 2.14.1.


## Known Issues (JAX)

- `path_idx` must be a concrete Python integer, so each scattering path is compiled separately the first time it is used inside a JIT-compiled function, and first-call compilation time grows with the number of paths. Applying `vmap` over separate path keys or indices is not supported.
- Per-parameter aggregation of θ-IS curvature estimates (the `agg` argument in PyTorch) is not implemented; all encoder weights are treated as one group.
- Float64 parity with PyTorch, GPU performance, and compilation costs for large filter banks have not yet been validated.


## Citation

If you find this project helpful for your own work, please consider citing it:

Accepted to the International Conference on Learning Representations (ICLR), Rio de Janeiro, Brazil, 23 - 27 April 2026.

<pre><code>
@inproceedings{mitcheltree2026scrapl,
    title={{SCRAPL}: Scattering Transform with Random Paths for Machine Learning},
    author={Christopher Mitcheltree and Vincent Lostanlen and Emmanouil Benetos and Mathieu Lagrange},
    booktitle={Proceedings of the International Conference on Learning Representations (ICLR)},
    month={April},
    year={2026}
}
</code></pre>
