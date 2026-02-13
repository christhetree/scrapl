<a name="top"/>

<div align="center">
<h1><b>SCRAPL</b>: <b>Sc</b>attering Transform with <b>Ra</b>ndom <b>P</b>aths for Machine <b>L</b>earning</h1>
<p>
<a href="https://christhetr.ee/" target=”_blank”>Christopher Mitcheltree</a>,
<a href="https://www.lostanlen.com/" target=”_blank”>Vincent Lostanlen</a>,
<a href="https://www.qmul.ac.uk/eecs/people/profiles/benetosemmanouil.html" target=”_blank”>Emmanouil Benetos</a>, and
<a href="https://mathieulagrange.github.io/" target=”_blank”>Mathieu Lagrange</a>
</p>

[![Release](https://img.shields.io/badge/PyPI-v0.1.0-green)](https://pypi.org/project/scrapl/)
[![OpenReview](https://img.shields.io/badge/OpenReview-ICLR_2026_Paper-b31b1b.svg)](https://openreview.net/forum?id=RuYwbd5xYa)
[![Listening Samples](https://img.shields.io/badge/%F0%9F%94%8A%F0%9F%8E%B6-Listening_Samples-blue)](https://christhetree.github.io/scrapl/)
[![License](https://img.shields.io/badge/License-MIT-blue)](https://opensource.org/license/mit)
</div>


`scrapl` is a Python package for efficient evaluation of multivariable scattering transforms, specifically designed for use as a differentiable loss function in machine learning applications and perceptual quality assessment of deep inverse problems in computer vision, speech, and audio processing. 
It implements the "Scattering Transform with Random Paths for Machine Learning" (SCRAPL) algorithm, which accelerates gradient-based optimization of neural networks with scattering transforms by stochastically sampling scattering paths to approximate the full transform's gradient.
This reduces the computational and memory costs by one to three orders of magnitude. 
SCRAPL also leverages specialized stochastic optimization techniques (𝒫-Adam, 𝒫-SAGA) and an architecture-informed importance sampling heuristic (θ-IS) to stabilize gradients and improve convergence.
For more details, please see the [ICLR 2026 paper](https://openreview.net/forum?id=RuYwbd5xYa).

`scrapl` currently contains a PyTorch implementation of the SCRAPL algorithm for the [joint time–frequency scattering transform (JTFS)](https://www.kymat.io/ismir23-tutorial/ch1_intro/why_wavelets.html), which demodulates spectrotemporal patterns at multiple scales and rates, and has been shown to correlate with human perception ([Lostanlen et al., 2021](https://link.springer.com/article/10.1186/s13636-020-00187-z); [Tian et al., 2025](https://doi.org/10.48550/arXiv.2507.07764)).
In our experiments, we find that SCRAPL accomplishes a favorable tradeoff between goodness of fit and computational efficiency on unsupervised sound matching, i.e., a nonlinear inverse problem in which the forward operator implements an audio synthesizer.
In the context of differentiable digital signal processing (DDSP), the state-of-the-art perceptual loss function for this task is multiscale spectral loss (MSS, [Yamamoto et al., 2020](https://doi.org/10.48550/arXiv.1910.11480); [Engel et al., 2020](https://openreview.net/forum?id=B1x1ma4tDr)). 
However, the gradient of MSS is uninformative when input and reconstruction are misaligned or when the synthesizer controls involve spectrotemporal modulations ([Vahidi et al., 2023](https://doi.org/10.48550/arXiv.2301.10183)). 
Taking advantage of the stability guarantees of JTFS, SCRAPL expands the class of synthesizers which can be effectively decoded via DDSP.

Additional scattering transform implementations and support for other machine learning frameworks (e.g. JAX) may be added to `scrapl` in the future.


![image](https://raw.githubusercontent.com/christhetree/scrapl/main/docs/figs/relative_param_error_vs_computation.svg)

_Figure 1: Mean average synthesizer parameter error (y-axis) versus computational cost (x-axis) of unsupervised sound matching models for the granular synthesis task. 
Both axes are rescaled by the performance of a supervised model with the same number of parameters. 
Whiskers denote 95% CI, estimated over 20 random seeds. Due to computational limitations, 
JTFS-based sound matching is evaluated only once._


## Citation

Accepted to the International Conference on Learning Representations (ICLR), Rio de Janeiro, Brazil, 23 - 27 April 2026.
<pre><code>
@inproceedings{mitcheltree2026scrapl,
    title={{SCRAPL}: Scattering Transform with Random Paths for Machine Learning},
    author={Christopher Mitcheltree and Vincent Lostanlen and Emmanouil Benetos and Mathieu Lagrange},
    booktitle={Proceedings of the International Conference on Learning Representations (ICLR)},
    year={2026}
}
</code></pre>


## Table of Contents

- [Python Package](#python-package)
  - [Installation](#installation)
  - [Examples](#examples)
  - [Hyperparameters](#hyperparameters)
  - [Best Practices](#best-practices)
  - [Algorithm](#algorithm)
  - [Known Issues](#known-issues)
- [Paper Experiments](#paper-experiments)
  - [Abstract](#abstract)
  - [Instructions for Reproducibility](#instructions-for-reproducibility)


# Python Package

## Installation

You can install `scrapl` using `pip`: 

```
pip install scrapl-loss
```

The package requires Python 3.10 or higher and `2.8.0 <= torch < 3.0.0` as well as `numpy` and `scipy`.


## Examples

Importing and initializing `SCRAPLLoss` with the minimum required arguments:

```python
# Import SCRAPLLoss from the scrapl Python package
from scrapl import SCRAPLLoss

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
INFO:scrapl.scrapl_loss:SCRAPLLoss:
J=12, Q1=8, Q2=2, Jfr=3, Qfr=2, T=None, F=None
use_rho_log1p          = False, eps = 0.001
grad_mult              = 100000000.0
use_p_adam             = True
use_p_saga             = True
sample_all_paths_first = False
n_theta                = 1
min_prob_frac          = 0.0
number of SCRAPL keys  = 315
unif_prob              = 0.00317460
```

Calculating the loss for two signals: 

```python 
import torch as tr

# Create two random tensors of shape (batch_size, num_channels, signal_length)
x = tr.randn(4, 1, 48000) # Example input tensor 
x_target = tr.randn(4, 1, 48000) # Example target tensor 
# Compute the SCRAPL loss between x and x_target. Since SCRAPL is stochastic,
# the loss value will be different each time `scrapl_loss()` is called.
loss = scrapl_loss(x, x_target)
print(f"SCRAPL loss: {loss.item()}")
```

`SCRAPLLoss` utility attributes and methods:

```python
print(f"Number of scattering paths: {scrapl_loss.n_paths}")
print(f"Uniform path sampling probability: {scrapl_loss.unif_prob:.6f}")
print(f"Most recently sampled path index (prev. example): {scrapl_loss.curr_path_idx}")

# Calculate the loss for a specific path index
loss = scrapl_loss(x, x_target, path_idx=8)
print(f"Most recently sampled path index (specific): {scrapl_loss.curr_path_idx}")

# Calculate the loss with a random seed for deterministic path sampling
# (this will sample the same path index every time)
loss = scrapl_loss(x, x_target, seed=42)
print(f"Most recently sampled path index (random seed): {scrapl_loss.curr_path_idx}")
print(f"Path sampling statistics (original): {scrapl_loss.path_counts}")

# Get the state dictionary of the SCRAPLLoss instance
state_dict = scrapl_loss.state_dict()

# Clear all state of the SCRAPLLoss instance
scrapl_loss.clear()
print(f"Path sampling statistics (cleared): {scrapl_loss.path_counts}")

# Load a state dictionary into the SCRAPLLoss instance
scrapl_loss.load_state_dict(state_dict)
print(f"Path sampling statistics (loaded): {scrapl_loss.path_counts}")
```

Console output:

```text
Number of scattering paths: 315
Uniform path sampling probability: 0.003175
Most recently sampled path index (prev. example): 106
Most recently sampled path index (specific): 8
Most recently sampled path index (random seed): 211
Path sampling statistics (original): defaultdict(<class 'int'>, {106: 1, 8: 1, 211: 1})
Path sampling statistics (cleared): defaultdict(<class 'int'>, {})
Path sampling statistics (loaded): defaultdict(<class 'int'>, {106: 1, 8: 1, 211: 1})
```

### Using 𝒫-Adam and 𝒫-SAGA

𝒫-Adam and 𝒫-SAGA are enabled by default when initializing `SCRAPLLoss`.
The Adam hyperparameters β1, β2, and ε can be set using the `p_adam_b1`, `p_adam_b2`, and `p_adam_eps` arguments in the `SCRAPLLoss` constructor.
However, the learnable weights of the neural network being optimized must be attached to `SCRAPLLoss` for 𝒫-Adam and 𝒫-SAGA to have any effect.
When 𝒫-Adam and 𝒫-SAGA are enabled, vanilla stochastic gradient descent (SGD) with no momentum and optional weight decay should be used as the downstream optimizer:

```python
from torch import nn

# Toy example model
model = nn.Sequential(
    nn.Linear(in_features=48000, out_features=8),
    nn.PReLU(),
    nn.Linear(in_features=8, out_features=48000),
    nn.Tanh(),
)

# Attach the learnable weights of the model to the SCRAPLLoss instance
# for P-Adam and P-SAGA
scrapl_loss.attach_params(model.parameters())

# Since we are using P-Adam and / or P-SAGA, we should use vanilla SGD with no
# momentum and optional weight decay as the downstream optimizer
from torch.optim import SGD
sgd_optimizer = SGD(model.parameters(), lr=1e-4, weight_decay=0.01)

# Example forward and backward step with P-Adam and P-SAGA now active
sgd_optimizer.zero_grad()
x_hat = model(x)
loss = scrapl_loss(x_hat, x_target)
loss.backward()
# Even though vanilla SGD is called here, the gradients of the model parameters
# have been modified by P-Adam and P-SAGA under the hood during the backward pass
sgd_optimizer.step()

# To detach parameters (generally not necessary), simply attach an empty list
scrapl_loss.attach_params([])
```

Console output:

```text
INFO:scrapl.scrapl_loss:Attached 5 parameter tensors (816009 weights)
INFO:scrapl.scrapl_loss:Detached 5 parameter tensors
```

### Importance Sampling Warmup

The SCRAPL algorithm includes an importance sampling heuristic (θ-IS) that learns a non-uniform sampling distribution over scattering paths given an encoder and decoder (synth) self-supervised training architecture. 
This is done by measuring the curvature of the loss landscape with respect to the encoder output / decoder (synth) input parameters θ<sub>synth</sub> for each path, see Sections 3.4, 4.3, and 5.2 in the [paper](https://openreview.net/forum?id=RuYwbd5xYa) for more details.
This warmup step is done once before training on a subset of the training data (typically 32-256 samples) and can be parallelized across paths (currently implemented) and θ<sub>synth</sub> (to be added in the future).

```python
import torch as tr
from scrapl import SCRAPLLoss

# Setup
tr.set_printoptions(precision=4, sci_mode=False)
tr.manual_seed(42)
bs = 4
n_ch = 1
n_samples = 8096
# Number of parameters output by the encoder and input to the decoder (synth)
n_theta = 3
# Number of batches to use for warmup; when possible, one large batch filling all
# available GPU memory is more efficient than many smaller batches
n_batches = 1

# Provide a neural network encoder that outputs `n_theta` parameters
encoder = nn.Sequential(
    nn.Linear(n_samples, n_theta),
    nn.PReLU(),
    nn.Linear(n_theta, n_theta),
    nn.Sigmoid(),
)
# Make the encoder forward call functional (stateless)
theta_fn = lambda x: encoder(x.squeeze(1))

# Provide a differentiable decoder (synth) that takes as input `n_theta` parameters
decoder = nn.Sequential(
    nn.Linear(n_theta, n_theta),
    nn.PReLU(),
    nn.Linear(n_theta, n_samples),
    nn.Tanh(),
)
# Make the decoder forward call functional (stateless) and ensure it outputs the
# correct shape for the SCRAPLLoss (bs, n_ch, n_samples)
synth_fn = lambda theta: decoder(theta).unsqueeze(1)

# Create a list of dictionaries with batches of input data for `theta_fn`;
# typically this would be gathered from a dataloader
theta_is_batches = [tr.rand((bs, n_ch, n_samples)) for _ in range(n_batches)]
theta_fn_kwargs = [{"x": batch} for batch in theta_is_batches]

# Get the trainable weights of the encoder to pass to the warmup function
params = [p for p in encoder.parameters()]

# Initialize SCRAPLLoss with the minimum required arguments and `n_theta`
scrapl_loss = SCRAPLLoss(
    shape=n_samples,
    J=3,
    Q1=1,
    Q2=1,
    J_fr=2,
    Q_fr=1,
    n_theta=n_theta,
)
# We see that at initialization, all path sampling probabilities are uniform
print(f"Uniform path sampling probability: {scrapl_loss.unif_prob:.6f}")
print(
    f"[min, max] path sampling probabilities (before warmup): "
    f"[{scrapl_loss.probs.min():.6f}, {scrapl_loss.probs.max():.6f}]"
)

# The encoder and decoder (synth) must be deterministic during warmup,
# but can be non-deterministic otherwise
scrapl_loss.check_is_deterministic(theta_fn, theta_fn_kwargs[0], synth_fn)

# Run the warmup. This can be parallelized across paths by specifying different
# `start_path_idx` and `end_path_idx` subsets on multiple devices.
scrapl_loss.warmup_lc_hvp(
    theta_fn=theta_fn,
    synth_fn=synth_fn,
    theta_fn_kwargs=theta_fn_kwargs,
    params=params,
)
# We see that after warmup, the path sampling probabilities are no longer uniform
print(
    f"[min, max] path sampling probabilities (after warmup): "
    f"[{scrapl_loss.probs.min():.6f}, {scrapl_loss.probs.max():.6f}]"
)

# If warmup was conducted in parallel across multiple devices, the path sampling
# probabilities can be loaded from a directory
scrapl_loss.load_probs_from_warmup_dir(warmup_dir="scrapl_warmup")
```

Console output:

```text
INFO:scrapl.scrapl_loss:SCRAPLLoss:
J=3, Q1=1, Q2=1, Jfr=2, Qfr=1, T=None, F=None
use_rho_log1p          = False, eps = 0.001
grad_mult              = 100000000.0
use_p_adam             = True
use_p_saga             = True
sample_all_paths_first = False
n_theta                = 3
min_prob_frac          = 0.0
number of SCRAPL paths = 7
unif_prob              = 0.14285714

Uniform path sampling probability: 0.142857
[min, max] path sampling probabilities (before warmup): [0.142857, 0.142857]

INFO:scrapl.scrapl_loss:Starting warmup_lc_hvp with agg = none for 5 parameter(s) and 1 batch(es), 20 iter (multibatch = False)
 [=========================== 20/20 ===========================>...]  Step: 38ms | Tot: 749ms | power iter error: 0.0001
INFO:scrapl.scrapl_loss:path_idx = 0, curr_vals = tensor([0.9898, 0.1256, 0.2016])
 [=========================== 20/20 ===========================>...]  Step: 59ms | Tot: 1s190ms | power iter error: 0.0117
INFO:scrapl.scrapl_loss:path_idx = 1, curr_vals = tensor([1.0082, 0.1349, 0.1406])
 [=========================== 20/20 ===========================>...]  Step: 58ms | Tot: 1s148ms | power iter error: 0.0045
INFO:scrapl.scrapl_loss:path_idx = 2, curr_vals = tensor([0.9389, 0.0788, 0.1565])
 [=========================== 20/20 ===========================>...]  Step: 62ms | Tot: 1s105ms | power iter error: 0.0016
INFO:scrapl.scrapl_loss:path_idx = 3, curr_vals = tensor([0.9059, 0.1199, 0.1678])
 [=========================== 20/20 ===========================>...]  Step: 56ms | Tot: 1s111ms | power iter error: 0.0113
INFO:scrapl.scrapl_loss:path_idx = 4, curr_vals = tensor([1.0138, 0.1352, 0.1418])
 [=========================== 20/20 ===========================>...]  Step: 56ms | Tot: 1s104ms | power iter error: 0.0038
INFO:scrapl.scrapl_loss:path_idx = 5, curr_vals = tensor([0.9248, 0.0741, 0.1556])
 [=========================== 20/20 ===========================>...]  Step: 56ms | Tot: 1s109ms | power iter error: 0.0014
INFO:scrapl.scrapl_loss:path_idx = 6, curr_vals = tensor([0.8879, 0.1157, 0.1654])
INFO:scrapl.scrapl_loss:Saving warmup SCRAPL sampling probabilities to scrapl_warmup/probs.pt

[min, max] path sampling probabilities (after warmup): [0.123653, 0.162354]

INFO:scrapl.scrapl_loss:Loading probs from directory:
	scrapl_warmup
	min prob = 0.123653, max prob = 0.162354
```


## Hyperparameters

### `SCRAPLLoss` (SCRAPL Algorithm Implementation for the JTFS in PyTorch)

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
- `T` (Optional[Union[str, int]])
  - Temporal averaging size in samples of the JTFS. If 'global', averages over the entire signal. If 'None', averages over J**2 samples. 
  - Defaults to None.
- `F` (Optional[Union[str, int]])
  - Frequential averaging size in frequency bins of the JTFS. If 'global', averages over all bins. If 'None', averages over J_fr * Q_fr bins. 
  - Defaults to None.
- `p` (int, optional)
  - The order of the norm used for the distance calculation. 
  - Defaults to 2 (Euclidean norm).
- `use_rho_log1p` (bool, optional)
  - If True, applies log1p scaling to the scattering coefficients (log(1 + x / `log1p_eps`)) before computing the distance. If True, `grad_mult` is no longer necessary and can be set to a value of 1.
  - Defaults to False.
- `log1p_eps` (float, optional)
  - The epsilon value used in the log1p scaling.
  - Defaults to 1e-3.
- `grad_mult` (float, optional)
  - A scalar multiplier applied to gradients to prevent JTFS precision errors when squaring gradient values in commonly used optimizers like Adam. 
  - See https://hal.science/hal-05124224v1 for more information.
  - When `use_rho_log1p` is True, `grad_mult` is no longer necessary and can be set to a value of 1.
  - When `grad_mult` is not 1, `attach_params()` must be called with the model parameters being optimized for this to have an effect. 
  - Defaults to 1e8.
- `use_p_adam` (bool, optional)
  - If True, enables the 𝒫-Adam algorithm. 
  - When True, `attach_params()` must be called before training with the model parameters being optimized and vanilla stochastic gradient descent (SGD) with no momentum and optional weight decay should be used as the downstream optimizer. 
  - Defaults to True.
- `use_p_saga` (bool, optional)
  - If True, enables the 𝒫-SAGA algorithm. 
  - When True, `attach_params()` must be called before training with the model parameters being optimized and vanilla stochastic gradient descent (SGD) with no momentum and optional weight decay should be used as the downstream optimizer. 
  - Defaults to True.
- `sample_all_paths_first` (bool, optional)
  - If True, forces the sampler to visit every possible scattering path once in order before switching to the internal path sampling distribution. 
  - sDefaults to False.
- `n_theta` (int, optional)
  - Given an encoder and decoder (synth) self-supervised training architecture, the number of encoder output / decoder (synth) input parameters θ<sub>synth</sub>. 
  - This is only required for the importance sampling heuristic (θ-IS) and calling the `warmup_lc_hvp()` method before training. 
  - Defaults to 1.
- `min_prob_frac` (float, optional)
  - When using θ-IS, the minimum fraction of the uniform sampling probability assigned to any path, ensuring no path has zero probability of being sampled. 
  - Defaults to 0.0.
- `probs_path` (Optional[str], optional)
  - File path to a `.pt` file containing pre-computed sampling probabilities for the scattering paths. 
  - Defaults to None.
- `eps` (float, optional)
  - A small value for numerical stability in probability calculations. 
  - Defaults to 1e-12.
- `p_adam_b1` (float, optional)
  - β1 Adam hyperparameter for the internal 𝒫-Adam algorithm. 
  - Defaults to 0.9.
- `p_adam_b2` (float, optional)
  - β2 Adam hyperparameter for the internal 𝒫-Adam algorithm. 
  - Defaults to 0.999.
- `p_adam_eps` (float, optional)
  - ε Adam hyperparameter for the internal 𝒫-Adam algorithm. 
  - Defaults to 1e-8.


### `SCRAPLLoss.warmup_lc_hvp` (Importance Sampling (θ-IS) Warmup)

**Parallelization:** To speed up warmup, this method can be run on multiple GPUs simultaneously by assigning disjoint [`start_path_idx`, `end_path_idx`) ranges to each process and providing a `save_dir`. After all processes finish, use `load_probs_from_warmup_dir` to load the aggregated path sampling probability distribution into a `SCRAPLLoss` instance.

- `theta_fn` (Callable[..., T])
  - The encoder function. It must accept arguments provided in `theta_fn_kwargs` and return a tensor θ<sub>synth</sub> of shape `(batch_size, n_theta)`. 
  - This function should be deterministic during warmup, but can be non-deterministic otherwise.
- `synth_fn` (Callable[[T, ...], T])
  - The decoder (synthesizer) function. It must accept the θ<sub>synth</sub> tensor output by `theta_fn` (and optional `synth_fn_kwargs`) and return a signal tensor `x_hat` of shape `(n_batches, n_ch, n_samples)`. 
  - This function should be deterministic during warmup, but can be non-deterministic otherwise.
- `theta_fn_kwargs` (List[Dict[str, Any]])
  - A list of dictionaries, where each dictionary contains a batch of input arguments for `theta_fn`. One of these arguments must be the input signal `x` of shape `(n_batches, n_ch, n_samples)`. 
  - The length of this list determines the number of batches used for the loss landscape curvature estimation and is a tradeoff between computational cost and curvature estimation accuracy.
- `params` (List[Parameter])
  - A list of encoder parameters (learnable weights) involved in the computation of θ<sub>synth</sub>. 
  - These parameters must have no prior gradients (i.e. `p.grad is None` for all `p` in `params`) before calling this method.
- `synth_fn_kwargs` (Optional[List[Dict[str, Any]]], optional)
  - A list of dictionaries corresponding to `theta_fn_kwargs`, containing additional arguments for `synth_fn`. 
  - If provided, must have the same length as `theta_fn_kwargs`. 
  - Defaults to None.
- `start_path_idx` (int, optional)
  - The starting index of the scattering paths to compute. Used for parallelizing the warmup across multiple devices. 
  - Defaults to 0.
- `end_path_idx` (Optional[int], optional)
  - The ending index (exclusive) of the scattering paths to compute. If None, defaults to `self.n_paths`. 
  - Defaults to None.
- `save_dir` (Optional[str], optional)
  - Directory path where intermediate eigenvalues (`vals_{path_idx}.pt`) and the final path sampling probability distribution (`probs.pt`) will be saved. 
  - Defaults to "scrapl_warmup".
- `n_iter` (int, optional)
  - The maximum number of power iteration steps used to approximate the largest eigenvalue of the Hessian. 
  - Higher values increase precision at the cost of computation time. 
  - Defaults to 20.
- `agg` (Literal["none", "mean", "max", "med"], optional)
  - The aggregation strategy if `params` are split into individual tensors. 
  - Usually kept as "none" to aggregate across the full parameter set provided unless it does not fit in GPU memory, in which case curvature is estimated for each parameter tensor separately and then aggregated at the end. 
  - If you are seeing "out of memory" errors, try switching this to "med" to reduce memory usage at the cost of computation time. 
  - Defaults to "none".
- `force_multibatch` (bool, optional)
  - Debugging tool. If True, forces the calculation to use multibatch logic even if `theta_fn_kwargs` contains only a single batch. 
  - Defaults to False.


## Best Practices

- SCRAPL and the JTFS are best suited for comparing audio signals that:
  - Contain spectrotemporal modulations
  - Benefit from multi-resolution analysis like percussive sounds
  - Are misaligned in time or frequency and therefore benefit from temporal and frequential shift invariance
- Choosing the best JTFS hyperparameters for a given task is very important and requires some understanding of how wavelet scattering transforms work. For an introduction to the JTFS for audio signal processing, check out our ISMIR 2023 tutorial: [Kymatio: Deep Learning meets Wavelet Theory for Music Signal Processing](https://www.kymat.io/ismir23-tutorial/intro.html)
- If GPU memory is becoming a bottleneck, try reducing the number of scattering paths by decreasing the required JTFS arguments or disabling 𝒫-SAGA and then 𝒫-Adam.
- If the SCRAPL loss is not converging and 𝒫-Adam and 𝒫-SAGA are enabled and the model parameters have been attached to the `SCRAPLLoss` instance, try reducing the learning rate of the downstream vanilla SGD optimizer.
- When using 𝒫-Adam and / or 𝒫-SAGA, use vanilla SGD with no momentum and optional weight decay as the downstream optimizer.
- When using 𝒫-Adam, 𝒫-SAGA, or `grad_mult != 1.0`, ensure that `attach_params()` is called before training for these features to have any effect.
- When using θ-IS, ensure that the encoder and decoder (synth) are deterministic during the warmup phase, but they can be non-deterministic otherwise.
- The `warmup_lc_hvp` method can be parallelized across paths by assigning disjoint [`start_path_idx`, `end_path_idx`) ranges to each process and providing a `save_dir`. After all processes finish, use `load_probs_from_warmup_dir` to load the aggregated path sampling probability distribution into a `SCRAPLLoss` instance.
- The `warmup_lc_hvp` method is most efficient when used with a single large batch filling all available GPU memory rather than many smaller batches.


## Algorithm

![image](https://raw.githubusercontent.com/christhetree/scrapl/main/docs/figs/scrapl_algorithm.png)


## Known Issues

- Resuming training from a checkpoint when model parameters have been attached (i.e. 𝒫-Adam, 𝒫-SAGA or `grad_mult` are enabled) is currently not supported.
- Parallelization of the `warmup_lc_hvp` method across θ<sub>synth</sub> is currently not implemented, which may lead to long warmup times when the number of encoder output / decoder (synth) input parameters `n_theta` is large.


# Paper Experiments

## Abstract

The Euclidean distance between wavelet scattering transform coefficients (known as <em>paths</em>) provides informative gradients for perceptual quality assessment of deep inverse problems in computer vision, speech, and audio processing.
However, these transforms are computationally expensive when employed as differentiable loss functions for stochastic gradient descent due to their numerous paths, which significantly limits their use in neural network training.
Against this problem, we propose "<b>Sc</b>attering transform with <b>Ra</b>ndom <b>P</b>aths for machine <b>L</b>earning" (SCRAPL): a stochastic optimization scheme for efficient evaluation of multivariable scattering transforms.
We implement SCRAPL for the joint time–frequency scattering transform (JTFS) which demodulates spectrotemporal patterns at multiple scales and rates, allowing a fine characterization of intermittent auditory textures.
We apply SCRAPL to differentiable digital signal processing (DDSP), specifically, unsupervised sound matching of a granular synthesizer and the Roland TR-808 drum machine.
We also propose an initialization heuristic based on importance sampling, which adapts SCRAPL to the perceptual content of the dataset, improving neural network convergence and evaluation performance.
We make our code and audio samples available and provide SCRAPL as a Python package.


## Instructions for Reproducibility

1. Clone this repository and open its directory.
1. Be sure to initialize the submodules:\
   `git submodule update --init --recursive`
1. Install the requirements:
   <br>`conda env create --file=conda_env_gpu.yml`
   <br>or\
   `conda env create --file=conda_env_cpu.yml`
   <br>depending on your computing environment.
   <br>For posterity, `requirements_all_gpu.txt` and `requirements_all_cpu.txt` are also provided.
1. The source code for the SCRAPL algorithm can be explored in the [`scrapl/`](./scrapl) directory.
1. The source code for the three experiments in the paper can be explored in the [`experiments/`](./experiments) directory.
1. All experiment config files can be found in the [`experiments/configs/`](./experiments/configs) directory.
1. The dataset for the Roland TR-808 sound matching task can be found [here](https://samplesfrommars.com/products/tr-808-samples) and needs to be placed in [`experiments/data/808_dataset/`](./experiments/data/808_dataset/file_list.txt).
1. Create an out directory (`mkdir experiments/out`).
1. Models for each experiment can be trained and evaluated by modifying [`experiments/scripts/train.py`](./experiments/scripts/train.py)\
   and the corresponding `experiments/configs/.../train_ ... .yml` config file and then running:\
   `python experiments/scripts/train.py`\
   Make sure your PYTHONPATH has been set correctly by running commands like:\
   `export PYTHONPATH=$PYTHONPATH:[ROOT_DIR]/experiments/`\
   `export PYTHONPATH=$PYTHONPATH:[ROOT_DIR]/scrapl/`\
   `export PYTHONPATH=$PYTHONPATH:[ROOT_DIR]/scrapl/kymatio/`\
   `export PYTHONPATH=$PYTHONPATH:[ROOT_DIR]/scrapl/pytorch_hessian_eigenthings/`\
   `export PYTHONPATH=$PYTHONPATH:[ROOT_DIR]/fadtk/`
1. The experiments source code is currently not documented, but don't hesitate to open an issue if you have any questions or comments.
1. A PyPI Python package of the SCRAPL algorithm for the joint time-frequency scattering transform (JTFS) is available and can be installed with:\
   `pip install scrapl-loss`\
   The package documentation can be found in the [`scrapl/`](./scrapl) directory.

If you would like to learn more about wavelets, scattering transforms, and deep learning for music and audio, check out our ISMIR 2023 tutorial:\
[Kymatio: Deep Learning meets Wavelet Theory for Music Signal Processing](https://www.kymat.io/ismir23-tutorial/intro.html)
