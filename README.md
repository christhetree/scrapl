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
SCRAPL also leverages specialized stochastic optimization techniques ($\mathcal{P}$-Adam, $\mathcal{P}$-SAGA) and an architecture-informed importance sampling heuristic ($\theta$-IS) to stabilize gradients and improve convergence.
For more details, please see the [ICLR 2026 paper](https://openreview.net/forum?id=RuYwbd5xYa).

`scrapl` currently contains a PyTorch implementation of the SCRAPL algorithm for the [joint time–frequency scattering transform (JTFS)](https://www.kymat.io/ismir23-tutorial/ch1_intro/why_wavelets.html), which demodulates spectrotemporal patterns at multiple scales and rates, and has been shown to correlate with human perception ([Lostanlen et al., 2021](https://link.springer.com/article/10.1186/s13636-020-00187-z); [Tian et al., 2025](https://doi.org/10.48550/arXiv.2507.07764)).
In our experiments, we find that SCRAPL accomplishes a favorable tradeoff between goodness of fit and computational efficiency on unsupervised sound matching, i.e., a nonlinear inverse problem in which the forward operator implements an audio synthesizer.
In the context of differentiable digital signal processing (DDSP), the state-of-the-art perceptual loss function for this task is multiscale spectral loss (MSS, [Yamamoto et al., 2020](https://doi.org/10.48550/arXiv.1910.11480); [Engel et al., 2020](https://openreview.net/forum?id=B1x1ma4tDr)). 
However, the gradient of MSS is uninformative when input and reconstruction are misaligned or when the synthesizer controls involve spectrotemporal modulations ([Vahidi et al., 2023](https://doi.org/10.48550/arXiv.2301.10183)). 
Taking advantage of the stability guarantees of JTFS, SCRAPL expands the class of synthesizers which can be effectively decoded via DDSP.

Additional scattering transform implementations and support for other machine learning frameworks (e.g. JAX) may be added to `scrapl` in the future.


![image](/docs/figs/relative_param_error_vs_computation.svg)

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
- [Paper Experiments](#paper-experiments)
  - [Abstract](#abstract)
  - [Instructions for Reproducibility](#instructions-for-reproducibility)


# Python Package

## Installation

You can install `scrapl` using `pip`: 

```
pip install scrapl
```

The package requires Python 3.8 or higher and `2.8.0 <= torch < 3.0.0` as well as `numpy` and `scipy`.


## Examples

Importing and initializing `SCRAPLLoss` with the minimum required hyperparameters:

```python
# Import SCRAPLLoss from the scrapl Python package
from scrapl import SCRAPLLoss

# Initialize SCRAPLLoss with the minimum required hyperparameters
scrapl_loss = SCRAPLLoss(
    shape=48000,  # Length of x and x_target in samples
    J=12,         # Number of octaves (1st and 2nd order temporal filters)
    Q1=8,         # Filters per octave (1st order temporal filters)
    Q2=2,         # Filters per octave (2nd order temporal filters)
    J_fr=3,       # Number of octaves (2nd order frequential filters)
    Q_fr=2,       # Filters per octave (2nd order frequential filters)
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

### Using $\mathcal{P}$-Adam and $\mathcal{P}$-SAGA

$\mathcal{P}$-Adam and $\mathcal{P}$-SAGA are enabled by default when initializing `SCRAPLLoss`. 
However, the learnable weights of the neural network being optimized must be attached to `SCRAPLLoss` for $\mathcal{P}$-Adam and $\mathcal{P}$-SAGA to have any effect.
When $\mathcal{P}$-Adam and $\mathcal{P}$-SAGA are enabled, vanilla stochastic gradient descent (SGD) with no momentum and optional weight decay should be used as the downstream optimizer:

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

### Importance Sampling ($\theta$-IS) Warmup


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

# Initialize SCRAPLLoss with the minimum required hyperparameters and `n_theta`
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
 [=============================================================>...]   20/20 
INFO:scrapl.scrapl_loss:path_idx = 0, curr_vals = tensor([0.9898, 0.1256, 0.2016])
 [=============================================================>...]   20/20 
INFO:scrapl.scrapl_loss:path_idx = 1, curr_vals = tensor([1.0082, 0.1349, 0.1406])
 [=============================================================>...]   20/20 
INFO:scrapl.scrapl_loss:path_idx = 2, curr_vals = tensor([0.9389, 0.0788, 0.1565])
 [=============================================================>...]   20/20 
INFO:scrapl.scrapl_loss:path_idx = 3, curr_vals = tensor([0.9059, 0.1199, 0.1678])
 [=============================================================>...]   20/20 
INFO:scrapl.scrapl_loss:path_idx = 4, curr_vals = tensor([1.0138, 0.1352, 0.1418])
 [=============================================================>...]   20/20 
INFO:scrapl.scrapl_loss:path_idx = 5, curr_vals = tensor([0.9248, 0.0741, 0.1556])
 [=============================================================>...]   20/20 
INFO:scrapl.scrapl_loss:path_idx = 6, curr_vals = tensor([0.8879, 0.1157, 0.1654])
INFO:scrapl.scrapl_loss:Saving warmup SCRAPL sampling probabilities to scrapl_warmup/probs.pt

[min, max] path sampling probabilities (after warmup): [0.123653, 0.162354]
```


## Hyperparameters


## Best Practices


## Algorithm

![image](/docs/figs/scrapl_algorithm.png)


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
   `pip install scrapl`\
   The package documentation can be found in the [`scrapl/`](./scrapl) directory.

If you would like to learn more about wavelets, scattering transforms, and deep learning for music and audio, check out our ISMIR 2023 tutorial:\
[Kymatio: Deep Learning meets Wavelet Theory for Music Signal Processing](https://www.kymat.io/ismir23-tutorial/intro.html)
