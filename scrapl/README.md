<div align="center">
<h1><b>SCRAPL</b>: <b>Sc</b>attering Transform with <b>Ra</b>ndom <b>P</b>aths for Machine <b>L</b>earning</h1>

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


## Table of Contents

- [Installation](#installation)
- [Examples](#examples)
- [Hyperparameters](#hyperparameters)
- [Best Practices](#best-practices)
- [Algorithm](#algorithm)
- [Citation](#citation)


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
use_log1p              = False, eps = 0.001
grad_mult              = 100000000.0
use_pwa                = True
use_saga               = True
sample_all_paths_first = False
n_theta                = 1
min_prob_frac          = 0.0
number of SCRAPL keys  = 315
unif_prob              = 0.00317460
```

Calculating the loss for two signals: 

```python 
import torch

# Create two random tensors of shape (batch_size, num_channels, signal_length)
x = torch.randn(4, 1, 48000) # Example input tensor 
x_target = torch.randn(4, 1, 48000) # Example target tensor 
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


## Hyperparameters


## Best Practices


## Algorithm

![image](/docs/figs/scrapl_algorithm.png)


## Citation

If you find this project helpful for your own work, please consider citing it:

<pre><code>
@inproceedings{mitcheltree2026scrapl,
    title={{SCRAPL}: Scattering Transform with Random Paths for Machine Learning},
    author={Christopher Mitcheltree and Vincent Lostanlen and Emmanouil Benetos and Mathieu Lagrange},
    booktitle={Proceedings of the International Conference on Learning Representations (ICLR)},
    year={2026}
}
</code></pre>
