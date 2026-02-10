<div align="center">
<h1><b>SCRAPL</b>: <b>Sc</b>attering Transform with <b>Ra</b>ndom <b>P</b>aths for Machine <b>L</b>earning</h1>
<p>
<a href="https://christhetr.ee/" target=”_blank”>Christopher Mitcheltree</a>,
<a href="https://www.lostanlen.com/" target=”_blank”>Vincent Lostanlen</a>,
<a href="https://www.qmul.ac.uk/eecs/people/profiles/benetosemmanouil.html" target=”_blank”>Emmanouil Benetos</a>, and
<a href="https://mathieulagrange.github.io/" target=”_blank”>Mathieu Lagrange</a>
</p>

[![OpenReview](https://img.shields.io/badge/OpenReview-ICLR_2026_Paper-b31b1b.svg)](https://openreview.net/forum?id=RuYwbd5xYa)
[![Listening Samples](https://img.shields.io/badge/%F0%9F%94%8A%F0%9F%8E%B6-Listening_Samples-blue)](https://christhetree.github.io/scrapl/)
[![Release](https://img.shields.io/badge/PyPI-v0.1.0-green)](https://pypi.org/project/scrapl/)
[![License](https://img.shields.io/badge/License-MIT-blue)](https://opensource.org/license/mit)
</div>

### This is the `README.md` for the ICLR 2026 paper and experiments; for the `SCRAPLLoss` PyTorch implementation and PyPI package documentation, see the [`scrapl/`](./scrapl) directory.

<h2>Abstract</h2>
<hr>
The Euclidean distance between wavelet scattering transform coefficients (known as <em>paths</em>) provides informative gradients for perceptual quality assessment of deep inverse problems in computer vision, speech, and audio processing.
However, these transforms are computationally expensive when employed as differentiable loss functions for stochastic gradient descent due to their numerous paths, which significantly limits their use in neural network training.
Against this problem, we propose "<b>Sc</b>attering transform with <b>Ra</b>ndom <b>P</b>aths for machine <b>L</b>earning" (SCRAPL): a stochastic optimization scheme for efficient evaluation of multivariable scattering transforms.
We implement SCRAPL for the joint time–frequency scattering transform (JTFS) which demodulates spectrotemporal patterns at multiple scales and rates, allowing a fine characterization of intermittent auditory textures.
We apply SCRAPL to differentiable digital signal processing (DDSP), specifically, unsupervised sound matching of a granular synthesizer and the Roland TR-808 drum machine.
We also propose an initialization heuristic based on importance sampling, which adapts SCRAPL to the perceptual content of the dataset, improving neural network convergence and evaluation performance.
We make our code and audio samples available and provide SCRAPL as a Python package.
<p>

![image](docs/figs/relative_param_error_vs_computation.svg)

_Figure1: Mean average synthesizer parameter error (y-axis) versus computational cost (x-axis) of unsupervised sound matching models for the granular synthesis task. 
Both axes are rescaled by the performance of a supervised model with the same number of parameters. 
Whiskers denote 95% CI, estimated over 20 random seeds. Due to computational limitations, 
JTFS-based sound matching is evaluated only once._

<h2>Citation</h2>
<hr>
Accepted to the International Conference on Learning Representations (ICLR), Rio de Janeiro, Brazil, 23 - 27 April 2026.
<pre><code>
@inproceedings{mitcheltree2026scrapl,
    title={{SCRAPL}: Scattering Transform with Random Paths for Machine Learning},
    author={Christopher Mitcheltree and Vincent Lostanlen and Emmanouil Benetos and Mathieu Lagrange},
    booktitle={Proceedings of the International Conference on Learning Representations (ICLR)},
    year={2026}
}
</code></pre>


<hr>
<h2>Instructions for Reproducibility</h2>

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
