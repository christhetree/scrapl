The `single_path_jtfs` package was written by Vincent Lostanlen.

The original repository for this package is located at: [https://github.com/lostanlen/scrapl](https://github.com/lostanlen/scrapl)

Minor modifications were made by Christopher Mitcheltree and can be viewed at: [https://github.com/christhetree/single_path_jtfs](https://github.com/christhetree/single_path_jtfs)

## Experimental JAX frontend

The JAX frontend implements single-path JTFS using the same scattering core as
the PyTorch frontend. A [JAX loss wrapper with θ-IS warmup and path sampling](../jax/README.md)
is now available, with optional P-Adam gradient normalisation and P-SAGA gradient correction.
The `[jax]` extra works without PyTorch. The independent `[torch]` extra
enables the PyTorch backend, and `[torch,jax]` installs both frameworks.

Install from this checkout, including the pinned submodules:

```sh
git submodule update --init scrapl/kymatio
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e '.[jax]'
```

Construct the transform outside JIT, then compile with a fixed path:

```python
import jax
import jax.numpy as jnp

from scrapl.single_path_jtfs.jax import TimeFrequencyScrapl

jtfs = TimeFrequencyScrapl(
    shape=(256,), J=4, Q=(2, 1), J_fr=2, Q_fr=1, T=16, F=2
)
paths = [key for key in jtfs.meta()["key"] if len(key) == 2]
n2, n_fr = paths[0]

@jax.jit
def coefficients(x):
    return jtfs.scattering_singlepath(x, n2, n_fr)["coef"]

x_key, target_key = jax.random.split(jax.random.key(42))
x = jax.random.normal(x_key, (2, 1, 256))
target = jax.random.normal(target_key, (2, 1, 256))
target_coef = coefficients(target)

def loss(x):
    difference = coefficients(x) - target_coef
    return jnp.linalg.norm(difference, axis=(-2, -1)).mean()

value, gradient = jax.jit(jax.value_and_grad(loss))(x)
```

The generic entry point also accepts `backend="jax"`. Path indices must be Python
integers fixed during tracing, since paths select different filters and shapes.
Alternatively, compile `jtfs.scattering_singlepath` with
`static_argnames=("n2", "n_fr")`. A new static path can cause another compilation.

Like the existing SCRAPL implementation, this frontend keeps padding and flattens
the input's leading batch/channel dimensions. Coefficients have shape
`(flattened_batch, 1, frequency, time)`. They use native JAX arrays and omit
PyTorch's final singleton axis. Compare them with
`torch_path["coef"].squeeze(-1)`; do not squeeze the JAX time axis.

Install both frameworks to run the parity checks:

```sh
git submodule update --init scrapl/pytorch_hessian_eigenthings
source .venv/bin/activate
uv pip install -e '.[torch,jax,test]'
python -m pytest tests/test_jax_single_path_jtfs.py
```

These checks cover all 14 second-order paths in a small filter bank with three
temporal/frequency averaging configurations, including global averaging. They
compare metadata and coefficients, raw and log-compressed L2 losses, input
gradients, and Hessian-vector products against PyTorch. They also exercise JIT,
batched stereo inputs of non-power-of-two length, and silence. The silence check
concerns the transform's derivatives; it does not establish differentiability of
the L2 loss at an exact match.

Initial validation uses float32 on CPU with Python 3.12, JAX 0.11.1 and PyTorch
2.14.0. End-to-end θ-IS validation is described in the JAX loss documentation.
GPU performance and large filter banks remain to be validated.
