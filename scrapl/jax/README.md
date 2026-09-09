# Experimental JAX SCRAPL loss

`scrapl.jax.SCRAPLLoss` provides an Lp scattering loss with uniform or weighted
random path sampling. It uses the same second-order paths and padding convention as the
PyTorch implementation. Each call selects one path for the entire batch and
averages its distances over batch and channels. Inputs must have matching shapes
`(batch, channels, samples)` and use float32 or float64.

This stage includes θ-IS warmup, uniform and importance sampling, fixed-path
evaluation, optional `log1p` compression, JAX differentiation, P-Adam gradient
normalisation and P-SAGA gradient correction. The loss itself returns ordinary
gradients; P-Adam and P-SAGA are optional, explicit transformations applied after
differentiation.

## Installation

Use this checkout's pinned Kymatio submodule and the `[jax]` extra. JAX
installation, imports, loss computation and warmup do not require PyTorch. The
independent `[torch]` extra enables the existing PyTorch API; `[torch,jax]`
installs both frameworks. A bare installation includes neither framework.

```sh
git submodule update --init scrapl/kymatio
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e '.[jax]'
```

## Training with uniform sampling

Construct the loss once, outside JIT. Pass a fresh random key for each step:

```python
import jax
import jax.numpy as jnp

from scrapl.jax import SCRAPLLoss

loss_fn = SCRAPLLoss(
    shape=128, J=3, Q1=2, Q2=1, J_fr=2, Q_fr=1, use_rho_log1p=True
)
target = jax.random.normal(jax.random.key(8), (1, 1, 128))

def objective(theta, path_key):
    prediction = jax.nn.sigmoid(theta) * target
    return loss_fn(prediction, target, key=path_key)

@jax.jit
def step(theta, key):
    key, path_key = jax.random.split(key)
    value, gradient = jax.value_and_grad(objective)(theta, path_key)
    return theta - 0.1 * gradient, key, value

theta = jnp.asarray(-1.0)
key = jax.random.key(10)
for _ in range(20):
    theta, key, sampled_loss = step(theta, key)

print("Fitted gain:", float(jax.nn.sigmoid(theta)))
```

The example fits a scalar gain; an encoder and differentiable synthesiser can
replace the prediction expression. Keys belong to the caller: the loss stores no
random state, counters or gradient history. Reusing a key reproduces its path.
PyTorch and JAX do not produce the same path sequence from the same numeric seed.

## Fixed paths and path logging

Pass exactly one of `key` or `path_idx`. The return value is a scalar JAX array.
To inspect the selected path, sample its index explicitly and pass that index:

```python
prediction = jax.nn.sigmoid(theta) * target
path_idx = loss_fn.sample_path(jax.random.key(42))
value = jax.jit(loss_fn)(prediction, target, path_idx=path_idx)
print("Path:", int(path_idx), "Loss:", float(value))
```

`loss_fn.scrapl_keys` maps indices to `(n2, n_fr)`. `loss_fn.n_paths` gives the
number of paths, and `loss_fn.unif_prob` is `1 / n_paths`. With uniform sampling,
the sampled loss's expectation is the arithmetic mean of the individual path losses. There is no
additional multiplication by the number of paths or division by the sampling
probability.

For a single fixed path, use a Python integer outside JIT or mark it static:

```python
fixed_loss = jax.jit(loss_fn, static_argnames=("path_idx",))
value = fixed_loss(prediction, target, path_idx=0)
```

## θ-IS warmup and importance sampling

Warmup estimates how each scattering path interacts with each synthesiser control
through the encoder's trainable weights. The encoder and synthesiser must use JAX
operations and be deterministic during warmup. Put model state and fixed synthesis
settings in closures; pass only trainable encoder arrays in the parameter PyTree.

`theta_fn(params, x)` receives a waveform batch and returns `(batch, n_theta)`.
`synth_fn(theta)` returns `(batch, channels, samples)`. Warmup data has shape
`(n_batches, batch, channels, samples)`; batches must have the same shape.

This self-contained example computes probabilities and uses them in a training
step:

```python
import jax
import jax.numpy as jnp

from scrapl.jax import SCRAPLLoss, warmup_lc_hvp

loss_fn = SCRAPLLoss(
    shape=128, J=3, Q1=2, Q2=1, J_fr=1, Q_fr=1,
    T=8, F=1, use_rho_log1p=True,
)
params = {
    "bias": jnp.array([-0.4, 0.2]),
    "matrix": jnp.array([[0.2, -0.1], [0.1, 0.3]]),
}
xs = jax.random.normal(jax.random.key(1), (2, 2, 1, 128))
basis = jax.random.normal(jax.random.key(2), (2, 128))

def encoder(params, x):
    return jax.nn.sigmoid(x[:, 0, :2] @ params["matrix"] + params["bias"])

def synthesiser(theta):
    return (theta @ basis)[:, None, :]

warmup = warmup_lc_hvp(
    loss_fn, params, encoder, synthesiser, xs,
    key=jax.random.key(3), n_iter=40, min_prob_frac=0.05,
)
print("Probabilities:", warmup.probs)
print("Largest relative residual:", float(warmup.relative_residuals.max()))

@jax.jit
def step(params, batch, key, probs):
    key, path_key = jax.random.split(key)

    def objective(params):
        prediction = synthesiser(encoder(params, batch))
        return loss_fn(batch, prediction, key=path_key, probs=probs)

    value, gradient = jax.value_and_grad(objective)(params)
    params = jax.tree_util.tree_map(lambda p, g: p - 1e-5 * g, params, gradient)
    return params, key, value

params, key, value = step(params, xs[0], jax.random.key(4), warmup.probs)
```

`warmup_lc_hvp` returns a `ThetaISResult` containing `curvatures` and
`relative_residuals` of shape `(n_paths, n_theta)`, and `probs` of shape
`(n_paths,)`. It leaves the loss and parameters unchanged. Keep this result in
your training/checkpoint state and pass its probabilities explicitly. To log a
weighted path, use `loss_fn.sample_path(key, probs=warmup.probs)` and then evaluate
that fixed index. Passing `probs` together with `path_idx` raises `ValueError`.

For curvature estimates `c[path, theta]`, the completed-warmup distribution is:

```text
scores = maximum(c, eps)
per_theta = scores / sum(scores, axis=paths)
probs = (1 - min_prob_frac) * mean(per_theta, axis=theta)
        + min_prob_frac / n_paths
```

The implementation normalises in log space. This matches the repo's completed
PyTorch warmup formula. A zero-curvature theta contributes a uniform distribution;
if all curvatures are zero, sampling stays uniform. `min_prob_frac` must be in
`[0, 1)` and sets a floor of `min_prob_frac / n_paths` on every probability.
`theta_importance_probs(curvatures, min_prob_frac=..., eps=...)` exposes this
conversion separately and supports JIT with those configuration options fixed.

Probability vectors must be finite, nonnegative, floating-point arrays with
`n_paths` entries summing to one. Invalid values return `-1` from `sample_path`
and `NaN` from a sampled loss, including inside JIT. Invalid shapes or dtypes raise
an exception. Supplying new probability values as a training-step argument does
not require changing the loss instance.

Weighted sampling retains the PyTorch implementation's scaling: the chosen path's
loss and gradient are unchanged. Its expectation is `sum(probs * path_losses)`,
so it is not an importance-corrected estimator of the uniform path mean.

Warmup uses all encoder parameter leaves as one group, corresponding to PyTorch's
`agg="none"`. Curvature products are summed over batches, as in PyTorch. Per-leaf
aggregation, partial-path jobs, saved warmup directories and automatic adaptive
refreshes are not part of this JAX API yet. Call warmup outside JIT; it compiles
each path separately and evaluates theta coordinates and batches sequentially.
Large models and filter banks may still make warmup expensive.

Each theta contributes a parameter-gradient vector whose Jacobian need not be
symmetric. The implementation applies its transpose, matching PyTorch's
grad-of-grad calculation. Power iteration estimates the magnitude of a dominant
eigenvalue; it is a heuristic, not a certified Lipschitz bound. Inspect
`relative_residuals`: they measure `norm(A v - lambda v) / max(norm(A v), eps)`
for the last unit iterate. A large residual indicates that the estimate has not
settled. Increasing `n_iter` can help, but does not guarantee convergence for
every non-symmetric operator. Iteration counts and initial vectors differ from
PyTorch, so finite-iteration estimates are not expected to be bit-identical.

## P-Adam gradient normalisation

`PAdam` keeps separate first and second moments for each path and parameter leaf.
Apply it to the gradient PyTree after differentiating the selected path's loss.
Its output is a gradient direction: apply a plain SGD step afterwards. Do not feed
the normalised gradients through another Adam optimiser.

Continuing from the warmup example above:

```python
from scrapl.jax import PAdam

p_adam = PAdam(loss_fn.n_paths, b1=0.9, b2=0.999, eps=1e-8, grad_mult=1.0)
p_adam_state = p_adam.init(params)

@jax.jit
def p_adam_step(params, state, batch, key, probs):
    key, path_key = jax.random.split(key)
    path_idx = loss_fn.sample_path(path_key, probs=probs)

    def objective(params):
        prediction = synthesiser(encoder(params, batch))
        return loss_fn(batch, prediction, path_idx=path_idx)

    value, grads = jax.value_and_grad(objective)(params)
    directions, state = p_adam.update(grads, state, path_idx=path_idx)
    params = jax.tree_util.tree_map(lambda p, g: p - 1e-3 * g, params, directions)
    return params, state, key, value

params, p_adam_state, key, value = p_adam_step(
    params, p_adam_state, xs[0], key, warmup.probs
)
```

Sample the path once and pass that same index to both the loss and P-Adam. The
example accepts θ-IS probabilities; passing `None` as `probs` uses uniform
sampling. No hooks or parameter attachment are needed.

With `s = state.count + 2`, `r = state.last_steps[path_idx]` and `N = n_paths`,
P-Adam uses `t = s / N` and `delta = (s - r) / N`. For each selected moment row:

```text
g = grad_mult * gradient
m = b1**delta * m + (1 - b1**delta) * g
v = b2**delta * v + (1 - b2**delta) * g**2
direction = (m / (1 - b1**t)) / (sqrt(v / (1 - b2**t)) + eps)
```

The first update uses timestamp **2**, matching this repo's PyTorch behaviour:
its forward pass increments the loss counter, then the gradient hook adds one.
The JAX counter counts completed P-Adam updates. Numerical parity assumes one
forward/backward training pass per update. Evaluating the loss, running warmup or
computing gradients alone does not advance JAX optimiser state.

Elapsed time is measured in global steps, including steps spent on other paths.
Only the selected path's moments and timestamp are updated. Every gradient leaf
is processed on every update; use zero arrays for unused leaves. This does not
model PyTorch hooks being skipped for parameters whose gradient is `None`.

The defaults are `b1=0.9`, `b2=0.999`, `eps=1e-8`, and **`grad_mult=1.0`**.
Set `grad_mult` explicitly to match an existing PyTorch run, whose loss defaults
to `1e8`. Scaling acts before the moment updates and changes the effective role
of epsilon, so it can matter for tiny raw scattering gradients. Decay complements
and bias corrections use `expm1` to retain precision for small fractional times.

`PAdamState` is a JAX PyTree containing `count`, `last_steps`, `m` and `v`. Save it
alongside parameters, PRNG keys, sampling probabilities and the P-Adam/loss
configuration. Restored NumPy array leaves are accepted as well. Use `init(params)`
only when starting or deliberately resetting the moment history.

Moment memory contains **two copies of the entire parameter tree per path**, plus
integer counters. This can be substantial for large models and filter banks.
Parameter and moment leaves must use matching float32 or float64 dtypes and shapes.
Invalid Python path indices raise; invalid scalar array indices, non-finite
gradients, numerical failures or counter exhaustion return NaN directions and
unchanged state. Treat NaN directions as a failed step before updating parameters.

## P-SAGA gradient correction

`PSAGA` keeps the most recent incoming gradient for each path and parameter leaf.
Use it on its own after differentiation, or after P-Adam to match the combined
PyTorch hook. Its output is a gradient direction for plain SGD.

Continuing with the encoder, synthesiser and warmup probabilities above, this
example starts fresh optimiser histories and enables both transformations:

```python
from scrapl.jax import PAdam, PSAGA

p_adam = PAdam(loss_fn.n_paths, grad_mult=1.0)
p_saga = PSAGA(loss_fn.n_paths)
p_adam_state = p_adam.init(params)
p_saga_state = p_saga.init(params)

@jax.jit
def p_saga_step(params, adam_state, saga_state, batch, key, probs):
    key, path_key = jax.random.split(key)
    path_idx = loss_fn.sample_path(path_key, probs=probs)

    def objective(params):
        prediction = synthesiser(encoder(params, batch))
        return loss_fn(batch, prediction, path_idx=path_idx)

    value, grads = jax.value_and_grad(objective)(params)
    grads, adam_state = p_adam.update(grads, adam_state, path_idx=path_idx)
    directions, saga_state = p_saga.update(grads, saga_state, path_idx=path_idx)
    params = jax.tree_util.tree_map(lambda p, g: p - 1e-3 * g, params, directions)
    return params, adam_state, saga_state, key, value

params, p_adam_state, p_saga_state, key, value = p_saga_step(
    params, p_adam_state, p_saga_state, xs[0], key, warmup.probs
)
```

For P-SAGA alone, omit the `p_adam.update` line; P-SAGA then receives raw
gradients. To reproduce a PyTorch `grad_mult` setting without P-Adam, scale every
gradient leaf before passing it to P-SAGA. With P-Adam enabled, set its
`grad_mult` argument instead. Apply scaling once, before either transformation.
P-SAGA has no additional gradient multiplier.

The same scalar `path_idx` must select the loss and both transformations. P-SAGA
records paths on successful updates; loss evaluation and warmup do not mark paths
as seen. As with P-Adam, parity assumes one forward/backward training pass per
update, processing every gradient leaf. Unused leaves should be zero arrays.

For path `i`, incoming gradient `g`, and the stored gradients `h` **before** the
update, the rule is:

```text
seen[i] = True
denominator = max(1, sum(seen) - 1)
direction = g - h[i] + sum(h, axis=paths) / denominator
h[i] = g
```

This preserves the repo's PyTorch averaging convention, including on repeated
visits. Once all `N > 1` paths have been seen, the denominator remains `N - 1`.
It differs from textbook SAGA's average over all `N` components. The initial
history is zero, so the first direction equals the incoming gradient. A later
zero input gradient can still produce a nonzero correction from the history.
Only the selected history row changes; it stores the input to P-SAGA, including
P-Adam normalisation if enabled, rather than the corrected output.

Uniform or θ-IS sampling can supply the path. P-SAGA neither chooses paths nor
uses probabilities in its correction. This matches PyTorch and does not add an
inverse-probability correction or an unbiased-gradient guarantee. The sampler
does not force every path to be visited before allowing repeats.

`PSAGAState` is a JAX PyTree with `seen`, a boolean vector of length `n_paths`, and
`path_grads`, a parameter-shaped PyTree with a leading path axis on every leaf.
Save it alongside parameters, P-Adam state if enabled, PRNG keys, probabilities
and configuration. Restored NumPy array leaves are accepted. Reset the histories
when changing path ordering or the gradient transformation that feeds P-SAGA.

P-SAGA stores **one copy of the entire parameter tree per path**, in addition to
P-Adam's two copies when used together. It sums that history on every update.
Leaves must have matching float32 or float64 dtypes and shapes. Invalid Python
path indices raise; invalid array indices, non-finite gradients or numerical
failures return NaN directions and unchanged P-SAGA state. If a composed step
fails, keep the original parameters and both original optimiser states; a
successful P-Adam call cannot automatically roll itself back when P-SAGA fails.

## Compilation and numerical behaviour

Dynamic indices use `jax.lax.switch`: JIT traces all path branches on the first
call, then runs only the selected branch. Each branch reduces its coefficients to
a scalar internally, so differing coefficient shapes are supported. First-call
compilation time and compiled-code size can grow with the filter bank. A static
Python path index compiles only that path; changing it causes another compilation.

Use one scalar path key per batch. Applying `vmap` over separate path keys or
indices can turn conditional execution into evaluation of all branches, losing
the computation savings. Vectorising `sample_path` alone is fine.

The norm parameter `p` supports values at least 1, including positive infinity.
Log compression is `log1p(coef / log1p_eps)` before taking distances. Exact matches
return zero loss and zero gradients, including when only some batch/channel pairs
match. This is a chosen subgradient at a nondifferentiable point, not a claim that
the norm has a classical Hessian there.

Configuration is frozen: construct another loss to change its settings, and do
not mutate the underlying JTFS filters after compilation. Invalid Python path
indices raise `ValueError`. Invalid scalar array indices produce `NaN` in both
eager and compiled calls, instead of silently accepting JAX's clamped index.

## Validation

Install both frameworks for the comparison tests. Without an optional framework,
tests requiring it are skipped; the import isolation tests still check the
installed backend.

```sh
git submodule update --init scrapl/pytorch_hessian_eigenthings
source .venv/bin/activate
uv pip install -e '.[torch,jax,test]'
python -m pytest
```

Tests compare all 14 paths of a small filter bank with PyTorch, including gradients
with respect to both signals, L1/L2/L3/L-infinity norms, log compression, and local
and global averaging. They also cover uniform sampling, static and dynamic path
agreement, runtime branch selection, exact matches, silence, Hessian-vector
products, input validation and a stochastic optimisation loop compiled with
`jax.lax.scan`. That loop reduces the mean loss over all its paths by more than
half in 20 steps.

θ-IS tests compare curvature-vector products and completed warmup estimates with
PyTorch for a small encoder with two controls and two waveform batches. They also
check a dense non-symmetric Jacobian, probability normalisation and its floor,
zero-curvature warmup, weighted sampling frequencies, invalid probability values,
convergence diagnostics and a weighted training step.

P-Adam tests compare moment histories, directions and complete parameter updates
with PyTorch across repeated and delayed path visits, different decay settings
and gradient multipliers. They also cover zero gradients, small fractional decay,
checkpoint continuation, invalid-update rejection and a compiled training loop
using weighted sampling.

P-SAGA tests compare gradient histories and full training updates with PyTorch,
both alone and following P-Adam. They cover new and repeated path visits, the
reference averaging convention, zero gradients, gradient scaling, checkpoint
continuation, invalid-update rejection and compiled training loops with weighted
sampling that reduce the mean loss across all paths.

Validation currently uses float32 on CPU with Python 3.12, JAX 0.11.1 and PyTorch
2.14.0. Float64 parity, GPU performance and compilation costs for large filter
banks remain to be validated.
