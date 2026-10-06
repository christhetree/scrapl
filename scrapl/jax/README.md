# Experimental JAX SCRAPL loss

`scrapl.jax.SCRAPLLoss` provides an Lp scattering loss with uniform or weighted
random path sampling. It uses the same second-order paths and padding convention as the
PyTorch implementation. Each call selects one path for the entire batch and
averages its distances over batch and channels. Inputs must have matching shapes
`(batch, channels, samples)` and use float32 or float64.

This stage includes θ-IS warmup, uniform and importance sampling, fixed-path
evaluation, optional `log1p` compression, JAX differentiation, P-Adam gradient
normalisation and P-SAGA gradient correction. The loss itself returns ordinary
gradients; P-Adam and P-SAGA are optional, explicit Optax transformations applied after
differentiation.

## Installation

Use this checkout's pinned Kymatio submodule and the `[jax]` extra. JAX
installation, imports, loss computation and warmup do not require PyTorch. The
independent `[torch]` extra enables the existing PyTorch API; `[torch,jax]`
installs both frameworks. A bare installation includes neither framework.

```sh
git submodule update --init scrapl/kymatio
uv sync --extra jax
```

## Examples

### Importing and initializing `SCRAPLLoss`

Initialize `SCRAPLLoss` with the minimum required arguments:

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

### Calculating the loss for two signals

In JAX, pseudo-randomness is managed explicitly using `jax.random.key`:

```python
import jax

# Create two random arrays of shape (batch_size, num_channels, signal_length)
key = jax.random.key(0)
key_x, key_target, key_loss = jax.random.split(key, 3)
x = jax.random.normal(key_x, (4, 1, 48000))
x_target = jax.random.normal(key_target, (4, 1, 48000))

# Compute the SCRAPL loss between x and x_target. Since SCRAPL is stochastic,
# passing a PRNG key will sample a random scattering path.
loss = scrapl_loss(x, x_target, key=key_loss)
print(f"SCRAPL loss: {float(loss):.4f}")
```

### `SCRAPLLoss` utility attributes and methods

```python
import jax

print(f"Number of scattering paths: {scrapl_loss.n_paths}")
print(f"Uniform path sampling probability: {scrapl_loss.unif_prob:.6f}")

# Sample a path index explicitly with a JAX PRNG key
key = jax.random.key(42)
path_idx = scrapl_loss.sample_path(key)
print(f"Sampled path index: {int(path_idx)}")

# Calculate the loss for a specific path index
loss = scrapl_loss(x, x_target, path_idx=8)
print(f"Loss for specific path: {float(loss):.6f}")
```

> [!NOTE]
> Unlike the PyTorch version which is stateful (tracking `curr_path_idx`, `path_counts`, `state_dict`, and `clear()`), the JAX `SCRAPLLoss` is a pure, immutable Equinox module. Path sampling is explicit via `scrapl_loss.sample_path(key)` and all state lives cleanly in caller-managed PyTrees or Optax optimizer states.

### Using $\mathcal{P}$-Adam and $\mathcal{P}$-SAGA

In JAX, $\mathcal{P}$-Adam and $\mathcal{P}$-SAGA are implemented as native [Optax](https://github.com/google-deepmind/optax) gradient transformations (`padam` and `psaga`). They do not require backward hooks or attaching parameters; instead, chain them directly into your Optax optimizer:

```python
import equinox as eqx
import jax
import optax
from scrapl.jax import SCRAPLLoss, padam, psaga

# Example loss and toy MLP model
scrapl_loss = SCRAPLLoss(shape=1024, J=3, Q1=1, Q2=1, J_fr=2, Q_fr=1)
key = jax.random.key(42)
key_model, key_data, key_step = jax.random.split(key, 3)

model = eqx.nn.MLP(
    in_size=1024,
    out_size=1024,
    width_size=8,
    depth=1,
    activation=jax.nn.relu,
    final_activation=jax.nn.tanh,
    key=key_model,
)

# Chain P-Adam, P-SAGA, weight decay, and learning rate scaling
optimiser = optax.chain(
    padam(scrapl_loss.n_paths, b1=0.9, b2=0.999, eps=1e-8, grad_mult=1.0),
    psaga(scrapl_loss.n_paths),
    optax.add_decayed_weights(0.01),
    optax.scale_by_learning_rate(1e-4),
)
opt_state = optimiser.init(eqx.filter(model, eqx.is_array))

# Create input signals
x = jax.random.normal(key_data, (4, 1, 1024))
x_target = jax.random.normal(key_data, (4, 1, 1024))

@eqx.filter_jit
def train_step(model, opt_state, x, x_target, key):
    path_key, _ = jax.random.split(key)
    path_idx = scrapl_loss.sample_path(path_key)

    def loss_fn(m):
        x_hat = jax.vmap(m)(x.squeeze(1))[:, None, :]
        return scrapl_loss(x_hat, x_target, path_idx=path_idx)

    loss_val, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimiser.update(
        grads, opt_state, params=eqx.filter(model, eqx.is_array), path_idx=path_idx
    )
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss_val

model, opt_state, loss_val = train_step(model, opt_state, x, x_target, key_step)
print(f"Step Loss: {float(loss_val):.4f}")
```

### Importance Sampling Warmup ($\theta$-IS)

The SCRAPL algorithm includes an architecture-informed importance sampling heuristic ($\theta$-IS) that estimates the loss landscape curvature with respect to synthesizer controls $\theta_\text{synth}$ across all scattering paths.

In JAX, `warmup_lc_hvp` runs curvature estimation and returns a `ThetaISResult` containing `probs`, `curvatures`, and `relative_residuals`:

```python
import equinox as eqx
import jax
import jax.numpy as jnp
from scrapl.jax import SCRAPLLoss, warmup_lc_hvp

# Setup dimensions
bs = 4
n_ch = 1
n_samples = 8096
n_theta = 3
n_batches = 1

# Provide an encoder that outputs n_theta parameters
encoder = eqx.nn.MLP(
    in_size=n_samples,
    out_size=n_theta,
    width_size=n_theta,
    depth=1,
    activation=jax.nn.relu,
    final_activation=jax.nn.sigmoid,
    key=jax.random.key(1),
)

# Provide a differentiable synthesizer taking n_theta parameters
decoder = eqx.nn.MLP(
    in_size=n_theta,
    out_size=n_samples,
    width_size=n_theta,
    depth=1,
    activation=jax.nn.relu,
    final_activation=jax.nn.tanh,
    key=jax.random.key(2),
)

# Functional encoder and synthesiser functions
def theta_fn(params, x):
    enc = eqx.combine(params, encoder)
    return jax.vmap(enc)(x.squeeze(1))

def synth_fn(theta):
    return jax.vmap(decoder)(theta)[:, None, :]

# Warmup data: (n_batches, batch_size, channels, samples)
xs = jax.random.normal(jax.random.key(3), (n_batches, bs, n_ch, n_samples))
params = eqx.filter(encoder, eqx.is_array)

loss_fn = SCRAPLLoss(shape=n_samples, J=3, Q1=1, Q2=1, J_fr=2, Q_fr=1)
print(f"Uniform path sampling probability: {float(loss_fn.unif_prob):.6f}")

# Run warmup
warmup = warmup_lc_hvp(
    loss_fn,
    params,
    theta_fn,
    synth_fn,
    xs,
    key=jax.random.key(4),
    n_iter=20,
    min_prob_frac=0.0,
)

print(
    f"[min, max] path sampling probabilities (after warmup): "
    f"[{float(warmup.probs.min()):.6f}, {float(warmup.probs.max()):.6f}]"
)
```

> [!NOTE]
> In PyTorch, multi-process path distribution files (`vals_{path_idx}.pt`) are saved and loaded from disk via `load_probs_from_warmup_dir`. In JAX, `warmup_lc_hvp` compiles and computes the full path distribution directly, returning `warmup.probs` as an array which can be passed to `loss_fn(..., probs=warmup.probs)` or saved/loaded with `jnp.save` / `jnp.load`.

---

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

`padam` creates an Optax transformation keeping separate first and second moments for each path and parameter leaf.
Its output is a normalised gradient direction: chain with `optax.scale_by_learning_rate` and apply with `optax.apply_updates`. Do not feed
the normalised gradients through another Adam optimiser.

Continuing from the warmup example above:

```python
import optax
from scrapl.jax import padam

optimiser = optax.chain(
    padam(loss_fn.n_paths, b1=0.9, b2=0.999, eps=1e-8, grad_mult=1.0),
    optax.scale_by_learning_rate(1e-3),
)
opt_state = optimiser.init(params)

@jax.jit
def p_adam_step(params, opt_state, batch, key, probs):
    key, path_key = jax.random.split(key)
    path_idx = loss_fn.sample_path(path_key, probs=probs)

    def objective(params):
        prediction = synthesiser(encoder(params, batch))
        return loss_fn(batch, prediction, path_idx=path_idx)

    value, grads = jax.value_and_grad(objective)(params)
    updates, opt_state = optimiser.update(grads, opt_state, params=params, path_idx=path_idx)
    params = optax.apply_updates(params, updates)
    return params, opt_state, key, value

params, opt_state, key, value = p_adam_step(
    params, opt_state, xs[0], key, warmup.probs
)
```

Sample the path once and pass that same index to both the loss and the Optax update. The
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

`PAdamState` is a NamedTuple containing `count`, `last_steps`, `m` and `v`. Save it
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

`psaga` creates an Optax transformation keeping the most recent incoming gradient for each path and parameter leaf.
Use it on its own after differentiation, or chain it after `padam` to match the combined
PyTorch hook.

Continuing with the encoder, synthesiser and warmup probabilities above, this
example starts fresh optimiser histories and enables both transformations via `optax.chain`:

```python
import optax
from scrapl.jax import padam, psaga

optimiser = optax.chain(
    padam(loss_fn.n_paths, grad_mult=1.0),
    psaga(loss_fn.n_paths),
    optax.scale_by_learning_rate(1e-3),
)
opt_state = optimiser.init(params)

@jax.jit
def p_saga_step(params, opt_state, batch, key, probs):
    key, path_key = jax.random.split(key)
    path_idx = loss_fn.sample_path(path_key, probs=probs)

    def objective(params):
        prediction = synthesiser(encoder(params, batch))
        return loss_fn(batch, prediction, path_idx=path_idx)

    value, grads = jax.value_and_grad(objective)(params)
    updates, opt_state = optimiser.update(grads, opt_state, params=params, path_idx=path_idx)
    params = optax.apply_updates(params, updates)
    return params, opt_state, key, value

params, opt_state, key, value = p_saga_step(
    params, opt_state, xs[0], key, warmup.probs
)
```

For P-SAGA alone, omit the `padam` transform in the chain; P-SAGA then receives raw
gradients. To reproduce a PyTorch `grad_mult` setting without P-Adam, scale every
gradient leaf before passing it to P-SAGA. With P-Adam enabled, set its
`grad_mult` argument instead. Apply scaling once, before either transformation.
P-SAGA has no additional gradient multiplier.

The same scalar `path_idx` must select the loss and the Optax update. P-SAGA
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

`PSAGAState` is a NamedTuple with `seen`, a boolean vector of length `n_paths`, and
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
uv sync --extra torch --extra jax --extra test
uv run pytest
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
