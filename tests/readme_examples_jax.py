"""Runs the code examples from scrapl/jax/README.md, concatenated in order.

The code below is copied exactly from the README so that its console output can be
compared against the "Console output" blocks in the README. Only the section comments
are added. The θ-IS warmup example writes its results to ./scrapl_warmup.

Run with: uv run python tests/readme_examples_jax.py
"""

# ============================================================================
# README example 1/5 (Examples (JAX))
# ============================================================================
# Import SCRAPLLoss from the scrapl.jax Python package
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

# ============================================================================
# README example 2/5 (Examples (JAX))
# ============================================================================
import jax

# Create two random arrays of shape (batch_size, num_channels, signal_length)
key = jax.random.key(0)
key_x, key_target, key_loss = jax.random.split(key, 3)
x = jax.random.normal(key_x, (4, 1, 48000))  # Example input array
x_target = jax.random.normal(key_target, (4, 1, 48000))  # Example target array
# Sample a scattering path index. Since SCRAPL is stochastic, a different PRNG key
# will usually sample a different path, resulting in a different loss value. 
# Sampling the path separately is necessary for JIT compilation (see the training 
# example below).
path_idx = scrapl_loss.sample_path(key_loss)
# Compute the SCRAPL loss between x and x_target for the sampled path
loss = scrapl_loss(x, x_target, path_idx=path_idx)
print(f"SCRAPL loss: {float(loss)}")

# ============================================================================
# README example 3/5 (Examples (JAX))
# ============================================================================
print(f"Number of scattering paths: {scrapl_loss.n_paths}")
print(f"Uniform path sampling probability: {scrapl_loss.unif_prob:.6f}")

# Sample a path index explicitly with a JAX PRNG key
# (the same key will sample the same path index every time)
path_idx = scrapl_loss.sample_path(jax.random.key(42))
print(f"Sampled path index (PRNG key): {path_idx}")

# Calculate the loss for a specific path index
loss = scrapl_loss(x, x_target, path_idx=path_idx)
print(f"SCRAPL loss (path {path_idx}): {float(loss)}")

# ============================================================================
# README example 4/5 (Using 𝒫-Adam and 𝒫-SAGA)
# ============================================================================
import equinox as eqx
import optax
from jaxtyping import Array, Float
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
def train_step(
    model: eqx.nn.MLP,
    opt_state: optax.OptState,
    x: Float[Array, "bs n_ch n_samples"],
    x_target: Float[Array, "bs n_ch n_samples"],
    path_idx: int,
) -> tuple[eqx.nn.MLP, optax.OptState, Float[Array, ""]]:
    def loss_fn(m: eqx.nn.MLP) -> Float[Array, ""]:
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

# ============================================================================
# README example 5/5 (Importance Sampling Warmup)
# ============================================================================
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree
from scrapl.jax import SCRAPLLoss, load_warmup_from_dir, warmup_lc_hvp

# Setup
bs = 4
n_ch = 1
n_samples = 8192
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
def enc_fn(
    params: PyTree[Float[Array, "..."]], x: Float[Array, "bs n_ch n_samples"]
) -> Float[Array, "bs n_theta"]:
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
def dec_fn(theta: Float[Array, "bs n_theta"]) -> Float[Array, "bs n_ch n_samples"]:
    return jax.vmap(decoder)(theta)[:, None, :]

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
# We see that at initialization, all path sampling probabilities are uniform
print(f"Uniform path sampling probability: {scrapl_loss.unif_prob:.6f}")
unif_probs = jnp.full((scrapl_loss.n_paths,), scrapl_loss.unif_prob)
print(
    f"[min, max] path sampling probabilities (before warmup): "
    f"[{float(unif_probs.min()):.6f}, {float(unif_probs.max()):.6f}]"
)

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
