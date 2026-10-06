import functools

import equinox as eqx
import jax

from scrapl.jax import SCRAPLLoss

# Initialize SCRAPLLoss with the minimum required arguments
scrapl_loss = SCRAPLLoss(
    shape=48000,  # Length of x and x_target in samples
    J=12,  # Number of octaves (1st and 2nd order temporal filters)
    Q1=8,  # Wavelets per octave (1st order temporal filters)
    Q2=2,  # Wavelets per octave (2nd order temporal filters)
    J_fr=3,  # Number of octaves (2nd order frequential filters)
    Q_fr=2,  # Wavelets per octave (2nd order frequential filters)
)

# Create two random arrays of shape (batch_size, num_channels, signal_length)
key = jax.random.key(42)
key_x, key_target, key_loss = jax.random.split(key, 3)
x = jax.random.normal(key_x, (4, 1, 48000))
x_target = jax.random.normal(key_target, (4, 1, 48000))


@eqx.filter_jit
# @jax.jit
# @functools.partial(jax.jit, static_argnames=("path_idx",))
# def step(x, x_target, key):
def step(x, x_target, path_idx):
    # Compute the SCRAPL loss between x and x_target. Since SCRAPL is stochastic,
    # passing a PRNG key will sample a random scattering path.
    # loss = scrapl_loss(x, x_target, key=key)
    loss = scrapl_loss(x, x_target, path_idx=path_idx)
    return loss


# Compute the SCRAPL loss between x and x_target. Since SCRAPL is stochastic,
# passing a PRNG key will sample a random scattering path.
# loss = scrapl_loss(x, x_target, key=key_loss)

# loss = step(x, x_target, key=key_loss)
path_idx = scrapl_loss.sample_path(key_loss)
print(f"Sampled path index: {path_idx}")
loss = step(x, x_target, path_idx=path_idx)

print(f"SCRAPL loss: {float(loss):.8f}")
