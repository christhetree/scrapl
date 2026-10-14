"""Toy JAX autoencoder models with matching PyTorch copies, shared by the JAX tests."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from scrapl._dependencies import require_backend

tr = require_backend("torch")
jax = require_backend("jax")
import equinox as eqx
import torch.nn as nn


class EncoderJAX(eqx.Module):
    layers: list[Any]

    def __init__(self, in_size: int, out_size: int, *, key: jax.Array) -> None:
        k1, k2 = jax.random.split(key)
        self.layers = [
            eqx.nn.Linear(in_size, out_size, key=k1),
            jax.nn.relu,
            eqx.nn.Linear(out_size, out_size, key=k2),
            jax.nn.sigmoid,
        ]

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.layers:
            x = layer(x)
        return x


class DecoderJAX(eqx.Module):
    layers: list[Any]

    def __init__(self, in_size: int, out_size: int, *, key: jax.Array) -> None:
        k1, k2 = jax.random.split(key)
        self.layers = [
            eqx.nn.Linear(in_size, in_size, key=k1),
            jax.nn.relu,
            eqx.nn.Linear(in_size, out_size, key=k2),
            jax.nn.tanh,
        ]

    def __call__(self, theta: jax.Array) -> jax.Array:
        for layer in self.layers:
            theta = layer(theta)
        return theta


@tr.no_grad()
def load_jax_to_pytorch(jax_model: eqx.Module, torch_model: nn.Sequential) -> None:
    """Copies weights and biases from an Equinox module to a PyTorch sequential model."""
    torch_model[0].weight.copy_(tr.from_dlpack(jax_model.layers[0].weight))
    torch_model[0].bias.copy_(tr.from_dlpack(jax_model.layers[0].bias))
    torch_model[2].weight.copy_(tr.from_dlpack(jax_model.layers[2].weight))
    torch_model[2].bias.copy_(tr.from_dlpack(jax_model.layers[2].bias))


@dataclass(frozen=True)
class TestSetup:
    __test__ = False
    encoder_jax: EncoderJAX
    decoder_jax: DecoderJAX
    encoder_torch: nn.Sequential
    decoder_torch: nn.Sequential
    x_jax: jax.Array
    x_target_jax: jax.Array
    x_tr: Any
    x_target_tr: Any
    torch_params: list[Any]
    loop_key: jax.Array


def setup_toy_models_and_data(
    seed: int,
    n_samples: int = 10000,
    n_theta: int = 4,
    bs: int = 2,
) -> TestSetup:
    """Initializes synthetic data and matching JAX/PyTorch autoencoder models for testing."""
    key = jax.random.key(seed)
    enc_key, dec_key, x_key, x_target_key, loop_key = jax.random.split(key, 5)

    x_jax = jax.random.normal(x_key, (bs, 1, n_samples))
    x_target_jax = jax.random.normal(x_target_key, (bs, 1, n_samples))

    x_tr = tr.from_dlpack(x_jax)
    x_target_tr = tr.from_dlpack(x_target_jax)

    encoder_jax = EncoderJAX(n_samples, n_theta, key=enc_key)
    decoder_jax = DecoderJAX(n_theta, n_samples, key=dec_key)

    encoder_torch = nn.Sequential(
        nn.Linear(n_samples, n_theta),
        nn.ReLU(),
        nn.Linear(n_theta, n_theta),
        nn.Sigmoid(),
    )
    decoder_torch = nn.Sequential(
        nn.Linear(n_theta, n_theta),
        nn.ReLU(),
        nn.Linear(n_theta, n_samples),
        nn.Tanh(),
    )

    load_jax_to_pytorch(encoder_jax, encoder_torch)
    load_jax_to_pytorch(decoder_jax, decoder_torch)

    # Verify initial outputs match
    theta_hat_jax = jax.vmap(encoder_jax)(x_jax[:, 0, :])
    theta_hat_tr = encoder_torch(x_tr[:, 0, :])
    np.testing.assert_allclose(
        np.asarray(theta_hat_jax),
        theta_hat_tr.detach().numpy(),
        rtol=1e-5,
        atol=1e-6,
    )

    x_hat_jax = jax.vmap(decoder_jax)(theta_hat_jax)[:, None, :]
    x_hat_tr = decoder_torch(theta_hat_tr).unsqueeze(1)
    np.testing.assert_allclose(
        np.asarray(x_hat_jax),
        x_hat_tr.detach().numpy(),
        rtol=1e-5,
        atol=1e-6,
    )

    torch_params = list(encoder_torch.parameters())
    jax_params = jax.tree.leaves(eqx.filter(encoder_jax, eqx.is_inexact_array))
    assert len(jax_params) == len(
        torch_params
    ), f"Parameter count mismatch: JAX ({len(jax_params)}) vs Torch ({len(torch_params)})"
    for jax_p, tr_p in zip(jax_params, torch_params, strict=True):
        assert jax_p.shape == tuple(
            tr_p.shape
        ), f"Parameter shape mismatch: JAX {jax_p.shape} vs Torch {tr_p.shape}"

    return TestSetup(
        encoder_jax=encoder_jax,
        decoder_jax=decoder_jax,
        encoder_torch=encoder_torch,
        decoder_torch=decoder_torch,
        x_jax=x_jax,
        x_target_jax=x_target_jax,
        x_tr=x_tr,
        x_target_tr=x_target_tr,
        torch_params=torch_params,
        loop_key=loop_key,
    )
