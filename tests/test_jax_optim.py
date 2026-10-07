from typing import Any

import numpy as np
import pytest
from tqdm import tqdm

from scrapl._dependencies import require_backend

try:
    tr = require_backend("torch")
    jax = require_backend("jax")
    import equinox as eqx
    import jax.numpy as jnp
    import optax
    import torch.nn as nn
except ModuleNotFoundError as e:
    pytest.skip(str(e), allow_module_level=True)


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
def load_jax_to_pytorch(
    jax_model: eqx.Module, torch_model: nn.Sequential
) -> None:
    torch_model[0].weight.copy_(tr.from_dlpack(jax_model.layers[0].weight))
    torch_model[0].bias.copy_(tr.from_dlpack(jax_model.layers[0].bias))
    torch_model[2].weight.copy_(tr.from_dlpack(jax_model.layers[2].weight))
    torch_model[2].bias.copy_(tr.from_dlpack(jax_model.layers[2].bias))


@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize("n_iters", [10])
def test_p_adam(seed: int, n_iters: int) -> None:
    from scrapl import SCRAPLLoss as TorchSCRAPLLoss
    from scrapl.jax import SCRAPLLoss as JaxSCRAPLLoss
    from scrapl.jax import p_adam

    n_samples = 10000
    n_theta = 4
    bs = 2
    b1, b2, eps, lr = 0.9, 0.999, 1e-8, 0.05

    config = dict(
        shape=n_samples,
        J=6,
        Q1=2,
        Q2=1,
        J_fr=1,
        Q_fr=1,
        use_rho_log1p=True,
    )

    torch_loss = TorchSCRAPLLoss(
        **config,
        use_p_adam=True,
        use_p_saga=False,
        grad_mult=1.0,
        p_adam_b1=b1,
        p_adam_b2=b2,
        p_adam_eps=eps,
    )
    jax_loss = JaxSCRAPLLoss(**config)

    assert jax_loss.scrapl_keys == tuple(torch_loss.scrapl_keys)
    assert jax_loss.n_paths == torch_loss.n_paths

    # Setup random keys and data
    key = jax.random.key(seed)
    enc_key, dec_key, x_key, x_target_key, loop_key = jax.random.split(key, 5)

    x_jax = jax.random.normal(x_key, (bs, 1, n_samples))
    x_target_jax = jax.random.normal(x_target_key, (bs, 1, n_samples))

    x_tr = tr.from_dlpack(x_jax)
    x_target_tr = tr.from_dlpack(x_target_jax)

    encoder_jax = EncoderJAX(n_samples, n_theta, key=enc_key)
    decoder_jax = DecoderJAX(n_theta, n_samples, key=dec_key)

    # Define PyTorch Encoder & Decoder
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

    # Attach encoder parameters to PyTorch SCRAPLLoss and setup optimizer
    torch_params = list(encoder_torch.parameters())
    torch_loss.attach_params(torch_params)
    torch_opt = tr.optim.SGD(torch_params, lr=lr)

    # Setup JAX Optimizer chained with SGD learning rate scaling
    opt = optax.chain(
        p_adam(n_paths=jax_loss.n_paths, b1=b1, b2=b2, eps=eps),
        optax.scale_by_learning_rate(lr),
    )
    opt_state = opt.init(eqx.filter(encoder_jax, eqx.is_inexact_array))

    @eqx.filter_jit
    def evaluate_and_update(
        enc: EncoderJAX, state: optax.OptState, p_idx: int
    ) -> tuple[jax.Array, optax.Updates, EncoderJAX, optax.OptState]:
        def loss_fn(e: EncoderJAX) -> jax.Array:
            th = jax.vmap(e)(x_jax[:, 0, :])
            pred = jax.vmap(decoder_jax)(th)[:, None, :]
            loss = jax_loss(pred, x_target_jax, path_idx=p_idx)
            return loss

        loss_val, grads = eqx.filter_value_and_grad(loss_fn)(enc)
        params = eqx.filter(enc, eqx.is_inexact_array)
        updates, new_state = opt.update(
            grads, state, params=params, path_idx=p_idx
        )
        new_enc = eqx.apply_updates(enc, updates)
        return loss_val, updates, new_enc, new_state

    pbar = tqdm(range(n_iters), desc="Testing P-Adam iterations")

    for step in pbar:
        loop_key, step_key = jax.random.split(loop_key)
        path_idx = int(jax_loss.sample_path(step_key))
        pbar.set_postfix(step=step, path_idx=path_idx)

        # PyTorch forward, backward hook, and optimizer step
        torch_opt.zero_grad()
        theta_tr = encoder_torch(x_tr[:, 0, :])
        pred_tr = decoder_torch(theta_tr).unsqueeze(1)
        loss_tr = torch_loss(pred_tr, x_target_tr, path_idx=path_idx)
        loss_tr.backward()

        torch_norm_grads = [p.grad.clone() for p in torch_params]
        torch_opt.step()

        # JAX step
        loss_jax, updates, encoder_jax, opt_state = evaluate_and_update(
            encoder_jax, opt_state, path_idx
        )
        padam_state = opt_state[0]

        # 1. Verify loss values match
        np.testing.assert_allclose(
            np.asarray(loss_jax),
            loss_tr.detach().numpy(),
            rtol=2e-5,
            atol=1e-7,
        )

        # 2. Verify normalized gradient directions match (scaled by -lr in optax)
        for jax_u, tr_g in zip(
            jax.tree.leaves(updates), torch_norm_grads
        ):
            np.testing.assert_allclose(
                np.asarray(-jax_u / lr),
                tr_g.numpy(),
                rtol=2e-4,
                atol=1e-6,
            )

        # 3. Verify P-Adam moment buffers match
        for i, (m_leaf, v_leaf) in enumerate(
            zip(
                jax.tree.leaves(padam_state.prev_m_s),
                jax.tree.leaves(padam_state.prev_v_s),
            )
        ):
            np.testing.assert_allclose(
                np.asarray(m_leaf),
                torch_loss.p_adam_m[i].numpy(),
                rtol=2e-4,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                np.asarray(v_leaf),
                torch_loss.p_adam_v[i].numpy(),
                rtol=2e-4,
                atol=1e-6,
            )

        # 4. Verify step timestamps match
        np.testing.assert_array_equal(
            np.asarray(padam_state.prev_t_s),
            [torch_loss.p_adam_t[0][i] for i in range(torch_loss.n_paths)],
        )
        assert int(padam_state.scrapl_t) == torch_loss.scrapl_t

        # 5. Verify updated parameter values match
        for jax_p, tr_p in zip(
            jax.tree.leaves(eqx.filter(encoder_jax, eqx.is_inexact_array)),
            torch_params,
        ):
            np.testing.assert_allclose(
                np.asarray(jax_p),
                tr_p.detach().numpy(),
                rtol=2e-4,
                atol=1e-6,
            )
