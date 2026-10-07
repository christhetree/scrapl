from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pytest
from tqdm import tqdm

from scrapl._dependencies import require_backend

try:
    tr = require_backend("torch")
    jax = require_backend("jax")
    import equinox as eqx
    import optax
    import torch.nn as nn
    from scrapl import SCRAPLLoss as TorchSCRAPLLoss
    from scrapl.jax import (
        SCRAPLLoss as JaxSCRAPLLoss,
        p_adam,
        p_saga,
        scale_by_gradient_multiplier,
    )
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


@dataclass(frozen=True)
class OptimExperiment:
    __test__ = False
    setup: TestSetup
    torch_loss: TorchSCRAPLLoss
    jax_loss: JaxSCRAPLLoss
    torch_opt: tr.optim.SGD
    optimizer: optax.GradientTransformationExtraArgs | optax.GradientTransformation
    evaluate_and_update: Callable[
        ..., tuple[jax.Array, optax.Updates, EncoderJAX, optax.OptState]
    ]
    opt_state: optax.OptState
    lr: float
    grad_mult: float


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


def build_evaluate_and_update_fn(
    optimizer: optax.GradientTransformationExtraArgs | optax.GradientTransformation,
    jax_loss: JaxSCRAPLLoss,
    decoder_jax: DecoderJAX,
    x_jax: jax.Array,
    x_target_jax: jax.Array,
) -> Callable[..., tuple[jax.Array, optax.Updates, EncoderJAX, optax.OptState]]:
    """Compiles a JIT-filtered step function computing loss, gradients, and model parameter updates."""

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
        updates, new_state = optimizer.update(
            grads, state, params=params, path_idx=p_idx
        )
        new_enc = eqx.apply_updates(enc, updates)
        return loss_val, updates, new_enc, new_state

    return evaluate_and_update


def setup_optim_experiment(
    seed: int,
    optimizer_fn: Callable[
        [int],
        optax.GradientTransformationExtraArgs | optax.GradientTransformation,
    ],
    *,
    use_p_adam: bool = False,
    use_p_saga: bool = False,
    p_adam_b1: float = 0.9,
    p_adam_b2: float = 0.999,
    p_adam_eps: float = 1e-8,
    grad_mult: float = 1e8,
    lr: float = 0.05,
    n_samples: int = 10000,
    n_theta: int = 4,
    bs: int = 2,
) -> OptimExperiment:
    """Sets up corresponding PyTorch and JAX loss functions, models, and optimizer states."""
    config = dict(
        shape=n_samples,
        J=6,
        Q1=2,
        Q2=1,
        J_fr=1,
        Q_fr=1,
        use_rho_log1p=True,
    )
    setup = setup_toy_models_and_data(
        seed=seed, n_samples=n_samples, n_theta=n_theta, bs=bs
    )
    torch_loss = TorchSCRAPLLoss(
        **config,
        use_p_adam=use_p_adam,
        use_p_saga=use_p_saga,
        grad_mult=grad_mult,
        p_adam_b1=p_adam_b1,
        p_adam_b2=p_adam_b2,
        p_adam_eps=p_adam_eps,
    )
    jax_loss = JaxSCRAPLLoss(**config)

    assert jax_loss.scrapl_keys == tuple(torch_loss.scrapl_keys)
    assert jax_loss.n_paths == torch_loss.n_paths

    torch_loss.attach_params(setup.torch_params)
    torch_opt = tr.optim.SGD(setup.torch_params, lr=lr)

    optimizer = optimizer_fn(jax_loss.n_paths)
    opt_state = optimizer.init(eqx.filter(setup.encoder_jax, eqx.is_inexact_array))
    evaluate_and_update = build_evaluate_and_update_fn(
        optimizer=optimizer,
        jax_loss=jax_loss,
        decoder_jax=setup.decoder_jax,
        x_jax=setup.x_jax,
        x_target_jax=setup.x_target_jax,
    )

    return OptimExperiment(
        setup=setup,
        torch_loss=torch_loss,
        jax_loss=jax_loss,
        torch_opt=torch_opt,
        optimizer=optimizer,
        evaluate_and_update=evaluate_and_update,
        opt_state=opt_state,
        lr=lr,
        grad_mult=grad_mult,
    )


def run_torch_step(
    encoder_torch: nn.Sequential,
    decoder_torch: nn.Sequential,
    torch_loss: TorchSCRAPLLoss,
    torch_opt: tr.optim.SGD,
    torch_params: list[Any],
    x_tr: Any,
    x_target_tr: Any,
    path_idx: int,
) -> tuple[Any, list[Any]]:
    """Runs forward/backward passes and SGD optimization step in PyTorch for a given path."""
    torch_opt.zero_grad()
    theta_tr = encoder_torch(x_tr[:, 0, :])
    pred_tr = decoder_torch(theta_tr).unsqueeze(1)
    loss_tr = torch_loss(pred_tr, x_target_tr, path_idx=path_idx)
    loss_tr.backward()

    torch_norm_grads = [p.grad.clone() for p in torch_params]
    torch_opt.step()
    return loss_tr, torch_norm_grads


def assert_step_matches(
    loss_jax: jax.Array,
    loss_tr: Any,
    updates: optax.Updates,
    torch_norm_grads: list[Any],
    encoder_jax: EncoderJAX,
    torch_params: list[Any],
    lr: float = 0.05,
    loss_rtol: float = 2e-5,
    loss_atol: float = 1e-7,
    grad_rtol: float = 2e-4,
    grad_atol: float = 1e-6,
    param_rtol: float = 2e-4,
    param_atol: float = 1e-6,
) -> None:
    """Verifies numerical equivalence of loss values, parameter updates, and model weights."""
    # 1. Verify loss values match
    np.testing.assert_allclose(
        np.asarray(loss_jax),
        loss_tr.detach().numpy(),
        rtol=loss_rtol,
        atol=loss_atol,
    )

    # 2. Verify parameter updates match (-lr * torch_norm_grads)
    for jax_u, tr_g in zip(jax.tree.leaves(updates), torch_norm_grads, strict=True):
        np.testing.assert_allclose(
            np.asarray(jax_u),
            -lr * tr_g.numpy(),
            rtol=grad_rtol,
            atol=grad_atol,
        )

    # 3. Verify updated parameter values match
    for jax_p, tr_p in zip(
        jax.tree.leaves(eqx.filter(encoder_jax, eqx.is_inexact_array)),
        torch_params,
        strict=True,
    ):
        np.testing.assert_allclose(
            np.asarray(jax_p),
            tr_p.detach().numpy(),
            rtol=param_rtol,
            atol=param_atol,
        )


def run_optim_step(
    exp: OptimExperiment,
    encoder_jax: EncoderJAX,
    opt_state: optax.OptState,
    loop_key: jax.Array,
) -> tuple[EncoderJAX, optax.OptState, jax.Array, int]:
    """Coordinates a single optimization step and asserts equivalence across JAX and PyTorch."""
    loop_key, step_key = jax.random.split(loop_key)
    path_idx = int(exp.jax_loss.sample_path(step_key))

    loss_tr, torch_norm_grads = run_torch_step(
        exp.setup.encoder_torch,
        exp.setup.decoder_torch,
        exp.torch_loss,
        exp.torch_opt,
        exp.setup.torch_params,
        exp.setup.x_tr,
        exp.setup.x_target_tr,
        path_idx=path_idx,
    )

    loss_jax, updates, encoder_jax, opt_state = exp.evaluate_and_update(
        encoder_jax, opt_state, path_idx
    )

    assert_step_matches(
        loss_jax=loss_jax,
        loss_tr=loss_tr,
        updates=updates,
        torch_norm_grads=torch_norm_grads,
        encoder_jax=encoder_jax,
        torch_params=exp.setup.torch_params,
        lr=exp.lr,
    )

    return encoder_jax, opt_state, loop_key, path_idx


@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize("n_iters", [3])
def test_grad_multiplier(seed: int, n_iters: int) -> None:
    grad_mult = 1e8
    lr = 0.05
    exp = setup_optim_experiment(
        seed=seed,
        optimizer_fn=lambda _: optax.chain(
            scale_by_gradient_multiplier(grad_mult=grad_mult),
            optax.scale_by_learning_rate(lr),
        ),
        use_p_adam=False,
        use_p_saga=False,
        grad_mult=grad_mult,
        lr=lr,
    )
    encoder_jax = exp.setup.encoder_jax
    opt_state = exp.opt_state
    loop_key = exp.setup.loop_key

    pbar = tqdm(range(n_iters), desc="Testing Gradient Multiplier iterations")

    for step in pbar:
        encoder_jax, opt_state, loop_key, path_idx = run_optim_step(
            exp, encoder_jax, opt_state, loop_key
        )
        pbar.set_postfix(step=step, path_idx=path_idx)


@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize("n_iters", [10])
def test_p_adam(seed: int, n_iters: int) -> None:
    grad_mult = 1e8
    b1, b2, eps, lr = 0.9, 0.999, 1e-8, 0.05

    exp = setup_optim_experiment(
        seed=seed,
        optimizer_fn=lambda n_paths: optax.chain(
            scale_by_gradient_multiplier(grad_mult=grad_mult),
            p_adam(n_paths=n_paths, b1=b1, b2=b2, eps=eps),
            optax.scale_by_learning_rate(lr),
        ),
        use_p_adam=True,
        use_p_saga=False,
        grad_mult=grad_mult,
        p_adam_b1=b1,
        p_adam_b2=b2,
        p_adam_eps=eps,
        lr=lr,
    )
    encoder_jax = exp.setup.encoder_jax
    opt_state = exp.opt_state
    loop_key = exp.setup.loop_key

    pbar = tqdm(range(n_iters), desc="Testing P-Adam iterations")

    for step in pbar:
        encoder_jax, opt_state, loop_key, path_idx = run_optim_step(
            exp, encoder_jax, opt_state, loop_key
        )
        pbar.set_postfix(step=step, path_idx=path_idx)
        padam_state = opt_state[1]

        # 3. Verify P-Adam moment buffers match
        for i, (m_leaf, v_leaf) in enumerate(
            zip(
                jax.tree.leaves(padam_state.prev_m_s),
                jax.tree.leaves(padam_state.prev_v_s),
                strict=True,
            )
        ):
            np.testing.assert_allclose(
                np.asarray(m_leaf),
                exp.torch_loss.p_adam_m[i].numpy(),
                rtol=2e-4,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                np.asarray(v_leaf),
                exp.torch_loss.p_adam_v[i].numpy(),
                rtol=3e-4,
                atol=1e-6,
            )

        # 4. Verify step timestamps match
        np.testing.assert_array_equal(
            np.asarray(padam_state.prev_t_s),
            [
                exp.torch_loss.p_adam_t[0].get(i, 0)
                for i in range(exp.torch_loss.n_paths)
            ],
        )
        assert int(padam_state.scrapl_t) == exp.torch_loss.scrapl_t


@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize("n_iters", [10])
def test_p_saga(seed: int, n_iters: int) -> None:
    grad_mult = 1e8
    lr = 0.05

    exp = setup_optim_experiment(
        seed=seed,
        optimizer_fn=lambda n_paths: optax.chain(
            scale_by_gradient_multiplier(grad_mult=grad_mult),
            p_saga(n_paths=n_paths),
            optax.scale_by_learning_rate(lr),
        ),
        use_p_adam=False,
        use_p_saga=True,
        grad_mult=grad_mult,
        lr=lr,
    )
    encoder_jax = exp.setup.encoder_jax
    opt_state = exp.opt_state
    loop_key = exp.setup.loop_key

    pbar = tqdm(range(n_iters), desc="Testing P-SAGA iterations")

    for step in pbar:
        encoder_jax, opt_state, loop_key, path_idx = run_optim_step(
            exp, encoder_jax, opt_state, loop_key
        )
        pbar.set_postfix(step=step, path_idx=path_idx)
        psaga_state = opt_state[1]

        # 3. Verify P-SAGA gradient history buffers match
        for i, g_leaf in enumerate(jax.tree.leaves(psaga_state.prev_path_grads)):
            np.testing.assert_allclose(
                np.asarray(g_leaf),
                exp.torch_loss.prev_path_grads[i].numpy(),
                rtol=2e-4,
                atol=1e-6,
            )

        # 4. Verify path visit counts match
        np.testing.assert_array_equal(
            np.asarray(psaga_state.path_counts),
            [
                exp.torch_loss.path_counts.get(i, 0)
                for i in range(exp.torch_loss.n_paths)
            ],
        )


@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize("n_iters", [10])
def test_grad_multiplier_p_adam_p_saga(seed: int, n_iters: int) -> None:
    grad_mult = 1e8
    b1, b2, eps, lr = 0.9, 0.999, 1e-8, 0.05

    exp = setup_optim_experiment(
        seed=seed,
        optimizer_fn=lambda n_paths: optax.chain(
            scale_by_gradient_multiplier(grad_mult=grad_mult),
            p_adam(n_paths=n_paths, b1=b1, b2=b2, eps=eps),
            p_saga(n_paths=n_paths),
            optax.scale_by_learning_rate(lr),
        ),
        use_p_adam=True,
        use_p_saga=True,
        grad_mult=grad_mult,
        p_adam_b1=b1,
        p_adam_b2=b2,
        p_adam_eps=eps,
        lr=lr,
    )
    encoder_jax = exp.setup.encoder_jax
    opt_state = exp.opt_state
    loop_key = exp.setup.loop_key

    pbar = tqdm(
        range(n_iters),
        desc="Testing Gradient Multiplier + P-Adam + P-SAGA iterations",
    )

    for step in pbar:
        encoder_jax, opt_state, loop_key, path_idx = run_optim_step(
            exp, encoder_jax, opt_state, loop_key
        )
        pbar.set_postfix(step=step, path_idx=path_idx)
