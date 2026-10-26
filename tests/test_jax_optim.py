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
        PAdamState,
        PSAGAState,
        SCRAPLLoss as JaxSCRAPLLoss,
        scrapl_optimizer,
    )
    from tests._toy_models import (
        DecoderJAX,
        EncoderJAX,
        TestSetup,
        setup_toy_models_and_data,
    )
except ModuleNotFoundError as e:
    pytest.skip(str(e), allow_module_level=True)


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
    weight_decay: float
    grad_mult: float


def build_evaluate_and_update_fn(
    optimizer: optax.GradientTransformationExtraArgs | optax.GradientTransformation,
    jax_loss: JaxSCRAPLLoss,
    decoder_jax: DecoderJAX,
    x_jax: jax.Array,
    x_target_jax: jax.Array,
) -> Callable[..., tuple[jax.Array, optax.Updates, EncoderJAX, optax.OptState]]:
    """
    Compiles a JIT-filtered step function computing loss, gradients, and model
    parameter updates.
    """

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
    *,
    use_p_adam: bool = False,
    use_p_saga: bool = False,
    p_adam_b1: float = 0.9,
    p_adam_b2: float = 0.999,
    p_adam_eps: float = 1e-8,
    grad_mult: float = 1e8,
    lr: float = 0.05,
    weight_decay: float = 0.01,
    n_samples: int = 10000,
    n_theta: int = 4,
    bs: int = 2,
) -> OptimExperiment:
    """
    Sets up corresponding PyTorch and JAX loss functions, models, and optimizer states.
    """
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
    torch_opt = tr.optim.SGD(setup.torch_params, lr=lr, weight_decay=weight_decay)

    optimizer = scrapl_optimizer(
        learning_rate=lr,
        weight_decay=weight_decay,
        n_paths=jax_loss.n_paths,
        grad_mult=grad_mult,
        use_p_adam=use_p_adam,
        use_p_saga=use_p_saga,
        p_adam_b1=p_adam_b1,
        p_adam_b2=p_adam_b2,
        p_adam_eps=p_adam_eps,
    )
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
        weight_decay=weight_decay,
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
) -> tuple[Any, list[Any], list[Any]]:
    """
    Runs forward/backward passes and SGD optimization step in PyTorch for a given path.
    """
    torch_opt.zero_grad()
    theta_tr = encoder_torch(x_tr[:, 0, :])
    pred_tr = decoder_torch(theta_tr).unsqueeze(1)
    loss_tr = torch_loss(pred_tr, x_target_tr, path_idx=path_idx)
    # We call the loss again before the backward step to ensure that its state is only
    # updated once per optimization step, as it is in the JAX implementation.
    loss_tr_2 = torch_loss(pred_tr, x_target_tr, path_idx=0)
    loss_tr.backward()

    torch_norm_grads = [p.grad.clone() for p in torch_params]
    torch_params_before_step = [p.clone() for p in torch_params]
    torch_opt.step()
    return loss_tr, torch_norm_grads, torch_params_before_step


def assert_step_matches(
    loss_jax: jax.Array,
    loss_tr: Any,
    updates: optax.Updates,
    torch_norm_grads: list[Any],
    torch_params_before_step: list[Any],
    encoder_jax: EncoderJAX,
    torch_params: list[Any],
    lr: float = 0.05,
    weight_decay: float = 0.01,
    loss_rtol: float = 1e-6,
    loss_atol: float = 1e-7,
    grad_rtol: float = 1e-5,
    grad_atol: float = 1e-6,
    param_rtol: float = 1e-5,
    param_atol: float = 1e-6,
) -> None:
    """
    Verifies numerical equivalence of loss values, parameter updates, and model weights.
    """
    # 1. Verify loss values match
    np.testing.assert_allclose(
        np.asarray(loss_jax),
        loss_tr.detach().numpy(),
        rtol=loss_rtol,
        atol=loss_atol,
    )

    # 2. Verify parameter updates match (-lr * (torch_norm_grads + weight_decay * params))
    for jax_u, tr_g, tr_p_before in zip(
        jax.tree.leaves(updates),
        torch_norm_grads,
        torch_params_before_step,
        strict=True,
    ):
        expected_update = -lr * (
            tr_g.numpy() + weight_decay * tr_p_before.detach().numpy()
        )
        np.testing.assert_allclose(
            np.asarray(jax_u),
            expected_update,
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
    """
    Coordinates a single optimization step and asserts equivalence across JAX and
    PyTorch.
    """
    loop_key, step_key = jax.random.split(loop_key)
    path_idx = int(exp.jax_loss.sample_path(step_key))

    loss_tr, torch_norm_grads, torch_params_before = run_torch_step(
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
        torch_params_before_step=torch_params_before,
        encoder_jax=encoder_jax,
        torch_params=exp.setup.torch_params,
        lr=exp.lr,
        weight_decay=exp.weight_decay,
    )

    return encoder_jax, opt_state, loop_key, path_idx


def find_state(opt_state: optax.OptState, state_type: type) -> Any:
    """Returns the state of the transform with the given state type in an Optax chain."""
    return next(s for s in opt_state if isinstance(s, state_type))


def assert_p_adam_state_matches(state: PAdamState, torch_loss: TorchSCRAPLLoss) -> None:
    """Verifies P-Adam moment buffers and step timestamps match the PyTorch hooks."""
    for i, (m_leaf, v_leaf) in enumerate(
        zip(
            jax.tree.leaves(state.prev_m_s),
            jax.tree.leaves(state.prev_v_s),
            strict=True,
        )
    ):
        np.testing.assert_allclose(
            np.asarray(m_leaf), torch_loss.p_adam_m[i].numpy(), rtol=1e-4, atol=1e-5
        )
        np.testing.assert_allclose(
            np.asarray(v_leaf),
            torch_loss.p_adam_v[i].numpy(),
            rtol=1e-3,  # v accumulates squared grads, roughly doubling float32 error
            atol=1e-5,
        )
    np.testing.assert_array_equal(
        np.asarray(state.prev_t_s),
        [torch_loss.p_adam_t[0].get(i, 0) for i in range(torch_loss.n_paths)],
    )
    assert int(state.scrapl_t) == torch_loss.scrapl_t


def assert_p_saga_state_matches(state: PSAGAState, torch_loss: TorchSCRAPLLoss) -> None:
    """Verifies P-SAGA gradient history buffers and path visit counts match PyTorch."""
    for i, g_leaf in enumerate(jax.tree.leaves(state.prev_path_grads)):
        np.testing.assert_allclose(
            np.asarray(g_leaf),
            torch_loss.prev_path_grads[i].numpy(),
            rtol=1e-5,  # Stored grads are scaled by grad_mult, see assert_step_matches
            atol=1e-6,
        )
    np.testing.assert_array_equal(
        np.asarray(state.path_counts),
        [torch_loss.path_counts.get(i, 0) for i in range(torch_loss.n_paths)],
    )


def array_leaves(tree: Any) -> list[jax.Array]:
    """Returns the array leaves of a pytree, skipping static leaves like activations."""
    return jax.tree.leaves(eqx.filter(tree, eqx.is_array))


# Every optimizer configuration also uses the gradient multiplier and weight decay
optimizer_configs = pytest.mark.parametrize(
    "use_p_adam, use_p_saga",
    [(False, False), (True, False), (False, True), (True, True)],
    ids=["grad_mult", "p_adam", "p_saga", "p_adam_p_saga"],
)


@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize("n_iters", [10])
@optimizer_configs
def test_optimizer_matches_pytorch(
    seed: int, n_iters: int, use_p_adam: bool, use_p_saga: bool
) -> None:
    exp = setup_optim_experiment(seed, use_p_adam=use_p_adam, use_p_saga=use_p_saga)
    encoder_jax = exp.setup.encoder_jax
    opt_state = exp.opt_state
    loop_key = exp.setup.loop_key

    pbar = tqdm(
        range(n_iters),
        desc=f"Testing optimizer (use_p_adam={use_p_adam}, use_p_saga={use_p_saga})",
    )
    for step in pbar:
        encoder_jax, opt_state, loop_key, path_idx = run_optim_step(
            exp, encoder_jax, opt_state, loop_key
        )
        pbar.set_postfix(step=step, path_idx=path_idx)
        if use_p_adam:
            assert_p_adam_state_matches(
                find_state(opt_state, PAdamState), exp.torch_loss
            )
        if use_p_saga:
            assert_p_saga_state_matches(
                find_state(opt_state, PSAGAState), exp.torch_loss
            )


@optimizer_configs
def test_resuming_saved_optimizer_state_matches_uninterrupted_training(
    tmp_path, use_p_adam: bool, use_p_saga: bool
) -> None:
    n_iters, save_at = 6, 3
    exp = setup_optim_experiment(42, use_p_adam=use_p_adam, use_p_saga=use_p_saga)
    step_keys = jax.random.split(exp.setup.loop_key, n_iters)
    path_indices = [int(exp.jax_loss.sample_path(k)) for k in step_keys]

    def train(
        encoder: EncoderJAX, opt_state: optax.OptState, steps: list[int]
    ) -> tuple[EncoderJAX, optax.OptState]:
        for path_idx in steps:
            _, _, encoder, opt_state = exp.evaluate_and_update(
                encoder, opt_state, path_idx
            )
        return encoder, opt_state

    uninterrupted = train(exp.setup.encoder_jax, exp.opt_state, path_indices)

    # 1. Train part of the way, then save the model and optimizer state to disk
    checkpoint = train(exp.setup.encoder_jax, exp.opt_state, path_indices[:save_at])
    checkpoint_path = tmp_path / "checkpoint.eqx"
    eqx.tree_serialise_leaves(checkpoint_path, checkpoint)

    # 2. Load it into a freshly initialized model and optimizer state, as a new
    # process would, and verify nothing was lost in the round trip
    template = (exp.setup.encoder_jax, exp.opt_state)
    restored = eqx.tree_deserialise_leaves(checkpoint_path, template)
    for restored_leaf, saved_leaf in zip(
        array_leaves(restored), array_leaves(checkpoint), strict=True
    ):
        np.testing.assert_array_equal(np.asarray(restored_leaf), np.asarray(saved_leaf))

    # 3. Resumed training matches uninterrupted training
    resumed = train(*restored, path_indices[save_at:])
    for resumed_leaf, expected_leaf in zip(
        array_leaves(resumed), array_leaves(uninterrupted), strict=True
    ):
        expected = np.asarray(expected_leaf)
        np.testing.assert_allclose(
            np.asarray(resumed_leaf),
            expected,
            rtol=1e-6,
            atol=1e-7,
        )
