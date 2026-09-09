import pickle
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from scrapl import SCRAPLLoss as TorchLoss
from scrapl.jax import PAdam, PSAGA, PSAGAState, SCRAPLLoss


def assert_tree_close(actual, expected, **kwargs):
    assert jax.tree_util.tree_structure(actual) == jax.tree_util.tree_structure(
        expected
    )
    for a, b in zip(
        jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)
    ):
        np.testing.assert_allclose(a, b, **kwargs)


@pytest.fixture
def params():
    return {
        "bias": jnp.zeros(2),
        "block": (jnp.ones((2, 3)),),
        "gain": jnp.asarray(1.0),
    }


def test_new_and_revisited_paths_use_reference_denominator():
    transform = PSAGA(4)
    state = transform.init(jnp.asarray(0.0))
    update = jax.jit(transform.update)
    for index, gradient, expected in [
        (0, 2.0, 2.0),
        (0, 5.0, 5.0),
        (1, 7.0, 12.0),
        (2, 11.0, 17.0),
        (1, 13.0, 17.5),
        (3, 17.0, 17.0 + 29.0 / 3),
        (0, 19.0, 14.0 + 46.0 / 3),
        (0, 0.0, -19.0 + 60.0 / 3),
    ]:
        direction, state = update(
            jnp.asarray(gradient), state, path_idx=jnp.asarray(index)
        )
        np.testing.assert_allclose(direction, expected, rtol=1e-6)
        assert state.path_grads[index] == gradient
    np.testing.assert_array_equal(state.seen, [True, True, True, True])
    np.testing.assert_array_equal(state.path_grads, [0.0, 13.0, 11.0, 17.0])


def test_single_path_reduces_to_incoming_gradient(params):
    transform = PSAGA(1)
    state = transform.init(params)
    for scale in (1.0, -2.0, 0.0, 3.0):
        gradient = jax.tree_util.tree_map(lambda p: p * scale, params)
        direction, state = jax.jit(transform.update)(
            gradient, state, path_idx=jnp.asarray(0)
        )
        assert_tree_close(direction, gradient, rtol=0, atol=0)
        assert state.seen[0]


@pytest.mark.parametrize("use_p_adam", [False, True])
@pytest.mark.parametrize("grad_mult", [1.0, 1e8])
def test_updates_and_history_match_pytorch_hooks(params, use_p_adam, grad_mult):
    reference = TorchLoss(
        shape=128,
        J=3,
        Q1=2,
        Q2=1,
        J_fr=1,
        Q_fr=1,
        use_rho_log1p=True,
        use_p_adam=use_p_adam,
        use_p_saga=True,
        grad_mult=grad_mult,
    )
    reference.attach_params(
        [
            torch.nn.Parameter(torch.tensor(np.asarray(p)))
            for p in jax.tree_util.tree_leaves(params)
        ]
    )
    saga = PSAGA(reference.n_paths)
    adam = PAdam(reference.n_paths)
    state, adam_state = saga.init(params), adam.init(params)

    @jax.jit
    def update(grads, state, adam_state, path_idx):
        incoming = jax.tree_util.tree_map(lambda g: g * grad_mult, grads)
        if use_p_adam:
            incoming, adam_state = adam.update(incoming, adam_state, path_idx=path_idx)
        direction, state = saga.update(incoming, state, path_idx=path_idx)
        return direction, state, adam_state

    rng = np.random.default_rng(42)
    schedule = [0, 0, 1, 4, 1, 2, 4, 3, 2, 0]
    for count, path_idx in enumerate(schedule):
        grads = jax.tree_util.tree_map(
            lambda p: jnp.asarray(
                rng.normal(size=p.shape) if count != 8 else np.zeros(p.shape),
                dtype=p.dtype,
            ),
            params,
        )
        before = jax.device_get(state)
        directions, state, adam_state = update(
            grads, state, adam_state, jnp.asarray(path_idx)
        )
        reference.scrapl_t = count + 1
        reference.curr_path_idx = path_idx
        reference.path_counts[path_idx] += 1
        for i, (g, direction, history) in enumerate(
            zip(
                jax.tree_util.tree_leaves(grads),
                jax.tree_util.tree_leaves(directions),
                jax.tree_util.tree_leaves(state.path_grads),
            )
        ):
            expected = reference.grad_hook(torch.tensor(np.asarray(g)), param_idx=i)
            np.testing.assert_allclose(
                direction, expected.numpy(), rtol=1e-5, atol=1e-6
            )
            np.testing.assert_allclose(
                history, reference.prev_path_grads[i].numpy(), rtol=3e-6, atol=1e-6
            )
            inactive = [p for p in range(reference.n_paths) if p != path_idx]
            np.testing.assert_array_equal(
                np.asarray(history)[inactive],
                jax.tree_util.tree_leaves(before.path_grads)[i][inactive],
            )
        np.testing.assert_array_equal(
            state.seen, [i in reference.path_counts for i in range(reference.n_paths)]
        )


def test_checkpoint_continuation_and_input_immutability(params):
    saga, adam = PSAGA(5), PAdam(5)
    initial = (saga.init(params), adam.init(params))
    snapshot = jax.device_get(initial)

    @jax.jit
    def update(grads, states, path_idx):
        saga_state, adam_state = states
        incoming, adam_state = adam.update(grads, adam_state, path_idx=path_idx)
        direction, saga_state = saga.update(incoming, saga_state, path_idx=path_idx)
        return direction, (saga_state, adam_state)

    _, states = update(params, initial, jnp.asarray(1))
    _, states = update(params, states, jnp.asarray(3))
    assert_tree_close(initial, snapshot, rtol=0, atol=0)
    restored = pickle.loads(pickle.dumps(jax.device_get(states)))
    assert isinstance(restored[0], PSAGAState)
    uninterrupted = update(params, states, jnp.asarray(1))
    resumed = update(params, restored, jnp.asarray(1))
    assert_tree_close(resumed, uninterrupted, rtol=0, atol=0)
    direction, state = saga.update(params, restored[0], path_idx=2)
    expected = jax.jit(saga.update)(params, states[0], path_idx=jnp.asarray(2))
    assert_tree_close((direction, state), expected, rtol=2e-6, atol=1e-7)


@pytest.mark.parametrize("use_p_adam", [False, True])
def test_real_loss_training_matches_pytorch_forward_backward_steps(use_p_adam):
    config = dict(shape=128, J=3, Q1=2, Q2=1, J_fr=1, Q_fr=1, use_rho_log1p=True)
    loss = SCRAPLLoss(**config)
    reference = TorchLoss(**config, use_p_adam=use_p_adam, use_p_saga=True, grad_mult=1)
    target = jax.random.normal(jax.random.key(8), (1, 1, 128))
    torch_target = torch.tensor(np.asarray(target))
    theta = jnp.asarray(-1.0)
    torch_theta = torch.nn.Parameter(torch.tensor(-1.0))
    reference.attach_params([torch_theta])
    saga, adam = PSAGA(loss.n_paths), PAdam(loss.n_paths)
    saga_state, adam_state = saga.init(theta), adam.init(theta)
    optimiser = torch.optim.SGD([torch_theta], lr=0.01)

    @jax.jit
    def step(theta, saga_state, adam_state, path_idx):
        value, gradient = jax.value_and_grad(
            lambda theta: loss(
                jax.nn.sigmoid(theta) * target, target, path_idx=path_idx
            )
        )(theta)
        if use_p_adam:
            gradient, adam_state = adam.update(gradient, adam_state, path_idx=path_idx)
        direction, saga_state = saga.update(gradient, saga_state, path_idx=path_idx)
        return theta - 0.01 * direction, saga_state, adam_state, value, direction

    for path_idx in [0, 0, 1, 4, 2, 1, 4, 3, 0]:
        theta, saga_state, adam_state, value, direction = step(
            theta, saga_state, adam_state, jnp.asarray(path_idx)
        )
        optimiser.zero_grad()
        expected = reference(
            torch.sigmoid(torch_theta) * torch_target, torch_target, path_idx=path_idx
        )
        expected.backward()
        np.testing.assert_allclose(value, expected.detach().numpy(), rtol=3e-5)
        np.testing.assert_allclose(direction, torch_theta.grad.numpy(), rtol=3e-5)
        optimiser.step()
        np.testing.assert_allclose(
            theta, torch_theta.detach().numpy(), rtol=3e-5, atol=1e-6
        )
    assert theta > -1
    np.testing.assert_allclose(
        saga_state.path_grads, reference.prev_path_grads[0].numpy(), rtol=3e-5
    )
    if use_p_adam:
        np.testing.assert_allclose(
            adam_state.m, reference.p_adam_m[0].numpy(), rtol=3e-5
        )
        np.testing.assert_allclose(
            adam_state.v, reference.p_adam_v[0].numpy(), rtol=3e-5
        )


@pytest.mark.parametrize("use_p_adam", [False, True])
def test_jitted_weighted_sampling_scan_reduces_full_path_mean(use_p_adam):
    loss = SCRAPLLoss(shape=128, J=3, Q1=2, Q2=1, J_fr=1, Q_fr=1, use_rho_log1p=True)
    target = jax.random.normal(jax.random.key(8), (1, 1, 128))
    saga, adam = PSAGA(loss.n_paths), PAdam(loss.n_paths)
    initial_theta = jnp.asarray(-1.0)
    initial = (saga.init(initial_theta), adam.init(initial_theta))
    probs = jnp.asarray([0.4, 0.1, 0.2, 0.15, 0.15])

    @jax.jit
    def train(theta, states, key, probs):
        def step(carry, _):
            theta, (saga_state, adam_state), key = carry
            key, path_key = jax.random.split(key)
            path_idx = loss.sample_path(path_key, probs=probs)
            value, gradient = jax.value_and_grad(
                lambda theta: loss(
                    jax.nn.sigmoid(theta) * target, target, path_idx=path_idx
                )
            )(theta)
            if use_p_adam:
                gradient, adam_state = adam.update(
                    gradient, adam_state, path_idx=path_idx
                )
            direction, saga_state = saga.update(gradient, saga_state, path_idx=path_idx)
            return (theta - 0.1 * direction, (saga_state, adam_state), key), value

        return jax.lax.scan(step, (theta, states, key), None, length=40)

    def full_mean(theta):
        return jnp.stack(
            [
                loss(jax.nn.sigmoid(theta) * target, target, path_idx=i)
                for i in range(loss.n_paths)
            ]
        ).mean()

    (theta, states, _), values = train(initial_theta, initial, jax.random.key(4), probs)
    assert np.isfinite(values).all()
    assert states[0].seen.all()
    assert full_mean(theta) < 0.5 * full_mean(initial_theta)
    before = jax.device_get(states)
    full_mean(theta)
    assert_tree_close(states, before, rtol=0, atol=0)


def test_invalid_updates_leave_all_history_unchanged(params):
    saga = PSAGA(5)
    _, state = saga.update(params, saga.init(params), path_idx=1)
    update = jax.jit(saga.update)
    for path_idx in (-1, 5):
        with pytest.raises(ValueError, match="path_idx"):
            saga.update(params, state, path_idx=path_idx)
    for path_idx in (
        jnp.asarray(-1),
        jnp.asarray(5),
        jnp.asarray(2**32 - 1, dtype=jnp.uint32),
    ):
        directions, returned = update(params, state, path_idx=path_idx)
        assert all(
            np.isnan(leaf).all() for leaf in jax.tree_util.tree_leaves(directions)
        )
        assert_tree_close(returned, state, rtol=0, atol=0)
    for path_idx in (True, 1.5, jnp.asarray([0])):
        with pytest.raises(TypeError, match="scalar integer"):
            saga.update(params, state, path_idx=path_idx)
    for value in (float("nan"), float("inf")):
        grads = {**params, "bias": jnp.full((2,), value)}
        directions, returned = update(grads, state, path_idx=jnp.asarray(0))
        assert all(
            np.isnan(leaf).all() for leaf in jax.tree_util.tree_leaves(directions)
        )
        assert_tree_close(returned, state, rtol=0, atol=0)


def test_history_sum_overflow_is_rejected():
    saga = PSAGA(5)
    state = saga.init(jnp.asarray(0.0))._replace(
        seen=jnp.ones(5, dtype=jnp.bool_),
        path_grads=jnp.full(5, 2e38),
    )
    direction, returned = jax.jit(saga.update)(
        jnp.asarray(1.0), state, path_idx=jnp.asarray(1)
    )
    assert np.isnan(direction)
    assert_tree_close(returned, state, rtol=0, atol=0)


@pytest.mark.parametrize("n_paths", [0, -1, 2.5, True])
def test_invalid_configuration(n_paths):
    with pytest.raises(ValueError, match="positive integer"):
        PSAGA(n_paths)


def test_parameter_and_state_validation(params):
    saga = PSAGA(5)
    for empty in ({}, jnp.zeros(0)):
        with pytest.raises(ValueError, match="contain"):
            saga.init(empty)
    for wrong_dtype in (jnp.int32, jnp.complex64, jnp.float16):
        with pytest.raises(TypeError, match="float32 or float64"):
            saga.init(jnp.ones(2, dtype=wrong_dtype))
    state = saga.init(params)
    with pytest.raises(TypeError, match="PSAGAState"):
        saga.update(params, tuple(state), path_idx=0)
    with pytest.raises(ValueError, match="structures"):
        saga.update(jnp.ones(2), state, path_idx=0)
    with pytest.raises(ValueError, match="shapes"):
        saga.update({**params, "bias": jnp.ones(3)}, state, path_idx=0)
    with pytest.raises(ValueError, match="shapes"):
        replace(saga, n_paths=4).update(params, state, path_idx=0)
    with pytest.raises(TypeError, match="dtypes"):
        saga.update(
            params,
            state._replace(
                path_grads=jax.tree_util.tree_map(
                    lambda g: g.astype(jnp.int32), state.path_grads
                )
            ),
            path_idx=0,
        )
    with pytest.raises(ValueError, match="seen mask"):
        saga.update(
            params, state._replace(seen=jnp.zeros(4, dtype=jnp.bool_)), path_idx=0
        )
    with pytest.raises(TypeError, match="seen mask"):
        saga.update(
            params, state._replace(seen=jnp.zeros(5, dtype=jnp.int32)), path_idx=0
        )
