import pickle
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from scrapl import SCRAPLLoss as TorchLoss
from scrapl.jax import PAdam, SCRAPLLoss


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


@pytest.mark.parametrize("grad_mult", [1.0, 1e8])
@pytest.mark.parametrize("betas", [(0.9, 0.999), (0.0, 0.0), (0.0, 0.8)])
def test_updates_and_state_match_pytorch_hooks(params, grad_mult, betas):
    reference = TorchLoss(
        shape=128,
        J=3,
        Q1=2,
        Q2=1,
        J_fr=1,
        Q_fr=1,
        use_rho_log1p=True,
        use_p_adam=True,
        use_p_saga=False,
        grad_mult=grad_mult,
        p_adam_b1=betas[0],
        p_adam_b2=betas[1],
    )
    torch_params = [
        torch.nn.Parameter(torch.tensor(np.asarray(p)))
        for p in jax.tree_util.tree_leaves(params)
    ]
    reference.attach_params(torch_params)
    normaliser = PAdam(reference.n_paths, b1=betas[0], b2=betas[1], grad_mult=grad_mult)
    state = normaliser.init(params)
    update = jax.jit(normaliser.update)
    rng = np.random.default_rng(42)
    schedule = [0, 1, 0, 4, 4, 2, 0, 3, 1]
    for count, path_idx in enumerate(schedule):
        grads = jax.tree_util.tree_map(
            lambda p: jnp.asarray(rng.normal(size=p.shape), dtype=p.dtype), params
        )
        old_state = jax.device_get(state)
        directions, state = update(grads, state, path_idx=jnp.asarray(path_idx))
        reference.scrapl_t = count + 1
        reference.curr_path_idx = path_idx
        for i, (g, direction, m, v) in enumerate(
            zip(
                jax.tree_util.tree_leaves(grads),
                jax.tree_util.tree_leaves(directions),
                jax.tree_util.tree_leaves(state.m),
                jax.tree_util.tree_leaves(state.v),
            )
        ):
            expected = reference.grad_hook(torch.tensor(np.asarray(g)), param_idx=i)
            np.testing.assert_allclose(
                direction, expected.numpy(), rtol=3e-6, atol=1e-6
            )
            np.testing.assert_allclose(
                m, reference.p_adam_m[i].numpy(), rtol=3e-6, atol=1e-6
            )
            np.testing.assert_allclose(
                v, reference.p_adam_v[i].numpy(), rtol=3e-6, atol=1e-6
            )
            for other_path in set(range(reference.n_paths)) - {path_idx}:
                np.testing.assert_array_equal(
                    m[other_path], jax.tree_util.tree_leaves(old_state.m)[i][other_path]
                )
                np.testing.assert_array_equal(
                    v[other_path], jax.tree_util.tree_leaves(old_state.v)[i][other_path]
                )
        assert state.count == count + 1
        assert state.last_steps[path_idx] == count + 2
        np.testing.assert_array_equal(
            state.last_steps,
            [reference.p_adam_t[0][i] for i in range(reference.n_paths)],
        )


def test_first_step_and_small_fractional_decay_match_pytorch():
    normaliser = PAdam(100_000, b1=0.999, b2=1 - 1e-10)
    grads = jnp.array([0.0, 1e-6, -2.0], dtype=jnp.float32)
    initial = normaliser.init(grads)
    direction, state = jax.jit(normaliser.update)(
        grads, initial, path_idx=jnp.asarray(17)
    )
    expected, m, v = TorchLoss.adam_grad_norm_cont(
        torch.tensor(np.asarray(grads)),
        torch.zeros(3),
        torch.zeros(3),
        t=2 / normaliser.n_paths,
        prev_t=0.0,
        b1=normaliser.b1,
        b2=normaliser.b2,
    )
    np.testing.assert_allclose(direction, expected.numpy(), rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(state.m[17], m.numpy(), rtol=1e-6, atol=1e-14)
    # Python subtraction in the reference loses precision for this tiny coefficient.
    assert np.isfinite(state.v).all()
    assert state.v[17, -1] > 0
    np.testing.assert_allclose(
        direction, np.asarray(grads) / (np.abs(grads) + normaliser.eps), rtol=2e-6
    )


def test_long_gap_uses_elapsed_steps_and_global_bias_correction():
    normaliser = PAdam(5)
    initial = normaliser.init(jnp.array([1.0, 1.0]))
    state = initial._replace(
        count=jnp.asarray(10_000, dtype=jnp.int32),
        last_steps=initial.last_steps.at[2].set(2),
        m=initial.m.at[2].set(jnp.array([1.0, -1.0])),
        v=initial.v.at[2].set(jnp.array([2.0, 3.0])),
    )
    grads = jnp.array([0.0, 0.0])
    direction, updated = normaliser.update(grads, state, path_idx=2)
    expected, m, v = TorchLoss.adam_grad_norm_cont(
        torch.tensor(np.asarray(grads)),
        torch.tensor([1.0, -1.0]),
        torch.tensor([2.0, 3.0]),
        t=10002 / 5,
        prev_t=2 / 5,
    )
    np.testing.assert_allclose(direction, expected.numpy(), atol=1e-7)
    np.testing.assert_allclose(updated.m[2], m.numpy(), atol=1e-7)
    np.testing.assert_allclose(updated.v[2], v.numpy(), rtol=2e-6)


def test_checkpoint_resume_and_input_immutability(params):
    normaliser = PAdam(5)
    initial = normaliser.init(params)
    initial_copy = jax.device_get(initial)
    update = jax.jit(normaliser.update)
    _, state = update(params, initial, path_idx=jnp.asarray(2))
    assert_tree_close(initial, initial_copy, rtol=0, atol=0)
    restored = pickle.loads(pickle.dumps(jax.device_get(state)))
    resumed = normaliser.update(params, restored, path_idx=1)
    uninterrupted = update(params, state, path_idx=jnp.asarray(1))
    assert_tree_close(resumed, uninterrupted, rtol=2e-6, atol=1e-7)


def test_real_loss_training_matches_pytorch_forward_backward_steps():
    config = dict(shape=128, J=3, Q1=2, Q2=1, J_fr=1, Q_fr=1, use_rho_log1p=True)
    loss = SCRAPLLoss(**config)
    reference = TorchLoss(**config, use_p_adam=True, use_p_saga=False, grad_mult=1)
    target = jax.random.normal(jax.random.key(8), (1, 1, 128))
    torch_target = torch.tensor(np.asarray(target))
    theta = jnp.asarray(-1.0)
    torch_theta = torch.nn.Parameter(torch.tensor(-1.0))
    reference.attach_params([torch_theta])
    normaliser = PAdam(loss.n_paths)
    state = normaliser.init(theta)
    torch_optimizer = torch.optim.SGD([torch_theta], lr=0.1)

    @jax.jit
    def step(theta, state, path_idx):
        value, gradient = jax.value_and_grad(
            lambda theta: loss(
                jax.nn.sigmoid(theta) * target, target, path_idx=path_idx
            )
        )(theta)
        direction, state = normaliser.update(gradient, state, path_idx=path_idx)
        return theta - 0.1 * direction, state, value, direction

    for path_idx in [0, 1, 0, 4, 2, 1, 4, 3]:
        theta, state, value, direction = step(theta, state, jnp.asarray(path_idx))
        torch_optimizer.zero_grad()
        expected_value = reference(
            torch.sigmoid(torch_theta) * torch_target, torch_target, path_idx=path_idx
        )
        expected_value.backward()
        np.testing.assert_allclose(value, expected_value.detach().numpy(), rtol=3e-5)
        np.testing.assert_allclose(direction, torch_theta.grad.numpy(), rtol=3e-5)
        torch_optimizer.step()
        np.testing.assert_allclose(
            theta, torch_theta.detach().numpy(), rtol=3e-5, atol=1e-6
        )
    assert theta > -1
    np.testing.assert_allclose(state.m, reference.p_adam_m[0].numpy(), rtol=3e-5)
    np.testing.assert_allclose(state.v, reference.p_adam_v[0].numpy(), rtol=3e-5)


def test_jitted_weighted_sampling_scan_reduces_full_path_mean():
    loss = SCRAPLLoss(shape=128, J=3, Q1=2, Q2=1, J_fr=1, Q_fr=1, use_rho_log1p=True)
    target = jax.random.normal(jax.random.key(8), (1, 1, 128))
    normaliser = PAdam(loss.n_paths)
    initial_theta = jnp.asarray(-1.0)
    initial = normaliser.init(initial_theta)
    probs = jnp.asarray([0.4, 0.1, 0.2, 0.15, 0.15])

    @jax.jit
    def train(theta, state, key, probs):
        def step(carry, _):
            theta, state, key = carry
            key, path_key = jax.random.split(key)
            path_idx = loss.sample_path(path_key, probs=probs)
            value, gradient = jax.value_and_grad(
                lambda theta: loss(
                    jax.nn.sigmoid(theta) * target, target, path_idx=path_idx
                )
            )(theta)
            direction, state = normaliser.update(gradient, state, path_idx=path_idx)
            return (theta - 0.1 * direction, state, key), value

        return jax.lax.scan(step, (theta, state, key), None, length=40)

    def full_mean(theta):
        return jnp.stack(
            [
                loss(jax.nn.sigmoid(theta) * target, target, path_idx=i)
                for i in range(loss.n_paths)
            ]
        ).mean()

    (theta, state, _), values = train(initial_theta, initial, jax.random.key(4), probs)
    assert np.isfinite(values).all()
    assert state.count == 40
    assert full_mean(theta) < 0.5 * full_mean(initial_theta)
    before = jax.device_get(state)
    full_mean(theta)
    assert_tree_close(state, before, rtol=0, atol=0)


def test_invalid_updates_leave_state_unchanged(params):
    normaliser = PAdam(5)
    initial = normaliser.init(params)
    update = jax.jit(normaliser.update)
    for path_idx in (-1, 5):
        with pytest.raises(ValueError, match="path_idx"):
            normaliser.update(params, initial, path_idx=path_idx)
        direction, state = update(params, initial, path_idx=jnp.asarray(path_idx))
        assert all(
            np.isnan(leaf).all() for leaf in jax.tree_util.tree_leaves(direction)
        )
        assert_tree_close(state, initial, rtol=0, atol=0)
    for path_idx in (True, 1.5, jnp.asarray([0])):
        with pytest.raises(TypeError, match="scalar integer"):
            normaliser.update(params, initial, path_idx=path_idx)
    for value in (float("nan"), float("inf"), 1e30):
        bad_grads = {**params, "bias": jnp.full((2,), value)}
        direction, state = update(bad_grads, initial, path_idx=jnp.asarray(0))
        assert all(
            np.isnan(leaf).all() for leaf in jax.tree_util.tree_leaves(direction)
        )
        assert_tree_close(state, initial, rtol=0, atol=0)


def test_counter_overflow_and_invalid_moments_are_rejected():
    normaliser = PAdam(5)
    params = jnp.ones(2)
    initial = normaliser.init(params)
    update = jax.jit(normaliser.update)
    states = [
        initial._replace(count=jnp.asarray(jnp.iinfo(jnp.int32).max, dtype=jnp.int32)),
        initial._replace(last_steps=initial.last_steps.at[0].set(3)),
        initial._replace(v=initial.v.at[0].set(-1)),
    ]
    for state in states:
        direction, returned = update(params, state, path_idx=jnp.asarray(0))
        assert np.isnan(direction).all()
        assert_tree_close(state, returned, rtol=0, atol=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_paths": 0},
        {"n_paths": 2.5},
        {"b1": 1},
        {"b2": -1},
        {"eps": 0},
        {"grad_mult": float("inf")},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        PAdam(**{**dict(n_paths=5), **kwargs})


def test_parameter_and_state_shape_validation(params):
    normaliser = PAdam(5)
    with pytest.raises(ValueError, match="contain"):
        normaliser.init({})
    with pytest.raises(TypeError, match="float32 or float64"):
        normaliser.init(jnp.ones(2, dtype=jnp.int32))
    state = normaliser.init(params)
    with pytest.raises(ValueError, match="structures"):
        normaliser.update(jnp.ones(2), state, path_idx=0)
    with pytest.raises(ValueError, match="shapes"):
        normaliser.update({**params, "bias": jnp.ones(3)}, state, path_idx=0)
    with pytest.raises(ValueError, match="shapes"):
        replace(normaliser, n_paths=4).update(params, state, path_idx=0)
