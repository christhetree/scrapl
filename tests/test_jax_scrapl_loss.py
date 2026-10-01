from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from scrapl import SCRAPLLoss as TorchLoss
from scrapl.jax import SCRAPLLoss


@pytest.fixture(scope="module")
def signals():
    rng = np.random.default_rng(123)
    return tuple(rng.normal(size=(2, 2, 257)).astype(np.float32) for _ in range(3))


@pytest.fixture(scope="module")
def base_loss():
    return SCRAPLLoss(shape=257, J=4, Q1=2, Q2=1, J_fr=2, Q_fr=1, T=16, F=2)


@pytest.fixture(
    scope="module",
    params=[
        (16, 2, 2, False),
        (8, 1, 1, True),
        ("global", "global", 2, True),
        (16, 2, 3, False),
        (16, 2, float("inf"), False),
    ],
)
def losses(request, base_loss):
    time_average, frequency_average, p, use_log = request.param
    jax_loss = replace(
        base_loss, T=time_average, F=frequency_average, p=p, use_rho_log1p=use_log
    )
    torch_loss = TorchLoss(
        shape=257,
        J=4,
        Q1=2,
        Q2=1,
        J_fr=2,
        Q_fr=1,
        T=time_average,
        F=frequency_average,
        p=p,
        use_rho_log1p=use_log,
        grad_mult=1,
        use_p_adam=False,
        use_p_saga=False,
    )
    return jax_loss, torch_loss


def test_every_dynamic_path_matches_pytorch(losses, signals):
    loss, torch_loss = losses
    x, target, _ = signals
    assert loss.scrapl_keys == tuple(torch_loss.scrapl_keys)
    evaluate = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))
    for path_idx in range(loss.n_paths):
        value, gradients = evaluate(
            jnp.asarray(x), jnp.asarray(target), path_idx=jnp.asarray(path_idx)
        )
        torch_x, torch_target = (
            torch.tensor(v, requires_grad=True) for v in (x, target)
        )
        expected_value = torch_loss(torch_x, torch_target, path_idx=path_idx)
        expected_gradients = torch.autograd.grad(
            expected_value, (torch_x, torch_target)
        )
        np.testing.assert_allclose(
            value, expected_value.detach().numpy(), rtol=2e-5, atol=1e-7
        )
        for actual, expected in zip(gradients, expected_gradients):
            np.testing.assert_allclose(actual, expected.numpy(), rtol=5e-4, atol=2e-6)


def test_uniform_sampling_is_reproducible_and_covers_all_paths(base_loss):
    keys = jax.random.split(jax.random.key(42), 32768)
    sample = jax.jit(jax.vmap(base_loss.sample_path))
    indices = sample(keys)
    np.testing.assert_array_equal(indices, sample(keys))
    assert indices.min() >= 0
    assert indices.max() < base_loss.n_paths
    counts = np.bincount(np.asarray(indices), minlength=base_loss.n_paths)
    expected = len(keys) * base_loss.unif_prob
    assert np.max(np.abs(counts - expected)) < 0.15 * expected


def test_sampled_and_fixed_paths_have_identical_loss_and_gradients(base_loss, signals):
    x, target, _ = map(jnp.asarray, signals)
    evaluate = jax.jit(jax.value_and_grad(base_loss))
    for key in jax.random.split(jax.random.key(7), 3):
        path_idx = int(base_loss.sample_path(key))
        value, gradient = evaluate(x, target, key=key)
        expected_value, expected_gradient = jax.value_and_grad(base_loss)(
            x, target, path_idx=path_idx
        )
        np.testing.assert_allclose(value, expected_value, rtol=2e-5, atol=1e-7)
        np.testing.assert_allclose(gradient, expected_gradient, rtol=5e-4, atol=2e-6)


def test_dynamic_dispatch_executes_only_the_selected_path(
    base_loss, signals, monkeypatch
):
    executed = []
    original = base_loss.jtfs.scattering_singlepath

    def record(n2, n_fr):
        executed.append((int(n2), int(n_fr)))

    def observed(x, n2, n_fr):
        jax.debug.callback(record, n2, n_fr, ordered=True)
        return original(x, n2, n_fr)

    monkeypatch.setattr(base_loss.jtfs, "scattering_singlepath", observed)
    x, target, _ = map(jnp.asarray, signals)
    key = jax.random.key(99)
    value = jax.jit(base_loss)(x, target, key=key)
    value.block_until_ready()
    jax.effects_barrier()
    selected = base_loss.scrapl_keys[int(base_loss.sample_path(key))]
    assert executed == [selected, selected]


@pytest.mark.parametrize("case", ["match", "partial_match", "silence"])
def test_zero_distances_and_silence_have_finite_gradients(losses, signals, case):
    loss, torch_loss = losses
    x, target, _ = (v.copy() for v in signals)
    if case == "match":
        target = x.copy()
    elif case == "partial_match":
        target[:, 0] = x[:, 0]
    else:
        x.fill(0)
    key = jax.random.key(5)
    path_idx = int(loss.sample_path(key))
    value, gradient = jax.jit(jax.value_and_grad(loss))(
        jnp.asarray(x), jnp.asarray(target), key=key
    )
    torch_x = torch.tensor(x, requires_grad=True)
    expected_value = torch_loss(torch_x, torch.from_numpy(target), path_idx=path_idx)
    (expected_gradient,) = torch.autograd.grad(expected_value, torch_x)
    assert np.isfinite(value)
    assert np.isfinite(gradient).all()
    np.testing.assert_allclose(
        value, expected_value.detach().numpy(), rtol=2e-5, atol=1e-7
    )
    np.testing.assert_allclose(
        gradient, expected_gradient.numpy(), rtol=5e-4, atol=2e-6
    )
    if case == "match":
        assert value == 0
        np.testing.assert_array_equal(gradient, np.zeros_like(x))


def test_dynamic_path_hessian_vector_product_matches_pytorch(base_loss, signals):
    loss = replace(base_loss, use_rho_log1p=True)
    torch_loss = TorchLoss(
        shape=257,
        J=4,
        Q1=2,
        Q2=1,
        J_fr=2,
        Q_fr=1,
        T=16,
        F=2,
        use_rho_log1p=True,
        grad_mult=1,
        use_p_adam=False,
        use_p_saga=False,
    )
    x, target, tangent = map(jnp.asarray, signals)

    @jax.jit
    def hvp(x, tangent, key):
        grad = jax.grad(lambda x: loss(x, target, key=key))
        return jax.jvp(grad, (x,), (tangent,))[1]

    key = jax.random.key(17)
    actual = hvp(x, tangent, key)
    torch_x = torch.tensor(signals[0], requires_grad=True)
    value = torch_loss(
        torch_x, torch.from_numpy(signals[1]), path_idx=int(loss.sample_path(key))
    )
    (gradient,) = torch.autograd.grad(value, torch_x, create_graph=True)
    (expected,) = torch.autograd.grad(
        gradient, torch_x, grad_outputs=torch.from_numpy(signals[2])
    )
    np.testing.assert_allclose(actual, expected.numpy(), rtol=2e-3, atol=2e-5)


def test_jitted_stochastic_optimisation_reduces_full_path_mean():
    loss = SCRAPLLoss(shape=128, J=3, Q1=2, Q2=1, J_fr=2, Q_fr=1, use_rho_log1p=True)
    target = jax.random.normal(jax.random.key(8), (1, 1, 128))

    @jax.jit
    def train(theta, key):
        def step(state, _):
            theta, key = state
            key, path_key = jax.random.split(key)
            value, gradient = jax.value_and_grad(
                lambda theta: loss(jax.nn.sigmoid(theta) * target, target, key=path_key)
            )(theta)
            return (theta - 0.1 * gradient, key), value

        return jax.lax.scan(step, (theta, key), None, length=20)

    def full_mean(theta):
        x = jax.nn.sigmoid(theta) * target
        return jnp.stack(
            [loss(x, target, path_idx=i) for i in range(loss.n_paths)]
        ).mean()

    initial = jnp.asarray(-1.0)
    (theta, _), history = train(initial, jax.random.key(10))
    assert np.isfinite(history).all()
    assert np.isfinite(theta)
    assert full_mean(theta) < 0.5 * full_mean(initial)


def test_invalid_path_indices_are_not_silently_clamped(base_loss, signals):
    x, target, _ = map(jnp.asarray, signals)
    for index in (-1, base_loss.n_paths):
        with pytest.raises(ValueError, match="path_idx"):
            base_loss(x, target, path_idx=index)
    evaluate = jax.jit(base_loss)
    for index in (-1, base_loss.n_paths):
        assert jnp.isnan(evaluate(x, target, path_idx=jnp.asarray(index)))
    for dtype in (jnp.int8, jnp.uint32):
        value = evaluate(x, target, path_idx=jnp.asarray(0, dtype=dtype))
        expected = base_loss(x, target, path_idx=0)
        np.testing.assert_allclose(value, expected, rtol=2e-5, atol=1e-7)
    assert jnp.isnan(
        evaluate(x, target, path_idx=jnp.asarray(2**32 - 1, dtype=jnp.uint32))
    )
    for index in (True, 1.5, jnp.array([0])):
        with pytest.raises(TypeError, match="scalar integer"):
            base_loss(x, target, path_idx=index)


def test_requires_explicit_unambiguous_randomness(base_loss, signals):
    x, target, _ = map(jnp.asarray, signals)
    with pytest.raises(ValueError, match="exactly one"):
        base_loss(x, target)
    with pytest.raises(ValueError, match="exactly one"):
        base_loss(x, target, key=jax.random.key(0), path_idx=0)


@pytest.mark.parametrize("shape", [(1, 1, 128), (0, 1, 257), (2, 257)])
def test_rejects_invalid_signal_shapes(base_loss, shape):
    x = jnp.zeros(shape)
    with pytest.raises(ValueError):
        base_loss(x, x, path_idx=0)


def test_rejects_broadcasting_and_unsupported_dtypes(base_loss, signals):
    x, target, _ = map(jnp.asarray, signals)
    with pytest.raises(ValueError, match="matching"):
        base_loss(x, target[:1], path_idx=0)
    with pytest.raises(TypeError, match="float32 or float64"):
        base_loss(x.astype(jnp.float16), target, path_idx=0)


@pytest.mark.parametrize(
    "kwargs", [{"p": 0}, {"p": float("nan")}, {"log1p_eps": 0}, {"J": 1}]
)
def test_rejects_invalid_configuration(base_loss, kwargs):
    with pytest.raises(ValueError):
        replace(base_loss, **kwargs)
