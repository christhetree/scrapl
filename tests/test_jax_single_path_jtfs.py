import numpy as np
import pytest

torch = pytest.importorskip("torch")

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from scrapl import SCRAPLLoss
from scrapl.single_path_jtfs import TimeFrequencyScrapl
from scrapl.single_path_jtfs.jax import TimeFrequencyScrapl as JaxJTFS


@pytest.fixture(scope="module")
def signals():
    rng = np.random.default_rng(42)
    return tuple(rng.normal(size=(2, 2, 257)).astype(np.float32) for _ in range(3))


@pytest.fixture(scope="module", params=[(16, 2), (8, 1), ("global", "global")])
def transforms(request):
    time_average, frequency_average = request.param
    torch_loss = SCRAPLLoss(
        shape=257,
        J=4,
        Q1=2,
        Q2=1,
        J_fr=2,
        Q_fr=1,
        T=time_average,
        F=frequency_average,
        grad_mult=1,
        use_p_adam=False,
        use_p_saga=False,
    )
    jax_jtfs = JaxJTFS(
        shape=(257,),
        J=4,
        Q=(2, 1),
        J_fr=2,
        Q_fr=1,
        T=time_average,
        F=frequency_average,
    )
    return jax_jtfs, torch_loss


def path_loss(jtfs, key, target, use_log):
    target_coef = jtfs.scattering_singlepath(target, *key)["coef"]
    if use_log:
        target_coef = jnp.log1p(target_coef / 1e-3)

    def loss(x):
        coef = jtfs.scattering_singlepath(x, *key)["coef"]
        if use_log:
            coef = jnp.log1p(coef / 1e-3)
        return jnp.linalg.norm(target_coef - coef, axis=(-2, -1)).mean()

    return loss


def key_for_spin(jtfs, keys, spin):
    return next(
        key
        for key in reversed(keys)
        if np.sign(jtfs.filters_fr[1][key[1]]["xi"]) == spin
    )


def test_all_path_coefficients_match_pytorch(transforms, signals):
    jax_jtfs, torch_loss = transforms
    keys = [key for key in jax_jtfs.meta()["key"] if len(key) == 2]
    assert keys == torch_loss.scrapl_keys
    x = signals[0]
    for key in keys:
        actual = jax_jtfs.scattering_singlepath(jnp.asarray(x), *key)
        expected = torch_loss.jtfs.scattering_singlepath(torch.from_numpy(x), *key)
        for field in expected.keys() - {"coef"}:
            np.testing.assert_array_equal(actual[field], expected[field])
        expected_coef = expected["coef"].squeeze(-1).numpy()
        assert actual["coef"].shape == expected_coef.shape
        assert actual["coef"].dtype == jnp.float32
        np.testing.assert_allclose(actual["coef"], expected_coef, rtol=2e-5, atol=1e-7)


@pytest.mark.parametrize("spin", [-1, 0, 1])
@pytest.mark.parametrize("use_log", [False, True])
def test_loss_and_input_gradients_match_pytorch(transforms, signals, spin, use_log):
    jax_jtfs, torch_loss = transforms
    x, target, _ = signals
    key = key_for_spin(jax_jtfs, torch_loss.scrapl_keys, spin)
    loss = path_loss(jax_jtfs, key, jnp.asarray(target), use_log)
    value, gradient = jax.jit(jax.value_and_grad(loss))(jnp.asarray(x))

    torch_loss.use_rho_log1p = use_log
    torch_x = torch.tensor(x, requires_grad=True)
    expected_value = torch_loss(
        torch_x, torch.from_numpy(target), path_idx=torch_loss.scrapl_keys.index(key)
    )
    (expected_gradient,) = torch.autograd.grad(expected_value, torch_x)
    np.testing.assert_allclose(
        value, expected_value.detach().numpy(), rtol=2e-5, atol=1e-7
    )
    np.testing.assert_allclose(
        gradient, expected_gradient.numpy(), rtol=5e-4, atol=2e-6
    )


@pytest.mark.parametrize("spin", [-1, 0, 1])
def test_hessian_vector_products_match_pytorch(transforms, signals, spin):
    jax_jtfs, torch_loss = transforms
    x, target, tangent = signals
    key = key_for_spin(jax_jtfs, torch_loss.scrapl_keys, spin)
    loss = path_loss(jax_jtfs, key, jnp.asarray(target), use_log=True)

    @jax.jit
    def hvp(x, tangent):
        return jax.jvp(jax.grad(loss), (x,), (tangent,))[1]

    actual = hvp(jnp.asarray(x), jnp.asarray(tangent))
    torch_loss.use_rho_log1p = True
    torch_x = torch.tensor(x, requires_grad=True)
    expected_value = torch_loss(
        torch_x, torch.from_numpy(target), path_idx=torch_loss.scrapl_keys.index(key)
    )
    (gradient,) = torch.autograd.grad(expected_value, torch_x, create_graph=True)
    (expected,) = torch.autograd.grad(
        gradient, torch_x, grad_outputs=torch.from_numpy(tangent)
    )
    assert np.isfinite(actual).all()
    assert np.isfinite(expected.numpy()).all()
    np.testing.assert_allclose(actual, expected.numpy(), rtol=2e-3, atol=2e-5)


def test_silence_has_zero_coefficients_and_finite_derivatives(transforms):
    jax_jtfs, torch_loss = transforms
    x = jnp.zeros((1, 1, 257))
    key = key_for_spin(jax_jtfs, torch_loss.scrapl_keys, 1)

    def coefficient_sum(x):
        return jax_jtfs.scattering_singlepath(x, *key)["coef"].sum()

    value, gradient = jax.jit(jax.value_and_grad(coefficient_sum))(x)
    hvp = jax.jvp(jax.grad(coefficient_sum), (x,), (jnp.ones_like(x),))[1]
    assert value == 0
    np.testing.assert_array_equal(gradient, np.zeros_like(x))
    assert np.isfinite(hvp).all()


def test_jax_entry_point_and_static_path_jit(signals):
    jtfs = TimeFrequencyScrapl(shape=(257,), J=4, Q=(2, 1), J_fr=2, backend="jax")
    assert isinstance(jtfs, JaxJTFS)
    key = next(key for key in jtfs.meta()["key"] if len(key) == 2)
    x = jnp.asarray(signals[0])
    compiled = jax.jit(jtfs.scattering_singlepath, static_argnames=("n2", "n_fr"))
    actual = compiled(x, *key)["coef"]
    expected = jtfs.scattering_singlepath(x, *key)["coef"]
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=1e-7)
