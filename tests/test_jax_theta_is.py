from functools import partial

import numpy as np
import pytest

torch = pytest.importorskip("torch")

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
eqx = pytest.importorskip("equinox")
from jax.flatten_util import ravel_pytree

from scrapl import SCRAPLLoss as TorchLoss
from scrapl.jax import SCRAPLLoss, theta_importance_probs, warmup_lc_hvp
from scrapl.jax.warmup import (
    calc_hvp_at_theta_multibatch,
    calc_largest_eig_power_iter,
    calc_loss,
    calc_s_dE_dw_at_theta,
    generate_random_tangent,
)


@pytest.fixture(scope="module")
def problem():
    config = dict(
        shape=128, J=3, Q1=2, Q2=1, J_fr=1, Q_fr=1, T=8, F=1, use_rho_log1p=True
    )
    loss = SCRAPLLoss(**config)
    params = {
        "bias": jnp.array([-0.4, 0.2]),
        "matrix": jnp.array([[0.2, -0.1], [0.1, 0.3]]),
    }
    xs = jax.random.normal(jax.random.key(1), (2, 2, 1, 128))
    basis = jax.random.normal(jax.random.key(2), (2, 128))

    def encoder(params, x):
        return jax.nn.sigmoid(x[:, 0, :2] @ params["matrix"] + params["bias"])

    def synth(theta):
        return (theta @ basis)[:, None, :]

    torch_params = {
        name: torch.tensor(np.asarray(value), requires_grad=True)
        for name, value in params.items()
    }
    torch_basis = torch.tensor(np.asarray(basis))

    def torch_encoder(x):
        return torch.sigmoid(
            x[:, 0, :2] @ torch_params["matrix"] + torch_params["bias"]
        )

    def torch_synth(theta):
        return (theta @ torch_basis)[:, None, :]

    torch_loss = TorchLoss(
        **config, n_theta=2, grad_mult=1, use_p_adam=False, use_p_saga=False
    )
    return dict(
        loss=loss,
        params=params,
        enc_fn=encoder,
        dec_fn=synth,
        xs=xs,
        torch_loss=torch_loss,
        torch_params=list(torch_params.values()),
        torch_encoder=torch_encoder,
        torch_synth=torch_synth,
    )


def arguments(problem):
    return {
        "enc_params": problem["params"],
        **{name: problem[name] for name in ("loss", "enc_fn", "dec_fn", "xs")},
    }


@pytest.fixture(scope="module")
def result(problem):
    return warmup_lc_hvp(
        **arguments(problem), key=jax.random.key(3), n_iter=40, min_prob_frac=0.05
    )


@pytest.mark.parametrize("theta_idx", [0, 1])
def test_curvature_product_matches_dense_matrix_and_pytorch_transpose(
    problem, theta_idx
):
    w = problem["params"]
    tangent = generate_random_tangent(jax.random.key(10), w)
    theta_loss_fn = partial(
        calc_loss,
        dec_fn=problem["dec_fn"],
        loss_fn=partial(problem["loss"], path_idx=0),
    )
    actual = jax.jit(
        partial(
            calc_hvp_at_theta_multibatch,
            theta_idx=theta_idx,
            xs=problem["xs"],
            enc_fn=problem["enc_fn"],
            theta_loss_fn=theta_loss_fn,
        )
    )(tangent, enc_params=w)
    actual, _ = ravel_pytree(actual)
    flat_tangent, _ = ravel_pytree(tangent)
    weights, unravel = ravel_pytree(w)

    def gradient(weights):
        return sum(
            ravel_pytree(
                calc_s_dE_dw_at_theta(
                    unravel(weights), x, theta_idx=theta_idx, enc_fn=problem["enc_fn"], theta_loss_fn=theta_loss_fn
                )
            )[0]
            for x in problem["xs"]
        )

    matrix = jax.jit(jax.jacrev(gradient))(weights)
    assert np.max(np.abs(matrix - matrix.T)) > 1e-3
    # Forward-over-reverse computes J @ v
    np.testing.assert_allclose(actual, matrix @ flat_tangent, rtol=2e-4, atol=2e-5)
    # PyTorch's grad-of-grad computes J.T @ v, which has the same eigenvalues
    torch_kwargs = [dict(x=torch.tensor(np.asarray(batch))) for batch in problem["xs"]]
    expected = problem["torch_loss"]._calc_param_hvp_multibatch(
        torch.tensor(np.asarray(flat_tangent)),
        path_idx=0,
        theta_idx=theta_idx,
        theta_fn=problem["torch_encoder"],
        synth_fn=problem["torch_synth"],
        theta_fn_kwargs=torch_kwargs,
        params=problem["torch_params"],
    )
    np.testing.assert_allclose(
        expected.detach().numpy(), matrix.T @ flat_tangent, rtol=2e-3, atol=2e-4
    )


def test_vmap_theta_matches_sequential_theta(problem, result):
    vmapped = warmup_lc_hvp(
        **arguments(problem),
        key=jax.random.key(3),
        n_iter=40,
        min_prob_frac=0.05,
        vmap_theta=True,
    )
    np.testing.assert_allclose(vmapped.curvatures, result.curvatures, rtol=1e-5)
    np.testing.assert_allclose(vmapped.probs, result.probs, rtol=1e-5)
    assert np.max(vmapped.relative_residuals) < 1e-3


def test_equinox_encoder_with_non_trainable_leaves_matches_dict_params(
    problem, result
):
    class Encoder(eqx.Module):
        bias: jax.Array
        matrix: jax.Array
        step: jax.Array
        activation: object

        def __call__(self, x):
            return self.activation(x[:, 0, :2] @ self.matrix + self.bias)

    params = problem["params"]
    encoder = Encoder(
        params["bias"], params["matrix"], jnp.array(7, dtype=jnp.int32), jax.nn.sigmoid
    )
    args = {**arguments(problem), "enc_params": encoder}
    args["enc_fn"] = lambda model, x: model(x)
    actual = warmup_lc_hvp(
        **args, key=jax.random.key(3), n_iter=40, min_prob_frac=0.05
    )
    np.testing.assert_allclose(actual.curvatures, result.curvatures, rtol=1e-5)
    np.testing.assert_allclose(actual.probs, result.probs, rtol=1e-5)
    assert np.max(actual.relative_residuals) < 1e-3


def test_completed_warmup_matches_pytorch(problem, result):
    torch.manual_seed(3)
    reference = problem["torch_loss"]
    reference.warmup_lc_hvp(
        theta_fn=problem["torch_encoder"],
        synth_fn=problem["torch_synth"],
        theta_fn_kwargs=[
            dict(x=torch.tensor(np.asarray(batch))) for batch in problem["xs"]
        ],
        params=problem["torch_params"],
        n_iter=40,
        save_dir=None,
    )
    expected_curvatures = reference.all_log_vals.exp().T
    assert result.curvatures.shape == (problem["loss"].n_paths, 2)
    assert np.max(result.relative_residuals) < 1e-3
    np.testing.assert_allclose(
        result.curvatures, expected_curvatures.numpy(), rtol=3e-3, atol=3e-4
    )
    expected_probs = 0.95 * reference.probs.numpy() + 0.05 / reference.n_paths
    np.testing.assert_allclose(result.probs, expected_probs, rtol=3e-3, atol=2e-5)
    assert np.min(result.probs) >= 0.05 / reference.n_paths


def test_theta_normalisation_and_floor_match_reference_updates(problem):
    n_paths = problem["loss"].n_paths
    values = np.array(
        [[1, 1000], [9, 1000], [0, 1000], [1e-5, 1000], [20, 1000]], dtype=np.float32
    )
    assert values.shape == (n_paths, 2)
    reference = TorchLoss(
        shape=128,
        J=3,
        Q1=2,
        Q2=1,
        J_fr=1,
        Q_fr=1,
        n_theta=2,
        grad_mult=1,
        use_p_adam=False,
        use_p_saga=False,
    )
    for path_idx, row in enumerate(values):
        for theta_idx, value in enumerate(row):
            reference.update_prob(path_idx, float(value), theta_idx)
    actual = jax.jit(partial(theta_importance_probs, min_prob_frac=0.1))(
        jnp.asarray(values)
    )
    expected = 0.9 * reference.probs.numpy() + 0.1 / n_paths
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(
        theta_importance_probs(jnp.zeros_like(values)), np.full(n_paths, 1 / n_paths)
    )


def test_weighted_sampling_distribution_and_dynamic_probability_updates(
    problem, result
):
    loss = problem["loss"]
    keys = jax.random.split(jax.random.key(123), 32768)
    sample = jax.jit(
        jax.vmap(
            lambda key, probs: loss.sample_path(key, probs=probs), in_axes=(0, None)
        )
    )
    draws = sample(keys, result.probs)
    counts = np.bincount(np.asarray(draws), minlength=loss.n_paths) / len(keys)
    np.testing.assert_allclose(counts, result.probs, rtol=0, atol=0.012)
    np.testing.assert_array_equal(draws, sample(keys, result.probs))
    for index in (0, loss.n_paths - 1):
        one_hot = jax.nn.one_hot(index, loss.n_paths)
        np.testing.assert_array_equal(sample(keys[:32], one_hot), np.full(32, index))


def test_weighted_training_step_preserves_selected_path_loss_and_gradient(
    problem, result
):
    loss, params, x = problem["loss"], problem["params"], problem["xs"][0]

    def objective(params, key, probs):
        prediction = problem["dec_fn"](problem["enc_fn"](params, x))
        return loss(x, prediction, key=key, probs=probs)

    evaluate = jax.jit(jax.value_and_grad(objective))
    key = jax.random.key(99)
    value, gradient = evaluate(params, key, result.probs)
    index = int(loss.sample_path(key, probs=result.probs))
    expected_value, expected_gradient = jax.value_and_grad(
        lambda params: loss(
            x, problem["dec_fn"](problem["enc_fn"](params, x)), path_idx=index
        )
    )(params)
    np.testing.assert_allclose(value, expected_value, rtol=2e-5)
    for actual, expected in zip(
        jax.tree_util.tree_leaves(gradient),
        jax.tree_util.tree_leaves(expected_gradient),
    ):
        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)
    updated = jax.tree_util.tree_map(lambda p, g: p - 1e-5 * g, params, gradient)
    assert evaluate(updated, key, result.probs)[0] < value


def test_zero_curvature_warmup_is_uniform(problem):
    args = arguments(problem)
    args["enc_fn"] = lambda params, x: jnp.full((x.shape[0], 2), 0.5)
    result = warmup_lc_hvp(**args, key=jax.random.key(5), n_iter=2, min_prob_frac=0.1)
    np.testing.assert_array_equal(result.curvatures, np.zeros_like(result.curvatures))
    np.testing.assert_array_equal(
        result.relative_residuals, np.zeros_like(result.relative_residuals)
    )
    np.testing.assert_allclose(
        result.probs, np.full(problem["loss"].n_paths, problem["loss"].unif_prob)
    )


def test_power_iteration_reports_nonconvergence_and_negative_dominant_eigenvalue():
    matrix = jnp.array([[-3.0, 1.0], [0.0, 1.0]])
    value, errors = calc_largest_eig_power_iter(
        lambda v: matrix @ v, jnp.array([1.0, 1.0]), n_iter=40, eps=1e-12
    )
    np.testing.assert_allclose(value, 3.0, rtol=1e-6)
    assert errors.shape == (40,)
    assert errors[-1] < 1e-6
    rotation = jnp.array([[0.0, -1.0], [1.0, 0.0]])
    _, errors = calc_largest_eig_power_iter(
        lambda v: rotation @ v, jnp.array([1.0, 0.0]), n_iter=20, eps=1e-12
    )
    assert errors[-1] > 0.9


@pytest.mark.parametrize("case", ["zero", "negative", "nan", "unnormalised"])
def test_invalid_probabilities_are_rejected_under_jit(problem, case):
    loss = problem["loss"]
    probs = jnp.full((loss.n_paths,), loss.unif_prob)
    if case == "zero":
        probs = jnp.zeros_like(probs)
    elif case == "negative":
        probs = probs.at[0].set(-1)
    elif case == "nan":
        probs = probs.at[0].set(jnp.nan)
    else:
        probs = probs * 2
    key = jax.random.key(0)
    assert jax.jit(loss.sample_path)(key, probs=probs) == -1
    x = problem["xs"][0]
    assert jnp.isnan(jax.jit(loss)(x, x * 0.5, key=key, probs=probs))


def test_probability_shapes_and_fixed_path_arguments(problem):
    loss, key = problem["loss"], jax.random.key(0)
    with pytest.raises(ValueError, match="shape"):
        loss.sample_path(key, probs=jnp.ones((1,)))
    with pytest.raises(TypeError, match="float"):
        loss.sample_path(key, probs=jnp.ones(loss.n_paths, dtype=jnp.int32))
    x = problem["xs"][0]
    with pytest.raises(ValueError, match="only when sampling"):
        loss(x, x, path_idx=0, probs=jnp.full(loss.n_paths, loss.unif_prob))


@pytest.mark.parametrize("kwargs", [{"n_iter": 0}, {"min_prob_frac": 1}, {"eps": 0}])
def test_invalid_warmup_settings(problem, kwargs):
    with pytest.raises(ValueError):
        warmup_lc_hvp(**arguments(problem), key=jax.random.key(0), **kwargs)


def test_invalid_training_arrays_and_encoder_shapes(problem):
    args = arguments(problem)
    with pytest.raises(ValueError, match="trainable"):
        warmup_lc_hvp(**{**args, "enc_params": {}}, key=jax.random.key(0))
    with pytest.raises(ValueError, match="xs must"):
        warmup_lc_hvp(**{**args, "xs": args["xs"][0]}, key=jax.random.key(0))
    with pytest.raises(ValueError, match="enc_fn"):
        warmup_lc_hvp(
            **{**args, "enc_fn": lambda params, x: jnp.ones((1,))},
            key=jax.random.key(0),
        )
    with pytest.raises(ValueError, match="finite"):
        warmup_lc_hvp(**{**args, "xs": args["xs"] * jnp.nan}, key=jax.random.key(0))


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_invalid_curvature_estimates_produce_nan_probabilities(value):
    values = jnp.array([[1.0, 2.0], [value, 4.0]])
    assert jnp.isnan(jax.jit(theta_importance_probs)(values)).all()
