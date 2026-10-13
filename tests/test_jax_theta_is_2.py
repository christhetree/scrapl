import numpy as np
import pytest

from scrapl._dependencies import require_backend

try:
    tr = require_backend("torch")
    jax = require_backend("jax")
    from scrapl import SCRAPLLoss as TorchSCRAPLLoss
    from scrapl.jax import (
        SCRAPLLoss as JaxSCRAPLLoss,
        theta_importance_probs,
        warmup_lc_hvp,
    )
    from tests.test_jax_optim import setup_toy_models_and_data
except ModuleNotFoundError as e:
    pytest.skip(str(e), allow_module_level=True)


@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize("tol", [0.0, 1e-2])
@pytest.mark.parametrize("n_batches", [1, 2])
def test_theta_is_matches_pytorch(
    seed: int, n_batches: int, tol: float
) -> None:
    print(f"\n\nRunning test_theta_is_matches_pytorch with seed={seed}, n_batches={n_batches}, tol={tol}")
    # Same hyperparameters as test_scrapl_loss_warmup in test_torch_scrapl_loss.py
    bs = 3
    n_samples = 2048
    n_theta = 2
    n_iter = 20
    config = dict(shape=n_samples, J=3, Q1=1, Q2=1, J_fr=1, Q_fr=1)
    rel_residual_tol = 1e-2

    setup = setup_toy_models_and_data(
        seed=seed, n_samples=n_samples, n_theta=n_theta, bs=bs
    )
    xs_jax = jax.numpy.stack([setup.x_jax, setup.x_target_jax])[:n_batches]
    xs_tr = [setup.x_tr, setup.x_target_tr][:n_batches]
    assert xs_jax.shape[0] == len(xs_tr) == n_batches

    jax_loss = JaxSCRAPLLoss(**config)
    result = warmup_lc_hvp(
        jax_loss,
        lambda enc, x: jax.vmap(enc)(x[:, 0, :]),
        lambda theta: jax.vmap(setup.decoder_jax)(theta)[:, None, :],
        xs_jax,
        setup.encoder_jax,
        key=setup.loop_key,
        n_iter=n_iter,
        tol=tol,
        vmap_theta=True,
    )

    torch_loss = TorchSCRAPLLoss(**config, n_theta=n_theta)
    assert jax_loss.scrapl_keys == tuple(torch_loss.scrapl_keys)
    tr.manual_seed(seed)
    torch_loss.warmup_lc_hvp(
        theta_fn=lambda x: setup.encoder_torch(x[:, 0, :]),
        synth_fn=lambda theta: setup.decoder_torch(theta).unsqueeze(1),
        theta_fn_kwargs=[dict(x=x) for x in xs_tr],
        params=setup.torch_params,
        n_iter=n_iter,
        save_dir=None,
    )
    torch_curvatures = torch_loss.all_log_vals.exp().T.numpy()
    torch_rel_residuals = torch_loss.all_rel_residuals.T.numpy()
    assert result.curvatures.shape == torch_curvatures.shape

    with np.printoptions(precision=4, suppress=True):
        print(f"\nJAX eigenvalues (n_paths, n_theta):\n{np.asarray(result.curvatures)}")
        print(f"PyTorch eigenvalues (n_paths, n_theta):\n{torch_curvatures}")
        print(
            f"JAX relative residuals (n_paths, n_theta):\n"
            f"{np.asarray(result.relative_residuals)}"
        )
        print(f"PyTorch relative residuals (n_paths, n_theta):\n{torch_rel_residuals}")

    # 1. Verify curvature estimates match where both power iterations converged.
    # Unconverged estimates depend on each implementation's random initial vector.
    converged = (np.asarray(result.relative_residuals) < rel_residual_tol) & (
        torch_rel_residuals < rel_residual_tol
    )
    print(
        f"Converged: {converged.sum()} / {converged.size} ({converged.mean():.0%})"
    )
    assert converged.mean() >= 0.8, f"Only {converged.mean():.0%} converged"
    np.testing.assert_allclose(
        np.asarray(result.curvatures)[converged],
        torch_curvatures[converged],
        rtol=3e-3,
        atol=3e-4,
    )

    # 2. Verify path sampling probabilities match for the same curvatures
    np.testing.assert_allclose(
        np.asarray(theta_importance_probs(jax.numpy.asarray(torch_curvatures))),
        torch_loss.probs.numpy(),
        rtol=3e-3,
        atol=2e-5,
    )
