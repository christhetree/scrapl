import os

import numpy as np
import pytest

from scrapl._dependencies import require_backend

try:
    tr = require_backend("torch")
    jax = require_backend("jax")
    import equinox as eqx
    from scrapl import SCRAPLLoss as TorchSCRAPLLoss
    from scrapl.jax import (
        SCRAPLLoss as JaxSCRAPLLoss,
        load_warmup_from_dir,
        theta_importance_probs,
        warmup_lc_hvp,
    )
    from tests._toy_models import setup_toy_models_and_data
except ModuleNotFoundError as e:
    pytest.skip(str(e), allow_module_level=True)


@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize("tol", [0.0, 1e-2])
@pytest.mark.parametrize("n_batches", [1, 2])
@pytest.mark.parametrize("use_kwargs", [False, True])
def test_theta_is_matches_pytorch(
    seed: int, n_batches: int, tol: float, use_kwargs: bool
) -> None:
    print(
        f"\n\nRunning test_theta_is_matches_pytorch with seed={seed}, "
        f"n_batches={n_batches}, tol={tol}, use_kwargs={use_kwargs}"
    )
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

    # Optional per-batch encoder input gains and decoder output shifts
    enc_kwargs, dec_kwargs = None, None
    tr_enc_kwargs, tr_dec_kwargs = [{}] * n_batches, None
    if use_kwargs:
        gain_key, shift_key = jax.random.split(jax.random.key(seed + 1))
        gains = jax.random.uniform(gain_key, (n_batches, bs, 1), minval=0.5, maxval=1.5)
        shifts = jax.random.uniform(shift_key, (n_batches,), minval=-0.1, maxval=0.1)
        enc_kwargs, dec_kwargs = dict(gain=gains), dict(shift=shifts)
        tr_enc_kwargs = [dict(gain=tr.tensor(np.asarray(g))) for g in gains]
        tr_dec_kwargs = [dict(shift=tr.tensor(np.asarray(s))) for s in shifts]

    jax_loss = JaxSCRAPLLoss(**config)
    enc_params, enc_static = eqx.partition(setup.encoder_jax, eqx.is_inexact_array)
    result = warmup_lc_hvp(
        jax_loss,
        lambda p, x, gain=1.0: jax.vmap(eqx.combine(p, enc_static))(
            x[:, 0, :] * gain
        ),
        lambda theta, shift=0.0: jax.vmap(setup.decoder_jax)(theta)[:, None, :]
        + shift,
        xs_jax,
        enc_params,
        enc_kwargs,
        dec_kwargs,
        key=setup.loop_key,
        n_iter=n_iter,
        tol=tol,
        vmap_theta=True,
    )

    torch_loss = TorchSCRAPLLoss(**config, n_theta=n_theta)
    assert jax_loss.scrapl_keys == tuple(torch_loss.scrapl_keys)
    tr.manual_seed(seed)
    torch_loss.warmup_lc_hvp(
        theta_fn=lambda x, gain=1.0: setup.encoder_torch(x[:, 0, :] * gain),
        synth_fn=lambda theta, shift=0.0: setup.decoder_torch(theta).unsqueeze(1)
        + shift,
        theta_fn_kwargs=[dict(x=x, **kw) for x, kw in zip(xs_tr, tr_enc_kwargs)],
        synth_fn_kwargs=tr_dec_kwargs,
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


def test_split_warmup_loaded_from_dir_matches_full_warmup(tmp_path) -> None:
    n_samples, n_theta, bs = 2048, 2, 3
    config = dict(shape=n_samples, J=3, Q1=1, Q2=1, J_fr=1, Q_fr=1)
    min_prob_frac = 0.05
    setup = setup_toy_models_and_data(
        seed=42, n_samples=n_samples, n_theta=n_theta, bs=bs
    )
    loss = JaxSCRAPLLoss(**config)
    enc_params, enc_static = eqx.partition(setup.encoder_jax, eqx.is_inexact_array)
    args = (
        loss,
        lambda p, x: jax.vmap(eqx.combine(p, enc_static))(x[:, 0, :]),
        lambda theta: jax.vmap(setup.decoder_jax)(theta)[:, None, :],
        setup.x_jax[None],
        enc_params,
    )
    kwargs = dict(key=setup.loop_key, n_iter=20, tol=0.0, min_prob_frac=min_prob_frac)

    full_dir = str(tmp_path / "full")
    full = warmup_lc_hvp(*args, save_dir=full_dir, **kwargs)
    np.testing.assert_allclose(np.load(os.path.join(full_dir, "probs.npy")), full.probs)

    # 1. A partial run leaves the other paths NaN and saves no probabilities
    split_dir = str(tmp_path / "split")
    mid = loss.n_paths // 2
    first = warmup_lc_hvp(*args, end_path_idx=mid, save_dir=split_dir, **kwargs)
    assert np.isfinite(first.curvatures[:mid]).all()
    assert np.isnan(first.curvatures[mid:]).all()
    assert np.isnan(first.probs).all()
    assert not os.path.exists(os.path.join(split_dir, "probs.npy"))
    with pytest.raises(AssertionError, match="missing"):
        load_warmup_from_dir(loss, split_dir, min_prob_frac=min_prob_frac)

    # 2. The combined split runs match the full run
    warmup_lc_hvp(*args, start_path_idx=mid, save_dir=split_dir, **kwargs)
    loaded = load_warmup_from_dir(loss, split_dir, min_prob_frac=min_prob_frac)
    np.testing.assert_allclose(loaded.curvatures, full.curvatures, rtol=1e-6)
    np.testing.assert_allclose(
        loaded.relative_residuals, full.relative_residuals, rtol=1e-6, atol=1e-7
    )
    np.testing.assert_allclose(loaded.probs, full.probs, rtol=1e-6)

    # 3. Existing results are never overwritten
    with pytest.raises(AssertionError, match="already exists"):
        warmup_lc_hvp(*args, end_path_idx=1, save_dir=split_dir, **kwargs)
