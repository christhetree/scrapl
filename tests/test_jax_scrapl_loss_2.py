import numpy as np
import pytest
from tqdm import tqdm

from scrapl._dependencies import require_backend

try:
    tr = require_backend("torch")
    jax = require_backend("jax")
except ModuleNotFoundError as e:
    pytest.skip(str(e), allow_module_level=True)


@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize("n_paths", [2])
def test_scrapl_loss(seed: int, n_paths: int) -> None:
    import equinox as eqx
    import jax.numpy as jnp
    from scrapl import SCRAPLLoss as TorchSCRAPLLoss
    from scrapl.jax import SCRAPLLoss as JaxSCRAPLLoss

    n_samples = 48000
    config = dict(
        shape=n_samples,
        J=12,
        Q1=8,
        Q2=2,
        J_fr=3,
        Q_fr=2,
        use_rho_log1p=True,
    )

    torch_loss = TorchSCRAPLLoss(**config)
    jax_loss = JaxSCRAPLLoss(**config)

    assert jax_loss.scrapl_keys == tuple(torch_loss.scrapl_keys)
    assert jax_loss.n_paths == torch_loss.n_paths

    tr.manual_seed(seed)
    x = tr.rand((2, 1, n_samples))  # Batch of 2 mono audio samples
    x_target = tr.rand((2, 1, n_samples))  # Batch of 2 mono audio samples

    x_jax = jnp.asarray(x.numpy())
    x_target_jax = jnp.asarray(x_target.numpy())

    rng = np.random.default_rng(seed)
    path_indices = rng.choice(jax_loss.n_paths, size=n_paths, replace=False)

    @eqx.filter_jit
    def evaluate_jax(x_in, target_in, p_idx):
        return eqx.filter_value_and_grad(jax_loss)(x_in, target_in, path_idx=p_idx)

    pbar = tqdm(path_indices, desc="Testing SCRAPL paths")
    for path_idx in pbar:
        path_idx_int = int(path_idx)
        pbar.set_postfix(path_idx=path_idx_int)

        # PyTorch forward and backward for x
        x_torch = x.clone().detach().requires_grad_(True)
        torch_dist = torch_loss(x_torch, x_target, path_idx=path_idx_int)
        (torch_grad_x,) = tr.autograd.grad(torch_dist, x_torch)

        # JAX forward and gradient for x using Equinox
        jax_dist, jax_grad_x = evaluate_jax(x_jax, x_target_jax, path_idx_int)

        # Verify distance values match
        np.testing.assert_allclose(
            np.asarray(jax_dist),
            torch_dist.detach().numpy(),
            rtol=2e-5,
            atol=1e-7,
        )

        # Verify x gradients match
        np.testing.assert_allclose(
            np.asarray(jax_grad_x),
            torch_grad_x.detach().numpy(),
            rtol=5e-4,
            atol=2e-6,
        )
