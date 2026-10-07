import importlib.util
import os
import subprocess
import sys
import textwrap
from importlib.metadata import requires

import pytest
from packaging.requirements import Requirement

IMPORT_BLOCKER = """
import importlib.abc
import sys

blocked = set(sys.argv[1].split(','))

class BlockFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        name = fullname.split('.')[0]
        if name in blocked:
            raise ModuleNotFoundError(f'Blocked optional dependency: {name}', name=name)

sys.meta_path.insert(0, BlockFrameworks())
"""


def is_backend_installed(name):
    spec = importlib.util.find_spec(name)
    return spec is not None and spec.origin is not None


def run_without(blocked, script):
    result = subprocess.run(
        [sys.executable, "-c", IMPORT_BLOCKER + textwrap.dedent(script), blocked],
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("extra", ["", "torch", "jax", "test"])
def test_installed_metadata_keeps_framework_dependencies_independent(extra):
    active = set()
    for entry in requires("scrapl-loss"):
        requirement = Requirement(entry)
        if requirement.marker is None or requirement.marker.evaluate({"extra": extra}):
            active.add(requirement.name)
    assert ("torch" in active) == (extra == "torch")
    assert ("jax" in active) == (extra == "jax")
    assert ("pytest" in active) == (extra == "test")


def test_base_imports_do_not_load_a_framework():
    run_without(
        "torch,jax,scipy,hessian_eigenthings",
        """
        import sys
        import scrapl
        from scrapl import single_path_jtfs

        loaded = set(sys.modules)
        assert not blocked.intersection(loaded)
        assert "scrapl.scrapl_loss" not in loaded
        assert "scrapl.torch" not in loaded
        assert "scrapl.single_path_jtfs.torch" not in loaded
        assert "scrapl.single_path_jtfs.jax" not in loaded
        assert "scrapl.jax" not in loaded
        assert "SCRAPLLoss" in dir(scrapl)
        assert not hasattr(scrapl, "unknown_attribute")
    """,
    )


@pytest.mark.parametrize(
    "backend, expression",
    [
        ("torch", "from scrapl import SCRAPLLoss"),
        ("torch", "import scrapl; scrapl.SCRAPLLoss"),
        ("torch", "import scrapl.torch"),
        ("torch", "from scrapl.torch import SCRAPLLoss"),
        ("torch", "import scrapl.scrapl_loss"),
        ("torch", "import scrapl.single_path_jtfs.torch"),
        ("jax", "import scrapl.jax"),
        ("jax", "from scrapl.jax import SCRAPLLoss"),
        ("jax", "import scrapl.single_path_jtfs.jax"),
    ],
)
def test_missing_framework_reports_the_required_extra(backend, expression):
    run_without(
        backend,
        f"""
        import sys
        import pytest
        try:
            {expression}
        except ModuleNotFoundError as error:
            assert error.name == {backend!r}
            assert 'scrapl-loss[{backend}]' in str(error)
        else:
            raise AssertionError('Missing backend should fail on use')
    """,
    )


def test_jax_loss_and_transformations_work_without_torch():
    if not is_backend_installed("jax"):
        pytest.skip("JAX is not installed")
    run_without(
        "torch,hessian_eigenthings",
        """
        import jax
        import jax.numpy as jnp
        from scrapl.jax import SCRAPLLoss, p_adam, p_saga, warmup_lc_hvp
        from scrapl.single_path_jtfs import TimeFrequencyScrapl

        config = dict(shape=128, J=3, Q1=2, Q2=1, J_fr=1, Q_fr=1, use_rho_log1p=True)
        loss = SCRAPLLoss(**config)
        target = jax.random.normal(jax.random.key(1), (1, 1, 128))
        value, gradient = jax.jit(jax.value_and_grad(
            lambda gain: loss(gain * target, target, path_idx=jnp.asarray(0))
        ))(jnp.asarray(0.5))
        assert jnp.isfinite(value) and value > 0
        assert jnp.isfinite(gradient) and gradient != 0

        adam, saga = p_adam(loss.n_paths), p_saga(loss.n_paths)
        direction, _ = jax.jit(adam.update)(gradient, adam.init(gradient), path_idx=0)
        direction, state = jax.jit(saga.update)(direction, saga.init(gradient), path_idx=0)
        assert jnp.isfinite(direction) and (state.path_counts > 0)[0]
        transform = TimeFrequencyScrapl(
            shape=128, J=3, Q=(2, 1), J_fr=1, Q_fr=1, backend='jax'
        )
        coefficients = transform.scattering_singlepath(target, *loss.scrapl_keys[0])['coef']
        assert jnp.isfinite(coefficients).all()

        result = warmup_lc_hvp(
            loss, jnp.asarray(0.5),
            lambda gain, batch: jnp.broadcast_to(gain, (batch.shape[0], 1)),
            lambda controls: controls[..., None] * target,
            target[None], key=jax.random.key(2), n_iter=2,
        )
        assert jnp.isfinite(result.probs).all()
        assert not blocked.intersection(sys.modules)
    """,
    )


def test_torch_loss_and_legacy_import_work_without_jax():
    if not is_backend_installed("torch"):
        pytest.skip("PyTorch is not installed")
    run_without(
        "jax",
        """
        import torch
        import scrapl
        from scrapl import SCRAPLLoss
        from scrapl.torch import SCRAPLLoss as TorchLoss
        from scrapl.scrapl_loss import SCRAPLLoss as DirectLoss
        from scrapl.single_path_jtfs.torch import TimeFrequencyScrapl

        assert SCRAPLLoss is DirectLoss is scrapl.SCRAPLLoss is TorchLoss
        loss = SCRAPLLoss(
            shape=128, J=3, Q1=2, Q2=1, J_fr=1, Q_fr=1,
            use_rho_log1p=True, grad_mult=1,
        )
        target = torch.randn(1, 1, 128)
        gain = torch.nn.Parameter(torch.tensor(0.5))
        loss.attach_params([gain])
        value = loss(gain * target, target, path_idx=0)
        value.backward()
        assert torch.isfinite(value) and value > 0
        assert torch.isfinite(gain.grad) and gain.grad != 0
        transform = TimeFrequencyScrapl(shape=128, J=3, Q=(2, 1), J_fr=1, Q_fr=1)
        coefficients = transform.scattering_singlepath(target, *loss.scrapl_keys[0])['coef']
        assert torch.isfinite(coefficients).all()
        assert not blocked.intersection(sys.modules)
    """,
    )


def test_backend_dependency_errors_are_not_misreported(monkeypatch):
    from scrapl import _dependencies

    missing_transitive_dependency = ModuleNotFoundError("missing numpy", name="numpy")

    def fail(name):
        raise missing_transitive_dependency

    monkeypatch.setattr(_dependencies, "import_module", fail)
    with pytest.raises(ModuleNotFoundError) as caught:
        _dependencies.require_backend("torch")
    assert caught.value is missing_transitive_dependency
