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


def test_base_imports_do_not_load_a_framework():
    run_without(
        "torch,jax,hessian_eigenthings",
        """
        import scrapl
        from scrapl.single_path_jtfs import TimeFrequencyScrapl

        assert 'SCRAPLLoss' in dir(scrapl)
        assert not hasattr(scrapl, 'unknown_attribute')
        assert not blocked.intersection(sys.modules)
    """,
    )


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_missing_framework_reports_the_required_extra(backend):
    run_without(
        backend,
        f"""
        import importlib
        from scrapl.single_path_jtfs import TimeFrequencyScrapl

        def import_loss():
            if {backend!r} == 'torch':
                from scrapl import SCRAPLLoss
            else:
                from scrapl.jax import SCRAPLLoss

        for action in (
            import_loss,
            lambda: importlib.import_module('scrapl.single_path_jtfs.{backend}'),
            lambda: TimeFrequencyScrapl(backend={backend!r}),
        ):
            try:
                action()
            except ModuleNotFoundError as error:
                assert error.name == {backend!r}
                assert 'scrapl-loss[{backend}]' in str(error)
            else:
                raise AssertionError('Missing backend should fail on use')
    """,
    )


def test_jax_loss_and_transformations_work_without_torch():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed")
    run_without(
        "torch,hessian_eigenthings",
        """
        import jax
        import jax.numpy as jnp
        from scrapl.jax import PAdam, PSAGA, SCRAPLLoss, warmup_lc_hvp
        from scrapl.single_path_jtfs import TimeFrequencyScrapl

        config = dict(shape=128, J=3, Q1=2, Q2=1, J_fr=1, Q_fr=1, use_rho_log1p=True)
        loss = SCRAPLLoss(**config)
        target = jax.random.normal(jax.random.key(1), (1, 1, 128))
        value, gradient = jax.jit(jax.value_and_grad(
            lambda gain: loss(gain * target, target, path_idx=jnp.asarray(0))
        ))(jnp.asarray(0.5))
        assert jnp.isfinite(value) and value > 0
        assert jnp.isfinite(gradient) and gradient != 0

        adam, saga = PAdam(loss.n_paths), PSAGA(loss.n_paths)
        direction, _ = jax.jit(adam.update)(gradient, adam.init(gradient), path_idx=0)
        direction, state = jax.jit(saga.update)(direction, saga.init(gradient), path_idx=0)
        assert jnp.isfinite(direction) and state.seen[0]
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
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")
    run_without(
        "jax",
        """
        import torch
        import scrapl
        from scrapl import SCRAPLLoss
        from scrapl.scrapl_loss import SCRAPLLoss as DirectLoss
        from scrapl.single_path_jtfs.torch import TimeFrequencyScrapl

        assert SCRAPLLoss is DirectLoss is scrapl.SCRAPLLoss
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
