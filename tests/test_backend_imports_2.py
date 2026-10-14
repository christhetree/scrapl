"""Checks that the torch and JAX backends can be imported independently of each other.

Each import runs in a fresh interpreter via _import_probe.py, because modules already
loaded in the pytest process cannot be unloaded or blocked.
"""

import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

PROBE = Path(__file__).with_name("_import_probe.py")

FRAMEWORK_FREE_IMPORTS = [
    "import scrapl",
    "import scrapl.single_path_jtfs",
    "from scrapl.single_path_jtfs import TimeFrequencyScrapl",
]
TORCH_IMPORTS = [
    "import scrapl.torch",
    "from scrapl.torch import SCRAPLLoss",
    "from scrapl import SCRAPLLoss",
    "import scrapl.scrapl_loss",
    "import scrapl.single_path_jtfs.torch",
]
JAX_IMPORTS = [
    "import scrapl.jax",
    "from scrapl.jax import SCRAPLLoss",
    "import scrapl.single_path_jtfs.jax",
]
BACKEND_IMPORTS = [("torch", s) for s in TORCH_IMPORTS] + [
    ("jax", s) for s in JAX_IMPORTS
]
OTHER_BACKEND = {"torch": "jax", "jax": "torch"}


@dataclass
class ImportResult:
    modules: set[str]
    error: dict[str, str] | None

    def loaded(self, package: str) -> bool:
        return any(m == package or m.startswith(f"{package}.") for m in self.modules)


def run_import(statement: str, block: list[str]) -> ImportResult:
    proc = subprocess.run(
        [sys.executable, str(PROBE), ",".join(block), statement],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout.strip().splitlines()[-1])
    return ImportResult(modules=set(report["modules"]), error=report["error"])


def skip_if_not_installed(backend: str) -> None:
    if importlib.util.find_spec(backend) is None:
        pytest.skip(f"{backend} is not installed")


@pytest.mark.parametrize("statement", FRAMEWORK_FREE_IMPORTS)
def test_base_package_loads_no_framework(statement):
    result = run_import(statement, block=["torch", "jax"])
    assert result.error is None
    assert not result.loaded("torch")
    assert not result.loaded("jax")


@pytest.mark.parametrize("backend, statement", BACKEND_IMPORTS)
def test_backend_imports_without_the_other_framework(backend, statement):
    skip_if_not_installed(backend)
    other = OTHER_BACKEND[backend]
    result = run_import(statement, block=[other])
    assert result.error is None
    assert result.loaded(backend)
    assert not result.loaded(other)


@pytest.mark.parametrize("backend, statement", BACKEND_IMPORTS)
def test_missing_backend_error_names_the_extra(backend, statement):
    result = run_import(statement, block=[backend])
    assert result.error is not None
    assert result.error["name"] == backend
    assert f"scrapl-loss[{backend}]" in result.error["message"]


def test_torch_loss_is_the_same_class_from_every_entry_point():
    skip_if_not_installed("torch")
    import scrapl
    import scrapl.scrapl_loss
    import scrapl.torch

    assert scrapl.SCRAPLLoss is scrapl.torch.SCRAPLLoss
    assert scrapl.SCRAPLLoss is scrapl.scrapl_loss.SCRAPLLoss


def test_lazy_top_level_attributes():
    import scrapl

    assert "SCRAPLLoss" in dir(scrapl)
    with pytest.raises(AttributeError):
        scrapl.unknown_attribute


def test_missing_transitive_dependency_is_not_reported_as_missing_backend(
    monkeypatch,
):
    from scrapl import _dependencies

    transitive_error = ModuleNotFoundError("No module named 'numpy'", name="numpy")

    def fail_import(name):
        raise transitive_error

    monkeypatch.setattr(_dependencies, "import_module", fail_import)
    with pytest.raises(ModuleNotFoundError) as caught:
        _dependencies.require_backend("torch")
    assert caught.value is transitive_error
