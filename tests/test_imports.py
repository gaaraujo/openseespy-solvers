"""Package imports, namespace exports, and backend attributes."""

from __future__ import annotations

import importlib

import pytest

import openseespy_solvers
from openseespy_solvers import scipy as scipy_ns
from openseespy_solvers.exceptions import BackendNotAvailableError


def test_version() -> None:
    assert openseespy_solvers.__version__ == "0.2.0"


def test_scipy_namespace_exports() -> None:
    assert set(scipy_ns.__all__) == {"cg", "gmres", "spsolve", "umfpack", "eigsh", "lobpcg"}


def test_lazy_scipy_attribute() -> None:
    assert openseespy_solvers.scipy is not None


def test_scipy_solver_backend_attr() -> None:
    assert scipy_ns.cg().backend == "scipy"
    assert scipy_ns.eigsh().backend == "scipy"
    assert scipy_ns.cg()._on_device is False


def test_unknown_namespace_raises() -> None:
    with pytest.raises(AttributeError):
        openseespy_solvers.unknown_backend  # noqa: B018


def test_cupy_namespace() -> None:
    try:
        import cupy  # noqa: F401
    except ImportError:
        with pytest.raises(BackendNotAvailableError):
            importlib.import_module("openseespy_solvers.cupy")
    else:
        cupy_ns = importlib.import_module("openseespy_solvers.cupy")
        assert set(cupy_ns.__all__) == {"cg", "gmres", "spsolve", "eigsh", "lobpcg"}
        assert cupy_ns.cg().backend == "cupy"
        assert cupy_ns.cg()._on_device is True
        assert cupy_ns.eigsh().backend == "cupy"


def test_hybrid_export() -> None:
    from openseespy_solvers import hybrid

    solver = hybrid(scipy_ns.spsolve())
    assert solver.backend == "scipy"
    assert solver._on_device is False


def test_nvmath_namespace() -> None:
    nvmath_ns = importlib.import_module("openseespy_solvers.nvmath")
    assert nvmath_ns.__all__ == ["direct_solver"]


def test_nvmath_direct_solver_requires_package() -> None:
    nvmath_ns = importlib.import_module("openseespy_solvers.nvmath")
    pytest.importorskip("nvmath")
    solver = nvmath_ns.direct_solver(device="cpu")
    assert solver.backend == "nvmath"
