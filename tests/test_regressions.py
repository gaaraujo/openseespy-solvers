"""Regression tests for silent wrong-answer bugs."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from conftest import csc_linear_kwargs, csr_linear_kwargs

from openseespy_solvers import hybrid
from openseespy_solvers.scipy import spsolve

# Nonsymmetric 4x4 with a full diagonal plus off-diagonal entries.
_A1 = np.array(
    [
        [4.0, 1.0, 0.0, 0.0],
        [0.0, 5.0, 2.0, 0.0],
        [0.0, 0.0, 6.0, 3.0],
        [1.0, 0.0, 0.0, 7.0],
    ]
)
# Same pattern as _A1, different nonsymmetric coefficients.
_A2 = np.array(
    [
        [9.0, -2.0, 0.0, 0.0],
        [0.0, 3.0, 0.5, 0.0],
        [0.0, 0.0, 8.0, -1.0],
        [4.0, 0.0, 0.0, 2.0],
    ]
)
# Same size and nnz as _A1, different pattern (transpose).
_A3 = 2.0 * _A1.T + np.diag([1.0, 2.0, 3.0, 4.0])
_B = np.array([1.0, -2.0, 3.0, 0.5])


def _nvmath_devices() -> list[str]:
    pytest.importorskip("nvmath")
    devices = ["cpu"]
    try:
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() > 0:
            devices.append("gpu")
    except Exception:  # pragma: no cover - no GPU
        pass
    return devices


@pytest.mark.parametrize("device", ["cpu", "gpu"])
def test_nvmath_csc_coefficient_update_nonsymmetric(device: str) -> None:
    if device not in _nvmath_devices():
        pytest.skip("no GPU")
    from openseespy_solvers.nvmath import direct_solver

    solver = direct_solver(device=device, scheme="CSC")
    x = np.zeros(4)
    assert solver.solve(**csc_linear_kwargs(_A1, _B, x=x)) == 0
    np.testing.assert_allclose(x, np.linalg.solve(_A1, _B), rtol=1e-10)

    kw = csc_linear_kwargs(_A2, _B, x=x, matrix_status="COEFFICIENTS_CHANGED")
    assert solver.solve(**kw) == 0
    np.testing.assert_allclose(x, np.linalg.solve(_A2, _B), rtol=1e-10)


def _hybrid_inners() -> list:
    inners = [("spsolve", spsolve)]
    try:
        from openseespy_solvers.scipy import umfpack

        umfpack()
        inners.append(("umfpack", umfpack))
    except ImportError:
        pass
    try:
        from openseespy_solvers.nvmath import direct_solver

        if "gpu" in _nvmath_devices():
            inners.append(("nvmath-gpu", lambda: direct_solver(device="gpu")))
        inners.append(("nvmath-cpu", lambda: direct_solver(device="cpu")))
    except Exception:  # pragma: no cover
        pass
    return inners


def _inner_id(value) -> str:  # noqa: ANN001
    return value if isinstance(value, str) else ""


@pytest.mark.parametrize("name,factory", _hybrid_inners(), ids=_inner_id)
@pytest.mark.parametrize("make_kwargs", [csr_linear_kwargs, csc_linear_kwargs], ids=["CSR", "CSC"])
def test_hybrid_fallback_honours_structure_change(name, factory, make_kwargs) -> None:
    scheme = "CSC" if make_kwargs is csc_linear_kwargs else "CSR"
    inner = factory()
    inner.scheme = scheme
    # One GMRES iteration cannot converge to 1e-14 with a stale preconditioner,
    # so the fallback refactorization runs on the new pattern.
    solver = hybrid(inner, rtol=1e-14, restart=1, maxiter=1)
    x = np.zeros(4)
    assert solver.solve(**make_kwargs(_A1, _B, x=x)) == 0
    np.testing.assert_allclose(x, np.linalg.solve(_A1, _B), rtol=1e-10)

    assert sp.csr_matrix(_A3).nnz == sp.csr_matrix(_A1).nnz
    assert solver.solve(**make_kwargs(_A3, _B, x=x, matrix_status="STRUCTURE_CHANGED")) == 0
    np.testing.assert_allclose(x, np.linalg.solve(_A3, _B), rtol=1e-10)

    # A later coefficient change on the new pattern must also be right.
    A4 = _A3 + np.diag([1.0, 1.0, 1.0, 1.0])
    assert solver.solve(**make_kwargs(A4, _B, x=x, matrix_status="COEFFICIENTS_CHANGED")) == 0
    np.testing.assert_allclose(x, np.linalg.solve(A4, _B), rtol=1e-10)


@pytest.mark.parametrize("make_kwargs", [csr_linear_kwargs, csc_linear_kwargs], ids=["CSR", "CSC"])
def test_nvmath_gpu_refactorizes_on_coefficient_change(make_kwargs) -> None:
    """GPU operands: a coefficient change must refactorize, not fail on a lost plan."""
    if "gpu" not in _nvmath_devices():
        pytest.skip("no GPU")
    from openseespy_solvers.nvmath import direct_solver

    scheme = "CSC" if make_kwargs is csc_linear_kwargs else "CSR"
    solver = direct_solver(device="gpu", scheme=scheme)
    x = np.zeros(4)
    assert solver.solve(**make_kwargs(_A1, _B, x=x)) == 0
    for k in range(3):
        A = _A2 + k * np.eye(4)
        assert solver.solve(**make_kwargs(A, _B, x=x, matrix_status="COEFFICIENTS_CHANGED")) == 0
        np.testing.assert_allclose(x, np.linalg.solve(A, _B), rtol=1e-10)
        assert solver.solve(**make_kwargs(A, 2 * _B, x=x, matrix_status="UNCHANGED")) == 0
        np.testing.assert_allclose(x, np.linalg.solve(A, 2 * _B), rtol=1e-10)


@pytest.mark.parametrize("name,factory", _hybrid_inners(), ids=_inner_id)
def test_hybrid_gmres_path_converges(name, factory) -> None:
    """The frozen factorization preconditions GMRES on a coefficient change."""
    solver = hybrid(factory(), rtol=1e-12, restart=20, maxiter=50, record_stats=True)
    x = np.zeros(4)
    assert solver.solve(**csr_linear_kwargs(_A1, _B, x=x)) == 0
    A = _A1 + 0.05 * _A2
    assert solver.solve(**csr_linear_kwargs(A, _B, x=x, matrix_status="COEFFICIENTS_CHANGED")) == 0
    np.testing.assert_allclose(x, np.linalg.solve(A, _B), rtol=1e-9)
    assert solver.stats.num_gmres_solves == 1
    assert solver.stats.num_factorizations == 1


def test_residual_check_is_opt_in() -> None:
    x = np.zeros(4)
    quiet = spsolve(record_stats=True)
    assert quiet.solve(**csr_linear_kwargs(_A1, _B, x=x)) == 0
    assert quiet.stats.num_solves == 1
    assert quiet.stats.last_residual_norm is None
    checked = spsolve(record_stats=True, debug=True)
    assert checked.solve(**csr_linear_kwargs(_A1, _B, x=x)) == 0
    assert checked.stats.last_residual_norm < 1e-12
