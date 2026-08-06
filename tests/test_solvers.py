"""Linear solver tests (SciPy backend)."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from openseespy_solvers.scipy import cg, gmres, spsolve
from openseespy_solvers.scipy import precond

from conftest import csr_linear_kwargs, form_ap_kwargs


def test_spsolve_2x2() -> None:
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])
    x_expected = np.linalg.solve(A, b)
    x_buf = np.zeros(2, dtype=np.float64)
    kwargs = csr_linear_kwargs(A, b, x=x_buf)
    solver = spsolve()
    status = solver.solve(**kwargs)
    assert status == 0
    np.testing.assert_allclose(x_buf, x_expected, rtol=1e-10)
    assert sp.issparse(solver.A)


def test_cg_with_jacobi_preconditioner_factory() -> None:
    A = np.diag([4.0, 5.0, 6.0])
    b = np.array([1.0, 1.0, 1.0])
    x_buf = np.zeros(3, dtype=np.float64)
    kwargs = csr_linear_kwargs(A, b, x=x_buf)
    solver = cg(rtol=1e-12, M=precond.jacobi)
    assert solver.solve(**kwargs) == 0
    np.testing.assert_allclose(x_buf, b / np.diag(A), rtol=1e-8)


def test_matrix_status_caching() -> None:
    A = sp.diags([4.0, 5.0, 6.0])
    b = np.ones(3)
    x_buf = np.zeros(3, dtype=np.float64)
    solver = spsolve()

    kwargs = csr_linear_kwargs(A, b, x=x_buf, matrix_status="STRUCTURE_CHANGED")
    assert solver.solve(**kwargs) == 0
    first_A = solver.A

    kwargs2 = csr_linear_kwargs(A, b, x=x_buf, matrix_status="UNCHANGED")
    assert solver.solve(**kwargs2) == 0
    assert solver.A is first_A

    A2 = sp.diags([8.0, 10.0, 12.0])
    kwargs3 = csr_linear_kwargs(A2, b, x=x_buf, matrix_status="COEFFICIENTS_CHANGED")
    assert solver.solve(**kwargs3) == 0
    assert solver.A is first_A
    np.testing.assert_allclose(x_buf, b / np.array([8.0, 10.0, 12.0]), rtol=1e-10)


def test_form_ap() -> None:
    A = np.array([[2.0, 1.0], [1.0, 3.0]])
    p = np.array([1.0, 2.0])
    kwargs = form_ap_kwargs(A, p)
    ap = np.frombuffer(kwargs["Ap"], dtype=np.float64, count=2)
    solver = cg()
    assert solver.formAp(**kwargs) == 0
    np.testing.assert_allclose(ap, A @ p)


def test_gmres_nonsymmetric() -> None:
    A = np.array([[3.0, 1.0, 0.0], [0.0, 4.0, 1.0], [1.0, 0.0, 5.0]])
    b = np.array([1.0, 2.0, 3.0])
    x_expected = np.linalg.solve(A, b)
    x_buf = np.zeros(3, dtype=np.float64)
    kwargs = csr_linear_kwargs(A, b, x=x_buf)
    solver = gmres(rtol=1e-10)
    assert solver.solve(**kwargs) == 0
    np.testing.assert_allclose(x_buf, x_expected, rtol=1e-8)


def test_cupy_cg() -> None:
    pytest.importorskip("cupy")
    import importlib

    cupy_cg = importlib.import_module("openseespy_solvers.cupy").cg
    A = np.diag([4.0, 5.0, 6.0])
    b = np.array([1.0, 1.0, 1.0])
    x_buf = np.zeros(3, dtype=np.float64)
    kwargs = csr_linear_kwargs(A, b, x=x_buf)
    solver = cupy_cg(rtol=1e-8)
    assert solver.solve(**kwargs) == 0
    np.testing.assert_allclose(x_buf, b / np.diag(A), rtol=1e-5)


def test_cupy_spsolve_matrix_status_caching() -> None:
    pytest.importorskip("cupy")
    import scipy.sparse as sp

    from openseespy_solvers.cupy import spsolve as cupy_spsolve

    A = sp.diags([4.0, 5.0, 6.0])
    b = np.ones(3)
    x_buf = np.zeros(3, dtype=np.float64)
    solver = cupy_spsolve()

    kwargs = csr_linear_kwargs(A, b, x=x_buf, matrix_status="STRUCTURE_CHANGED")
    assert solver.solve(**kwargs) == 0
    first_A = solver.A
    first_solve = solver._solve_func

    kwargs2 = csr_linear_kwargs(A, b, x=x_buf, matrix_status="UNCHANGED")
    assert solver.solve(**kwargs2) == 0
    assert solver.A is first_A
    assert solver._solve_func is first_solve

    A2 = sp.diags([8.0, 10.0, 12.0])
    kwargs3 = csr_linear_kwargs(A2, b, x=x_buf, matrix_status="COEFFICIENTS_CHANGED")
    assert solver.solve(**kwargs3) == 0
    assert solver.A is first_A
    assert solver._solve_func is not first_solve
    np.testing.assert_allclose(x_buf, b / np.array([8.0, 10.0, 12.0]), rtol=1e-5)


def test_cupy_cg_jacobi_preconditioner() -> None:
    pytest.importorskip("cupy")
    from openseespy_solvers.cupy import cg, precond

    A = np.diag([4.0, 5.0, 6.0])
    b = np.array([1.0, 1.0, 1.0])
    x_buf = np.zeros(3, dtype=np.float64)
    kwargs = csr_linear_kwargs(A, b, x=x_buf)
    solver = cg(rtol=1e-8, M=precond.jacobi)
    assert solver.solve(**kwargs) == 0
    np.testing.assert_allclose(x_buf, b / np.diag(A), rtol=1e-5)


def test_cupy_ilu_defaults_fill_factor_one() -> None:
    pytest.importorskip("cupy")
    import cupy as cp
    import cupyx.scipy.sparse as csp

    from openseespy_solvers.cupy.precond import ilu

    A = csp.csr_matrix(cp.array([[4.0, 1.0], [1.0, 3.0]], dtype=cp.float64))
    M = ilu(A)
    x = cp.array([1.0, 2.0], dtype=cp.float64)
    y = M @ x
    assert y.shape == (2,)
    assert float(cp.linalg.norm(y)) > 0.0


def test_cg_float32_compute_dtype() -> None:
    A = np.diag([4.0, 5.0, 6.0])
    b = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    x_buf = np.zeros(3, dtype=np.float64)
    kwargs = csr_linear_kwargs(A, b, x=x_buf)
    solver = cg(rtol=1e-6, dtype=np.float32)
    assert solver.solve(**kwargs) == 0
    assert solver.A.dtype == np.float32
    np.testing.assert_allclose(x_buf, b / np.diag(A), rtol=1e-5)


def test_hybrid_first_solve_factorizes() -> None:
    from openseespy_solvers import hybrid

    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])
    x_expected = np.linalg.solve(A, b)
    x_buf = np.zeros(2, dtype=np.float64)
    kwargs = csr_linear_kwargs(A, b, x=x_buf, matrix_status="STRUCTURE_CHANGED")
    solver = hybrid(spsolve(), record_stats=True)
    assert solver.solve(**kwargs) == 0
    np.testing.assert_allclose(x_buf, x_expected, rtol=1e-10)
    assert solver.stats.num_factorizations == 1
    assert solver.stats.num_gmres_solves == 0


def test_hybrid_reuses_factorization_with_gmres() -> None:
    from openseespy_solvers import hybrid

    A = sp.csr_matrix(np.array([[4.0, 1.0], [1.0, 3.0]]))
    b = np.array([1.0, 2.0])
    x_buf = np.zeros(2, dtype=np.float64)
    inner = spsolve()
    solver = hybrid(inner, rtol=1e-12, record_stats=True)

    kwargs1 = csr_linear_kwargs(A, b, x=x_buf, matrix_status="STRUCTURE_CHANGED")
    assert solver.solve(**kwargs1) == 0
    assert solver.stats.num_factorizations == 1

    A2 = sp.csr_matrix(np.array([[4.1, 1.0], [1.0, 3.1]]))
    kwargs2 = csr_linear_kwargs(A2, b, x=x_buf, matrix_status="COEFFICIENTS_CHANGED")
    assert solver.solve(**kwargs2) == 0
    assert solver.stats.num_factorizations == 1
    assert solver.stats.num_gmres_solves == 1
    np.testing.assert_allclose(x_buf, np.linalg.solve(A2.toarray(), b), rtol=1e-8)


def test_hybrid_reuses_factorization_after_model_rebuild_same_size() -> None:
    """Frozen factorization survives wipe-style STRUCTURE_CHANGED when n is unchanged."""
    import copy

    from openseespy_solvers import hybrid

    A1 = sp.csr_matrix(np.array([[4.0, 1.0], [1.0, 3.0]]))
    b = np.array([1.0, 2.0])
    x_buf = np.zeros(2, dtype=np.float64)
    solver = hybrid(spsolve(), rtol=1e-12, record_stats=True)

    kwargs1 = csr_linear_kwargs(A1, b, x=x_buf, matrix_status="STRUCTURE_CHANGED")
    assert solver.solve(**kwargs1) == 0
    assert solver.stats.num_factorizations == 1

    # After wipe()/rebuild, OpenSees reports STRUCTURE_CHANGED even for the same mesh.
    A2 = sp.csr_matrix(np.array([[4.2, 0.9], [0.8, 3.2]]))
    kwargs2 = csr_linear_kwargs(A2, b, x=x_buf, matrix_status="STRUCTURE_CHANGED")
    assert solver.solve(**kwargs2) == 0
    assert solver.stats.num_factorizations == 1
    assert solver.stats.num_gmres_solves == 1
    np.testing.assert_allclose(x_buf, np.linalg.solve(A2.toarray(), b), rtol=1e-8)

    clone = copy.copy(solver)
    assert clone.stats.num_factorizations == 0
    clone.solve(**kwargs2)
    assert clone.stats.num_factorizations == 1


def test_scipy_precond_direct_callable() -> None:
    A = sp.diags([4.0, 5.0, 6.0]).tocsr()
    M_factory = precond.direct(spsolve())
    M = M_factory(A)
    x = np.array([1.0, 1.0, 1.0])
    y = M @ x
    np.testing.assert_allclose(y, x / np.array([4.0, 5.0, 6.0]), rtol=1e-10)


def test_unsupported_compute_dtype() -> None:
    from openseespy_solvers._dtype import resolve_compute_dtype
    from openseespy_solvers.exceptions import UnsupportedComputeDtypeError

    with pytest.raises(UnsupportedComputeDtypeError):
        resolve_compute_dtype(np.float16)
    with pytest.raises(UnsupportedComputeDtypeError):
        cg(dtype=np.complex128)


def test_residual_opt_in_record_stats() -> None:
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])
    x_buf = np.zeros(2, dtype=np.float64)
    kwargs = csr_linear_kwargs(A, b, x=x_buf)

    default = spsolve()
    assert default.solve(**kwargs) == 0
    assert default.stats.num_solves == 0
    assert default.stats.last_residual_norm is None
    assert default.stats.last_solve_time is None

    stats_only = spsolve(record_stats=True)
    x_buf2 = np.zeros(2, dtype=np.float64)
    assert stats_only.solve(**csr_linear_kwargs(A, b, x=x_buf2)) == 0
    assert stats_only.stats.num_solves == 1
    assert stats_only.stats.last_solve_time is not None
    assert stats_only.stats.last_residual_norm is None

    debug_only = spsolve(debug=True)
    x_buf3 = np.zeros(2, dtype=np.float64)
    assert debug_only.solve(**csr_linear_kwargs(A, b, x=x_buf3)) == 0
    assert debug_only.stats.num_solves == 0
    assert debug_only.stats.last_residual_norm is None

    both = spsolve(record_stats=True, debug=True)
    x_buf4 = np.zeros(2, dtype=np.float64)
    assert both.solve(**csr_linear_kwargs(A, b, x=x_buf4)) == 0
    residual = both.stats.last_residual_norm
    assert both.stats.num_solves == 1
    assert residual is not None
    assert np.isfinite(residual)
    assert residual < 1e-10
