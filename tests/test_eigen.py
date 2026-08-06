"""Eigen solver tests."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from openseespy_solvers.scipy import eigsh, lobpcg

from conftest import csr_eigen_kwargs, csc_eigen_kwargs


def test_eigsh_largest_diagonal() -> None:
    n = 6
    K = sp.diags(np.arange(2.0, 2.0 + n, 1.0))
    M = sp.eye(n)
    num_modes = 3
    kwargs = csr_eigen_kwargs(K, M, num_modes=num_modes, find_smallest=False)
    ev = np.frombuffer(kwargs["eigenvalues"], dtype=np.float64, count=num_modes)
    eigsh(tol=1e-10).solve(**kwargs)
    np.testing.assert_allclose(ev, [7.0, 6.0, 5.0], rtol=1e-6)


def test_cupy_eigsh_largest_general() -> None:
    pytest.importorskip("cupy")
    from openseespy_solvers.cupy import eigsh as cupy_eigsh

    n = 6
    K = sp.diags(np.arange(2.0, 2.0 + n, 1.0))
    M = sp.eye(n)
    num_modes = 3
    kwargs = csr_eigen_kwargs(K, M, num_modes=num_modes, find_smallest=False)
    ev = np.frombuffer(kwargs["eigenvalues"], dtype=np.float64, count=num_modes)
    cupy_eigsh(tol=1e-10).solve(**kwargs)
    np.testing.assert_allclose(ev, [7.0, 6.0, 5.0], rtol=1e-5)


def test_eigsh_diagonal() -> None:
    n = 6
    K = sp.diags(np.arange(2.0, 2.0 + n, 1.0))
    M = sp.eye(n)
    num_modes = 3
    kwargs = csr_eigen_kwargs(K, M, num_modes=num_modes, find_smallest=True)
    ev = np.frombuffer(kwargs["eigenvalues"], dtype=np.float64, count=num_modes)
    solver = eigsh(tol=1e-10)
    solver.solve(**kwargs)
    np.testing.assert_allclose(ev, [2.0, 3.0, 4.0], rtol=1e-6)
    assert sp.issparse(solver.K)
    assert sp.issparse(solver.M)


def test_eigsh_shift_invert_csc_scheme() -> None:
    """CSC assembly matches SciPy shift-invert LU (SuperLU/UMFPACK) preference."""
    n = 6
    K = sp.diags(np.arange(2.0, 2.0 + n, 1.0))
    M = sp.eye(n)
    num_modes = 3
    kwargs = csc_eigen_kwargs(K, M, num_modes=num_modes, find_smallest=True)
    ev = np.frombuffer(kwargs["eigenvalues"], dtype=np.float64, count=num_modes)
    solver = eigsh(tol=1e-10, scheme="CSC")
    assert solver.scheme == "CSC"
    solver.solve(**kwargs)
    np.testing.assert_allclose(ev, [2.0, 3.0, 4.0], rtol=1e-6)
    assert solver.K.format == "csc"
    assert solver.M.format == "csc"


def test_eigsh_to_openseespy() -> None:
    solver = eigsh()
    cfg = solver.to_openseespy()
    assert cfg["solver"] is solver
    assert "writable" not in cfg


def test_cupy_eigsh_diagonal() -> None:
    pytest.importorskip("cupy")
    from openseespy_solvers.cupy import eigsh as cupy_eigsh, spsolve as cupy_spsolve

    n = 6
    K = sp.diags(np.arange(2.0, 2.0 + n, 1.0))
    M = sp.eye(n)
    num_modes = 3
    kwargs = csr_eigen_kwargs(K, M, num_modes=num_modes, find_smallest=True)
    ev = np.frombuffer(kwargs["eigenvalues"], dtype=np.float64, count=num_modes)
    solver = cupy_eigsh(tol=1e-10, linear_solver=cupy_spsolve(), mass_mode="diagonal")
    solver.solve(**kwargs)
    np.testing.assert_allclose(ev, [2.0, 3.0, 4.0], rtol=1e-5)


def test_cupy_eigsh_generalized_spd() -> None:
    """Non-diagonal K: shift-invert via SciPy ARPACK + GPU OPinv inner solves."""
    pytest.importorskip("cupy")
    from openseespy_solvers.cupy import eigsh as cupy_eigsh, spsolve as cupy_spsolve

    n = 24
    rng = np.random.default_rng(0)
    r = sp.random(n, n, density=0.12, random_state=rng)
    K = (r.T @ r + 4.0 * sp.eye(n)).tocsr()
    # OpenSees uses the same CSR pattern for K and M buffers.
    M = sp.csr_matrix((np.ones(K.nnz), K.indices, K.indptr), shape=(n, n))
    num_modes = 3
    # ARPACK shift-invert is sensitive to the starting vector; pin v0 so this
    # test stays stable when other eigen tests run first in the same process.
    v0 = np.ones(n, dtype=np.float64)
    ref_kwargs = csr_eigen_kwargs(K, M, num_modes=num_modes, find_smallest=True)
    eigsh(tol=1e-10, v0=v0).solve(**ref_kwargs)
    ref_ev = np.frombuffer(
        ref_kwargs["eigenvalues"], dtype=np.float64, count=num_modes
    ).copy()

    kwargs = csr_eigen_kwargs(K, M, num_modes=num_modes, find_smallest=True)
    cupy_eigsh(tol=1e-10, linear_solver=cupy_spsolve(), v0=v0).solve(**kwargs)
    ev = np.frombuffer(kwargs["eigenvalues"], dtype=np.float64, count=num_modes).copy()
    np.testing.assert_allclose(np.sort(ev), np.sort(ref_ev), rtol=0.15)


def test_cupy_eigsh_diagonal_rejects_offdiag_mass() -> None:
    pytest.importorskip("cupy")
    from openseespy_solvers.cupy import eigsh as cupy_eigsh

    n = 8
    K = sp.diags(
        [np.full(n - 1, -1.0), np.full(n, 4.0), np.full(n - 1, -1.0)],
        offsets=[-1, 0, 1],
        format="csr",
    )
    M = sp.csr_matrix((np.ones(K.nnz), K.indices, K.indptr), shape=(n, n))
    kwargs = csr_eigen_kwargs(K, M, num_modes=2, find_smallest=True)
    with pytest.raises(ValueError, match="mass_mode='diagonal' requires a diagonal mass matrix"):
        cupy_eigsh(mass_mode="diagonal").solve(**kwargs)


def test_cupy_eigsh_lumped_allows_offdiag_mass() -> None:
    pytest.importorskip("cupy")
    from openseespy_solvers.cupy import eigsh as cupy_eigsh

    n = 8
    K = sp.diags(
        [np.full(n - 1, -1.0), np.full(n, 4.0), np.full(n - 1, -1.0)],
        offsets=[-1, 0, 1],
        format="csr",
    )
    M = sp.csr_matrix((np.ones(K.nnz), K.indices, K.indptr), shape=(n, n))
    kwargs = csr_eigen_kwargs(K, M, num_modes=2, find_smallest=True)
    ev = np.frombuffer(kwargs["eigenvalues"], dtype=np.float64, count=2)
    cupy_eigsh(mass_mode="lumped").solve(**kwargs)
    assert np.all(np.isfinite(ev))


@pytest.mark.parametrize("mass_mode", ["diagonal", "lumped"])
def test_cupy_eigsh_shift_invert_caches_on_unchanged(mass_mode: str) -> None:
    pytest.importorskip("cupy")
    from openseespy_solvers.cupy import eigsh as cupy_eigsh

    n = 12
    K = sp.diags(np.arange(2.0, 2.0 + n, 1.0))
    M = sp.eye(n)
    num_modes = 2
    solver = cupy_eigsh(mass_mode=mass_mode, tol=1e-10)
    kwargs = csr_eigen_kwargs(K, M, num_modes=num_modes, matrix_status="STRUCTURE_CHANGED")
    solver.solve(**kwargs)
    first_m = solver._diag_m
    first_shifted = solver._shifted

    kwargs2 = csr_eigen_kwargs(K, M, num_modes=num_modes, matrix_status="UNCHANGED")
    solver.solve(**kwargs2)
    assert solver._diag_m is first_m
    assert solver._shifted is first_shifted


def test_cupy_lobpcg_diagonal() -> None:
    pytest.importorskip("cupy")
    from openseespy_solvers.cupy import lobpcg as cupy_lobpcg

    n = 6
    K = sp.diags(2.0 * np.arange(1, n + 1, dtype=float))
    M = sp.eye(n)
    num_modes = 2
    kwargs = csr_eigen_kwargs(K, M, num_modes=num_modes, find_smallest=True)
    ev = np.frombuffer(kwargs["eigenvalues"], dtype=np.float64, count=num_modes)
    solver = cupy_lobpcg(maxiter=40, tol=1e-8, rng=0)
    solver.solve(**kwargs)
    np.testing.assert_allclose(ev, [2.0, 4.0], rtol=1e-5)


def test_cupy_lobpcg_jacobi_preconditioner() -> None:
    pytest.importorskip("cupy")
    from openseespy_solvers.cupy import lobpcg as cupy_lobpcg, precond

    n = 6
    K = sp.diags(2.0 * np.arange(1, n + 1, dtype=float))
    M = sp.eye(n)
    num_modes = 5
    kwargs = csr_eigen_kwargs(K, M, num_modes=num_modes, find_smallest=True)
    ev = np.frombuffer(kwargs["eigenvalues"], dtype=np.float64, count=num_modes)
    solver = cupy_lobpcg(M=precond.jacobi, maxiter=40, tol=1e-8, rng=0)
    solver.solve(**kwargs)
    np.testing.assert_allclose(ev[:2], [2.0, 4.0], rtol=1e-5)


def test_cupy_precond_jacobi_block_matmat() -> None:
    """LOBPCG applies preconditioners to dense blocks via matmat."""
    pytest.importorskip("cupy")
    import cupy as cp

    from openseespy_solvers.cupy import precond
    from openseespy_solvers.cupy._base import _import_cupy

    _cp, csp, _ = _import_cupy()
    n = 32
    K = csp.diags(_cp.arange(2.0, 2.0 + n, dtype=_cp.float64))
    op = precond.jacobi(K)
    block = _cp.random.standard_normal((n, 5), dtype=_cp.float64)
    applied = op @ block
    assert applied.shape == (n, 5)
    assert cp.isfinite(applied).all()


@pytest.mark.filterwarnings(
    "ignore:The problem size.*too small relative to the block size:UserWarning"
)
def test_lobpcg_diagonal() -> None:
    # n=6 is below SciPy's 5*num_modes threshold, so LOBPCG uses its dense fallback here.
    n = 6
    K = sp.diags(2.0 * np.arange(1, n + 1, dtype=float))
    M = sp.eye(n)
    num_modes = 2
    kwargs = csr_eigen_kwargs(K, M, num_modes=num_modes, find_smallest=True)
    ev = np.frombuffer(kwargs["eigenvalues"], dtype=np.float64, count=num_modes)
    solver = lobpcg(maxiter=40, tol=1e-8)
    solver.solve(**kwargs)
    np.testing.assert_allclose(ev, [2.0, 4.0], rtol=1e-5)
