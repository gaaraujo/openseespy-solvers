"""Preconditioner factories for the cupy backend.

These callables are intended for the ``M`` argument of :func:`cg`,
:func:`gmres`, and :func:`lobpcg`. Each factory accepts the assembled sparse
matrix from OpenSees and returns a preconditioner on device.
"""

from __future__ import annotations

from typing import Any

from openseespy_solvers._base import LinearSolver
from openseespy_solvers._factorization import apply_inner_factorization
from openseespy_solvers.cupy._base import _import_cupy

__all__ = ["jacobi", "ilu", "direct"]


def _linear_operator_with_matmat(
    cspla: Any,
    n: int,
    dtype: Any,
    matvec: Any,
    matmat: Any,
) -> Any:
    """LOBPCG applies preconditioners to block vectors via ``matmat``."""
    return cspla.LinearOperator((n, n), matvec=matvec, matmat=matmat, dtype=dtype)


def jacobi(A: Any) -> Any:
    """Return a Jacobi (diagonal) preconditioner on GPU.

    Computes ``M = diag(1 / diag(A))``. Zero diagonal entries are left as ``1``.

    Parameters
    ----------
    A : cupyx.scipy.sparse.spmatrix
        System matrix assembled by OpenSees on device.

    Returns
    -------
    M : cupyx.scipy.sparse.linalg.LinearOperator
        Diagonal preconditioner (``y = inv * x`` on device) for ``M=`` in
        :func:`~openseespy_solvers.cupy.cg`.

    See Also
    --------
    direct
    ilu
    openseespy_solvers.cupy.cg
    openseespy_solvers.cupy.lobpcg
    openseespy_solvers.scipy.precond.jacobi

    Examples
    --------
    >>> from openseespy_solvers.cupy import cg, lobpcg, precond
    >>> cg(M=precond.jacobi)
    >>> lobpcg(M=precond.jacobi)
    """
    cp, _csp, cspla = _import_cupy()
    diag = A.diagonal()
    inv = cp.ones_like(diag)
    mask = diag != 0
    inv[mask] = 1.0 / diag[mask]

    dtype = A.dtype
    n = A.shape[0]

    def matvec(x: Any) -> Any:
        arr = cp.asarray(x, dtype=dtype)
        if arr.ndim == 1:
            return inv * arr
        return inv[:, None] * arr

    def matmat(X: Any) -> Any:
        return inv[:, None] * cp.asarray(X, dtype=dtype)

    return _linear_operator_with_matmat(cspla, n, dtype, matvec, matmat)


def ilu(A: Any, **opts: Any) -> Any:
    """Return an incomplete LU preconditioner as a ``LinearOperator`` on GPU.

    By default ``fill_factor=1``, which triggers cupy's GPU ILU path with no
    fill-in and no pivoting. See
    :func:`cupyx.scipy.sparse.linalg.spilu` — only ``fill_factor=1`` runs the
    factorization on GPU; other settings delegate to `scipy.sparse.linalg` on CPU.

    Parameters
    ----------
    A : cupyx.scipy.sparse.spmatrix
        System matrix assembled by OpenSees on device.
    **opts
        Keyword arguments forwarded to
        :func:`cupyx.scipy.sparse.linalg.spilu`. ``fill_factor`` defaults to
        ``1`` when omitted.

    Returns
    -------
    M : cupyx.scipy.sparse.linalg.LinearOperator
        Preconditioner implementing ``M @ x`` via ``solve`` on the factorization.

    See Also
    --------
    cupyx.scipy.sparse.linalg.spilu
    jacobi
    direct
    openseespy_solvers.scipy.precond.ilu

    Notes
    -----
    With ``fill_factor=1``, factorization and application stay on the GPU but
    the pattern is restricted to the sparsity structure of ``A`` (ILU(0)-like,
    no fill-in).

    Examples
    --------
    >>> from openseespy_solvers.cupy import cg, precond
    >>> solver = cg(M=precond.ilu)
    """
    cp, _csp, cspla = _import_cupy()
    kwargs = dict(opts)
    kwargs.setdefault("fill_factor", 1)
    ilu_factor = cspla.spilu(A.tocsc(), **kwargs)

    dtype = A.dtype
    n = A.shape[0]

    def matvec(x: Any) -> Any:
        arr = cp.asarray(x, dtype=dtype)
        if arr.ndim == 1:
            return ilu_factor.solve(arr)
        return cp.column_stack([ilu_factor.solve(arr[:, j]) for j in range(arr.shape[1])])

    def matmat(X: Any) -> Any:
        X = cp.asarray(X, dtype=dtype)
        return cp.column_stack([ilu_factor.solve(X[:, j]) for j in range(X.shape[1])])

    return _linear_operator_with_matmat(cspla, n, dtype, matvec, matmat)


def direct(solver: LinearSolver):
    """Return a callable ``M(A)`` that approximates ``A^{-1}`` via a direct solver.

    Intended primarily for :func:`~openseespy_solvers.cupy.lobpcg` as ``M=`` on
    stiffness ``K``. The first application factors ``A``; later applications in
    the same eigen solve reuse the factorization when the matrix is unchanged.

    Parameters
    ----------
    solver : LinearSolver
        Direct solver for ``A x = b``, for example
        :func:`~openseespy_solvers.nvmath.direct_solver` or
        :func:`~openseespy_solvers.cupy.spsolve`.

    Returns
    -------
    preconditioner : callable
        Function ``M(A)`` returning a ``LinearOperator`` suitable for ``M=``.

    See Also
    --------
    jacobi
    ilu
    openseespy_solvers.nvmath.direct_solver
    openseespy_solvers.cupy.lobpcg
    openseespy_solvers.hybrid

    Examples
    --------
    >>> from openseespy_solvers.cupy import lobpcg, precond, spsolve
    >>> eigsolver = lobpcg(M=precond.direct(spsolve()), tol=1e-8)
    """
    if not isinstance(solver, LinearSolver):
        raise TypeError("precond.direct() requires a LinearSolver instance.")

    cp, _csp, cspla = _import_cupy()

    def preconditioner(A: Any) -> Any:
        needs_refactor = True
        dtype = A.dtype
        n = A.shape[0]

        def _apply_column(rhs: Any) -> Any:
            nonlocal needs_refactor
            result = apply_inner_factorization(
                solver,
                A,
                rhs,
                refactor=needs_refactor,
                on_device=False,
                # Each preconditioner is built for a new matrix, whose pattern
                # may differ from the one the inner solver analysed last.
                structure_changed=needs_refactor,
            )
            needs_refactor = False
            return cp.asarray(result, dtype=dtype)

        def matvec(x: Any) -> Any:
            arr = cp.asarray(x, dtype=dtype)
            if arr.ndim == 1:
                return _apply_column(arr)
            return cp.column_stack([_apply_column(arr[:, j]) for j in range(arr.shape[1])])

        def matmat(X: Any) -> Any:
            X = cp.asarray(X, dtype=dtype)
            return cp.column_stack([_apply_column(X[:, j]) for j in range(X.shape[1])])

        return _linear_operator_with_matmat(cspla, n, dtype, matvec, matmat)

    return preconditioner
