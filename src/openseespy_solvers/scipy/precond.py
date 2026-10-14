"""Preconditioner factories for the scipy backend.

These callables are intended for the ``M`` argument of :func:`cg`,
:func:`gmres`, and :func:`lobpcg`. Each factory accepts the assembled sparse
matrix ``A`` from OpenSees and returns a preconditioner object.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from openseespy_solvers._base import LinearSolver
from openseespy_solvers._factorization import apply_inner_factorization


def jacobi(A: sp.spmatrix) -> sp.spmatrix:
    """Return a Jacobi (diagonal) preconditioner.

    Computes ``M = diag(1 / diag(A))``. Zero diagonal entries are left as ``1``.

    Parameters
    ----------
    A : sparse matrix
        System matrix assembled by OpenSees.

    Returns
    -------
    M : scipy.sparse.spmatrix
        Diagonal preconditioner suitable for ``M=`` in :func:`~openseespy_solvers.scipy.cg`.

    See Also
    --------
    ilu
    direct
    openseespy_solvers.scipy.cg

    Examples
    --------
    >>> import scipy.sparse as sp
    >>> from openseespy_solvers.scipy import cg, precond
    >>> A = sp.diags([2.0, 3.0, 4.0])
    >>> M = precond.jacobi(A)
    >>> solver = cg(M=precond.jacobi)
    """
    diag = A.diagonal()
    inv = np.ones_like(diag, dtype=np.float64)
    nz = diag != 0.0
    inv[nz] = 1.0 / diag[nz]
    return sp.diags(inv)


def ilu(A: sp.spmatrix, **opts: Any) -> spla.LinearOperator:
    """Return an incomplete LU preconditioner as a ``LinearOperator``.

    Parameters
    ----------
    A : sparse matrix
        System matrix assembled by OpenSees.
    **opts
        Additional keyword arguments forwarded to
        :func:`scipy.sparse.linalg.spilu` (for example ``drop_tol``,
        ``fill_factor``).

    Returns
    -------
    M : scipy.sparse.linalg.LinearOperator
        Preconditioner implementing ``M @ x`` via forward/back substitution on
        the ILU factorization.

    See Also
    --------
    scipy.sparse.linalg.spilu
    jacobi
    direct

    Examples
    --------
    >>> from openseespy_solvers.scipy import cg, precond
    >>> solver = cg(M=lambda A: precond.ilu(A, drop_tol=1e-4))
    """
    ilu_factor = spla.spilu(A.tocsc(), **opts)

    def matvec(x: np.ndarray) -> np.ndarray:
        return ilu_factor.solve(x)

    n = A.shape[0]
    return spla.LinearOperator((n, n), matvec=matvec, dtype=A.dtype)


def direct(solver: LinearSolver):
    """Return a callable ``M(A)`` that approximates ``A^{-1}`` via a direct solver.

    Intended primarily for :func:`~openseespy_solvers.scipy.lobpcg` as ``M=``.
    The first application factors ``A``; later applications in the same eigen
    solve reuse the factorization when the matrix is unchanged.

    Parameters
    ----------
    solver : LinearSolver
        Direct solver for ``A x = b``, for example :func:`~openseespy_solvers.scipy.spsolve`
        or :func:`~openseespy_solvers.scipy.umfpack`.

    Returns
    -------
    preconditioner : callable
        Function ``M(A)`` returning a ``LinearOperator`` suitable for ``M=``.

    See Also
    --------
    jacobi
    ilu
    openseespy_solvers.scipy.lobpcg
    openseespy_solvers.hybrid

    Examples
    --------
    >>> from openseespy_solvers.scipy import lobpcg, precond, spsolve
    >>> eigsolver = lobpcg(M=precond.direct(spsolve()), tol=1e-8)
    """
    if not isinstance(solver, LinearSolver):
        raise TypeError("precond.direct() requires a LinearSolver instance.")

    def preconditioner(A: sp.spmatrix) -> spla.LinearOperator:
        needs_refactor = True
        n = A.shape[0]

        def matvec(x: np.ndarray) -> np.ndarray:
            nonlocal needs_refactor
            result = apply_inner_factorization(
                solver,
                A,
                x,
                refactor=needs_refactor,
                on_device=False,
                # Each preconditioner is built for a new matrix, whose pattern
                # may differ from the one the inner solver analysed last.
                structure_changed=needs_refactor,
            )
            needs_refactor = False
            return np.asarray(result, dtype=A.dtype).ravel()

        return spla.LinearOperator((n, n), matvec=matvec, dtype=A.dtype)

    return preconditioner
