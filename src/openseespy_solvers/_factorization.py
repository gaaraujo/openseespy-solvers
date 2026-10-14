"""Apply an inner direct solver's cached factorization to a vector."""

from __future__ import annotations

from typing import Any

import numpy as np

from openseespy_solvers._base import LinearSolver
from openseespy_solvers._sparse import _host_array, csr_linear_kwargs_from_matrix
from openseespy_solvers.exceptions import SolverConvergenceError


def apply_inner_factorization(
    inner: LinearSolver,
    matrix: Any,
    vec: Any,
    *,
    refactor: bool,
    on_device: bool,
    structure_changed: bool = False,
    native: bool = False,
) -> Any:
    """Apply ``inner`` direct solver's factorization of ``matrix`` to ``vec``.

    When ``refactor`` is ``True``, the inner solver (re)factorizes ``matrix`` and
    applies the factorization. When ``False``, the cached factorization is reused
    (``matrix_status='UNCHANGED'``) even if ``matrix`` coefficients were updated
    elsewhere.

    Parameters
    ----------
    inner : LinearSolver
        Direct solver whose factorization is applied.
    matrix : sparse matrix
        System matrix in the inner solver's backend format.
    vec : array
        Right-hand side vector (host or device).
    refactor : bool
        If ``True``, refresh the factorization before applying it.
    on_device : bool
        If ``True``, call ``inner._solve_system`` without a host round-trip.
        If ``False``, marshal through OpenSees-style buffers and ``inner.solve``.
    structure_changed : bool, optional
        When ``refactor`` is ``True``, use ``STRUCTURE_CHANGED`` instead of
        ``COEFFICIENTS_CHANGED``.
    native : bool, optional
        ``True`` when ``matrix`` was built by ``inner``'s own ``_build_matrix``
        hook (as in :func:`~openseespy_solvers.hybrid`). The matrix and ``vec``
        then go straight to ``inner._solve_system``, with no re-marshalling or
        host round trip; this runs once per GMRES iteration.

    Returns
    -------
    array
        Solution vector on the inner solver's compute device.

    Raises
    ------
    SolverConvergenceError
        If the inner solve reports failure.
    """
    if refactor:
        matrix_status = "STRUCTURE_CHANGED" if structure_changed else "COEFFICIENTS_CHANGED"
    else:
        matrix_status = "UNCHANGED"

    if native:
        if on_device:
            rhs = vec.astype(inner._cupy_dtype, copy=False).ravel()
        else:
            rhs = np.asarray(vec, dtype=inner._compute_dtype).ravel()
        result, info, _ = inner._solve_system(matrix, rhs, None, matrix_status)
        if info != 0:
            raise SolverConvergenceError(
                f"Inner direct solve failed with info={info} (matrix_status={matrix_status!r})"
            )
        # Some backends return an internal buffer that the next solve overwrites.
        return result.copy()

    if on_device:
        rhs = inner._to_device(np.asarray(_host_array(vec), dtype=inner._compute_dtype))
        result, info, _ = inner._solve_system(matrix, rhs, None, matrix_status)
        if info != 0:
            raise SolverConvergenceError(
                f"Inner direct solve failed with info={info} (matrix_status={matrix_status!r})"
            )
        return result

    n = matrix.shape[0]
    rhs_host = np.asarray(_host_array(vec), dtype=np.float64).ravel()
    x_host = np.zeros(n, dtype=np.float64)
    lin_kwargs = csr_linear_kwargs_from_matrix(
        matrix,
        rhs_host,
        matrix_status=matrix_status,
        x=x_host,
        scheme="CSC" if getattr(inner, "scheme", "CSR") == "CSC" else "CSR",
    )
    info = inner.solve(**lin_kwargs)
    if info != 0:
        raise SolverConvergenceError(
            f"Inner direct solve failed with info={info} (matrix_status={matrix_status!r})"
        )
    solution = np.frombuffer(lin_kwargs["x"], dtype=np.float64, count=n).copy()
    return inner._to_device(solution)


class FactorizedOperator:
    """``x = A^{-1} v`` through a package direct solver, factorized once.

    The matrix is handed to ``inner.solve`` once with ``STRUCTURE_CHANGED``
    (or ``COEFFICIENTS_CHANGED`` after :meth:`update`); every later
    application reuses the same buffers with ``matrix_status='UNCHANGED'``, so
    it costs one triangular solve plus O(n) copies. Used as ``OPinv`` for
    shift-invert eigen solves.

    The marshalled layout follows the inner solver: CSC when its ``scheme`` is
    ``'CSC'``, and only the lower triangle (``matrix_type='SYMMETRIC_LOWER'``)
    when it requests symmetric storage, in which case ``A`` must be symmetric.
    """

    def __init__(self, inner: LinearSolver, matrix: Any) -> None:
        self.inner = inner
        self.lower = inner._matrix_type_request() == "SYMMETRIC_LOWER"
        scheme = "CSC" if getattr(inner, "scheme", "CSR") == "CSC" else "CSR"
        n = matrix.shape[0]
        self._rhs = np.zeros(n, dtype=np.float64)
        self._x = np.zeros(n, dtype=np.float64)
        self._kwargs = csr_linear_kwargs_from_matrix(
            sp_tril(matrix) if self.lower else matrix,
            self._rhs,
            matrix_status="STRUCTURE_CHANGED",
            x=self._x,
            scheme=scheme,
        )
        # Keep the marshalled arrays alive; the inner solver may reference them.
        self._values = np.frombuffer(self._kwargs["values"], dtype=np.float64)
        if self.lower:
            self._kwargs["matrix_type"] = "SYMMETRIC_LOWER"
        self._pending = "STRUCTURE_CHANGED"
        self.shape = matrix.shape

    def update(self, matrix: Any) -> None:
        """New coefficients on the same pattern (refactorize on next use)."""
        src = sp_tril(matrix) if self.lower else matrix
        src = src.tocsc() if self._kwargs["storage_scheme"] == "CSC" else src.tocsr()
        src.sort_indices()
        self._values[:] = _host_array(src.data)
        if self._pending != "STRUCTURE_CHANGED":
            self._pending = "COEFFICIENTS_CHANGED"

    def matvec(self, v: Any) -> np.ndarray:
        self._rhs[:] = np.asarray(_host_array(v), dtype=np.float64).ravel()
        self._kwargs["matrix_status"] = self._pending
        info = self.inner.solve(**self._kwargs)
        if info != 0:
            raise SolverConvergenceError(
                f"Inner direct solve failed with info={info} (matrix_status={self._pending!r})"
            )
        self._pending = "UNCHANGED"
        return self._x.copy()


def sp_tril(matrix: Any) -> Any:
    import scipy.sparse as sp

    return sp.tril(matrix)
