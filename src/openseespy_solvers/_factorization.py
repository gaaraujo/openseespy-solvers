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
        scheme=getattr(inner, "scheme", "CSR") or "CSR",
    )
    info = inner.solve(**lin_kwargs)
    if info != 0:
        raise SolverConvergenceError(
            f"Inner direct solve failed with info={info} (matrix_status={matrix_status!r})"
        )
    solution = np.frombuffer(lin_kwargs["x"], dtype=np.float64, count=n).copy()
    return inner._to_device(solution)
