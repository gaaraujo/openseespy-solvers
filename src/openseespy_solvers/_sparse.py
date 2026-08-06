"""Parsing of OpenSees PythonSparse buffers into NumPy arrays.

These helpers are backend-agnostic: they only turn the raw memoryviews OpenSees
passes into NumPy views. Building the actual sparse matrix (scipy, cupy, ...) is
the job of each namespace's solver, via its ``_build_matrix`` hook.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from openseespy_solvers.exceptions import InvalidOpenSeesDataError, UnsupportedStorageSchemeError


class SparseArrays(NamedTuple):
    """Parsed sparse matrix arrays plus shape/format metadata."""

    values: np.ndarray
    indices: np.ndarray
    indptr: np.ndarray
    shape: tuple[int, int]
    fmt: str


def parse_sparse_arrays(
    kwargs: dict[str, Any],
    *,
    values_key: str = "values",
) -> SparseArrays:
    """Parse a single sparse matrix's buffers from OpenSees kwargs."""
    storage_scheme = kwargs.get("storage_scheme", "CSR")
    num_eqn = int(kwargs["num_eqn"])
    nnz = int(kwargs["nnz"])

    if values_key not in kwargs:
        raise InvalidOpenSeesDataError(f"Missing buffer: {values_key}")
    values = np.frombuffer(kwargs[values_key], dtype=np.float64, count=nnz)

    if storage_scheme in ("CSR", "CSC"):
        if "index_ptr" not in kwargs or "indices" not in kwargs:
            raise InvalidOpenSeesDataError("CSR/CSC requires index_ptr and indices buffers")
        indptr = np.frombuffer(kwargs["index_ptr"], dtype=np.int32, count=num_eqn + 1)
        indices = np.frombuffer(kwargs["indices"], dtype=np.int32, count=nnz)
        return SparseArrays(values, indices, indptr, (num_eqn, num_eqn), storage_scheme)

    raise UnsupportedStorageSchemeError(
        f"Storage scheme {storage_scheme!r} is not supported in this version."
    )


def parse_eigen_values(kwargs: dict[str, Any]) -> np.ndarray:
    """Parse the mass-matrix coefficient buffer for an eigen solve."""
    nnz = int(kwargs["nnz"])
    if "m_values" not in kwargs:
        raise InvalidOpenSeesDataError("Missing buffer: m_values")
    return np.frombuffer(kwargs["m_values"], dtype=np.float64, count=nnz)


def _host_array(array: Any) -> np.ndarray:
    get = getattr(array, "get", None)
    if get is not None:
        return get()
    return np.asarray(array, dtype=np.float64)


# OpenSees PythonSparse reference (benchmark_python_sparse_eigen.py) always uses LM.
OPENSEES_EIGSH_WHICH = "LM"


def opensees_eigsh_sigma(find_smallest: bool, sigma: float | None) -> float | None:
    """Shift for ``eigsh`` matching OpenSees.

    Uses ``0.0`` when smallest, else optional user ``sigma``.
    """
    if find_smallest:
        return 0.0 if sigma is None else sigma
    return sigma


def eigsh_arpack_kwargs(
    *,
    num_modes: int,
    which: str,
    tol: float,
    v0: Any | None,
    ncv: int | None,
    maxiter: int | None,
    mode: str | None = None,
) -> dict[str, Any]:
    """Keyword arguments forwarded to scipy/cupy ``eigsh`` (ARPACK), except ``A``/``M``."""
    kwargs: dict[str, Any] = {"k": num_modes, "which": which}
    if mode is not None:
        kwargs["mode"] = mode
    if v0 is not None:
        kwargs["v0"] = v0
    if ncv is not None:
        kwargs["ncv"] = ncv
    if maxiter is not None:
        kwargs["maxiter"] = maxiter
    if tol > 0.0:
        kwargs["tol"] = tol
    return kwargs


def csr_linear_kwargs_from_matrix(
    matrix: Any,
    rhs: Any,
    *,
    matrix_status: str,
    x: np.ndarray | None = None,
    scheme: str = "CSR",
) -> dict:
    """Build OpenSees-style linear ``solve`` kwargs from a sparse matrix and RHS.

    Parameters
    ----------
    scheme : {'CSR', 'CSC'}, optional
        Storage layout of the marshalled buffers. Use ``'CSC'`` when the inner
        solver prefers CSC (e.g. ``umfpack``), to avoid an extra convert.
    """
    fmt = scheme.upper()
    if fmt not in ("CSR", "CSC"):
        raise UnsupportedStorageSchemeError(
            f"Storage scheme {scheme!r} is not supported in this version."
        )
    if fmt == "CSC":
        if hasattr(matrix, "tocsc"):
            matrix = matrix.tocsc()
    elif hasattr(matrix, "tocsr"):
        matrix = matrix.tocsr()
    indptr = _host_array(matrix.indptr).astype(np.int32, copy=False)
    indices = _host_array(matrix.indices).astype(np.int32, copy=False)
    values = _host_array(matrix.data).astype(np.float64, copy=False)
    b = _host_array(rhs).astype(np.float64, copy=False)
    n = matrix.shape[0]
    if x is None:
        x = np.zeros(n, dtype=np.float64)
    return {
        "index_ptr": memoryview(indptr),
        "indices": memoryview(indices),
        "values": memoryview(values),
        "rhs": memoryview(b),
        "x": memoryview(x),
        "num_eqn": n,
        "nnz": values.size,
        "matrix_status": matrix_status,
        "storage_scheme": fmt,
    }
