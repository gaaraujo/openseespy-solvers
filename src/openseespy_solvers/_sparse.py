"""Parsing of OpenSees PythonSparse buffers.

OpenSees hands its sparse matrix over as raw memoryviews in one of several
layouts: CSR, CSC or COO, optionally holding only the lower triangle of a
symmetric matrix (``matrix_type='SYMMETRIC_LOWER'``, OpenSees builds that
support it). Backends want one specific compressed format. :class:`Layout`
reconciles the two once per sparsity pattern (``STRUCTURE_CHANGED``): it holds
the target ``indptr``/``indices`` and, when the orders differ, a permutation
that maps OpenSees' value order onto the backend's ``data`` order. Coefficient
updates are then a single gather (or a plain copy) with no index work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp

from openseespy_solvers.exceptions import InvalidOpenSeesDataError, UnsupportedStorageSchemeError

GENERAL = "GENERAL"
SYMMETRIC_LOWER = "SYMMETRIC_LOWER"


@dataclass(frozen=True)
class Layout:
    """Backend-side description of one OpenSees sparsity pattern.

    Attributes
    ----------
    fmt : {'CSR', 'CSC'}
        Compressed format handed to the backend.
    indptr, indices : ndarray of int32
        Pattern in ``fmt`` with sorted indices (owned copies).
    shape : tuple of int
        Matrix shape ``(n, n)``.
    perm : ndarray of intp or None
        ``backend_data = opensees_values[perm]``; ``None`` when the orders agree.
    lower : bool
        ``True`` when the backend receives only the lower triangle (row >= col)
        of a symmetric matrix.
    nnz_in : int
        Number of values in the OpenSees buffer.
    """

    fmt: str
    indptr: np.ndarray
    indices: np.ndarray
    shape: tuple[int, int]
    perm: np.ndarray | None
    lower: bool
    nnz_in: int

    def reorder(self, values: np.ndarray) -> np.ndarray:
        """Return ``values`` in backend order (a view when no reordering is needed)."""
        return values if self.perm is None else values[self.perm]


def _positions(n: int) -> np.ndarray:
    # 1-based so that no entry is an explicit zero that scipy might drop.
    return np.arange(1, n + 1, dtype=np.int64)


def _pattern_matrix(scheme: str, kwargs: dict[str, Any], n: int, nnz: int) -> sp.spmatrix:
    """Sparse matrix whose ``data`` holds the 1-based position of each OpenSees value."""
    pos = _positions(nnz)
    if scheme in ("CSR", "CSC"):
        if "index_ptr" not in kwargs or "indices" not in kwargs:
            raise InvalidOpenSeesDataError("CSR/CSC requires index_ptr and indices buffers")
        indptr = np.frombuffer(kwargs["index_ptr"], dtype=np.int32, count=n + 1)
        indices = np.frombuffer(kwargs["indices"], dtype=np.int32, count=nnz)
        cls = sp.csr_matrix if scheme == "CSR" else sp.csc_matrix
        return cls((pos, indices.copy(), indptr.copy()), shape=(n, n))
    if scheme == "COO":
        if "row" not in kwargs or "col" not in kwargs:
            raise InvalidOpenSeesDataError("COO requires row and col buffers")
        row = np.frombuffer(kwargs["row"], dtype=np.int32, count=nnz)
        col = np.frombuffer(kwargs["col"], dtype=np.int32, count=nnz)
        return sp.coo_matrix((pos, (row, col)), shape=(n, n))
    raise UnsupportedStorageSchemeError(
        f"Storage scheme {scheme!r} is not supported (expected CSR, CSC or COO)."
    )


def plan_layout(
    kwargs: dict[str, Any],
    *,
    formats: tuple[str, ...] = ("CSR", "CSC"),
    accepts_lower: bool = False,
) -> Layout:
    """Plan how OpenSees values map onto a backend matrix.

    Parameters
    ----------
    kwargs : dict
        OpenSees ``solve`` keyword arguments (``STRUCTURE_CHANGED`` call).
    formats : tuple of {'CSR', 'CSC'}
        Formats the backend accepts, in order of preference.
    accepts_lower : bool
        Whether the backend can use the lower triangle of a symmetric matrix
        directly. Otherwise a lower-triangle buffer is expanded to the full
        matrix.
    """
    scheme = kwargs.get("storage_scheme", "CSR")
    n = int(kwargs["num_eqn"])
    nnz = int(kwargs["nnz"])
    lower = kwargs.get("matrix_type", GENERAL) == SYMMETRIC_LOWER

    # Fast path: the OpenSees layout is already acceptable as is.
    if scheme in formats and scheme in ("CSR", "CSC") and (accepts_lower or not lower):
        indptr = np.array(np.frombuffer(kwargs["index_ptr"], dtype=np.int32, count=n + 1))
        indices = np.array(np.frombuffer(kwargs["indices"], dtype=np.int32, count=nnz))
        return Layout(scheme, indptr, indices, (n, n), None, lower, nnz)

    pattern = _pattern_matrix(scheme, kwargs, n, nnz)
    if lower and not accepts_lower:
        # Mirror the strictly lower part; the two parts have disjoint patterns,
        # so positions are carried over without summation.
        pattern = (pattern + sp.tril(pattern, k=-1).T).tocsr()
        lower = False
    fmt = formats[0]
    pattern = pattern.tocsr() if fmt == "CSR" else pattern.tocsc()
    pattern.sort_indices()
    perm = pattern.data.astype(np.intp) - 1
    if perm.shape[0] == nnz and np.array_equal(perm, np.arange(nnz)):
        perm = None
    return Layout(
        fmt,
        pattern.indptr.astype(np.int32),
        pattern.indices.astype(np.int32),
        (n, n),
        perm,
        lower,
        nnz,
    )


def read_values(kwargs: dict[str, Any], key: str = "values") -> np.ndarray:
    """Zero-copy float64 view of an OpenSees coefficient buffer."""
    if key not in kwargs:
        raise InvalidOpenSeesDataError(f"Missing buffer: {key}")
    return np.frombuffer(kwargs[key], dtype=np.float64, count=int(kwargs["nnz"]))


def read_vector(kwargs: dict[str, Any], key: str) -> np.ndarray:
    """Zero-copy float64 view of an OpenSees length-``num_eqn`` vector buffer."""
    if key not in kwargs:
        raise InvalidOpenSeesDataError(f"Missing buffer: {key}")
    return np.frombuffer(kwargs[key], dtype=np.float64, count=int(kwargs["num_eqn"]))


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
            f"Storage scheme {scheme!r} is not supported for marshalled buffers."
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
