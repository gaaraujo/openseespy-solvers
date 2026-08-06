"""scipy backend hooks shared by all scipy-namespace solvers."""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from openseespy_solvers.exceptions import UnsupportedStorageSchemeError


def _import_umfpack() -> Any:
    """Import :mod:`scikits.umfpack` lazily.

    ``scikit-umfpack`` is an optional dependency, so importing the scipy
    namespace must never require it. The import is deferred until a UMFPACK
    solver is actually instantiated.

    Returns
    -------
    module
        The imported :mod:`scikits.umfpack` module.

    Raises
    ------
    ImportError
        If ``scikit-umfpack`` is not installed.
    """
    try:
        import scikits.umfpack as umfpack
    except ImportError as exc:  # pragma: no cover - exercised only without umfpack
        raise ImportError(
            "The 'umfpack' solver requires scikit-umfpack. "
            "Install with: python -m pip install \"openseespy-solvers[umfpack]\""
        ) from exc
    return umfpack


class ScipyMixin:
    """Implements backend hooks using scipy / NumPy (CPU)."""

    backend = "scipy"
    _on_device = False

    def _build_matrix(
        self,
        values: np.ndarray,
        indices: np.ndarray,
        indptr: np.ndarray,
        shape: tuple[int, int],
        fmt: str,
    ) -> sp.spmatrix:
        # Owned contiguous arrays (UMFPACK C API rejects non-C / non-owned views).
        vals = np.array(values, dtype=self._compute_dtype, order="C", copy=True)
        idx = np.array(indices, order="C", copy=True)
        ip = np.array(indptr, order="C", copy=True)
        if fmt == "CSR":
            return sp.csr_matrix((vals, idx, ip), shape=shape)
        if fmt == "CSC":
            return sp.csc_matrix((vals, idx, ip), shape=shape)
        raise UnsupportedStorageSchemeError(f"scipy backend does not support scheme {fmt!r}")

    def _update_matrix(self, matrix: sp.spmatrix, values: np.ndarray) -> sp.spmatrix:
        matrix.data[:] = np.asarray(values, dtype=self._compute_dtype)
        return matrix

    def _to_device(self, array: np.ndarray) -> np.ndarray:
        return np.asarray(array, dtype=self._compute_dtype)

    def _to_host(self, array: Any) -> np.ndarray:
        return np.asarray(array, dtype=self._compute_dtype)

    def _matvec(self, matrix: sp.spmatrix, vector: Any) -> np.ndarray:
        return matrix @ np.asarray(vector, dtype=self._compute_dtype)

    def _is_sparse(self, obj: Any) -> bool:
        return sp.issparse(obj)

    def _is_linear_operator(self, obj: Any) -> bool:
        return isinstance(obj, spla.LinearOperator)
