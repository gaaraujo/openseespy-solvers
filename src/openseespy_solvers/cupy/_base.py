"""cupy backend hooks shared by all cupy-namespace solvers."""

from __future__ import annotations

import functools
import inspect
from typing import Any

import numpy as np

from openseespy_solvers.exceptions import BackendNotAvailableError, UnsupportedStorageSchemeError


def _import_cupy() -> tuple[Any, Any, Any]:
    try:
        import cupy as cp
        import cupyx.scipy.sparse as csp
        import cupyx.scipy.sparse.linalg as cspla
    except ImportError as exc:  # pragma: no cover - exercised only without cupy
        raise BackendNotAvailableError(
            "The 'cupy' backend requires cupy. Install a CUDA-matched wheel, "
            "for example: python -m pip install \"openseespy-solvers[cuda13]\""
        ) from exc
    return cp, csp, cspla


@functools.cache
def _signature_parameters(func: Any) -> Any:
    return inspect.signature(func).parameters


class CupyMixin:
    """Implements backend hooks using cupy / cupyx (GPU)."""

    backend = "cupy"
    _on_device = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._cp, self._csp, self._cspla = _import_cupy()
        super().__init__(*args, **kwargs)

    @property
    def _cupy_dtype(self) -> Any:
        return self._cp.float32 if self._compute_dtype == np.float32 else self._cp.float64

    def _build_matrix(
        self,
        values: np.ndarray,
        indices: np.ndarray,
        indptr: np.ndarray,
        shape: tuple[int, int],
        fmt: str,
    ) -> Any:
        cp, csp = self._cp, self._csp
        data = cp.asarray(values, dtype=self._cupy_dtype)
        ind = cp.asarray(indices, dtype=cp.int32)
        ptr = cp.asarray(indptr, dtype=cp.int32)
        if fmt == "CSR":
            return csp.csr_matrix((data, ind, ptr), shape=shape)
        if fmt == "CSC":
            return csp.csc_matrix((data, ind, ptr), shape=shape)
        raise UnsupportedStorageSchemeError(f"cupy backend does not support scheme {fmt!r}")

    def _update_matrix(self, matrix: Any, values: np.ndarray) -> Any:
        # One host-to-device copy into a preallocated buffer; any cast to
        # float32 then runs on the device.
        staging = getattr(self, "_values_staging", None)
        if staging is None or staging.shape[0] != values.shape[0]:
            staging = self._cp.empty(values.shape[0], dtype=self._cp.float64)
            self._values_staging = staging
        staging.set(np.ascontiguousarray(values, dtype=np.float64))
        if matrix.data.dtype == staging.dtype:
            matrix.data[...] = staging
        else:
            matrix.data[...] = staging.astype(matrix.data.dtype)
        return matrix

    # -- persistent device buffers ---------------------------------------

    def _rhs_to_compute(self, rhs: np.ndarray) -> Any:
        """Copy the OpenSees RHS into one preallocated device buffer."""
        if not self._on_device:
            return np.asarray(rhs, dtype=self._compute_dtype)
        cp = self._cp
        buf = getattr(self, "_rhs_device", None)
        if buf is None or buf.shape[0] != rhs.shape[0]:
            buf = cp.empty(rhs.shape[0], dtype=self._cupy_dtype)
            self._rhs_device = buf
        if buf.dtype == np.float64:
            buf.set(np.ascontiguousarray(rhs))
        else:
            buf.set(np.ascontiguousarray(rhs, dtype=buf.dtype))
        return buf

    def _write_opensees_buffer(self, buf: np.ndarray, value: Any) -> None:
        """Device-to-host copy straight into the OpenSees buffer."""
        if not self._on_device or not hasattr(value, "get"):
            buf[:] = np.asarray(self._to_host(value), dtype=np.float64)
            return
        if value.dtype != np.float64:
            value = value.astype(np.float64)
        value.ravel().get(out=buf)

    def _to_device(self, array: np.ndarray) -> Any:
        return self._cp.asarray(array, dtype=self._cupy_dtype)

    def _to_host(self, array: Any) -> np.ndarray:
        return self._cp.asnumpy(array)

    def _matvec(self, matrix: Any, vector: Any) -> Any:
        return matrix @ self._cp.asarray(vector, dtype=self._cupy_dtype)

    def _is_sparse(self, obj: Any) -> bool:
        return self._csp.issparse(obj)

    def _is_linear_operator(self, obj: Any) -> bool:
        return isinstance(obj, self._cspla.LinearOperator)

    def _iterative_kwargs(
        self,
        func: Any,
        *,
        rtol: float,
        atol: float,
        maxiter: int | None,
        M: Any | None,
        callback: Any | None,
    ) -> dict[str, Any]:
        """Map scipy-style rtol/atol onto the installed cupy iterative API."""
        params = _signature_parameters(func)
        kwargs: dict[str, Any] = {"maxiter": maxiter, "M": M}
        if callback is not None:
            kwargs["callback"] = callback
        if "rtol" in params:
            kwargs["rtol"] = rtol
            if "atol" in params:
                kwargs["atol"] = atol
        else:
            kwargs["tol"] = rtol
            if "atol" in params and atol != 0.0:
                kwargs["atol"] = atol
        return kwargs
