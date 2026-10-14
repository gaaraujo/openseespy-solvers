"""nvmath.sparse backend hooks shared by all nvmath-namespace solvers."""

from __future__ import annotations

import importlib.resources
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from openseespy_solvers.exceptions import UnsupportedStorageSchemeError


def _import_nvmath() -> Any:
    """Import :mod:`nvmath` lazily.

    ``nvmath-python`` is an optional dependency; importing the nvmath namespace
    must never require it. The import is deferred until a solver is constructed.

    Returns
    -------
    module
        The imported :mod:`nvmath` module.

    Raises
    ------
    ImportError
        If ``nvmath-python`` is not installed.
    """
    try:
        import nvmath
    except ImportError as exc:  # pragma: no cover - exercised only without nvmath
        raise ImportError(
            "The 'nvmath' backend requires nvmath-python. "
            "Install with: python -m pip install \"openseespy-solvers[cuda13]\" "
            "(or [cuda12]; see installation docs)"
        ) from exc
    return nvmath


def _is_cupy_sparse(sp_module: Any) -> bool:
    name = getattr(sp_module, "__name__", "")
    return "cupy" in name.lower()


def _resolve_sp_module(
    sp_module: Any | None,
    *,
    device: str | None = None,
) -> tuple[Any, bool]:
    """Return ``(sparse_module, is_gpu)`` for matrix assembly."""
    if sp_module is not None:
        is_gpu = _is_cupy_sparse(sp_module)
        if is_gpu:
            try:
                import cupy  # noqa: F401
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "cupy sparse matrices require cupy. "
                    "Install with: python -m pip install "
                    '"openseespy-solvers[cuda13]"'
                ) from exc
        return sp_module, is_gpu

    if device == "cpu":
        return sp, False
    if device == "gpu":
        try:
            import cupyx.scipy.sparse as csp
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "device='gpu' requires cupy. "
                "Install with: python -m pip install "
                '"openseespy-solvers[cuda13]"'
            ) from exc
        return csp, True

    try:
        import cupyx.scipy.sparse as csp
    except ImportError:
        return sp, False
    return csp, True


def _find_multithreading_lib() -> str | None:
    """Locate the cuDSS multithreading layer library, if installed.

    nvmath.sparse uses this for multi-threaded CPU work during plan/factorize (especially
    with hybrid execution). See ``CUDSS_THREADING_LIB`` and NVIDIA's cuDSS docs.
    """
    env = os.environ.get("CUDSS_THREADING_LIB")
    if env:
        path = Path(env)
        if path.is_file():
            return str(path.resolve())

    if sys.platform == "win32":
        lib_names = (
            "cudss_mtlayer_vcomp140.dll",
            "libcudss_mtlayer_vcomp.dll",
            "cudss_mtlayer_vcomp.dll",
        )
        subdirs = ("bin", "lib")
    else:
        lib_names = (
            "libcudss_mtlayer_gomp.so.0",
            "libcudss_mtlayer_gomp.so",
        )
        subdirs = ("lib",)

    search_roots: list[Path] = []

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        search_roots.append(Path(conda_prefix))

    for cuda_pkg in ("nvidia.cu13", "nvidia.cu12"):
        try:
            search_roots.append(Path(str(importlib.resources.files(cuda_pkg))))
        except (ImportError, AttributeError, TypeError, ModuleNotFoundError):
            pass

    for entry in sys.path:
        base = Path(entry)
        if not base.is_dir():
            continue
        for cuda_ver in ("cu13", "cu12"):
            root = base / "nvidia" / cuda_ver
            if root.is_dir():
                search_roots.append(root)

    seen: set[Path] = set()
    for root in search_roots:
        root = root.resolve()
        if root in seen:
            continue
        seen.add(root)
        for subdir in subdirs:
            for lib_name in lib_names:
                lib_path = root / subdir / lib_name
                if lib_path.is_file():
                    return str(lib_path.resolve())

    if sys.platform != "win32":
        for sys_path in (Path("/usr/local/lib"), Path("/usr/lib"), Path("/lib")):
            for lib_name in lib_names:
                lib_path = sys_path / lib_name
                if lib_path.is_file():
                    return str(lib_path.resolve())

    return None


class NvMathMixin:
    """Implements backend hooks for nvmath.sparse direct solvers (CPU or GPU sparse)."""

    backend = "nvmath"
    _formats = ("CSR",)

    def __init__(
        self,
        *args: Any,
        sp_module: Any | None = None,
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._nvmath = _import_nvmath()
        self._sp_module, self._on_device = _resolve_sp_module(sp_module, device=device)
        self._values_staging: Any | None = None
        if self._on_device:
            import cupy as cp

            self._np_module = cp
        else:
            self._np_module = np
        super().__init__(*args, **kwargs)

    @property
    def _cupy_dtype(self) -> Any:
        cp = self._np_module
        return cp.float32 if self._compute_dtype == np.float32 else cp.float64

    def _build_matrix(
        self,
        values: np.ndarray,
        indices: np.ndarray,
        indptr: np.ndarray,
        shape: tuple[int, int],
        fmt: str,
    ) -> Any:
        if fmt != "CSR":
            raise UnsupportedStorageSchemeError(f"nvmath backend expects CSR, got {fmt!r}")
        if self._on_device:
            cp = self._np_module
            self._values_staging = cp.empty(values.shape[0], dtype=cp.float64)
            return self._sp_module.csr_matrix(
                (
                    cp.asarray(values, dtype=self._cupy_dtype),
                    cp.asarray(indices, dtype=cp.int32),
                    cp.asarray(indptr, dtype=cp.int32),
                ),
                shape=shape,
            )
        return self._sp_module.csr_matrix(
            (
                np.array(values, dtype=self._compute_dtype),
                np.array(indices, dtype=np.int32),
                np.array(indptr, dtype=np.int32),
            ),
            shape=shape,
        )

    def _update_matrix(self, matrix: Any, values: np.ndarray) -> Any:
        if self._on_device:
            # One host-to-device copy into a preallocated buffer; any cast to
            # float32 then runs on the device.
            staging = self._values_staging
            staging.set(np.ascontiguousarray(values, dtype=np.float64))
            if matrix.data.dtype == staging.dtype:
                matrix.data[...] = staging
            else:
                matrix.data[...] = staging.astype(matrix.data.dtype)
        else:
            matrix.data[...] = values
        return matrix

    # -- persistent device buffers ---------------------------------------

    def _rhs_to_compute(self, rhs: np.ndarray) -> Any:
        """Copy the OpenSees RHS into one preallocated device buffer."""
        if not self._on_device:
            return np.asarray(rhs, dtype=self._compute_dtype)
        cp = self._np_module
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
        if self._on_device:
            return self._np_module.asarray(array, dtype=self._cupy_dtype)
        return np.asarray(array, dtype=self._compute_dtype)

    def _to_host(self, array: Any) -> np.ndarray:
        if self._on_device:
            return self._np_module.asnumpy(array)
        return np.asarray(array, dtype=self._compute_dtype)

    def _matvec(self, matrix: Any, vector: Any) -> Any:
        if self._on_device:
            return matrix @ self._np_module.asarray(vector, dtype=self._cupy_dtype)
        return matrix @ np.asarray(vector, dtype=self._compute_dtype)

    def _is_sparse(self, obj: Any) -> bool:
        return self._sp_module.issparse(obj)

    def _is_linear_operator(self, obj: Any) -> bool:
        if self._on_device:
            from cupyx.scipy.sparse import linalg as cspla

            return isinstance(obj, cspla.LinearOperator)
        return isinstance(obj, spla.LinearOperator)
