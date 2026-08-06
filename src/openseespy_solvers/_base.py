"""Base solver classes for the OpenSees PythonSparse interfaces.

These bases own the parts that are *identical for every backend*: wrapping the
OpenSees memoryviews, the ``matrix_status`` caching pattern, writing the solution
in place, statistics, ``formAp``, ``to_openseespy`` and ``copy`` support.

Backend-specific numerics are provided by small hooks that each namespace
(``openseespy_solvers.scipy``, ``.cupy``, ...) implements:

* ``_build_matrix`` / ``_update_matrix`` - assemble the cached sparse matrix
* ``_to_device`` / ``_to_host`` - host<->device transfer (identity on CPU)
* ``_matvec`` - ``A @ v``
* ``_is_sparse`` / ``_is_linear_operator`` - preconditioner detection
* ``_solve_system`` (linear) / ``_solve_eigen`` (eigen) - the actual library call

A backend that does not fit this shape (e.g. a distributed PETSc backend) can
override ``solve`` directly; nothing here is mandatory beyond the hooks a given
solver uses.
"""

from __future__ import annotations

import copy
import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from openseespy_solvers._dtype import OPENSEES_BUFFER_DTYPE, resolve_compute_dtype
from openseespy_solvers._sparse import parse_eigen_values, parse_sparse_arrays
from openseespy_solvers.exceptions import InvalidOpenSeesDataError
from openseespy_solvers.stats import EigenSolverStats, LinearSolverStats


def _normalize_writable(writable: str | list[str]) -> str:
    if isinstance(writable, list):
        return ",".join(writable)
    return writable


class BaseOpenSeesSolver(ABC):
    """Shared behavior for linear and eigen solvers."""

    #: Backend name, set as a class attribute by each namespace mixin.
    backend: str = "base"
    #: Whether matrices/vectors live on a device (GPU) rather than host memory.
    _on_device: bool = False

    def __init__(
        self,
        *,
        scheme: str = "CSR",
        debug: bool = False,
        dtype: Any = np.float64,
        record_stats: bool = False,
    ) -> None:
        self.scheme = scheme
        self.debug = debug
        self.record_stats = record_stats
        self._compute_dtype = resolve_compute_dtype(dtype)
        self._matrix: Any | None = None
        self._k_matrix: Any | None = None
        self._m_matrix: Any | None = None
        self._params: dict[str, Any] = {}

    # -- backend hooks ---------------------------------------------------

    @abstractmethod
    def _build_matrix(
        self,
        values: np.ndarray,
        indices: np.ndarray,
        indptr: np.ndarray,
        shape: tuple[int, int],
        fmt: str,
    ) -> Any:
        """Build a new sparse matrix in the backend's format (owns its data)."""

    @abstractmethod
    def _update_matrix(self, matrix: Any, values: np.ndarray) -> Any:
        """Refresh coefficients (in place where possible) and return the matrix."""

    @abstractmethod
    def _to_device(self, array: np.ndarray) -> Any:
        """Transfer a host array to the backend (identity on CPU)."""

    @abstractmethod
    def _to_host(self, array: Any) -> np.ndarray:
        """Transfer a backend array back to a NumPy host array."""

    @abstractmethod
    def _matvec(self, matrix: Any, vector: Any) -> Any:
        """Return ``matrix @ vector`` in the backend."""

    # -- OpenSees config -------------------------------------------------

    def to_openseespy(
        self,
        *,
        scheme: str | None = None,
        writable: str | list[str] | None = None,
    ) -> dict[str, Any]:
        """Return the configuration dict for OpenSeesPy ``PythonSparse`` commands.

        Parameters
        ----------
        scheme : {'CSR', 'CSC'}, optional
            Sparse storage scheme. Default is the value given at construction.
        writable : str or list of str, optional
            Writable buffers declared to OpenSees. Overrides the constructor
            value when provided.

        Returns
        -------
        config : dict
            Dictionary with keys ``solver``, ``scheme``, and optionally
            ``writable``. Pass directly to ``ops.system('PythonSparse', ...)``
            or ``ops.eigen('PythonSparse', ...)``.

        Examples
        --------
        >>> from openseespy_solvers.scipy import cg
        >>> solver = cg()
        >>> cfg = solver.to_openseespy()
        >>> sorted(cfg.keys())
        ['scheme', 'solver', 'writable']
        """
        cfg: dict[str, Any] = {
            "solver": self,
            "scheme": scheme or self.scheme,
        }
        if writable is not None:
            cfg["writable"] = _normalize_writable(writable)
        elif hasattr(self, "writable"):
            cfg["writable"] = _normalize_writable(self.writable)
        return cfg

    # -- matrix caching --------------------------------------------------

    def _values_for_compute(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=self._compute_dtype)

    def _write_opensees_buffer(self, buf: np.ndarray, value: Any) -> None:
        buf[:] = np.asarray(self._to_host(value), dtype=OPENSEES_BUFFER_DTYPE)

    def _current_matrix(self, kwargs: dict[str, Any], *, values_key: str = "values") -> Any:
        arrays = parse_sparse_arrays(kwargs, values_key=values_key)
        values = self._values_for_compute(arrays.values)
        status = kwargs["matrix_status"]
        if status == "STRUCTURE_CHANGED" or self._matrix is None:
            self._matrix = self._build_matrix(
                values, arrays.indices, arrays.indptr, arrays.shape, arrays.fmt
            )
        elif status == "COEFFICIENTS_CHANGED":
            self._matrix = self._update_matrix(self._matrix, values)
        return self._matrix

    def _current_eigen_matrices(self, kwargs: dict[str, Any]) -> tuple[Any, Any]:
        arrays = parse_sparse_arrays(kwargs, values_key="k_values")
        k_values = self._values_for_compute(arrays.values)
        m_values = self._values_for_compute(parse_eigen_values(kwargs))
        status = kwargs["matrix_status"]
        if status == "STRUCTURE_CHANGED" or self._k_matrix is None:
            self._k_matrix = self._build_matrix(
                k_values, arrays.indices, arrays.indptr, arrays.shape, arrays.fmt
            )
            self._m_matrix = self._build_matrix(
                m_values, arrays.indices, arrays.indptr, arrays.shape, arrays.fmt
            )
        elif status == "COEFFICIENTS_CHANGED":
            self._k_matrix = self._update_matrix(self._k_matrix, k_values)
            self._m_matrix = self._update_matrix(self._m_matrix, m_values)
        return self._k_matrix, self._m_matrix

    # -- copy ------------------------------------------------------------

    def copy(self) -> BaseOpenSeesSolver:
        return copy.copy(self)

    def __copy__(self) -> BaseOpenSeesSolver:
        return type(self)(**self._params)

    def __deepcopy__(self, memo: dict[int, Any]) -> BaseOpenSeesSolver:
        return self.__copy__()


class LinearSolver(BaseOpenSeesSolver, ABC):
    """Base class for linear system solvers (``Ax = b``).

    Instances are created by backend factories such as :func:`openseespy_solvers.scipy.cg`.
    OpenSees calls :meth:`solve` with sparse buffer memoryviews; the solution is
    written in place to the ``x`` buffer.

    Attributes
    ----------
    A : sparse matrix or None
        Cached system matrix from the last :meth:`solve` or :meth:`formAp` call.
    b : ndarray
        Right-hand side from the last :meth:`solve` call.
    x : ndarray
        Last solution vector.
    stats : LinearSolverStats
        Runtime statistics updated after each :meth:`solve`.
    """

    stats: LinearSolverStats

    def __init__(
        self,
        *,
        scheme: str = "CSR",
        writable: str | list[str] = "none",
        debug: bool = False,
        preconditioner: Any = None,
        dtype: Any = np.float64,
        record_stats: bool = False,
    ) -> None:
        super().__init__(
            scheme=scheme, debug=debug, dtype=dtype, record_stats=record_stats
        )
        self.writable = _normalize_writable(writable)
        self._preconditioner = preconditioner
        self._preconditioner_cached: Any | None = None
        self.stats = LinearSolverStats()
        self._A: Any | None = None
        self._b: Any | None = None
        self._x: Any | None = None

    @property
    def A(self) -> Any | None:
        return self._A

    @property
    def b(self) -> Any | None:
        return self._b

    @property
    def x(self) -> Any | None:
        return self._x

    def _resolve_preconditioner(self, A: Any, matrix_status: str) -> Any | None:
        M = self._preconditioner
        if M is None:
            return None
        if self._is_sparse(M) or self._is_linear_operator(M):
            return M
        if callable(M):
            if matrix_status == "UNCHANGED" and self._preconditioner_cached is not None:
                return self._preconditioner_cached
            self._preconditioner_cached = M(A)
            return self._preconditioner_cached
        return M

    def solve(self, **kwargs: Any) -> int:
        """Solve ``Ax = b`` using buffers supplied by OpenSees.

        This method is called by OpenSeesPy; application code normally does not
        invoke it directly.

        Parameters
        ----------
        **kwargs
            OpenSees ``PythonSparse`` buffers, including ``values``, ``rhs``,
            ``x``, ``num_eqn``, ``nnz``, ``matrix_status``, and
            ``storage_scheme``.

        Returns
        -------
        info : int
            ``0`` if the solve succeeded; a negative value otherwise. When
            ``debug=True``, failures raise the underlying exception instead.
        """
        try:
            matrix_status = kwargs["matrix_status"]
            num_eqn = int(kwargs["num_eqn"])
            if "rhs" not in kwargs or "x" not in kwargs:
                raise InvalidOpenSeesDataError("Linear solve requires rhs and x buffers")

            A = self._current_matrix(kwargs)
            rhs = np.frombuffer(kwargs["rhs"], dtype=OPENSEES_BUFFER_DTYPE, count=num_eqn)
            x_buf = np.frombuffer(kwargs["x"], dtype=OPENSEES_BUFFER_DTYPE, count=num_eqn)

            self._A = A
            b = self._to_device(np.asarray(rhs, dtype=self._compute_dtype))
            self._b = b

            M = self._resolve_preconditioner(A, matrix_status)
            start = time.perf_counter()
            result, info, num_iter = self._solve_system(A, b, M, matrix_status)
            elapsed = time.perf_counter() - start

            self._write_opensees_buffer(x_buf, result)
            self._x = result if self._on_device else x_buf

            residual = None
            # Residual SpMV doubles matvec work; only with record_stats+debug.
            if self.record_stats and self.debug and num_eqn > 0:
                ax = np.asarray(self._to_host(self._matvec(A, result)), dtype=OPENSEES_BUFFER_DTYPE)
                r = rhs - ax
                norm_b = float(np.linalg.norm(rhs))
                residual = (
                    float(np.linalg.norm(r) / norm_b) if norm_b > 0 else float(np.linalg.norm(r))
                )

            if self.record_stats:
                self.stats.num_solves += 1
                self.stats.last_solve_time = elapsed
                self.stats.last_info = info
                self.stats.last_num_iterations = num_iter
                self.stats.last_residual_norm = residual
                self.stats.last_error = None
            return 0 if info == 0 else -abs(int(info))
        except Exception as exc:
            if self.record_stats:
                self.stats.last_error = exc
            if self.debug:
                raise
            return -1

    def formAp(self, **kwargs: Any) -> int:
        try:
            num_eqn = int(kwargs["num_eqn"])
            if "p" not in kwargs or "Ap" not in kwargs:
                raise InvalidOpenSeesDataError("formAp requires p and Ap buffers")

            A = self._current_matrix(kwargs)
            self._A = A
            p = np.frombuffer(kwargs["p"], dtype=OPENSEES_BUFFER_DTYPE, count=num_eqn)
            ap_buf = np.frombuffer(kwargs["Ap"], dtype=OPENSEES_BUFFER_DTYPE, count=num_eqn)

            result = self._matvec(A, self._to_device(np.asarray(p, dtype=self._compute_dtype)))
            self._write_opensees_buffer(ap_buf, result)
            return 0
        except Exception as exc:
            if self.record_stats:
                self.stats.last_error = exc
            if self.debug:
                raise
            return -1

    @abstractmethod
    def _is_sparse(self, obj: Any) -> bool:
        """Return True if *obj* is a sparse matrix for this backend."""

    @abstractmethod
    def _is_linear_operator(self, obj: Any) -> bool:
        """Return True if *obj* is a LinearOperator for this backend."""

    @abstractmethod
    def _solve_system(
        self,
        A: Any,
        b: Any,
        M: Any | None,
        matrix_status: str,
    ) -> tuple[Any, int, int | None]:
        """Return (solution, info, num_iterations)."""


class EigenSolver(BaseOpenSeesSolver, ABC):
    """Base class for generalized eigenvalue solvers (``K phi = lambda M phi``)."""

    stats: EigenSolverStats

    def __init__(
        self, *, scheme: str = "CSR", debug: bool = False, dtype: Any = np.float64
    ) -> None:
        super().__init__(scheme=scheme, debug=debug, dtype=dtype)
        self.stats = EigenSolverStats()
        self._K: Any | None = None
        self._M: Any | None = None

    @property
    def K(self) -> Any | None:
        return self._K

    @property
    def M(self) -> Any | None:
        return self._M

    def solve(self, **kwargs: Any) -> None:
        num_modes = int(kwargs["num_modes"])
        num_eqn = int(kwargs["num_eqn"])
        find_smallest = bool(kwargs["find_smallest"])

        eigenvalues_buf = np.frombuffer(
            kwargs["eigenvalues"], dtype=OPENSEES_BUFFER_DTYPE, count=num_modes
        )
        eigenvectors_buf = np.frombuffer(
            kwargs["eigenvectors"], dtype=OPENSEES_BUFFER_DTYPE, count=num_modes * num_eqn
        )

        K, M = self._current_eigen_matrices(kwargs)
        self._K = K
        self._M = M

        start = time.perf_counter()
        try:
            eigvals, eigvecs = self._solve_eigen(
                K, M, num_modes=num_modes, find_smallest=find_smallest
            )
            elapsed = time.perf_counter() - start

            eigvals_host = np.asarray(self._to_host(eigvals), dtype=OPENSEES_BUFFER_DTYPE)
            eigvecs_host = np.asarray(self._to_host(eigvecs), dtype=OPENSEES_BUFFER_DTYPE)

            order = np.argsort(eigvals_host)
            if not find_smallest:
                order = order[::-1]
            eigvals_host = eigvals_host[order]
            eigvecs_host = eigvecs_host[:, order]

            eigenvalues_buf[:] = eigvals_host[:num_modes]
            eigenvectors_buf[:] = eigvecs_host[:, :num_modes].T.reshape(-1)

            self.stats.num_solves += 1
            self.stats.last_solve_time = elapsed
            self.stats.last_num_modes = num_modes
            self.stats.last_info = 0
            self.stats.last_eigenvalues = eigvals_host[:num_modes].copy()
            self.stats.last_error = None
        except Exception as exc:
            self.stats.last_error = exc
            self.stats.last_info = -1
            raise

    @abstractmethod
    def _solve_eigen(
        self,
        K: Any,
        M: Any,
        *,
        num_modes: int,
        find_smallest: bool,
    ) -> tuple[Any, Any]:
        """Return (eigenvalues, eigenvectors) with eigenvectors column-major."""
