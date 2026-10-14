"""Base solver classes for the OpenSees PythonSparse interfaces.

These bases own the parts that are *identical for every backend*: reading the
OpenSees memoryviews, mapping OpenSees' sparse layout onto the layout a backend
wants (:class:`~openseespy_solvers._sparse.Layout`), the ``matrix_status``
caching pattern, writing the solution in place, statistics, error reporting,
``formAp``, ``to_openseespy`` and ``copy`` support.

Backend-specific numerics are provided by small hooks that each namespace
(``openseespy_solvers.scipy``, ``.cupy``, ...) implements:

* ``_build_matrix`` / ``_update_matrix`` - assemble the cached sparse matrix from
  values already in backend order
* ``_to_device`` / ``_to_host`` - host<->device transfer (identity on CPU)
* ``_matvec`` - ``A @ v``
* ``_is_sparse`` / ``_is_linear_operator`` - preconditioner detection
* ``_solve_system`` (linear) / ``_solve_eigen`` (eigen) - the actual library call

and two class attributes:

* ``_formats`` - compressed formats the backend accepts, in order of preference
  (``None``: CSR or CSC, preferring whatever OpenSees sends)
* ``_accepts_lower`` - whether the backend can use the lower triangle of a
  symmetric matrix directly

A backend that does not fit this shape can override ``solve`` directly.
"""

from __future__ import annotations

import copy
import time
import warnings
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from openseespy_solvers._dtype import OPENSEES_BUFFER_DTYPE, resolve_compute_dtype
from openseespy_solvers._sparse import (
    SYMMETRIC_LOWER,
    Layout,
    plan_layout,
    read_values,
    read_vector,
)
from openseespy_solvers.exceptions import InvalidOpenSeesDataError, SolverWarning
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
    #: Compressed formats the backend accepts, in order of preference. ``None``
    #: accepts CSR and CSC and keeps whichever OpenSees sends.
    _formats: tuple[str, ...] | None = None
    #: Whether the backend can use a lower-triangle (symmetric) matrix directly.
    _accepts_lower: bool = False

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
        self._layout: Layout | None = None
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
        """Build a new sparse matrix in the backend's format (owns its data).

        ``values`` are in ``fmt`` order and already cast to the compute dtype.
        """

    @abstractmethod
    def _update_matrix(self, matrix: Any, values: np.ndarray) -> Any:
        """Refresh coefficients in place (``values`` in backend order, float64)."""

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

    def _matrix_type_request(self) -> str | None:
        """``'SYMMETRIC_LOWER'`` when this solver asks OpenSees for lower storage."""
        return None

    def to_openseespy(
        self,
        *,
        scheme: str | None = None,
        writable: str | list[str] | None = None,
    ) -> dict[str, Any]:
        """Return the configuration dict for OpenSeesPy ``PythonSparse`` commands.

        Parameters
        ----------
        scheme : {'CSR', 'CSC', 'COO'}, optional
            Sparse storage scheme. Default is the value given at construction.
        writable : str or list of str, optional
            Writable buffers declared to OpenSees. Overrides the constructor
            value when provided.

        Returns
        -------
        config : dict
            Dictionary with keys ``solver``, ``scheme``, and optionally
            ``writable`` and ``matrix_type``. Pass directly to
            ``ops.system('PythonSparse', ...)`` or ``ops.eigen('PythonSparse', ...)``.

        Notes
        -----
        ``matrix_type`` is included only for solvers built with
        ``symmetric=True``. OpenSees builds without lower-triangle storage
        ignore the key and send the full matrix, which the solver detects.

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
        matrix_type = self._matrix_type_request()
        if matrix_type is not None:
            cfg["matrix_type"] = matrix_type
        return cfg

    # -- matrix caching --------------------------------------------------

    def _values_for_compute(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=self._compute_dtype)

    def _write_opensees_buffer(self, buf: np.ndarray, value: Any) -> None:
        buf[:] = np.asarray(self._to_host(value), dtype=OPENSEES_BUFFER_DTYPE)

    def _plan_layout(self, kwargs: dict[str, Any]) -> Layout:
        scheme = kwargs.get("storage_scheme", "CSR")
        formats = self._formats
        if formats is None:
            formats = ("CSC", "CSR") if scheme == "CSC" else ("CSR", "CSC")
        return plan_layout(kwargs, formats=formats, accepts_lower=self._accepts_lower)

    @property
    def symmetric_storage(self) -> bool:
        """``True`` when the cached matrix holds only the lower triangle."""
        return self._layout is not None and self._layout.lower

    def _new_matrix(self, values: np.ndarray) -> Any:
        layout = self._layout
        return self._build_matrix(
            self._values_for_compute(layout.reorder(values)),
            layout.indices,
            layout.indptr,
            layout.shape,
            layout.fmt,
        )

    def _current_matrix(self, kwargs: dict[str, Any], *, values_key: str = "values") -> Any:
        # The O(nnz) value buffer is read only when OpenSees reports a change,
        # so an UNCHANGED solve costs O(n).
        status = kwargs["matrix_status"]
        if status == "STRUCTURE_CHANGED" or self._matrix is None:
            self._layout = self._plan_layout(kwargs)
            self._matrix = self._new_matrix(read_values(kwargs, values_key))
        elif status == "COEFFICIENTS_CHANGED":
            values = self._layout.reorder(read_values(kwargs, values_key))
            self._matrix = self._update_matrix(self._matrix, values)
        return self._matrix

    def _current_eigen_matrices(self, kwargs: dict[str, Any]) -> tuple[Any, Any]:
        status = kwargs["matrix_status"]
        if status == "UNCHANGED" and self._k_matrix is not None:
            return self._k_matrix, self._m_matrix
        if status == "STRUCTURE_CHANGED" or self._k_matrix is None:
            self._layout = self._plan_layout(kwargs)
            self._k_matrix = self._new_matrix(read_values(kwargs, "k_values"))
            self._m_matrix = self._new_matrix(read_values(kwargs, "m_values"))
        else:
            layout = self._layout
            self._k_matrix = self._update_matrix(
                self._k_matrix, layout.reorder(read_values(kwargs, "k_values"))
            )
            self._m_matrix = self._update_matrix(
                self._m_matrix, layout.reorder(read_values(kwargs, "m_values"))
            )
        return self._k_matrix, self._m_matrix

    # -- error reporting -------------------------------------------------

    def _report_failure(self, where: str, exc: BaseException) -> None:
        """Emit the exception text (OpenSees only sees a negative status)."""
        warnings.warn(
            f"{type(self).__name__}.{where} failed: {type(exc).__name__}: {exc}",
            SolverWarning,
            stacklevel=3,
        )

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
        Cached system matrix from the last :meth:`solve` or :meth:`formAp` call,
        in the backend's preferred format (which may differ from ``scheme``).
    b : ndarray
        Right-hand side from the last :meth:`solve` call.
    x : ndarray
        Last solution vector.
    stats : LinearSolverStats
        Runtime statistics, updated after each :meth:`solve` when the solver was
        constructed with ``record_stats=True``. The relative residual
        (``stats.last_residual_norm``) costs one extra sparse matrix-vector
        product per solve and is computed only when ``record_stats`` and
        ``debug`` are both ``True``.
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
        super().__init__(scheme=scheme, debug=debug, dtype=dtype, record_stats=record_stats)
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

    def _symmetric_matvec(self, A: Any, v: Any) -> Any:
        """``A @ v`` for the full matrix when ``A`` holds only its lower triangle."""
        if not self.symmetric_storage:
            return self._matvec(A, v)
        lower = self._matvec(A, v)
        upper = self._matvec(A.T, v)
        diag = A.diagonal() * v
        return lower + upper - diag

    def _rhs_to_compute(self, rhs: np.ndarray) -> Any:
        return self._to_device(np.asarray(rhs, dtype=self._compute_dtype))

    def solve(self, **kwargs: Any) -> int:
        """Solve ``Ax = b`` using buffers supplied by OpenSees.

        This method is called by OpenSeesPy; application code normally does not
        invoke it directly.

        Parameters
        ----------
        **kwargs
            OpenSees ``PythonSparse`` buffers, including ``values``, ``rhs``,
            ``x``, ``num_eqn``, ``nnz``, ``matrix_status``, and
            ``storage_scheme`` (and ``matrix_type`` on builds that support it).

        Returns
        -------
        info : int
            ``0`` if the solve succeeded; a negative value otherwise. Failures
            emit a :class:`~openseespy_solvers.exceptions.SolverWarning` with the
            exception text; with ``debug=True`` the exception is raised instead.
        """
        try:
            matrix_status = kwargs["matrix_status"]
            num_eqn = int(kwargs["num_eqn"])
            if "rhs" not in kwargs or "x" not in kwargs:
                raise InvalidOpenSeesDataError("Linear solve requires rhs and x buffers")

            A = self._current_matrix(kwargs)
            rhs = read_vector(kwargs, "rhs")
            x_buf = read_vector(kwargs, "x")

            self._A = A
            b = self._rhs_to_compute(rhs)
            self._b = b

            M = self._resolve_preconditioner(A, matrix_status)
            start = time.perf_counter()
            result, info, num_iter = self._solve_system(A, b, M, matrix_status)
            elapsed = time.perf_counter() - start

            self._write_opensees_buffer(x_buf, result)
            self._x = result if self._on_device else x_buf

            residual = None
            # The residual doubles the matvec work; only with record_stats + debug.
            if self.record_stats and self.debug and num_eqn > 0:
                ax = np.asarray(
                    self._to_host(self._symmetric_matvec(A, result)), dtype=OPENSEES_BUFFER_DTYPE
                )
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
            self._report_failure("solve", exc)
            return -1

    def formAp(self, **kwargs: Any) -> int:
        try:
            if "p" not in kwargs or "Ap" not in kwargs:
                raise InvalidOpenSeesDataError("formAp requires p and Ap buffers")

            A = self._current_matrix(kwargs)
            self._A = A
            p = read_vector(kwargs, "p")
            ap_buf = read_vector(kwargs, "Ap")

            result = self._symmetric_matvec(
                A, self._to_device(np.asarray(p, dtype=self._compute_dtype))
            )
            self._write_opensees_buffer(ap_buf, result)
            return 0
        except Exception as exc:
            if self.record_stats:
                self.stats.last_error = exc
            if self.debug:
                raise
            self._report_failure("formAp", exc)
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

    def solve(self, **kwargs: Any) -> int:
        """Solve ``K phi = lambda M phi`` using buffers supplied by OpenSees.

        Returns
        -------
        info : int
            ``0`` on success, ``-1`` on failure (with a
            :class:`~openseespy_solvers.exceptions.SolverWarning`). With
            ``debug=True`` the exception is raised instead.
        """
        start = time.perf_counter()
        try:
            num_modes = int(kwargs["num_modes"])
            num_eqn = int(kwargs["num_eqn"])
            find_smallest = bool(kwargs["find_smallest"])
            if kwargs.get("storage_scheme", "CSR") == "COO":
                raise InvalidOpenSeesDataError(
                    "Eigen solvers support scheme='CSR' or 'CSC' (the OpenSees COO "
                    "eigen interface returns lists and is not supported here)."
                )

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
            return 0
        except Exception as exc:
            self.stats.last_error = exc
            self.stats.last_info = -1
            self.stats.last_solve_time = time.perf_counter() - start
            if self.debug:
                raise
            self._report_failure("solve", exc)
            return -1

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


# Re-exported for backends that need to know whether a request was honoured.
__all__ = ["BaseOpenSeesSolver", "EigenSolver", "LinearSolver", "SYMMETRIC_LOWER"]
