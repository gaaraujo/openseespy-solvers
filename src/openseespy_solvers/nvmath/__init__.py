"""Sparse direct solvers for OpenSeesPy (NVIDIA nvmath.sparse backend, GPU).

This module wraps :class:`nvmath.sparse.advanced.DirectSolver` (cuDSS) for
OpenSeesPy's ``PythonSparse`` linear system command on the GPU
(``cupyx.scipy.sparse`` matrices). Requires cupy and a CUDA-capable GPU; use
scipy solvers on CPU instead.

Importing this module does **not** require ``nvmath-python``; the dependency is
loaded when ``direct_solver()`` is called. Install a CUDA-matched extra, for example
``python -m pip install "openseespy-solvers[cuda13]"``.

See Also
--------
nvmath.sparse.advanced.DirectSolver
openseespy_solvers.cupy.spsolve
"""

from __future__ import annotations

from typing import Any

import numpy as np

from openseespy_solvers._base import LinearSolver
from openseespy_solvers._docstrings import (
    _LINEAR_NOTES,
    _LINEAR_RETURNS,
    _OPENSEES_LINEAR,
)
from openseespy_solvers.nvmath._base import (
    NvMathMixin,
    _find_multithreading_lib,
    _import_nvmath,
)

__all__ = ["direct_solver"]

_MATRIX_TYPES = ("general", "symmetric", "spd")


class _DirectSolver(NvMathMixin, LinearSolver):
    def __init__(
        self,
        *,
        sp_module: Any | None = None,
        device: str = "gpu",
        execution: Any | None = None,
        plan_algorithm: Any | None = None,
        multithreading_lib: str | None = None,
        matrix_type: str = "general",
        scheme: str = "CSR",
        writable: str | list[str] = "none",
        debug: bool = False,
        dtype: Any = np.float64,
        record_stats: bool = False,
    ) -> None:
        matrix_type = matrix_type.lower()
        if matrix_type not in _MATRIX_TYPES:
            raise ValueError(f"matrix_type must be one of {_MATRIX_TYPES}, got {matrix_type!r}")
        self._execution = execution
        self._plan_algorithm = plan_algorithm
        self._multithreading_lib = multithreading_lib
        self._matrix_type = matrix_type
        # cuDSS reads the lower triangle directly for symmetric types.
        self._accepts_lower = matrix_type != "general"
        self._solver: Any | None = None
        self._solver_shape: tuple[int, int] | None = None
        self._solver_matrix: Any | None = None
        self._solver_rhs: Any | None = None
        self._is_planned = False
        self._is_factorized = False
        self._params = {
            "sp_module": sp_module,
            "device": device,
            "execution": execution,
            "plan_algorithm": plan_algorithm,
            "multithreading_lib": multithreading_lib,
            "matrix_type": matrix_type,
            "scheme": scheme,
            "writable": writable,
            "debug": debug,
            "dtype": dtype,
            "record_stats": record_stats,
        }
        super().__init__(
            scheme=scheme,
            writable=writable,
            debug=debug,
            preconditioner=None,
            dtype=dtype,
            record_stats=record_stats,
            sp_module=sp_module,
            device=device,
        )

    def _matrix_type_request(self) -> str | None:
        return None if self._matrix_type == "general" else "SYMMETRIC_LOWER"

    def cleanup(self) -> None:
        """Release the underlying :class:`~nvmath.sparse.advanced.DirectSolver`."""
        if self._solver is not None:
            try:
                self._solver.__exit__(None, None, None)
            except Exception:  # pragma: no cover - best-effort cleanup
                pass
            finally:
                self._solver = None
                self._solver_shape = None
                self._solver_matrix = None
                self._solver_rhs = None
                self._is_planned = False
                self._is_factorized = False

    def __del__(self) -> None:
        self.cleanup()

    def __copy__(self) -> _DirectSolver:
        self.cleanup()
        return type(self)(**self._params)

    def _make_options(self) -> Any:
        advanced = self._nvmath.sparse.advanced
        kwargs: dict[str, Any] = {}
        multithreading_lib = self._multithreading_lib or _find_multithreading_lib()
        if multithreading_lib:
            kwargs["multithreading_lib"] = multithreading_lib
        if self._matrix_type != "general":
            kwargs["sparse_system_type"] = (
                advanced.DirectSolverMatrixType.SPD
                if self._matrix_type == "spd"
                else advanced.DirectSolverMatrixType.SYMMETRIC
            )
            # OpenSees sends the lower triangle when it supports
            # matrix_type='SYMMETRIC_LOWER'; older builds send the full matrix.
            kwargs["sparse_system_view"] = (
                advanced.DirectSolverMatrixViewType.LOWER
                if self.symmetric_storage
                else advanced.DirectSolverMatrixViewType.FULL
            )
        return advanced.DirectSolverOptions(**kwargs) if kwargs else None

    def _solve_system(self, A, b, M, matrix_status):  # noqa: ANN001
        if A.shape[0] == 0:
            return b, 0, None

        shape = tuple(A.shape)
        if self._solver is not None and (
            matrix_status == "STRUCTURE_CHANGED" or self._solver_shape != shape
        ):
            self.cleanup()

        if self._solver is None:
            solver_kwargs: dict[str, Any] = {}
            options = self._make_options()
            if options is not None:
                solver_kwargs["options"] = options
            if self._execution is not None:
                solver_kwargs["execution"] = self._execution
            self._solver = self._nvmath.sparse.advanced.DirectSolver(A, b, **solver_kwargs)
            self._solver.__enter__()
            self._solver_matrix = A
            self._solver_rhs = b
            self._solver_shape = shape
            self._is_planned = False
            self._is_factorized = False
        else:
            # GPU operands: the solver holds references to our buffers, which
            # are updated in place, so only a different object needs a reset
            # (resetting 'a' with GPU operands discards the plan). CPU operands
            # are copied into nvmath's device mirror on every reset.
            if matrix_status == "COEFFICIENTS_CHANGED":
                if not self._on_device or A is not self._solver_matrix:
                    self._solver.reset_operands(a=A)
                    if self._on_device:
                        self._is_planned = False
                    self._solver_matrix = A
                self._is_factorized = False
            if not self._on_device or b is not self._solver_rhs:
                self._solver.reset_operands(b=b)
                self._solver_rhs = b

        if not self._is_planned:
            if self._plan_algorithm is not None:
                self._solver.plan_config.algorithm = self._plan_algorithm
            self._solver.plan()
            self._is_planned = True

        if not self._is_factorized:
            self._solver.factorize()
            self._is_factorized = True

        result = self._solver.solve()
        if self._on_device:
            self._np_module.cuda.get_current_stream().synchronize()
        return result, 0, None


def direct_solver(
    *,
    sp_module: Any | None = None,
    device: str = "gpu",
    execution: Any | None = None,
    plan_algorithm: Any | None = None,
    multithreading_lib: str | None = None,
    matrix_type: str = "general",
    scheme: str | None = None,
    writable: str | list[str] = "none",
    debug: bool = False,
    dtype: Any = np.float64,
    record_stats: bool = False,
) -> _DirectSolver:
    r"""Configure an nvmath.sparse direct solver for OpenSees ``PythonSparse``.

    Uses :class:`nvmath.sparse.advanced.DirectSolver` (cuDSS) with separate plan,
    factorize, and solve phases. The factorization is reused while OpenSees
    reports ``matrix_status='UNCHANGED'``; it is refreshed on
    ``'COEFFICIENTS_CHANGED'`` and replanned on ``'STRUCTURE_CHANGED'``.

    Parameters
    ----------
    sp_module : module, optional
        Sparse matrix module: ``scipy.sparse`` for CPU or
        ``cupyx.scipy.sparse`` for GPU. Overrides ``device`` when given.
    device : {'gpu', 'cpu'}, optional
        Execution device when ``sp_module`` is not set. Default is ``'gpu'``
        (``cupyx.scipy.sparse``). Pass ``device='cpu'`` only for testing or
        environments without a GPU; CPU linear solves should normally use
        :func:`openseespy_solvers.scipy.spsolve` or :func:`~openseespy_solvers.scipy.umfpack`.
    execution : ExecutionCUDA or ExecutionHybrid, optional
        nvmath execution policy forwarded to :class:`~nvmath.sparse.advanced.DirectSolver`.
    plan_algorithm : DirectSolverAlgType, optional
        Planning algorithm assigned to ``solver.plan_config.algorithm`` before
        the first :meth:`~nvmath.sparse.advanced.DirectSolver.plan` call.
    multithreading_lib : str, optional
        Full path to the cuDSS threading-layer library (e.g.
        ``cudss_mtlayer_vcomp140.dll`` on Windows or ``libcudss_mtlayer_gomp.so.0``
        on Linux). When omitted, the backend searches ``CUDSS_THREADING_LIB`` and
        the ``nvidia-cudss-cu12`` / ``nvidia-cudss-cu13`` wheel layout.
    matrix_type : {'general', 'symmetric', 'spd'}, optional
        Factorization type. ``'symmetric'`` (LDL^T) and ``'spd'`` (Cholesky) are
        valid only for symmetric tangents. They also ask OpenSees to store only
        the lower triangle (``matrix_type='SYMMETRIC_LOWER'``), which halves the
        values copied to the GPU on OpenSees builds that support it; older builds
        send the full matrix and the factorization type still applies. Default
        is ``'general'`` (LU).
    """ + _OPENSEES_LINEAR + _LINEAR_RETURNS + _LINEAR_NOTES + """
    Raises
    ------
    ImportError
        If ``nvmath-python`` is not installed (raised when the solver is
        instantiated, not on import).

    Notes
    -----
    Prefer ``scheme='CSR'`` (the default). cuDSS operates on CSR; ``scheme='CSC'``
    or ``'COO'`` is mapped onto CSR once per sparsity pattern.

    Install cupy and ``nvmath-python`` wheels matching your CUDA driver; see
    :doc:`installation`.

    See Also
    --------
    nvmath.sparse.advanced.DirectSolver
    openseespy_solvers.scipy.spsolve
    openseespy_solvers.cupy.spsolve

    Examples
    --------
    >>> from openseespy_solvers.nvmath import direct_solver  # doctest: +SKIP
    >>> solver = direct_solver(device="cpu")  # doctest: +SKIP
    >>> solver.backend  # doctest: +SKIP
    'nvmath'
    """
    _import_nvmath()
    return _DirectSolver(
        sp_module=sp_module,
        device=device,
        execution=execution,
        plan_algorithm=plan_algorithm,
        multithreading_lib=multithreading_lib,
        matrix_type=matrix_type,
        scheme=scheme or "CSR",
        writable=writable,
        debug=debug,
        dtype=dtype,
        record_stats=record_stats,
    )
