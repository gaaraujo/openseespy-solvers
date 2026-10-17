"""Sparse linear algebra solvers for OpenSeesPy (`scipy.sparse.linalg` backend).

This module provides solver objects that wrap :mod:`scipy.sparse.linalg`,
UMFPACK (``scikit-umfpack``) and Intel MKL PARDISO for OpenSeesPy's
``PythonSparse`` linear and eigen commands. Factory signatures match
`scipy.sparse.linalg` except that ``A`` and ``b`` (or ``K`` and ``M``) are
supplied by OpenSees at solve time.

Each factory returns a solver object. Register it with OpenSees using
:meth:`~openseespy_solvers._base.BaseOpenSeesSolver.to_openseespy`::

    from openseespy_solvers.scipy import cg, precond
    solver = cg(rtol=1e-8, M=precond.jacobi)
    ops.system("PythonSparse", solver.to_openseespy())

Submodules
----------
precond
    Preconditioner factories for use with ``M=``.

See Also
--------
scipy.sparse.linalg
    Underlying sparse linear algebra routines.
openseespy_solvers.cupy
    GPU backend with the same interface.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.sparse.linalg as spla

from openseespy_solvers._base import EigenSolver, LinearSolver
from openseespy_solvers._docstrings import (
    _EIGEN_NOTES,
    _EIGEN_RETURNS,
    _LINEAR_NOTES,
    _LINEAR_RETURNS,
    _OPENSEES_EIGEN,
    _OPENSEES_LINEAR,
)
from openseespy_solvers._factorization import FactorizedOperator
from openseespy_solvers._sparse import (
    OPENSEES_EIGSH_WHICH,
    eigsh_arpack_kwargs,
    opensees_eigsh_sigma,
)
from openseespy_solvers.scipy._base import ScipyMixin, _import_umfpack

__all__ = ["cg", "gmres", "spsolve", "umfpack", "pardiso", "eigsh", "lobpcg"]


class _CG(ScipyMixin, LinearSolver):
    def __init__(
        self,
        x0: Any = None,
        *,
        rtol: float = 1e-5,
        atol: float = 0.0,
        maxiter: int | None = None,
        M: Any = None,
        callback: Any = None,
        scheme: str = "CSR",
        writable: str | list[str] = "none",
        debug: bool = False,
        dtype: Any = np.float64,
        record_stats: bool = False,
    ) -> None:
        super().__init__(
            scheme=scheme,
            writable=writable,
            debug=debug,
            preconditioner=M,
            dtype=dtype,
            record_stats=record_stats,
        )
        self._x0 = x0
        self._rtol = rtol
        self._atol = atol
        self._maxiter = maxiter
        self._callback = callback
        self._params = {
            "x0": x0,
            "rtol": rtol,
            "atol": atol,
            "maxiter": maxiter,
            "M": M,
            "callback": callback,
            "scheme": scheme,
            "writable": writable,
            "debug": debug,
            "dtype": dtype,
            "record_stats": record_stats,
        }

    def _solve_system(self, A, b, M, matrix_status):  # noqa: ANN001
        count = {"n": 0}
        callback = None
        if self._callback is not None:

            def callback(_xk=None):  # noqa: ANN001
                count["n"] += 1
                self._callback()

        kwargs: dict[str, Any] = {
            "rtol": self._rtol,
            "atol": self._atol,
            "maxiter": self._maxiter,
            "M": M,
        }
        if callback is not None:
            kwargs["callback"] = callback
        if self._x0 is not None:
            kwargs["x0"] = np.asarray(self._x0, dtype=self._compute_dtype)
        result, info = spla.cg(A, b, **kwargs)
        return result, int(info), (count["n"] or None)


class _GMRES(ScipyMixin, LinearSolver):
    def __init__(
        self,
        x0: Any = None,
        *,
        rtol: float = 1e-5,
        atol: float = 0.0,
        restart: int | None = None,
        maxiter: int | None = None,
        M: Any = None,
        callback: Any = None,
        scheme: str = "CSR",
        writable: str | list[str] = "none",
        debug: bool = False,
        dtype: Any = np.float64,
        record_stats: bool = False,
    ) -> None:
        super().__init__(
            scheme=scheme,
            writable=writable,
            debug=debug,
            preconditioner=M,
            dtype=dtype,
            record_stats=record_stats,
        )
        self._x0 = x0
        self._rtol = rtol
        self._atol = atol
        self._restart = restart
        self._maxiter = maxiter
        self._callback = callback
        self._params = {
            "x0": x0,
            "rtol": rtol,
            "atol": atol,
            "restart": restart,
            "maxiter": maxiter,
            "M": M,
            "callback": callback,
            "scheme": scheme,
            "writable": writable,
            "debug": debug,
            "dtype": dtype,
            "record_stats": record_stats,
        }

    def _solve_system(self, A, b, M, matrix_status):  # noqa: ANN001
        count = {"n": 0}
        callback = None
        if self._callback is not None:

            def callback(_xk=None):  # noqa: ANN001
                count["n"] += 1
                self._callback()

        kwargs: dict[str, Any] = {
            "rtol": self._rtol,
            "atol": self._atol,
            "restart": self._restart,
            "maxiter": self._maxiter,
            "M": M,
        }
        if callback is not None:
            kwargs["callback"] = callback
            # 'pr_norm': one call per inner iteration with maxiter still counting
            # restart cycles. SciPy's fallback, 'legacy', silently makes maxiter
            # count inner iterations instead.
            kwargs["callback_type"] = "pr_norm"
        if self._x0 is not None:
            kwargs["x0"] = np.asarray(self._x0, dtype=self._compute_dtype)
        result, info = spla.gmres(A, b, **kwargs)
        return result, int(info), (count["n"] or None)


class _SpSolve(ScipyMixin, LinearSolver):
    # SuperLU factorizes CSC; other schemes are mapped once per pattern.
    _formats = ("CSC",)

    def __init__(
        self,
        *,
        permc_spec: str = "COLAMD",
        scheme: str = "CSR",
        writable: str | list[str] = "none",
        debug: bool = False,
        dtype: Any = np.float64,
        record_stats: bool = False,
    ) -> None:
        super().__init__(
            scheme=scheme,
            writable=writable,
            debug=debug,
            preconditioner=None,
            dtype=dtype,
            record_stats=record_stats,
        )
        self._permc_spec = permc_spec
        self._solve_func = None
        self._params = {
            "permc_spec": permc_spec,
            "scheme": scheme,
            "writable": writable,
            "debug": debug,
            "dtype": dtype,
            "record_stats": record_stats,
        }

    def _solve_system(self, A, b, M, matrix_status):  # noqa: ANN001
        if matrix_status != "UNCHANGED" or self._solve_func is None:
            self._solve_func = spla.splu(A, permc_spec=self._permc_spec).solve
        return self._solve_func(b), 0, None


class _Umfpack(ScipyMixin, LinearSolver):
    # UMFPACK factorizes CSC; other schemes are mapped once per pattern, so
    # A.tocsc() below returns A itself and its values are shared, not copied.
    _formats = ("CSC",)

    def __init__(
        self,
        *,
        scheme: str = "CSC",
        writable: str | list[str] = "none",
        debug: bool = False,
        dtype: Any = np.float64,
        index_dtype: str = "auto",
        record_stats: bool = False,
    ) -> None:
        super().__init__(
            scheme=scheme,
            writable=writable,
            debug=debug,
            preconditioner=None,
            dtype=dtype,
            record_stats=record_stats,
        )
        self._umfpack = _import_umfpack()
        # 'di' (int32) matches OpenSees native UmfPack; 'dl' only when needed.
        self._index_dtype = index_dtype
        self._umf = None
        self._csc = None
        self._params = {
            "scheme": scheme,
            "writable": writable,
            "debug": debug,
            "dtype": dtype,
            "index_dtype": index_dtype,
            "record_stats": record_stats,
        }

    def _select_family(self, A) -> str:  # noqa: ANN001
        if self._index_dtype in ("int64", "dl", "long"):
            return "dl"
        if self._index_dtype in ("int32", "di"):
            return "di"
        # auto: prefer int32 like native OpenSees when indices fit
        idx = A.indices
        if idx.dtype == np.int32 or idx.size == 0 or int(idx.max()) < 2**31:
            return "di"
        return "dl"

    def _to_csc_umf(self, A):  # noqa: ANN001, ANN201
        csc = A.tocsc()
        family = self._select_family(csc)
        if self._umf is None or getattr(self._umf, "family", None) != family:
            self._umf = self._umfpack.UmfpackContext(family)
        # Keep index width matching the UMFPACK family; avoid gratuitous int64 widen.
        idx_dtype = np.int64 if family == "dl" else np.int32
        if csc.data.dtype != np.float64 or not csc.data.flags["C_CONTIGUOUS"]:
            csc.data = np.ascontiguousarray(csc.data, dtype=np.float64)
        if csc.indices.dtype != idx_dtype or not csc.indices.flags["C_CONTIGUOUS"]:
            csc.indices = np.ascontiguousarray(csc.indices, dtype=idx_dtype)
        if csc.indptr.dtype != idx_dtype or not csc.indptr.flags["C_CONTIGUOUS"]:
            csc.indptr = np.ascontiguousarray(csc.indptr, dtype=idx_dtype)
        return csc

    def _solve_system(self, A, b, M, matrix_status):  # noqa: ANN001
        rhs = np.ascontiguousarray(b, dtype=np.float64)
        if A.shape[0] == 0:
            return rhs, 0, None
        need_symbolic = (
            matrix_status == "STRUCTURE_CHANGED"
            or self._csc is None
            or self._umf is None
            or getattr(self._umf, "_symbolic", None) is None
        )
        need_numeric = (
            matrix_status == "COEFFICIENTS_CHANGED"
            or self._umf is None
            or getattr(self._umf, "_numeric", None) is None
        )
        if need_symbolic:
            self._csc = self._to_csc_umf(A)
            self._umf.symbolic(self._csc)
            self._umf.numeric(self._csc)
        elif need_numeric:
            self._csc = self._to_csc_umf(A)
            self._umf.numeric(self._csc)
        x = self._umf.solve(self._umfpack.UMFPACK_A, self._csc, rhs)
        return x, 0, None


class _Pardiso(ScipyMixin, LinearSolver):
    _formats = ("CSR",)

    def __init__(
        self,
        *,
        matrix_type: str = "general",
        num_threads: int | None = None,
        iparm: dict[int, int] | None = None,
        scheme: str = "CSR",
        writable: str | list[str] = "none",
        debug: bool = False,
        dtype: Any = np.float64,
        record_stats: bool = False,
    ) -> None:
        from openseespy_solvers.scipy import _pardiso

        matrix_type = matrix_type.lower()
        mtypes = {
            "general": _pardiso.MTYPE_REAL_NONSYMMETRIC,
            "symmetric": _pardiso.MTYPE_REAL_SYMMETRIC_INDEFINITE,
            "spd": _pardiso.MTYPE_REAL_SPD,
        }
        if matrix_type not in mtypes:
            raise ValueError(f"matrix_type must be one of {tuple(mtypes)}, got {matrix_type!r}")
        if np.dtype(dtype) != np.float64:
            raise ValueError("pardiso() supports dtype=float64 only")
        super().__init__(
            scheme=scheme,
            writable=writable,
            debug=debug,
            preconditioner=None,
            dtype=dtype,
            record_stats=record_stats,
        )
        _pardiso.load_mkl()  # fail at construction if MKL is missing
        self._pardiso = _pardiso
        self._matrix_type = matrix_type
        self._mtype = mtypes[matrix_type]
        if matrix_type != "general":
            # PARDISO's symmetric types read the upper triangle in CSR, whose
            # arrays are exactly the lower triangle in CSC.
            self._formats = ("CSC",)
            self._lower_only = True
        self._num_threads = num_threads
        self._iparm = iparm
        self._handle: Any | None = None
        self._handle_matrix: Any | None = None
        self._params = {
            "matrix_type": matrix_type,
            "num_threads": num_threads,
            "iparm": iparm,
            "scheme": scheme,
            "writable": writable,
            "debug": debug,
            "dtype": dtype,
            "record_stats": record_stats,
        }

    def _matrix_type_request(self) -> str | None:
        return None if self._matrix_type == "general" else "SYMMETRIC_LOWER"

    def cleanup(self) -> None:
        """Release the PARDISO factorization."""
        if self._handle is not None:
            self._handle.release()
        self._handle_matrix = None

    def __copy__(self) -> _Pardiso:
        return type(self)(**self._params)

    def _solve_system(self, A, b, M, matrix_status):  # noqa: ANN001
        n = A.shape[0]
        if n == 0:
            return np.asarray(b, dtype=np.float64), 0, None
        if self._handle is None:
            self._handle = self._pardiso.PardisoHandle(
                self._mtype, num_threads=self._num_threads, iparm=self._iparm
            )
        if matrix_status == "STRUCTURE_CHANGED" or A is not self._handle_matrix:
            # PARDISO keeps pointers to these arrays; A's buffers are updated in
            # place on COEFFICIENTS_CHANGED, so they stay valid.
            self._handle.analyze(A.data, A.indptr, A.indices, n)
            self._handle.factorize()
            self._handle_matrix = A
        elif matrix_status == "COEFFICIENTS_CHANGED":
            self._handle.factorize()
        x = np.empty(n, dtype=np.float64)
        self._handle.solve(np.ascontiguousarray(b, dtype=np.float64), x)
        return x, 0, None


class _Eigsh(ScipyMixin, EigenSolver):
    def __init__(
        self,
        *,
        sigma: float | None = None,
        which: str = "LM",
        v0: Any = None,
        ncv: int | None = None,
        maxiter: int | None = None,
        tol: float = 0.0,
        mode: str = "normal",
        linear_solver: LinearSolver | None = None,
        scheme: str = "CSR",
        debug: bool = False,
        dtype: Any = np.float64,
    ) -> None:
        super().__init__(scheme=scheme, debug=debug, dtype=dtype)
        if linear_solver is not None and not isinstance(linear_solver, LinearSolver):
            raise TypeError("linear_solver must be a LinearSolver (e.g. pardiso(), umfpack())")
        self._linear_solver = linear_solver
        self._eigen_status: str | None = None
        self._opinv: Any | None = None
        self._opinv_sigma: float | None = None
        self._sigma = sigma
        self._which = which
        self._v0 = v0
        self._ncv = ncv
        self._maxiter = maxiter
        self._tol = tol
        self._mode = mode
        self._params = {
            "sigma": sigma,
            "which": which,
            "v0": v0,
            "ncv": ncv,
            "maxiter": maxiter,
            "tol": tol,
            "mode": mode,
            "linear_solver": linear_solver,
            "scheme": scheme,
            "debug": debug,
            "dtype": dtype,
        }

    def _solve_eigen(self, K, M, *, num_modes, find_smallest):  # noqa: ANN001
        kwargs = eigsh_arpack_kwargs(
            num_modes=num_modes,
            which=OPENSEES_EIGSH_WHICH,
            tol=self._tol,
            v0=self._v0,
            ncv=self._ncv,
            maxiter=self._maxiter,
            mode=self._mode,
        )
        kwargs["M"] = M
        sigma = opensees_eigsh_sigma(find_smallest, self._sigma)
        if sigma is not None:
            kwargs["sigma"] = sigma
            if self._linear_solver is not None:
                kwargs["OPinv"] = self._shift_invert_operator(K, M, sigma)
        return spla.eigsh(K, **kwargs)

    def solve(self, **kwargs: Any) -> int:
        if kwargs.get("matrix_status") == "STRUCTURE_CHANGED":
            self._opinv = None
        self._eigen_status = kwargs.get("matrix_status")
        return super().solve(**kwargs)

    def _shift_invert_operator(self, K, M, sigma: float):  # noqa: ANN001, ANN202
        """(K - sigma M)^{-1} through ``linear_solver``, factorized once per change."""
        shifted = (K - sigma * M).tocsr()
        if self._opinv is None or self._opinv.shape != shifted.shape or self._opinv_sigma != sigma:
            self._opinv = FactorizedOperator(self._linear_solver.copy(), shifted)
            self._opinv_sigma = sigma
        elif self._eigen_status != "UNCHANGED":
            self._opinv.update(shifted)
        op = self._opinv
        return spla.LinearOperator(shifted.shape, matvec=op.matvec, dtype=np.float64)


class _Lobpcg(ScipyMixin, EigenSolver):
    def __init__(
        self,
        *,
        X: Any = None,
        M: Any = None,
        tol: float | None = None,
        maxiter: int = 20,
        largest: bool = False,
        rng: Any = None,
        scheme: str = "CSR",
        debug: bool = False,
        dtype: Any = np.float64,
    ) -> None:
        super().__init__(scheme=scheme, debug=debug, dtype=dtype)
        self._X = X
        self._preconditioner = M
        self._preconditioner_cached: Any | None = None
        self._eigen_matrix_status = "STRUCTURE_CHANGED"
        self._tol = tol
        self._maxiter = maxiter
        self._largest = largest
        self._rng = rng
        self._params = {
            "X": X,
            "M": M,
            "tol": tol,
            "maxiter": maxiter,
            "largest": largest,
            "rng": rng,
            "scheme": scheme,
            "debug": debug,
            "dtype": dtype,
        }

    def solve(self, **kwargs: Any) -> int:
        status = kwargs["matrix_status"]
        self._eigen_matrix_status = status
        if status != "UNCHANGED":
            self._preconditioner_cached = None
        return super().solve(**kwargs)

    def _resolve_preconditioner(self, K: Any) -> Any | None:
        M = self._preconditioner
        if M is None:
            return None
        if self._is_sparse(M) or self._is_linear_operator(M):
            return M
        if callable(M):
            if self._eigen_matrix_status == "UNCHANGED" and self._preconditioner_cached is not None:
                return self._preconditioner_cached
            self._preconditioner_cached = M(K)
            return self._preconditioner_cached
        return M

    def _solve_eigen(self, K, M, *, num_modes, find_smallest):  # noqa: ANN001
        X = self._X
        if X is None:
            X = np.random.default_rng(self._rng).standard_normal(
                (K.shape[0], num_modes), dtype=self._compute_dtype
            )
        else:
            X = np.asarray(X, dtype=self._compute_dtype)
        kwargs: dict[str, Any] = {
            "B": M,
            "M": self._resolve_preconditioner(K),
            "maxiter": self._maxiter,
            "largest": not find_smallest,
        }
        if self._tol is not None:
            kwargs["tol"] = self._tol
        return spla.lobpcg(K, X, **kwargs)


def spsolve(
    *,
    permc_spec: str = "COLAMD",
    scheme: str | None = None,
    writable: str | list[str] = "none",
    debug: bool = False,
    dtype: Any = np.float64,
    record_stats: bool = False,
) -> _SpSolve:
    r"""Configure a sparse direct solver for OpenSees ``PythonSparse``.

    Uses :func:`scipy.sparse.linalg.splu` internally. The LU factorization is
    reused when OpenSees reports ``matrix_status='UNCHANGED'``. SuperLU works on
    CSC, so the default ``scheme`` is ``'CSC'`` (OpenSees then stores CSC and no
    conversion is needed); other schemes are mapped once per sparsity pattern.

    Parameters
    ----------
    permc_spec : str, optional
        Column permutation applied during LU factorization. Passed to
        :func:`scipy.sparse.linalg.splu`. Default is ``'COLAMD'``.
    """ + _OPENSEES_LINEAR + _LINEAR_RETURNS + _LINEAR_NOTES + """
    See Also
    --------
    scipy.sparse.linalg.spsolve
    scipy.sparse.linalg.splu

    Examples
    --------
    >>> from openseespy_solvers.scipy import spsolve
    >>> solver = spsolve(permc_spec="COLAMD")
    >>> cfg = solver.to_openseespy()
    >>> cfg["scheme"]
    'CSC'
    """
    return _SpSolve(
        permc_spec=permc_spec,
        scheme=scheme or "CSC",
        writable=writable,
        debug=debug,
        dtype=dtype,
        record_stats=record_stats,
    )


def umfpack(
    *,
    scheme: str | None = None,
    writable: str | list[str] = "none",
    debug: bool = False,
    dtype: Any = np.float64,
    index_dtype: str = "auto",
    record_stats: bool = False,
) -> _Umfpack:
    r"""Configure a UMFPACK direct solver for OpenSees ``PythonSparse``.

    Wraps ``scikits.umfpack.UmfpackContext``. By default prefers the ``di``
    (int32) family — matching OpenSees native ``UmfPack`` — and only widens to
    ``dl`` (int64) when indices require it (``index_dtype='auto'``). Prefer
    ``scheme='CSC'`` so OpenSees stores UMFPACK's native layout (avoids a
    CSR→CSC conversion).

    The symbolic factorization is reused while sparsity is unchanged and is
    recomputed only on ``matrix_status='STRUCTURE_CHANGED'``; numeric is
    refreshed on ``'COEFFICIENTS_CHANGED'`` and reused on ``'UNCHANGED'``.

    Requires the optional ``scikit-umfpack`` package
    (``python -m pip install "openseespy-solvers[umfpack]"``).

    Parameters
    ----------
    index_dtype : {'auto', 'int32', 'int64', 'di', 'dl'}, optional
        UMFPACK index width. ``auto`` (default) uses int32 when safe.
    """ + _OPENSEES_LINEAR + _LINEAR_RETURNS + _LINEAR_NOTES + """
    Raises
    ------
    ImportError
        If ``scikit-umfpack`` is not installed (raised when the solver is
        instantiated, not on import).

    See Also
    --------
    spsolve
    scipy.sparse.linalg.spsolve

    Examples
    --------
    >>> from openseespy_solvers.scipy import umfpack  # doctest: +SKIP
    >>> solver = umfpack(scheme="CSC")  # doctest: +SKIP
    >>> solver.to_openseespy()["scheme"]  # doctest: +SKIP
    'CSC'
    """
    return _Umfpack(
        scheme=scheme or "CSC",
        writable=writable,
        debug=debug,
        dtype=dtype,
        index_dtype=index_dtype,
        record_stats=record_stats,
    )


def cg(
    x0: Any = None,
    *,
    rtol: float = 1e-5,
    atol: float = 0.0,
    maxiter: int | None = None,
    M: Any = None,
    callback: Any = None,
    scheme: str | None = None,
    writable: str | list[str] = "none",
    debug: bool = False,
    dtype: Any = np.float64,
    record_stats: bool = False,
) -> _CG:
    r"""Configure a Conjugate Gradient solver for OpenSees ``PythonSparse``.

    Uses :func:`scipy.sparse.linalg.cg` internally to solve ``Ax = b`` for
    symmetric, positive-definite ``A``.

    Parameters
    ----------
    x0 : ndarray, optional
        Starting guess for the solution. Forwarded to `scipy.sparse.linalg` when not ``None``.
    rtol, atol : float, optional
        Relative and absolute tolerances for convergence. For convergence,
        ``norm(b - A @ x) <= max(rtol*norm(b), atol)`` must hold. Defaults are
        ``rtol=1e-5`` and ``atol=0.0``.
    maxiter : int, optional
        Maximum number of iterations.
    M : {sparse matrix, LinearOperator, callable}, optional
        Preconditioner for ``A``. If ``M`` is callable, it must accept the
        assembled matrix ``A`` and return a preconditioner; see
        :mod:`openseespy_solvers.scipy.precond`.
    callback : callable, optional
        User-supplied function called after each iteration as ``callback()``.
    """ + _OPENSEES_LINEAR + _LINEAR_RETURNS + _LINEAR_NOTES + """
    See Also
    --------
    scipy.sparse.linalg.cg
    openseespy_solvers.scipy.precond.jacobi
    openseespy_solvers.scipy.precond.ilu

    Examples
    --------
    >>> from openseespy_solvers.scipy import cg, precond
    >>> solver = cg(rtol=1e-8, M=precond.jacobi)
    >>> solver.to_openseespy()["scheme"]
    'CSR'
    """
    return _CG(
        x0,
        rtol=rtol,
        atol=atol,
        maxiter=maxiter,
        M=M,
        callback=callback,
        scheme=scheme or "CSR",
        writable=writable,
        debug=debug,
        dtype=dtype,
        record_stats=record_stats,
    )


def gmres(
    x0: Any = None,
    *,
    rtol: float = 1e-5,
    atol: float = 0.0,
    restart: int | None = None,
    maxiter: int | None = None,
    M: Any = None,
    callback: Any = None,
    scheme: str | None = None,
    writable: str | list[str] = "none",
    debug: bool = False,
    dtype: Any = np.float64,
    record_stats: bool = False,
) -> _GMRES:
    r"""Configure a GMRES solver for OpenSees ``PythonSparse``.

    Uses :func:`scipy.sparse.linalg.gmres` internally to solve ``Ax = b``.

    Parameters
    ----------
    x0 : ndarray, optional
        Starting guess for the solution.
    rtol, atol : float, optional
        Relative and absolute tolerances for convergence. Defaults are
        ``rtol=1e-5`` and ``atol=0.0``.
    restart : int, optional
        Number of iterations between restarts.
    maxiter : int, optional
        Maximum number of restart cycles (outer iterations), as in
        :func:`scipy.sparse.linalg.gmres`; up to ``maxiter * restart`` iterations
        in total. :func:`openseespy_solvers.cupy.gmres` and
        :func:`openseespy_solvers.hybrid` use the same meaning.
    M : {sparse matrix, LinearOperator, callable}, optional
        Preconditioner for ``A``.
    callback : callable, optional
        User-supplied function called after each inner iteration (no arguments).
        The iteration count in ``solver.stats`` counts the same calls.
    """ + _OPENSEES_LINEAR + _LINEAR_RETURNS + _LINEAR_NOTES + """
    See Also
    --------
    scipy.sparse.linalg.gmres
    cg

    Examples
    --------
    >>> from openseespy_solvers.scipy import gmres
    >>> solver = gmres(rtol=1e-8, restart=30)
    >>> solver.backend
    'scipy'
    """
    return _GMRES(
        x0,
        rtol=rtol,
        atol=atol,
        restart=restart,
        maxiter=maxiter,
        M=M,
        callback=callback,
        scheme=scheme or "CSR",
        writable=writable,
        debug=debug,
        dtype=dtype,
        record_stats=record_stats,
    )


def pardiso(
    *,
    matrix_type: str = "general",
    num_threads: int | None = None,
    iparm: dict[int, int] | None = None,
    scheme: str | None = None,
    writable: str | list[str] = "none",
    debug: bool = False,
    dtype: Any = np.float64,
    record_stats: bool = False,
) -> _Pardiso:
    r"""Configure an Intel MKL PARDISO direct solver for OpenSees ``PythonSparse``.

    Calls the PARDISO phases directly: analysis (11) on
    ``matrix_status='STRUCTURE_CHANGED'``, numerical factorization (22) on
    ``'COEFFICIENTS_CHANGED'``, and solve (33) on every call, so an
    ``'UNCHANGED'`` solve costs only the triangular solves. Needs the MKL
    runtime (``mkl_rt``) from conda ``mkl``, the ``mkl`` wheel, or Intel oneAPI;
    no Python wrapper package is required.

    Parameters
    ----------
    matrix_type : {'general', 'symmetric', 'spd'}, optional
        ``'general'`` (LU, mtype 11) for any tangent; ``'symmetric'`` (LDL^T,
        mtype -2) or ``'spd'`` (Cholesky, mtype 2) for symmetric tangents only.
        Symmetric types ask OpenSees for the lower triangle
        (``matrix_type='SYMMETRIC_LOWER'``), which PARDISO reads directly; older
        OpenSees builds send the full matrix and its upper part is dropped once
        per pattern. Default is ``'general'``.
    num_threads : int, optional
        MKL thread count (``MKL_Set_Num_Threads``). Default: MKL's setting.
    iparm : dict of int to int, optional
        Overrides for PARDISO ``iparm`` entries (zero-based index). The defaults
        are METIS ordering, zero-based indexing, and scaling plus weighted
        matching for ``'general'``.
    """ + _OPENSEES_LINEAR + _LINEAR_RETURNS + _LINEAR_NOTES + """
    Raises
    ------
    BackendNotAvailableError
        If the MKL runtime cannot be loaded.

    Examples
    --------
    >>> from openseespy_solvers.scipy import pardiso  # doctest: +SKIP
    >>> solver = pardiso(matrix_type="symmetric")  # doctest: +SKIP
    >>> solver.to_openseespy()["matrix_type"]  # doctest: +SKIP
    'SYMMETRIC_LOWER'
    """
    if scheme is None:
        scheme = "CSR" if matrix_type.lower() == "general" else "CSC"
    return _Pardiso(
        matrix_type=matrix_type,
        num_threads=num_threads,
        iparm=iparm,
        scheme=scheme,
        writable=writable,
        debug=debug,
        dtype=dtype,
        record_stats=record_stats,
    )


def eigsh(
    *,
    sigma: float | None = None,
    which: str = "LM",
    v0: Any = None,
    ncv: int | None = None,
    maxiter: int | None = None,
    tol: float = 0.0,
    mode: str = "normal",
    linear_solver: LinearSolver | None = None,
    scheme: str | None = None,
    debug: bool = False,
    dtype: Any = np.float64,
) -> _Eigsh:
    r"""Configure a sparse eigenvalue solver for OpenSees ``PythonSparse``.

    Matches OpenSees ``eigsh``: shift-invert with ``sigma=0`` when
    ``find_smallest=True``; plain ARPACK when ``find_smallest=False`` unless
    ``sigma`` is set. ARPACK always uses ``which='LM'`` (OpenSees reference).

    For shift-invert (smallest modes), prefer ``scheme='CSC'`` so OpenSees
    assembles ``K``/``M`` in the layout SciPy's SuperLU/UMFPACK factor expects.
    Default remains ``'CSR'`` (better for plain ARPACK matvecs).

    Parameters
    ----------
    sigma : float, optional
        Shift for shift-invert. Default ``0.0`` when ``find_smallest=True``.
    which : {'LM', 'SM', ...}, optional
        Accepted for API compatibility; solves always use ``'LM'`` like OpenSees.
    v0 : ndarray, optional
        Starting vector for ARPACK.
    ncv : int, optional
        Number of Lanczos vectors.
    maxiter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Convergence tolerance. Default is ``0.0``.
    mode : {'normal', 'buckling', 'cayley'}, optional
        Eigenproblem mode. Default is ``'normal'``.
    linear_solver : LinearSolver, optional
        Direct solver for the shift-invert operator ``(K - sigma M)^{-1}``, for
        example :func:`pardiso` or :func:`umfpack`. It factorizes once per
        change of ``K``/``M`` and each Lanczos iteration costs one triangular
        solve. Default: SciPy's internal SuperLU factorization.
    """ + _OPENSEES_EIGEN + _EIGEN_RETURNS + _EIGEN_NOTES + """
    See Also
    --------
    scipy.sparse.linalg.eigsh
    lobpcg

    Examples
    --------
    >>> from openseespy_solvers.scipy import eigsh
    >>> solver = eigsh(tol=1e-8)
    >>> solver.scheme
    'CSR'
    """
    return _Eigsh(
        sigma=sigma,
        which=which,
        v0=v0,
        ncv=ncv,
        maxiter=maxiter,
        tol=tol,
        mode=mode,
        linear_solver=linear_solver,
        scheme=scheme or "CSR",
        debug=debug,
        dtype=dtype,
    )


def lobpcg(
    *,
    X: Any = None,
    M: Any = None,
    tol: float | None = None,
    maxiter: int = 20,
    largest: bool = False,
    rng: Any = None,
    scheme: str | None = None,
    debug: bool = False,
    dtype: Any = np.float64,
) -> _Lobpcg:
    r"""Configure a LOBPCG eigenvalue solver for OpenSees ``PythonSparse``.

    Uses :func:`scipy.sparse.linalg.lobpcg` internally to solve the generalized
    eigenproblem ``K x = \lambda M x``.

    Parameters
    ----------
    X : ndarray, optional
        Initial approximation to the eigenvectors. If ``None``, a random matrix
        with ``num_modes`` columns is generated using ``rng``.
    M : {sparse matrix, LinearOperator, callable}, optional
        Optional preconditioner passed to LOBPCG as ``M=``. A callable is
        invoked as ``M(K)``; see :mod:`openseespy_solvers.scipy.precond`.
    tol : float, optional
        Convergence tolerance.
    maxiter : int, optional
        Maximum number of iterations. Default is ``20``.
    largest : bool, optional
        If ``True``, compute largest eigenvalues. Default is ``False``.
    rng : {None, int, `numpy.random.Generator`}, optional
        Random number generator used when ``X`` is ``None``.
    """ + _OPENSEES_EIGEN + _EIGEN_RETURNS + _EIGEN_NOTES + """
    See Also
    --------
    scipy.sparse.linalg.lobpcg
    eigsh

    Examples
    --------
    >>> from openseespy_solvers.scipy import lobpcg
    >>> solver = lobpcg(tol=1e-8, maxiter=40)
    >>> solver.backend
    'scipy'
    """
    return _Lobpcg(
        X=X,
        M=M,
        tol=tol,
        maxiter=maxiter,
        largest=largest,
        rng=rng,
        scheme=scheme or "CSR",
        debug=debug,
        dtype=dtype,
    )
