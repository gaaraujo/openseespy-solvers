"""Sparse linear algebra solvers for OpenSeesPy (`scipy.sparse.linalg` backend).

This module provides solver objects that wrap :mod:`scipy.sparse.linalg` for
OpenSeesPy's ``PythonSparse`` linear and eigen commands. Factory signatures
match `scipy.sparse.linalg` except that ``A`` and ``b`` (or ``K`` and ``M``) are supplied by
OpenSees at solve time.

Each factory returns a solver object. Register it with OpenSees using
:meth:`~openseespy_solvers._base.BaseOpenSeesSolver.to_openseespy`::

    from openseespy_solvers.scipy import cg
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
from openseespy_solvers._sparse import (
    OPENSEES_EIGSH_WHICH,
    eigsh_arpack_kwargs,
    opensees_eigsh_sigma,
)
from openseespy_solvers.scipy._base import ScipyMixin, _import_umfpack

__all__ = ["cg", "gmres", "spsolve", "umfpack", "eigsh", "lobpcg"]


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
        if self._x0 is not None:
            kwargs["x0"] = np.asarray(self._x0, dtype=self._compute_dtype)
        result, info = spla.gmres(A, b, **kwargs)
        return result, int(info), (count["n"] or None)


class _SpSolve(ScipyMixin, LinearSolver):
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
            self._solve_func = spla.splu(A.tocsc(), permc_spec=self._permc_spec).solve
        return self._solve_func(b), 0, None


class _Umfpack(ScipyMixin, LinearSolver):
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
        scheme: str = "CSR",
        debug: bool = False,
        dtype: Any = np.float64,
    ) -> None:
        super().__init__(scheme=scheme, debug=debug, dtype=dtype)
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
        return spla.eigsh(K, **kwargs)


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

    def solve(self, **kwargs: Any) -> None:
        status = kwargs["matrix_status"]
        self._eigen_matrix_status = status
        if status != "UNCHANGED":
            self._preconditioner_cached = None
        super().solve(**kwargs)

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
    reused when OpenSees reports ``matrix_status='UNCHANGED'``.

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
    'CSR'
    """
    return _SpSolve(
        permc_spec=permc_spec,
        scheme=scheme or "CSR",
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
        Maximum number of iterations.
    M : {sparse matrix, LinearOperator, callable}, optional
        Preconditioner for ``A``.
    callback : callable, optional
        User-supplied function called after each iteration.
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


def eigsh(
    *,
    sigma: float | None = None,
    which: str = "LM",
    v0: Any = None,
    ncv: int | None = None,
    maxiter: int | None = None,
    tol: float = 0.0,
    mode: str = "normal",
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
