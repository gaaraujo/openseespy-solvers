"""Hybrid direct-iterative linear solver (frozen factorization + GMRES)."""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np

from openseespy_solvers._base import LinearSolver
from openseespy_solvers._factorization import apply_inner_factorization
from openseespy_solvers.exceptions import InvalidOpenSeesDataError


def _iterative_kwargs(
    func: Any,
    *,
    rtol: float,
    atol: float,
    restart: int | None,
    maxiter: int | None,
    M: Any | None,
    callback: Any | None,
    x0: Any | None,
) -> dict[str, Any]:
    params = inspect.signature(func).parameters
    kwargs: dict[str, Any] = {"maxiter": maxiter, "M": M}
    if callback is not None:
        kwargs["callback"] = callback
    if x0 is not None:
        kwargs["x0"] = x0
    if "rtol" in params:
        kwargs["rtol"] = rtol
        if "atol" in params:
            kwargs["atol"] = atol
    else:
        kwargs["tol"] = rtol
        if "atol" in params and atol != 0.0:
            kwargs["atol"] = atol
    if restart is not None and "restart" in params:
        kwargs["restart"] = restart
    if "callback_type" in params:
        kwargs["callback_type"] = "legacy"
    return kwargs


class _Hybrid(LinearSolver):
    """Direct solver wrapper that reuses a frozen factorization as a GMRES preconditioner."""

    def __init__(
        self,
        direct: LinearSolver,
        *,
        rtol: float = 1e-6,
        atol: float = 0.0,
        restart: int | None = None,
        maxiter: int | None = None,
        x0: Any = None,
        refresh_every: int | None = None,
        debug: bool = False,
        record_stats: bool = False,
    ) -> None:
        if not isinstance(direct, LinearSolver):
            raise InvalidOpenSeesDataError(
                "hybrid() requires a LinearSolver instance from spsolve, umfpack, "
                "direct_solver, or similar."
            )
        params = getattr(direct, "_params", {})
        scheme = params.get("scheme", getattr(direct, "scheme", "CSR"))
        writable = params.get("writable", getattr(direct, "writable", "none"))
        dtype = params.get("dtype", np.float64)
        super().__init__(
            scheme=scheme,
            writable=writable,
            debug=debug,
            preconditioner=None,
            dtype=dtype,
            record_stats=record_stats,
        )
        self._inner = direct
        self.backend = direct.backend
        self._on_device = direct._on_device
        self._rtol = rtol
        self._atol = atol
        self._restart = restart
        self._maxiter = maxiter
        self._x0 = x0
        self._refresh_every = refresh_every
        self._frozen_n: int | None = None
        self._has_factor = False
        self._step = 0
        self._last_x: Any | None = None
        self._params = {
            "direct": direct,
            "rtol": rtol,
            "atol": atol,
            "restart": restart,
            "maxiter": maxiter,
            "x0": x0,
            "refresh_every": refresh_every,
            "debug": debug,
            "record_stats": record_stats,
        }

    def __copy__(self) -> _Hybrid:
        params = {k: v for k, v in self._params.items() if k != "direct"}
        return type(self)(self._inner.copy(), **params)

    def _build_matrix(self, values, indices, indptr, shape, fmt):  # noqa: ANN001
        return self._inner._build_matrix(values, indices, indptr, shape, fmt)

    def _update_matrix(self, matrix, values):  # noqa: ANN001
        return self._inner._update_matrix(matrix, values)

    def _to_device(self, array):  # noqa: ANN001
        return self._inner._to_device(array)

    def _to_host(self, array):  # noqa: ANN001
        return self._inner._to_host(array)

    def _matvec(self, matrix, vector):  # noqa: ANN001
        return self._inner._matvec(matrix, vector)

    def _is_sparse(self, obj):  # noqa: ANN001
        return self._inner._is_sparse(obj)

    def _is_linear_operator(self, obj):  # noqa: ANN001
        return self._inner._is_linear_operator(obj)

    def _gmres_callables(self) -> tuple[Any, Any]:
        if self._on_device:
            from openseespy_solvers.cupy._base import _import_cupy

            _, _, cspla = _import_cupy()
            return cspla.gmres, cspla.LinearOperator
        import scipy.sparse.linalg as spla

        return spla.gmres, spla.LinearOperator

    def _force_refresh(self) -> bool:
        if self._refresh_every is None:
            return False
        self._step += 1
        return self._step % self._refresh_every == 0

    def _apply_factorization(
        self,
        A: Any,
        vec: Any,
        *,
        refactor: bool,
        structure_changed: bool = False,
    ) -> Any:
        return apply_inner_factorization(
            self._inner,
            A,
            vec,
            refactor=refactor,
            on_device=self._on_device,
            structure_changed=structure_changed,
        )

    def _solve_system(self, A, b, M, matrix_status):  # noqa: ANN001
        n = A.shape[0]
        if n == 0:
            return b, 0, None

        structure_changed = self._frozen_n is not None and self._frozen_n != n
        need_factor = (
            self._frozen_n != n or not self._has_factor or self._force_refresh()
        )

        if (
            matrix_status == "UNCHANGED"
            and self._has_factor
            and self._frozen_n == n
            and not need_factor
        ):
            result = self._apply_factorization(A, b, refactor=False)
            self._last_x = result
            return result, 0, None

        if need_factor:
            result = self._apply_factorization(
                A,
                b,
                refactor=True,
                structure_changed=structure_changed or self._frozen_n is None,
            )
            self._has_factor = True
            self._frozen_n = n
            self._last_x = result
            if self.record_stats:
                self.stats.num_factorizations += 1
            return result, 0, None

        gmres_fn, LinearOperator = self._gmres_callables()
        count = {"n": 0}

        def callback(_xk=None):  # noqa: ANN001
            count["n"] += 1

        x0 = self._x0
        if x0 is not None:
            x0 = self._to_device(np.asarray(x0, dtype=self._compute_dtype))
        elif self._last_x is not None:
            last = self._last_x
            if hasattr(last, "shape") and len(last.shape) == 1 and last.shape[0] == n:
                x0 = last

        precond_op = LinearOperator(
            (n, n),
            matvec=lambda v: self._apply_factorization(A, v, refactor=False),
            dtype=getattr(A, "dtype", self._compute_dtype),
        )
        kwargs = _iterative_kwargs(
            gmres_fn,
            rtol=self._rtol,
            atol=self._atol,
            restart=self._restart,
            maxiter=self._maxiter,
            M=precond_op,
            callback=callback,
            x0=x0,
        )
        result, info = gmres_fn(A, b, **kwargs)
        if self.record_stats:
            self.stats.num_gmres_solves += 1

        if info != 0:
            result = self._apply_factorization(
                A,
                b,
                refactor=True,
                structure_changed=structure_changed,
            )
            self._has_factor = True
            self._frozen_n = n
            self._last_x = result
            if self.record_stats:
                self.stats.num_factorizations += 1
            return result, 0, (count["n"] or None)

        self._last_x = result
        return result, int(info), (count["n"] or None)


def hybrid(
    direct: LinearSolver,
    *,
    rtol: float = 1e-6,
    atol: float = 0.0,
    restart: int | None = None,
    maxiter: int | None = None,
    x0: Any = None,
    refresh_every: int | None = None,
    debug: bool = False,
    record_stats: bool = False,
) -> _Hybrid:
    r"""Configure a hybrid direct-iterative solver for OpenSees ``PythonSparse``.

    On the first solve (or when the number of equations changes), the inner
    *direct* solver builds and factorizes the matrix and returns a direct
    solution. On later solves with the same system size, the cached
    factorization is kept frozen and used as a GMRES preconditioner while the
    current matrix coefficients are applied to ``A``. The factorization is
    refreshed when:

    * the number of equations changes,
    * GMRES fails to converge,
    * ``refresh_every`` steps elapse (if set).

    The frozen factorization lives on the Python solver object, not inside
    OpenSees. Reuse the **same** ``solver`` instance across ``wipeAnalysis()``
    or ``wipe()``/model rebuilds (for example multiple ground motions with the
    same mesh): as long as ``num_eqn`` is unchanged, the first run's
    factorization is reused even when OpenSees reports
    ``matrix_status='STRUCTURE_CHANGED'``. Creating a new ``hybrid(...)`` or
    ``copy.copy(solver)`` starts with an empty factorization cache.

    Parameters
    ----------
    direct : LinearSolver
        Inner direct solver from :func:`~openseespy_solvers.scipy.spsolve`,
        :func:`~openseespy_solvers.scipy.umfpack`,
        :func:`~openseespy_solvers.cupy.spsolve`, or
        :func:`~openseespy_solvers.nvmath.direct_solver`. GMRES runs on the
        same device as the inner solver (CPU or GPU).
    rtol, atol : float, optional
        GMRES convergence tolerances. Defaults are ``rtol=1e-6`` and ``atol=0.0``.
    restart : int, optional
        GMRES restart length.
    maxiter : int, optional
        Maximum GMRES iterations.
    x0 : ndarray, optional
        Initial guess for GMRES. When omitted, the previous solution is reused
        when the system size is unchanged.
    refresh_every : int, optional
        Force a factorization refresh every this many solves. Default is
        ``None`` (refresh only on size change or GMRES failure).
    debug : bool, optional
        Re-raise exceptions instead of returning a failure code.
    record_stats : bool, optional
        If ``True``, update ``solver.stats`` after each solve. Relative residual
        requires both ``record_stats`` and ``debug``. Default is ``False``.

    Returns
    -------
    solver : _Hybrid
        Solver object for :meth:`~openseespy_solvers._base.BaseOpenSeesSolver.to_openseespy`.

    Examples
    --------
    >>> from openseespy_solvers import hybrid
    >>> from openseespy_solvers.scipy import spsolve
    >>> solver = hybrid(spsolve(), rtol=1e-6, restart=50)
    >>> solver.backend
    'scipy'
    """
    return _Hybrid(
        direct,
        rtol=rtol,
        atol=atol,
        restart=restart,
        maxiter=maxiter,
        x0=x0,
        refresh_every=refresh_every,
        debug=debug,
        record_stats=record_stats,
    )
