"""Shared helpers for the Krylov (iterative) solvers."""

from __future__ import annotations

from typing import Any

#: Restart length cupyx.scipy.sparse.linalg.gmres uses when ``restart`` is None.
CUPYX_DEFAULT_RESTART = 20


def gmres_maxiter(func: Any, maxiter: int | None, restart: int | None, n: int) -> int | None:
    """Translate a SciPy-style GMRES ``maxiter`` for the GMRES function ``func``.

    The package uses SciPy's meaning everywhere: ``maxiter`` is the maximum
    number of restart cycles, so up to ``maxiter * restart`` iterations.
    :func:`cupyx.scipy.sparse.linalg.gmres` counts total iterations instead,
    so for it the value is multiplied by the effective restart length
    (``min(restart, n)``, CuPy's default restart being 20). ``None`` keeps the
    library's own default.
    """
    if maxiter is None:
        return None
    module = getattr(func, "__module__", "") or ""
    if not module.startswith("cupyx"):
        return maxiter
    r = restart if restart else CUPYX_DEFAULT_RESTART
    return int(maxiter) * max(1, min(int(r), int(n)))
