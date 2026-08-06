"""Lightweight solver statistics objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LinearSolverStats:
    """Statistics for a linear (``Ax = b``) solver.

    Counters and timing update when the solver was constructed with
    ``record_stats=True``. ``last_residual_norm`` is set only when both
    ``record_stats`` and ``debug`` are ``True``; otherwise it remains ``None``.
    """

    num_solves: int = 0
    last_solve_time: float | None = None
    last_info: int | None = None
    last_residual_norm: float | None = None
    last_num_iterations: int | None = None
    num_factorizations: int = 0
    num_gmres_solves: int = 0
    last_error: BaseException | None = field(default=None, repr=False)


@dataclass
class EigenSolverStats:
    """Statistics for a generalized eigenvalue solver."""

    num_solves: int = 0
    last_solve_time: float | None = None
    last_num_modes: int | None = None
    last_info: int | None = None
    last_eigenvalues: Any = None
    last_error: BaseException | None = field(default=None, repr=False)
