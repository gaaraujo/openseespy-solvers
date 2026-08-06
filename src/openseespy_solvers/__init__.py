"""scipy-style PythonSparse solvers for OpenSeesPy."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from openseespy_solvers.exceptions import (
    BackendNotAvailableError,
    InvalidOpenSeesDataError,
    OpenSeesSolverError,
    SolverConvergenceError,
    UnsupportedComputeDtypeError,
    UnsupportedStorageSchemeError,
)
from openseespy_solvers.hybrid import hybrid

__version__ = "0.2.0"

__all__ = [
    "BackendNotAvailableError",
    "InvalidOpenSeesDataError",
    "OpenSeesSolverError",
    "SolverConvergenceError",
    "UnsupportedComputeDtypeError",
    "UnsupportedStorageSchemeError",
    "__version__",
    "cupy",
    "hybrid",
    "scipy",
]

if TYPE_CHECKING:
    from openseespy_solvers import cupy as cupy
    from openseespy_solvers import scipy as scipy


def __getattr__(name: str):
    if name in {"scipy", "cupy"}:
        return import_module(f"openseespy_solvers.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
