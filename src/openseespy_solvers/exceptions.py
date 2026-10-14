"""Exception types raised by openseespy-solvers."""

from __future__ import annotations


class OpenSeesSolverError(Exception):
    """Base class for all openseespy-solvers exceptions."""


class UnsupportedStorageSchemeError(OpenSeesSolverError, NotImplementedError):
    """Sparse storage scheme is not supported.

    Raised when ``storage_scheme`` is not ``'CSR'``, ``'CSC'`` or ``'COO'``, or
    when a solver does not support the scheme it was given.
    """


class SolverConvergenceError(OpenSeesSolverError):
    """Iterative or eigen solver failed to converge."""


class InvalidOpenSeesDataError(OpenSeesSolverError, ValueError):
    """OpenSees passed missing or malformed buffer arguments."""


class UnsupportedComputeDtypeError(OpenSeesSolverError, ValueError):
    """Requested compute dtype is not supported (only float32 and float64)."""


class BackendNotAvailableError(OpenSeesSolverError, ImportError):
    """Optional backend library is not installed.

    For example, importing :mod:`openseespy_solvers.cupy` without `cupy` raises
    this exception. Install the CUDA 13 extra, for example
    ``python -m pip install "openseespy-solvers[cuda13]"``.
    """


class SolverWarning(UserWarning):
    """A solve called by OpenSees failed.

    OpenSees only sees the negative status code, so the exception text is
    reported through :mod:`warnings`. Pass ``debug=True`` to a solver
    constructor to raise the exception instead.
    """
