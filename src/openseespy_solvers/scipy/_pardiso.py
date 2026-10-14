"""Minimal ctypes binding to Intel MKL PARDISO.

Only the three phases a Newton loop needs are exposed (analysis, numerical
factorization, solve) plus release. No Python package beyond NumPy is
required: the MKL single dynamic library (``mkl_rt``) is loaded from the
active environment (conda ``mkl``, the ``mkl`` wheel, or Intel oneAPI).
Set ``OPENSEESPY_SOLVERS_MKL_RT`` to a full path to override the search.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
from pathlib import Path

import numpy as np

from openseespy_solvers.exceptions import BackendNotAvailableError, SolverConvergenceError

# PARDISO matrix types used here.
MTYPE_REAL_NONSYMMETRIC = 11
MTYPE_REAL_SYMMETRIC_INDEFINITE = -2
MTYPE_REAL_SPD = 2

PHASE_ANALYSIS = 11
PHASE_FACTORIZE = 22
PHASE_SOLVE = 33
PHASE_RELEASE = -1

_LIB = None


def _candidate_paths() -> list[str]:
    env = os.environ.get("OPENSEESPY_SOLVERS_MKL_RT")
    if env:
        return [env]
    if sys.platform == "win32":
        names = ["mkl_rt.3.dll", "mkl_rt.2.dll", "mkl_rt.dll"]
        dirs = [Path(sys.prefix) / "Library" / "bin", Path(sys.prefix) / "bin"]
        oneapi = os.environ.get("ONEAPI_ROOT", r"C:\Program Files (x86)\Intel\oneAPI")
        dirs.append(Path(oneapi) / "mkl" / "latest" / "bin")
    else:
        names = ["libmkl_rt.so.2", "libmkl_rt.so", "libmkl_rt.2.dylib", "libmkl_rt.dylib"]
        dirs = [Path(sys.prefix) / "lib"]
        oneapi = os.environ.get("ONEAPI_ROOT", "/opt/intel/oneapi")
        dirs.append(Path(oneapi) / "mkl" / "latest" / "lib")
    paths = [str(d / n) for d in dirs for n in names if (d / n).is_file()]
    found = ctypes.util.find_library("mkl_rt")
    if found:
        paths.append(found)
    return paths


def load_mkl() -> ctypes.CDLL:
    """Load ``mkl_rt`` (once) and declare ``pardiso``."""
    global _LIB
    if _LIB is not None:
        return _LIB
    errors = []
    for path in _candidate_paths():
        try:
            lib = ctypes.CDLL(path)
            lib.pardiso  # noqa: B018 - check the symbol exists
        except (OSError, AttributeError) as exc:
            errors.append(f"{path}: {exc}")
            continue
        lib.pardiso.restype = None
        lib.pardiso.argtypes = [
            ctypes.c_void_p,  # pt
            ctypes.POINTER(ctypes.c_int32),  # maxfct
            ctypes.POINTER(ctypes.c_int32),  # mnum
            ctypes.POINTER(ctypes.c_int32),  # mtype
            ctypes.POINTER(ctypes.c_int32),  # phase
            ctypes.POINTER(ctypes.c_int32),  # n
            ctypes.c_void_p,  # a
            ctypes.c_void_p,  # ia
            ctypes.c_void_p,  # ja
            ctypes.c_void_p,  # perm
            ctypes.POINTER(ctypes.c_int32),  # nrhs
            ctypes.c_void_p,  # iparm
            ctypes.POINTER(ctypes.c_int32),  # msglvl
            ctypes.c_void_p,  # b
            ctypes.c_void_p,  # x
            ctypes.POINTER(ctypes.c_int32),  # error
        ]
        _LIB = lib
        return lib
    detail = "; ".join(errors) if errors else "mkl_rt not found"
    raise BackendNotAvailableError(
        "The 'pardiso' solver needs the Intel MKL runtime (mkl_rt). Install it with "
        "'conda install mkl' or 'python -m pip install mkl', or set "
        f"OPENSEESPY_SOLVERS_MKL_RT to the library path ({detail})."
    )


class PardisoHandle:
    """One PARDISO factorization (internal pointer array + parameters).

    Arrays passed to :meth:`analyze`/:meth:`factorize` must stay alive and
    unchanged in address until :meth:`release`, because PARDISO keeps
    pointers to them between phases.
    """

    def __init__(self, mtype: int, *, num_threads: int | None = None,
                 iparm: dict[int, int] | None = None) -> None:
        self._lib = load_mkl()
        self.mtype = mtype
        self.pt = np.zeros(64, dtype=np.int64)
        self.iparm = np.zeros(64, dtype=np.int32)
        p = self.iparm
        p[0] = 1  # use the values below, not the defaults
        p[1] = 2  # METIS nested-dissection ordering
        p[9] = 13 if mtype == MTYPE_REAL_NONSYMMETRIC else 8  # pivot perturbation 1e-13 / 1e-8
        p[10] = 1 if mtype == MTYPE_REAL_NONSYMMETRIC else 0  # scaling
        p[12] = 1 if mtype == MTYPE_REAL_NONSYMMETRIC else 0  # weighted matching
        p[17] = -1  # report nnz in factors
        p[34] = 1  # zero-based ia/ja
        for key, value in (iparm or {}).items():
            p[int(key)] = int(value)
        if num_threads is not None:
            self._lib.MKL_Set_Num_Threads(ctypes.c_int(int(num_threads)))
        self.n = 0
        self._arrays: tuple | None = None
        self._initialized = False

    def _call(self, phase: int, a, ia, ja, b=None, x=None, nrhs: int = 1) -> int:
        i32 = ctypes.c_int32
        err = i32(0)
        null = ctypes.c_void_p()
        self._lib.pardiso(
            self.pt.ctypes.data_as(ctypes.c_void_p),
            ctypes.byref(i32(1)), ctypes.byref(i32(1)), ctypes.byref(i32(self.mtype)),
            ctypes.byref(i32(phase)), ctypes.byref(i32(self.n)),
            a.ctypes.data_as(ctypes.c_void_p) if a is not None else null,
            ia.ctypes.data_as(ctypes.c_void_p) if ia is not None else null,
            ja.ctypes.data_as(ctypes.c_void_p) if ja is not None else null,
            null, ctypes.byref(i32(nrhs)),
            self.iparm.ctypes.data_as(ctypes.c_void_p), ctypes.byref(i32(0)),
            b.ctypes.data_as(ctypes.c_void_p) if b is not None else null,
            x.ctypes.data_as(ctypes.c_void_p) if x is not None else null,
            ctypes.byref(err),
        )
        return int(err.value)

    def analyze(self, a: np.ndarray, ia: np.ndarray, ja: np.ndarray, n: int) -> None:
        self.release()
        self.n = int(n)
        self._arrays = (a, ia, ja)
        self._initialized = True
        self._check(self._call(PHASE_ANALYSIS, a, ia, ja), "analysis")

    def factorize(self) -> None:
        a, ia, ja = self._arrays
        self._check(self._call(PHASE_FACTORIZE, a, ia, ja), "factorization")

    def solve(self, b: np.ndarray, x: np.ndarray) -> None:
        a, ia, ja = self._arrays
        self._check(self._call(PHASE_SOLVE, a, ia, ja, b=b, x=x), "solve")

    def release(self) -> None:
        if self._initialized:
            self._call(PHASE_RELEASE, None, None, None)
            self.pt[:] = 0
            self._initialized = False
            self._arrays = None

    @staticmethod
    def _check(error: int, what: str) -> None:
        if error != 0:
            raise SolverConvergenceError(f"PARDISO {what} failed with error {error}")

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:  # pragma: no cover - interpreter shutdown
            pass
