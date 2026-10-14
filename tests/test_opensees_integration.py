"""End-to-end tests: real OpenSeesPy Newton loops through package solvers.

Each case runs the same analysis with ``system UmfPack`` (compiled) and with
``system PythonSparse`` + a package solver, and compares the displacements.

Set ``OPENSEESPY_PATH`` to a directory holding a locally built ``opensees``
module to test that build instead of the installed ``openseespy`` wheel.
Options that need a newer OpenSees (lower-triangle storage) are detected at
run time, not from the version number.
"""

from __future__ import annotations

import importlib
import os
import sys

import numpy as np
import pytest

GPU = pytest.mark.gpu


def _import_ops():
    path = os.environ.get("OPENSEESPY_PATH")
    if path:
        sys.path.insert(0, path)
        try:
            return importlib.import_module("opensees")
        finally:
            sys.path.remove(path)
    return pytest.importorskip("openseespy.opensees")


ops = _import_ops()


# -- models -------------------------------------------------------------------


def frame_model() -> int:
    """Nonlinear fiber cantilever with P-Delta: Newton iterates, tangent changes."""
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)
    ops.node(1, 0.0, 0.0)
    ops.node(2, 0.0, 120.0)
    ops.node(3, 0.0, 240.0)
    ops.fix(1, 1, 1, 1)
    ops.uniaxialMaterial("Steel01", 1, 50.0, 29000.0, 0.02)
    ops.section("Fiber", 1)
    ops.patch("rect", 1, 10, 1, -6.0, -6.0, 6.0, 6.0)
    ops.geomTransf("PDelta", 1)
    ops.beamIntegration("Lobatto", 1, 1, 5)
    ops.element("forceBeamColumn", 1, 1, 2, 1, 1)
    ops.element("forceBeamColumn", 2, 2, 3, 1, 1)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(3, 60.0, -200.0, 0.0)
    return 3


def contact_model() -> int:
    """Frictional contact past the slip limit: the tangent is nonsymmetric."""
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 2)
    for tag in (1, 2, 3):
        ops.node(tag, 0.0, 0.0)
    ops.fix(1, 1, 1)
    ops.fix(3, 1, 1)
    ops.element("zeroLengthContact2D", 1, 2, 1, 1.0e6, 1.0e6, 0.1, "-normal", 0.0, 1.0)
    ops.uniaxialMaterial("Elastic", 1, 5.0)
    ops.element("zeroLength", 2, 3, 2, "-mat", 1, "-dir", 1)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(2, 40.0, -200.0)
    return 2


def brick_model() -> int:
    """Elastic brick cantilever (symmetric positive definite)."""
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 3)
    ops.nDMaterial("ElasticIsotropic", 1, 29000.0, 0.3, 7.3e-7)
    ops.block3D(
        4, 2, 2, 1, 1, "stdBrick", 1,
        1, 0.0, 0.0, 0.0, 2, 10.0, 0.0, 0.0, 3, 10.0, 1.0, 0.0, 4, 0.0, 1.0, 0.0,
        5, 0.0, 0.0, 2.0, 6, 10.0, 0.0, 2.0, 7, 10.0, 1.0, 2.0, 8, 0.0, 1.0, 2.0,
    )
    ops.fixX(0.0, 1, 1, 1)
    tip = [n for n in ops.getNodeTags() if abs(ops.nodeCoord(n, 1) - 10.0) < 1e-9]
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    for n in tip:
        ops.load(n, 0.0, 0.0, -1.0)
    return tip[-1]


def run(build, system_args, *, algorithm="Newton", steps=10) -> np.ndarray:
    node = build()
    ops.constraints("Plain")
    ops.numberer("RCM")
    ops.system(*system_args)
    ops.test("NormDispIncr", 1e-10, 50)
    ops.algorithm(algorithm)
    ops.integrator("LoadControl", 1.0 / steps)
    ops.analysis("Static")
    assert ops.analyze(steps) == 0, f"analysis failed with {system_args[0]}"
    u = np.array(ops.nodeDisp(node))
    ops.wipe()
    return u


def reference(build, algorithm="Newton") -> np.ndarray:
    return run(build, ("UmfPack",), algorithm=algorithm)


def pythonsparse(solver, scheme):
    return ("PythonSparse", solver.to_openseespy(scheme=scheme))


def _supports_lower_storage() -> bool:
    seen = []

    class Probe:
        def solve(self, **kwargs):
            seen.append(kwargs.get("matrix_type"))
            from openseespy_solvers.scipy import spsolve

            return spsolve().solve(**kwargs)

    run(brick_model, ("PythonSparse", {"solver": Probe(), "matrix_type": "SYMMETRIC_LOWER"}),
        steps=1)
    return seen[:1] == ["SYMMETRIC_LOWER"]


# -- CPU ----------------------------------------------------------------------

SCHEMES = ["CSR", "CSC", "COO"]


def _cpu_direct():
    from openseespy_solvers.scipy import spsolve

    factories = {"spsolve": spsolve}
    try:
        from openseespy_solvers.scipy import umfpack

        umfpack()
        factories["umfpack"] = umfpack
    except ImportError:
        pass
    return factories


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("name", list(_cpu_direct()))
@pytest.mark.parametrize("build", [frame_model, contact_model], ids=["frame", "contact"])
def test_cpu_direct_newton(build, name, scheme) -> None:
    solver = _cpu_direct()[name]()
    np.testing.assert_allclose(
        run(build, pythonsparse(solver, scheme)), reference(build), rtol=1e-9, atol=1e-12
    )


@pytest.mark.parametrize("scheme", SCHEMES)
def test_cpu_gmres_newton_nonsymmetric(scheme) -> None:
    from openseespy_solvers.scipy import gmres

    solver = gmres(rtol=1e-13, restart=50, maxiter=200)
    np.testing.assert_allclose(
        run(contact_model, pythonsparse(solver, scheme)), reference(contact_model),
        rtol=1e-7, atol=1e-10,
    )


@pytest.mark.parametrize("scheme", SCHEMES)
def test_cpu_hybrid_modified_newton(scheme) -> None:
    from openseespy_solvers import hybrid
    from openseespy_solvers.scipy import spsolve

    solver = hybrid(spsolve(), rtol=1e-12, restart=30, record_stats=True)
    np.testing.assert_allclose(
        run(frame_model, pythonsparse(solver, scheme), algorithm="Newton"),
        reference(frame_model), rtol=1e-7, atol=1e-10,
    )
    assert solver.stats.num_factorizations >= 1


def test_cpu_hybrid_fallback_refactorizes() -> None:
    """GMRES limited to one iteration must fail and fall back to a refactorization."""
    from openseespy_solvers import hybrid
    from openseespy_solvers.scipy import spsolve

    solver = hybrid(spsolve(), rtol=1e-15, restart=1, maxiter=1, record_stats=True)
    np.testing.assert_allclose(
        run(contact_model, pythonsparse(solver, "CSR")), reference(contact_model),
        rtol=1e-9, atol=1e-12,
    )
    assert solver.stats.num_factorizations > 1


@pytest.mark.parametrize("scheme", ["CSR", "CSC"])
def test_symmetric_storage_request_is_safe_on_any_build(scheme) -> None:
    """matrix_type is ignored by older builds; the solver must stay correct."""
    from openseespy_solvers.scipy import spsolve

    class LowerSpsolve:
        def __init__(self):
            self.inner = spsolve()
            self.types = []

        def solve(self, **kwargs):
            self.types.append(kwargs.get("matrix_type"))
            return self.inner.solve(**kwargs)

    solver = LowerSpsolve()
    cfg = {"solver": solver, "scheme": scheme, "matrix_type": "SYMMETRIC_LOWER"}
    np.testing.assert_allclose(
        run(brick_model, ("PythonSparse", cfg), steps=1), reference(brick_model),
        rtol=1e-10,
    )
    expected = "SYMMETRIC_LOWER" if _supports_lower_storage() else None
    assert solver.types[0] == expected


# -- GPU ----------------------------------------------------------------------


def _gpu_available() -> bool:
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:  # pragma: no cover
        return False


@GPU
@pytest.mark.skipif(not _gpu_available(), reason="no GPU")
@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("build", [frame_model, contact_model], ids=["frame", "contact"])
def test_gpu_cudss_newton(build, scheme) -> None:
    pytest.importorskip("nvmath")
    from openseespy_solvers.nvmath import direct_solver

    solver = direct_solver()
    np.testing.assert_allclose(
        run(build, pythonsparse(solver, scheme)), reference(build), rtol=1e-9, atol=1e-12
    )


@GPU
@pytest.mark.skipif(not _gpu_available(), reason="no GPU")
@pytest.mark.parametrize("matrix_type", ["symmetric", "spd"])
@pytest.mark.parametrize("scheme", ["CSR", "CSC", "COO"])
def test_gpu_cudss_symmetric(matrix_type, scheme) -> None:
    pytest.importorskip("nvmath")
    from openseespy_solvers.nvmath import direct_solver

    solver = direct_solver(matrix_type=matrix_type)
    np.testing.assert_allclose(
        run(brick_model, pythonsparse(solver, scheme), steps=2), reference(brick_model),
        rtol=1e-9,
    )
    assert solver.symmetric_storage == _supports_lower_storage()


@GPU
@pytest.mark.skipif(not _gpu_available(), reason="no GPU")
def test_gpu_hybrid_modified_newton() -> None:
    pytest.importorskip("nvmath")
    from openseespy_solvers import hybrid
    from openseespy_solvers.nvmath import direct_solver

    solver = hybrid(direct_solver(), rtol=1e-12, restart=30)
    np.testing.assert_allclose(
        run(frame_model, pythonsparse(solver, "CSR")), reference(frame_model),
        rtol=1e-7, atol=1e-10,
    )
