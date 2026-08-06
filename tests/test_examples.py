"""OpenSeesPy integration: brick-bar helpers and example scripts."""

from __future__ import annotations

import io
import runpy
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
SOLVERS = EXAMPLES / "solvers"
sys.path.insert(0, str(EXAMPLES))
sys.path.insert(0, str(SOLVERS))

try:
    import openseespy.opensees as ops  # noqa: E402
except (ImportError, RuntimeError):
    pytest.skip("openseespy is not available", allow_module_level=True)

from conftest import apply_face_load, build_brick_bar  # noqa: E402
from openseespy_solvers.scipy import cg, precond, spsolve  # noqa: E402

# LOBPCG examples use a larger mesh and are not suited to the tiny smoke meshes
# used for other solvers; API coverage lives in tests/test_eigen.py.
SCIPY_SOLVER_SCRIPTS = sorted(
    p for p in SOLVERS.glob("scipy_*.py") if "lobpcg" not in p.stem
)
BENCHMARK_SCRIPTS = [EXAMPLES / "brick_bar.py", EXAMPLES / "brick_bar_eigen.py"]


def _run_example_script(path: Path) -> str:
    """Execute an example as ``__main__`` without inheriting pytest's argv."""
    out = io.StringIO()
    old_argv = sys.argv
    sys.argv = [str(path)]
    try:
        with redirect_stdout(out):
            runpy.run_path(str(path), run_name="__main__")
    finally:
        sys.argv = old_argv
    return out.getvalue()


def _run_static(ops, solver, *, steps: int) -> None:
    ops.system("PythonSparse", solver.to_openseespy())
    ops.numberer("RCM")
    ops.constraints("Plain")
    ops.integrator("LoadControl", 1.0 / steps)
    ops.test("NormUnbalance", 1.0e-7, 50)
    ops.algorithm("ModifiedNewton", "-FactorOnce")
    ops.analysis("Static", "-noWarnings")
    assert ops.analyze(steps) == 0


@pytest.mark.parametrize("script", SCIPY_SOLVER_SCRIPTS, ids=lambda p: p.stem)
def test_scipy_solver_example_runs(script: Path) -> None:
    if script.name == "scipy_umfpack.py":
        pytest.importorskip("scikits.umfpack")
    output = _run_example_script(script)
    assert "Passed!" in output


@pytest.mark.parametrize("script", BENCHMARK_SCRIPTS, ids=lambda p: p.stem)
def test_benchmark_example_runs(script: Path) -> None:
    output = _run_example_script(script)
    assert "Passed!" in output


def test_brick_bar_spsolve_integration() -> None:
    far_node = build_brick_bar(ops, nx=4, ny=1, nz=2)
    apply_face_load(ops)
    _run_static(ops, spsolve(), steps=2)
    assert far_node is not None
    assert ops.nodeDisp(far_node, 3) < 0.0


def test_brick_bar_cg_jacobi_integration() -> None:
    far_node = build_brick_bar(ops, nx=4, ny=1, nz=2)
    apply_face_load(ops)
    _run_static(ops, cg(rtol=1e-8, M=precond.jacobi), steps=2)
    assert far_node is not None
    assert ops.nodeDisp(far_node, 3) < 0.0


def test_brick_bar_cupy_spsolve() -> None:
    pytest.importorskip("cupy")
    from openseespy_solvers.cupy import spsolve as cupy_spsolve

    far_node = build_brick_bar(ops, nx=4, ny=1, nz=2)
    apply_face_load(ops)
    _run_static(ops, cupy_spsolve(), steps=2)
    assert far_node is not None
    assert ops.nodeDisp(far_node, 3) < 0.0


def test_cupy_solver_example_runs() -> None:
    pytest.importorskip("cupy")
    output = _run_example_script(SOLVERS / "cupy_spsolve.py")
    assert "Passed!" in output


def test_brick_bar_eigsh_two_tier_verification() -> None:
    """PythonSparse eigen must match genBandArpack or fullGenLapack tiebreaker."""
    import _brick_common as brick
    from openseespy_solvers.scipy import eigsh

    mesh = (3, 1, 2)

    def rebuild() -> None:
        build_brick_bar(ops, *mesh)

    build_brick_bar(ops, *mesh)
    far_node = brick.far_corner_node(ops)
    status, _ = brick.run_eigen_verified(
        ops, eigsh(tol=0.0), 3, far_node, rebuild, ev_rel_tol=1e-4, vec_rel_tol=0.02
    )
    assert status == 0, (
        f"PythonSparse eigsh failed two-tier verification "
        f"({brick.FAST_EIGEN_SOLVER} / {brick.TRUSTED_EIGEN_SOLVER})"
    )


def test_nvmath_solver_example_runs() -> None:
    pytest.importorskip("nvmath")
    pytest.importorskip("cupy")
    output = _run_example_script(SOLVERS / "nvmath_direct_solver.py")
    assert "Passed!" in output
