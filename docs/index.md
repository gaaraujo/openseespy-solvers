# openseespy-solvers

`openseespy-solvers` provides `scipy`-style sparse linear and eigen solvers for
[OpenSeesPy](https://openseespydoc.readthedocs.io/) `PythonSparse` commands.

The package wraps numerical libraries such as
[`scipy.sparse.linalg`](https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html),
[`cupyx.scipy.sparse.linalg`](https://docs.cupy.dev/en/stable/reference/scipy_sparse.html),
and NVIDIA `nvmath.sparse` as solver objects. OpenSeesPy assembles the model matrices; the solver
object receives those arrays, runs the selected backend, and writes results back to
OpenSeesPy.

## How it works

OpenSeesPy runs the analysis loop on the CPU (state determination, assembly, Newton
iterations). When a linear solve is needed, it calls the `PythonSparse` solver object with
the assembled buffers; GPU backends copy the data to the device, solve, and copy the result back.

![OpenSeesPy PythonSparse workflow with optional GPU solve](assets/pythonsparse-gpu-workflow.png)

More detail on callbacks, matrix status, and what runs where:
[PythonSparse interface](development/pythonsparse-interface.md#parallelism).

## Start Here

Install the package with its CPU solvers (SciPy):

```bash
python -m pip install openseespy-solvers
```

Optional backends are installed as extras or separate packages:

```bash
python -m pip install "openseespy-solvers[opensees]"   # OpenSeesPy from PyPI
python -m pip install "openseespy-solvers[umfpack]"    # UMFPACK (Linux, macOS)
python -m pip install mkl                              # MKL runtime for PARDISO
python -m pip install "openseespy-solvers[cuda13]"     # or [cuda12]: cupy and nvmath
```

Match `cuda12` or `cuda13` to the CUDA version reported by `nvidia-smi`. On
Windows, install UMFPACK from conda-forge. Details:
[Installation](installation.md).

Then wire a solver into OpenSeesPy:

```python
import openseespy.opensees as ops
from openseespy_solvers.scipy import spsolve

solver = spsolve()
ops.system("PythonSparse", solver.to_openseespy())
```

For a full model example, continue with the [tutorial](getting-started.md).

## Recommended Defaults

| Analysis | CPU | NVIDIA GPU |
|----------|-----|-------------|
| Static or transient linear solve | [`spsolve`](api/scipy.md#openseespy_solvers.scipy.spsolve); [`umfpack`](api/scipy.md#openseespy_solvers.scipy.umfpack) for larger CPU systems | [`direct_solver`](api/nvmath.md#openseespy_solvers.nvmath.direct_solver) |
| Generalized eigen solve | [`eigsh`](api/scipy.md#openseespy_solvers.scipy.eigsh) | [`eigsh`](api/cupy.md#openseespy_solvers.cupy.eigsh) |

For iterative solvers, preconditioners, and [`hybrid()`](api/hybrid.md#openseespy_solvers.hybrid.hybrid), see the [API overview](api/index.md) and
[Tutorial](getting-started.md).

## Modules

| Module | Provides |
|--------|----------|
| [`scipy`](api/scipy.md) | CPU solvers: [`spsolve`](api/scipy.md#openseespy_solvers.scipy.spsolve), [`umfpack`](api/scipy.md#openseespy_solvers.scipy.umfpack), [`pardiso`](api/scipy.md#openseespy_solvers.scipy.pardiso), [`cg`](api/scipy.md#openseespy_solvers.scipy.cg), [`gmres`](api/scipy.md#openseespy_solvers.scipy.gmres), [`eigsh`](api/scipy.md#openseespy_solvers.scipy.eigsh), [`lobpcg`](api/scipy.md#openseespy_solvers.scipy.lobpcg) |
| [`scipy.precond`](api/precond.md) | CPU preconditioners: [`jacobi`](api/precond.md#openseespy_solvers.scipy.precond.jacobi), [`ilu`](api/precond.md#openseespy_solvers.scipy.precond.ilu), [`direct`](api/precond.md#openseespy_solvers.scipy.precond.direct) |
| [`cupy`](api/cupy.md) | GPU solvers: [`spsolve`](api/cupy.md#openseespy_solvers.cupy.spsolve), [`cg`](api/cupy.md#openseespy_solvers.cupy.cg), [`gmres`](api/cupy.md#openseespy_solvers.cupy.gmres), [`eigsh`](api/cupy.md#openseespy_solvers.cupy.eigsh), [`lobpcg`](api/cupy.md#openseespy_solvers.cupy.lobpcg) |
| [`cupy.precond`](api/precond_cupy.md) | GPU preconditioners: [`jacobi`](api/precond_cupy.md#openseespy_solvers.cupy.precond.jacobi), [`ilu`](api/precond_cupy.md#openseespy_solvers.cupy.precond.ilu), [`direct`](api/precond_cupy.md#openseespy_solvers.cupy.precond.direct) |
| [`nvmath`](api/nvmath.md) | GPU direct sparse solver: [`direct_solver`](api/nvmath.md#openseespy_solvers.nvmath.direct_solver) |
| [`hybrid`](api/hybrid.md) | [`hybrid()`](api/hybrid.md#openseespy_solvers.hybrid.hybrid): direct factorization reused as a GMRES preconditioner |

Constructor signatures match the underlying `scipy` or `cupy` functions where possible. The matrix
and right-hand side are supplied by OpenSeesPy at solve time.

## Helpful Pages

- [Installation](installation.md)
- [Tutorial](getting-started.md)
- [Examples](examples.md)
- [User guide](user-guide/index.md)
- [Solver objects](user-guide/solver-objects.md)
- [API reference](api/index.md)

## Support

Source, issues, and contributions are hosted on
[GitHub](https://github.com/gaaraujo/openseespy-solvers).
