# API Reference

This section documents the public Python API for `openseespy-solvers`.

The public modules are:

| Module | Purpose |
|--------|---------|
| [`openseespy_solvers.scipy`](scipy.md) | CPU linear and eigen solver constructors |
| [`openseespy_solvers.scipy.precond`](precond.md) | CPU built-in preconditioners |
| [`openseespy_solvers.cupy`](cupy.md) | CUDA linear and eigen solver constructors |
| [`openseespy_solvers.cupy.precond`](precond_cupy.md) | CUDA built-in preconditioners |
| [`openseespy_solvers.nvmath`](nvmath.md) | NVIDIA `nvmath.sparse` direct solver constructor |
| [`openseespy_solvers.hybrid`](hybrid.md) | Hybrid direct-factorization/GMRES solver constructor |
| [`openseespy_solvers.exceptions`](exceptions.md) | Public exception types |

Private modules and names beginning with `_` are implementation details.

## Importing

Use the backend namespace that matches the solver you want:

```python
from openseespy_solvers.scipy import spsolve, eigsh
from openseespy_solvers.scipy import precond
```

For optional GPU backends:

```python
from openseespy_solvers.cupy import eigsh
from openseespy_solvers.nvmath import direct_solver
```

All solver constructors return solver objects. Register those objects with OpenSeesPy by
passing `solver.to_openseespy()` to `ops.system("PythonSparse", ...)` or
`ops.eigen("PythonSparse", ...)`.

## Linear Solvers

| Constructor | Backend | Purpose |
|-------------|---------|---------|
| [`scipy.spsolve`](scipy.md#openseespy_solvers.scipy.spsolve) | `scipy.sparse.linalg` / SuperLU | CPU sparse direct solve |
| [`scipy.umfpack`](scipy.md#openseespy_solvers.scipy.umfpack) | scikit-umfpack | CPU sparse direct solve using UMFPACK |
| [`scipy.pardiso`](scipy.md#openseespy_solvers.scipy.pardiso) | Intel MKL runtime | CPU sparse direct solve using PARDISO (LU, LDL^T, Cholesky) |
| [`scipy.cg`](scipy.md#openseespy_solvers.scipy.cg) | `scipy.sparse.linalg` | CPU Conjugate Gradient |
| [`scipy.gmres`](scipy.md#openseespy_solvers.scipy.gmres) | `scipy.sparse.linalg` | CPU GMRES |
| [`cupy.spsolve`](cupy.md#openseespy_solvers.cupy.spsolve) | `cupyx.scipy.sparse.linalg` | CUDA sparse direct solve |
| [`cupy.cg`](cupy.md#openseespy_solvers.cupy.cg) | `cupyx.scipy.sparse.linalg` | CUDA Conjugate Gradient |
| [`cupy.gmres`](cupy.md#openseespy_solvers.cupy.gmres) | `cupyx.scipy.sparse.linalg` | CUDA GMRES |
| [`nvmath.direct_solver`](nvmath.md#openseespy_solvers.nvmath.direct_solver) | NVIDIA `nvmath.sparse` | CUDA sparse direct solve |
| [`hybrid`](hybrid.md#openseespy_solvers.hybrid.hybrid) | `scipy`-compatible direct solver + GMRES | Reuse a direct factorization as a GMRES preconditioner |

## Eigen Solvers

| Constructor | Backend | Purpose |
|-------------|---------|---------|
| [`scipy.eigsh`](scipy.md#openseespy_solvers.scipy.eigsh) | `scipy.sparse.linalg` / ARPACK | CPU generalized symmetric eigen solve |
| [`scipy.lobpcg`](scipy.md#openseespy_solvers.scipy.lobpcg) | `scipy.sparse.linalg` | CPU LOBPCG eigen solve |
| [`cupy.eigsh`](cupy.md#openseespy_solvers.cupy.eigsh) | `cupyx.scipy.sparse.linalg` + `scipy.sparse.linalg` | CUDA-assisted generalized symmetric eigen solve |
| [`cupy.lobpcg`](cupy.md#openseespy_solvers.cupy.lobpcg) | `cupyx.scipy.sparse.linalg` | CUDA LOBPCG eigen solve |

## Preconditioners

| Preconditioner | Backend | Purpose |
|----------------|---------|---------|
| [`scipy.precond.jacobi`](precond.md#openseespy_solvers.scipy.precond.jacobi) | `scipy.sparse` | Diagonal/Jacobi preconditioner |
| [`scipy.precond.ilu`](precond.md#openseespy_solvers.scipy.precond.ilu) | `scipy.sparse.linalg` | Incomplete LU preconditioner |
| [`scipy.precond.direct`](precond.md#openseespy_solvers.scipy.precond.direct) | `scipy.sparse.linalg` | Direct-solver preconditioner |
| [`cupy.precond.jacobi`](precond_cupy.md#openseespy_solvers.cupy.precond.jacobi) | `cupyx.scipy.sparse` | Diagonal/Jacobi preconditioner |
| [`cupy.precond.ilu`](precond_cupy.md#openseespy_solvers.cupy.precond.ilu) | `cupyx.scipy.sparse.linalg` | Incomplete LU preconditioner |
| [`cupy.precond.direct`](precond_cupy.md#openseespy_solvers.cupy.precond.direct) | `cupyx.scipy.sparse.linalg` | Direct-solver preconditioner |

## API compatibility with scipy, cupy, and nvmath

If you already use [`scipy.sparse.linalg`](https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html)
or [`cupyx.scipy.sparse.linalg`](https://docs.cupy.dev/en/stable/reference/scipy_sparse_linalg.html),
the solver constructors in [`openseespy_solvers.scipy`](scipy.md) and
[`openseespy_solvers.cupy`](cupy.md) follow the same names and keyword arguments. The only
routine change is that you **do not pass `A` and `b` (or `K` and `M`)** — OpenSees
assembles those during analysis — and you **register** the solver with OpenSeesPy
instead of calling the solve function yourself.

### Direct solve

[`scipy.sparse.linalg`](https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html):

```python
import scipy.sparse.linalg as spla

x = spla.spsolve(A, b)
```

[`spsolve()`](scipy.md#openseespy_solvers.scipy.spsolve) in [`openseespy_solvers.scipy`](scipy.md) — same solver options, no matrix or RHS at construction:

```python
import openseespy.opensees as ops
from openseespy_solvers.scipy import spsolve

solver = spsolve()  # e.g. permc_spec=... like splu
ops.system("PythonSparse", solver.to_openseespy())
ops.analyze(num_steps)
```

### Iterative solve

[`scipy.sparse.linalg`](https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html):

```python
import scipy.sparse.linalg as spla

x, info = spla.cg(A, b, rtol=1e-8, M=M, maxiter=1000)
```

[`cg()`](scipy.md#openseespy_solvers.scipy.cg) in [`openseespy_solvers.scipy`](scipy.md) — same `rtol`, `atol`, `maxiter`, `M`, and related keywords:

```python
import openseespy.opensees as ops
from openseespy_solvers.scipy import cg, precond

solver = cg(rtol=1e-8, M=precond.jacobi, maxiter=1000)
ops.system("PythonSparse", solver.to_openseespy())
ops.analyze(num_steps)
```

[`openseespy_solvers.cupy`](cupy.md) mirrors the same pattern ([`spsolve()`](cupy.md#openseespy_solvers.cupy.spsolve), [`cg()`](cupy.md#openseespy_solvers.cupy.cg), …) for GPU backends.

### nvmath direct solve

[`nvmath.sparse.advanced.DirectSolver`](https://docs.nvidia.com/cuda/nvmath-python/latest/host-apis/sparse/generated/nvmath.sparse.advanced.DirectSolver.html)
takes the sparse matrix `a`, dense RHS `b`, and optional `options`, `execution`, and
`stream` keyword arguments:

```python
import nvmath

solver = nvmath.sparse.advanced.DirectSolver(a, b, execution=execution)
with solver:
    solver.plan()
    solver.factorize()
    x = solver.solve()
```

[`direct_solver()`](nvmath.md#openseespy_solvers.nvmath.direct_solver) in [`openseespy_solvers.nvmath`](nvmath.md) — pass the nvmath keywords you need at construction; OpenSees
supplies `a` and `b` on each solve and runs the `plan` / `factorize` / `solve` workflow
for you:

```python
import openseespy.opensees as ops
from openseespy_solvers.nvmath import direct_solver

solver = direct_solver(execution=execution, plan_algorithm=plan_algorithm)
ops.system("PythonSparse", solver.to_openseespy())
ops.analyze(num_steps)
```

[`direct_solver()`](nvmath.md#openseespy_solvers.nvmath.direct_solver) exposes `execution=`, `multithreading_lib=`, and `plan_algorithm=`;
`stream` and other `DirectSolverOptions` fields are not. See [nvmath](nvmath.md) for
details.

See [Solver objects](../user-guide/solver-objects.md) and
[PythonSparse interface](../development/pythonsparse-interface.md) for lifecycle details.
