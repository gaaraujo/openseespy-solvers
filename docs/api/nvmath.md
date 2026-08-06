# openseespy_solvers.nvmath

NVIDIA [`nvmath.sparse`](https://docs.nvidia.com/cuda/nvmath-python/latest/host-apis/sparse/index.html) direct solver constructor.

This module provides a CUDA direct solver for OpenSeesPy `PythonSparse` linear systems.
It wraps [`nvmath.sparse.advanced.DirectSolver`](https://docs.nvidia.com/cuda/nvmath-python/latest/host-apis/sparse/generated/nvmath.sparse.advanced.DirectSolver.html) and returns a solver object that can be
registered with `ops.system("PythonSparse", ...)`.

## Solving Linear Problems

| Constructor | nvmath analogue | Description |
|---------|-----------------|-------------|
| [`direct_solver`](#openseespy_solvers.nvmath.direct_solver) | [`nvmath.sparse.advanced.DirectSolver`](https://docs.nvidia.com/cuda/nvmath-python/latest/host-apis/sparse/generated/nvmath.sparse.advanced.DirectSolver.html) | Sparse direct solver on CUDA |

## Installation

Install the CUDA extra matching your driver:

```bash
python -m pip install "openseespy-solvers[cuda13]"   # or [cuda12]
```

## OpenSeesPy Usage

```python
from openseespy_solvers.nvmath import direct_solver

solver = direct_solver()
ops.system("PythonSparse", solver.to_openseespy())
```

## API compatibility with nvmath

[`direct_solver()`](#openseespy_solvers.nvmath.direct_solver) mirrors
[`nvmath.sparse.advanced.DirectSolver`](https://docs.nvidia.com/cuda/nvmath-python/latest/host-apis/sparse/generated/nvmath.sparse.advanced.DirectSolver.html)
where it can: pass `execution=` through to the underlying constructor, map
`multithreading_lib=` to `DirectSolverOptions`, and apply `plan_algorithm=` on
`solver.plan_config` before planning. OpenSees supplies `a` and `b`; the wrapper
calls `plan()`, `factorize()`, and `solve()` and reuses the factorization when
OpenSees reports `matrix_status='UNCHANGED'`.

Not exposed: `stream`, a full `options=` object, and other `plan_config` /
`factorization_config` settings. Use nvmath directly if you need those.

## Notes

[`direct_solver()`](#openseespy_solvers.nvmath.direct_solver) runs on the GPU by default. OpenSeesPy supplies CPU buffers; the solver
copies them to the device, factors the sparse matrix with `nvmath.sparse`/cuDSS, and writes the
solution back through the OpenSeesPy callback.

Prefer `scheme='CSR'` (the factory default): cuDSS uses CSR internally, so
`scheme='CSC'` forces an extra CSC→CSR conversion.

## Function Reference

::: openseespy_solvers.nvmath
    options:
      members:
        - direct_solver
      show_root_heading: false
      heading_level: 3
