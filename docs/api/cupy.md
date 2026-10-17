# openseespy_solvers.cupy

CUDA sparse linear and eigen solver constructors implemented with
[`cupyx.scipy.sparse.linalg`](https://docs.cupy.dev/en/stable/reference/scipy_sparse_linalg.html).

This module follows [`cupyx.scipy.sparse.linalg`](https://docs.cupy.dev/en/stable/reference/scipy_sparse_linalg.html) naming where possible. Constructors return
OpenSeesPy-compatible solver objects; OpenSeesPy supplies `A`, `b`, `K`, and `M` at solve
time. Importing this module requires `cupy`.

## Solving Linear Problems

Direct methods:

| Constructor | `cupy` analogue | Description |
|---------|---------------|-------------|
| [`spsolve`](#openseespy_solvers.cupy.spsolve) | `cupyx.scipy.sparse.linalg.splu` | Sparse direct solver; factorization on the CPU, triangular solves on the GPU |

Iterative methods:

| Constructor | `cupy` analogue | Description |
|---------|---------------|-------------|
| [`cg`](#openseespy_solvers.cupy.cg) | `cupyx.scipy.sparse.linalg.cg` | Conjugate Gradient on CUDA |
| [`gmres`](#openseespy_solvers.cupy.gmres) | `cupyx.scipy.sparse.linalg.gmres` | GMRES on CUDA |

## Eigenvalue Problems

| Constructor | `cupy`/`scipy` analogue | Description |
|---------|---------------------|-------------|
| [`eigsh`](#openseespy_solvers.cupy.eigsh) | `cupyx.scipy.sparse.linalg.eigsh` / `scipy.sparse.linalg.eigsh` | CUDA-assisted generalized symmetric eigen solve |
| [`lobpcg`](#openseespy_solvers.cupy.lobpcg) | `cupyx.scipy.sparse.linalg.lobpcg` | LOBPCG on CUDA |

## Installation

Install the CUDA extra matching your driver:

```bash
python -m pip install "openseespy-solvers[cuda13]"   # or [cuda12]
```

## OpenSeesPy Usage

```python
from openseespy_solvers.cupy import eigsh

solver = eigsh(tol=1e-8)
lam = ops.eigen("PythonSparse", num_modes, solver.to_openseespy())
```

## Notes

OpenSeesPy assembles matrices on the CPU. `cupy` solvers copy matrix data to the GPU before
solving, so GPU speedups depend on problem size, sparsity, and transfer overhead.

[`eigsh`](#openseespy_solvers.cupy.eigsh) solves `K x = lambda M x` with shift-invert support:

- `mass_mode="general"` uses `scipy.sparse.linalg.eigsh` with GPU inner solves when shift-invert is active.
- `mass_mode="diagonal"` and `mass_mode="lumped"` use GPU shift-invert with diagonal mass.

Raw [`cupyx.scipy.sparse.linalg.eigsh`](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.sparse.linalg.eigsh.html) does not accept a mass matrix; use [`lobpcg`](#openseespy_solvers.cupy.lobpcg) when you
need a different iterative eigen strategy.

## Function Reference

::: openseespy_solvers.cupy
    options:
      members:
        - spsolve
        - cg
        - gmres
        - eigsh
        - lobpcg
      show_root_heading: false
      heading_level: 3
