# openseespy_solvers.scipy

CPU sparse linear and eigen solver constructors.

This module follows the naming and keyword conventions of
[`scipy.sparse.linalg`](https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html) where possible. The main difference is that constructors return
OpenSeesPy-compatible solver objects; OpenSeesPy supplies `A`, `b`, `K`, and `M` at solve
time.

## Solving Linear Problems

Direct methods:

| Constructor | `scipy` analogue | Description |
|---------|----------------|-------------|
| [`spsolve`](#openseespy_solvers.scipy.spsolve) | `scipy.sparse.linalg.spsolve` / `splu` | Sparse direct solver using SuperLU |
| [`umfpack`](#openseespy_solvers.scipy.umfpack) | `scikits.umfpack` | Sparse direct solver using UMFPACK |

Iterative methods:

| Constructor | `scipy` analogue | Description |
|---------|----------------|-------------|
| [`cg`](#openseespy_solvers.scipy.cg) | `scipy.sparse.linalg.cg` | Conjugate Gradient |
| [`gmres`](#openseespy_solvers.scipy.gmres) | `scipy.sparse.linalg.gmres` | Generalized Minimal Residual |

## Eigenvalue Problems

| Constructor | `scipy` analogue | Description |
|---------|----------------|-------------|
| [`eigsh`](#openseespy_solvers.scipy.eigsh) | `scipy.sparse.linalg.eigsh` | ARPACK-based generalized symmetric eigen solve |
| [`lobpcg`](#openseespy_solvers.scipy.lobpcg) | `scipy.sparse.linalg.lobpcg` | Locally Optimal Block Preconditioned Conjugate Gradient |

## OpenSeesPy Usage

```python
from openseespy_solvers.scipy import spsolve

solver = spsolve()
ops.system("PythonSparse", solver.to_openseespy())
```

For eigen analysis:

```python
from openseespy_solvers.scipy import eigsh

solver = eigsh(tol=1e-8)
lam = ops.eigen("PythonSparse", num_modes, solver.to_openseespy())
```

## Notes

[`spsolve`](#openseespy_solvers.scipy.spsolve) and [`umfpack`](#openseespy_solvers.scipy.umfpack) are both sparse direct solvers. [`spsolve`](#openseespy_solvers.scipy.spsolve) uses [`scipy.sparse.linalg`](https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html)'s
SuperLU path and is available with the base install. `umfpack` requires the optional
`scikit-umfpack` package:

```bash
python -m pip install "openseespy-solvers[umfpack]"
```

On Windows, install `scikit-umfpack` from conda-forge instead.

[`umfpack`](#openseespy_solvers.scipy.umfpack) defaults to `scheme='CSC'` and prefers the
`di` (int32) family — matching OpenSees native `UmfPack` — widening to `dl`
(int64) only when needed (`index_dtype='auto'`). For shift-invert
[`eigsh`](#openseespy_solvers.scipy.eigsh), consider `scheme='CSC'` so OpenSees
assembles `K`/`M` for SciPy's SuperLU/UMFPACK factor; plain ARPACK matvecs are
usually happier with CSR (the default).

Parameters documented below follow `scipy` naming where possible. Extra parameters such as
`scheme`, `writable`, `debug`, `record_stats`, and `dtype` control the OpenSeesPy integration.

## Function Reference

::: openseespy_solvers.scipy
    options:
      members:
        - spsolve
        - umfpack
        - cg
        - gmres
        - eigsh
        - lobpcg
      show_root_heading: false
      heading_level: 3
