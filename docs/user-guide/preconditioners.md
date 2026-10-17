# Preconditioners

Iterative solvers accept an optional preconditioner through the `M` keyword,
consistent with [`scipy.sparse.linalg.cg`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.cg.html)
and [`cupyx.scipy.sparse.linalg.cg`](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.sparse.linalg.cg.html).

OpenSees assembles the system matrix `A` during `solve`, not when you construct
the solver. Pass a callable `M(A)`; the solver calls it when the matrix
structure or coefficients change and reuses the cached result when
`matrix_status='UNCHANGED'`. Use a built-in preconditioner from
[`openseespy_solvers.scipy.precond`](../api/precond.md) or
[`openseespy_solvers.cupy.precond`](../api/precond_cupy.md), or supply your own callable `M(A)`.

## Built-in preconditioners

Pass the built-in callable to `M=` on an iterative solver such as [`cg()`](../api/scipy.md#openseespy_solvers.scipy.cg) (do not call it yourself):

```python
from openseespy_solvers.scipy import cg, precond

solver = cg(rtol=1e-8, M=precond.jacobi)
```

```python
from openseespy_solvers.cupy import cg, precond

solver = cg(rtol=1e-8, M=precond.jacobi)
```

When a built-in accepts extra keyword arguments (for example ``drop_tol`` on
[`ilu`](../api/precond.md#openseespy_solvers.scipy.precond.ilu)), wrap it in a
lambda so `M` stays a single-argument callable:

```python
solver = cg(
    rtol=1e-8,
    M=lambda A: precond.ilu(A, drop_tol=1e-4, fill_factor=10),
)
```

### scipy

[`openseespy_solvers.scipy.precond`](../api/precond.md) provides:

[`jacobi`](../api/precond.md#openseespy_solvers.scipy.precond.jacobi)
: `M = diag(1 / diag(A))`.

[`ilu`](../api/precond.md#openseespy_solvers.scipy.precond.ilu)
: Incomplete LU as a `LinearOperator`; extra keywords are forwarded to
  `scipy.sparse.linalg.spilu`.

[`direct`](../api/precond.md#openseespy_solvers.scipy.precond.direct)
: Full sparse factorization as a `LinearOperator`, primarily for
  [`lobpcg`](../api/scipy.md#openseespy_solvers.scipy.lobpcg) ``M=``.
  Pass a direct solver such as [`spsolve()`](../api/scipy.md#openseespy_solvers.scipy.spsolve), e.g. ``M=precond.direct(spsolve())``.

### cupy

[`openseespy_solvers.cupy.precond`](../api/precond_cupy.md) provides the same
names on GPU:

[`jacobi`](../api/precond_cupy.md#openseespy_solvers.cupy.precond.jacobi)
: Diagonal preconditioner as a device ``LinearOperator`` (element-wise apply).

[`ilu`](../api/precond_cupy.md#openseespy_solvers.cupy.precond.ilu)
: Incomplete LU via [`cupyx.scipy.sparse.linalg.spilu`](https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.sparse.linalg.spilu.html).
Defaults to ``fill_factor=1``, which factorizes on the GPU with no fill-in
(ILU(0)). Other ``fill_factor`` values factorize on the CPU with
`scipy.sparse.linalg`; only the triangular solves run on the GPU.

[`direct`](../api/precond_cupy.md#openseespy_solvers.cupy.precond.direct)
: Full sparse factorization as a ``LinearOperator``, primarily for
  [`lobpcg`](../api/cupy.md#openseespy_solvers.cupy.lobpcg) ``M=``.
  Pass a direct solver such as [`direct_solver()`](../api/nvmath.md#openseespy_solvers.nvmath.direct_solver), e.g. ``M=precond.direct(direct_solver())``.

## Your own preconditioner

Define a function `M(A)` that returns a `LinearOperator` or sparse matrix in the
same backend as the solver:

```python
import numpy as np
import scipy.sparse as sp
from openseespy_solvers.scipy import cg

def my_precond(A):
    d = A.diagonal()
    inv = np.ones_like(d)
    inv[d != 0] = 1.0 / d[d != 0]
    return sp.diags(inv)

solver = cg(rtol=1e-8, M=my_precond)
```

Other accepted values for `M`:

`None`
: No preconditioning.

`LinearOperator` or sparse matrix
: A fixed, pre-built `M`. Not recommended, because the number of equations
  is unknown until OpenSees builds the analysis model.

## See also

- [User guide overview](index.md)
- [`scipy.cg`](../api/scipy.md#openseespy_solvers.scipy.cg)
- [`scipy.gmres`](../api/scipy.md#openseespy_solvers.scipy.gmres)
- [scipy tutorial: Preconditioned conjugate gradient](https://docs.scipy.org/doc/scipy/tutorial/sparse.html)
