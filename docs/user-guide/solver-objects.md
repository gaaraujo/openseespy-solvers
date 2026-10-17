# Solver objects

Solver constructors (for example [`spsolve()`](../api/scipy.md#openseespy_solvers.scipy.spsolve), [`cg()`](../api/scipy.md#openseespy_solvers.scipy.cg)) return objects that implement the OpenSees
`PythonSparse` protocol. Application code configures the solver, passes
`solver.to_openseespy()` to OpenSeesPy, and inspects cached data after analysis.

## Common methods

`to_openseespy()`
: Return a dict for OpenSeesPy; pass it unchanged. See
  [to_openseespy()](index.md#to_openseespy).

`solve(**kwargs)`
: Called by OpenSees; not normally invoked from application code.

`formAp(**kwargs)` *(linear solvers only)*
: Matrix-vector product `Ap = A @ p`.

## Linear solver attributes

All cached attributes are `None` until the first `solve` (or `formAp` for `A`).

`A`
: `scipy.sparse.spmatrix` or `cupyx.scipy.sparse.spmatrix`. Cached system
  matrix from the last solve or `formAp` call, in the format the backend
  factorizes (CSC for `spsolve`/`umfpack`, CSR for nvmath), which may differ
  from `scheme`. With symmetric storage it holds only the lower triangle.

`b`
: `numpy.ndarray` (CPU backends) or `cupy.ndarray` (GPU) — 1-D right-hand
  side in the solver's compute `dtype` from the last solve.

`x`
: `numpy.ndarray` (CPU) or `cupy.ndarray` (GPU) — 1-D solution from the last
  solve. On CPU backends this is the OpenSees `x` buffer (`float64`).

`stats`
: Runtime statistics (`num_solves`, `last_solve_time`, `last_info`,
  `last_num_iterations`, `last_residual_norm`, `last_error`).
  Counters and timing update only when constructed with `record_stats=True`.
  `last_residual_norm` additionally requires `debug=True` (extra SpMV);
  otherwise it remains `None`.

## Eigen solver attributes

`K`, `M`
: `scipy.sparse.spmatrix` or `cupyx.scipy.sparse.spmatrix` — cached stiffness
  and mass matrices from the last eigen solve, or `None` before the first solve.

`stats`
: Runtime statistics (`num_solves`, `last_solve_time`, `last_num_modes`,
  `last_eigenvalues`, `last_info`, `last_error`).

## copy.copy

`copy.copy(solver)` returns a new instance with the same keyword configuration
but reset internal cache and statistics. OpenSees uses this when cloning a
system of equations.

## Errors and statistics

A failed solve returns a negative status to OpenSees and emits a
`SolverWarning` with the exception text. Set `debug=True` at construction to
raise the exception instead (linear and eigen solvers).

Set `record_stats=True` to update `solver.stats` after each linear solve.
Relative residual (`last_residual_norm`) is computed only when both
`record_stats=True` and `debug=True` (extra sparse matvec).

## Compute precision (`dtype`) *(experimental)*

Solver constructors accept ``dtype`` (default ``numpy.float64``). Supported values are
``float32`` and ``float64``. `pardiso` accepts ``float64`` only, and `umfpack` always factorizes in ``float64``. The internal sparse solve, matrix cache (``A``,
``K``, ``M``), and right-hand sides use that precision.

## Symmetric tangents

`pardiso(matrix_type='symmetric' | 'spd')` and
`nvmath.direct_solver(matrix_type='symmetric' | 'spd')` factorize with LDL^T
or Cholesky and read only the lower triangle of the matrix. Use them only when
the assembled tangent is symmetric; the upper triangle is ignored, so a
nonsymmetric tangent gives a wrong solution without an error.

These solvers also ask OpenSees to send only the lower triangle
(`matrix_type='SYMMETRIC_LOWER'` in the configuration dict), which roughly halves the
values copied per coefficient update. That option requires a future OpenSees
release. The current `openseespy` wheel (3.8.0) ignores it and sends the full
matrix. `pardiso` then drops the upper triangle once per sparsity pattern, and
`nvmath.direct_solver` passes the full matrix to cuDSS; the factorization type
is the same in both cases.

## Iterative solver tolerances

The smallest relative residual an iterative solver can reach is limited by
round-off in the assembled tangent. On the solid brick-mesh models used to
benchmark this package (up to about 5,400 equations), a direct LU solve reaches
relative residuals of 1.4e-12 to 3.8e-12. Requesting `rtol` at or below that level makes
restarted GMRES or CG run to `maxiter` and report failure. With `rtol=1e-8`,
iterative and direct solutions agreed to 5e-10 or better on those models.

Inside a Newton iteration, the linear tolerance acts as the forcing term of an
inexact Newton method. A fixed tolerance gives only linear convergence of the
Newton loop, so use a direct solver, or a tolerance well below the Newton
convergence tolerance, when the number of Newton iterations matters.

## See also

- [User guide overview](index.md)
- [PythonSparse interface](../development/pythonsparse-interface.md)
- [Tutorial](../getting-started.md)
