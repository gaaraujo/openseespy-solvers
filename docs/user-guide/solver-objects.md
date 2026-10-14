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
: `scipy.sparse.spmatrix` or `cupyx.scipy.sparse.spmatrix` — cached system
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

Every solver constructor accepts ``dtype`` (default ``numpy.float64``). Supported values are
``float32`` and ``float64``. The internal sparse solve, matrix cache (``A``,
``K``, ``M``), and right-hand sides use that precision.

## Symmetric tangents

`pardiso(matrix_type='symmetric' | 'spd')` and
`nvmath.direct_solver(matrix_type='symmetric' | 'spd')` use LDL^T or Cholesky
and ask OpenSees to store only the lower triangle, which halves the values
copied per coefficient update. Use them only when the tangent is symmetric
(elastic, most plasticity with associated flow, no frictional contact or
follower loads). OpenSees builds without lower-triangle storage send the full
matrix; the factorization type still applies. Lower-triangle storage is an
upcoming OpenSees feature: the current `openseespy` wheel (3.8.0) always sends
the full matrix, which these solvers handle.

## Iterative solver tolerances

The smallest relative residual an iterative solver can reach is set by
round-off in the assembled tangent; on typical structural models a direct LU
solve itself stops near 1e-12 to 1e-11. Requesting `rtol` at or below that
level makes restarted GMRES or CG run to `maxiter` and report failure. Values
around `rtol=1e-8` give solutions that agree with a direct solve to about
1e-10 on such models. Inside Newton iterations the linear tolerance is the
inexact-Newton forcing term: a fixed tolerance gives linear convergence of
the Newton loop, so prefer a direct solver (or a tight tolerance) when the
Newton iteration count matters.

## See also

- [User guide overview](index.md)
- [PythonSparse interface](../development/pythonsparse-interface.md)
- [Tutorial](../getting-started.md)
