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
  matrix from the last solve or `formAp` call.

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

## Compute precision (`dtype`) *(experimental)*

Set `debug=True` at construction to propagate exceptions from `solve` and
`formAp` instead of returning a negative status code (linear solvers only).

Set `record_stats=True` to update `solver.stats` after each linear solve.
Relative residual (`last_residual_norm`) is computed only when both
`record_stats=True` and `debug=True` (extra sparse matvec).

## Compute precision (`dtype`) *(experimental)*

Every solver constructor accepts ``dtype`` (default ``numpy.float64``). Supported values are
``float32`` and ``float64``. The internal sparse solve, matrix cache (``A``,
``K``, ``M``), and right-hand sides use that precision.

## See also

- [User guide overview](index.md)
- [PythonSparse interface](../development/pythonsparse-interface.md)
- [Tutorial](../getting-started.md)
