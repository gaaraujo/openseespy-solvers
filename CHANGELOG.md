# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-10-16

### Added

- `scipy.pardiso(matrix_type="general" | "symmetric" | "spd")`: Intel MKL
  PARDISO through a ctypes binding; needs only the MKL runtime (`mkl_rt`).
  Analysis runs on `STRUCTURE_CHANGED`, numerical factorization on
  `COEFFICIENTS_CHANGED`, and only the solve phase on `UNCHANGED`.
- `scipy.eigsh(linear_solver=...)`: shift-invert with a package direct solver
  (for example `pardiso()` or `umfpack()`). `K - sigma M` is factorized once per
  eigen call.
- `nvmath.direct_solver(matrix_type="symmetric" | "spd")`: LDL^T or Cholesky
  in cuDSS.
- `scheme="COO"` for linear solvers.
- Symmetric solvers request lower-triangle storage
  (`"matrix_type": "SYMMETRIC_LOWER"` in the configuration dict), and every
  solver accepts lower-triangle buffers. This storage requires a future
  OpenSees release; the current `openseespy` wheel (3.8.0) ignores the key and
  sends the full matrix, which the solvers detect on each solve.
- `SolverWarning`: a failed solve reports the exception text.
- Documentation for `DistributedPythonSparse` (MPI OpenSeesPy builds from a
  future OpenSees release; not available in the 3.8.0 wheel) and for the
  `openseespy-solvers` branch of gaaraujo/OpenSees, which provides the
  unreleased OpenSees features ahead of an OpenSees release.
- End-to-end tests with OpenSeesPy (CSR, CSC and COO, nonsymmetric tangents,
  coefficient updates, `hybrid()` fallback, cuDSS) and a CI job that runs them
  against the PyPI wheel. GPU tests are marked `gpu`.

### Changed

- GMRES `maxiter` counts restart cycles on the CPU and the GPU, as in SciPy.
  `cupy.gmres` and `hybrid()` on the GPU convert it to the total iteration
  count that `cupyx` expects.
- The mapping from the OpenSees layout to the format each backend factorizes is
  computed once per sparsity pattern. SuperLU and UMFPACK no longer convert CSR
  to CSC on every factorization, and `spsolve()` and `cupy.spsolve()` default
  to `scheme="CSC"`.
- `UNCHANGED` solves no longer read or cast the matrix values.
- GPU solvers reuse preallocated device buffers for the matrix values and the
  right-hand side, and copy the solution directly into the OpenSees buffer.
- `hybrid()` applies the frozen factorization without re-marshalling the
  matrix or copying vectors through the host on each GMRES iteration.
- `EigenSolver.solve` returns `0` or `-1` like `LinearSolver.solve` instead of
  raising; `debug=True` still raises.
- The `[cuda12]` and `[cuda13]` extras require `nvmath-python>=1.0.0`
  (cuDSS 0.8).
- The `[opensees]` extra requires `openseespy>=3.8.0`.
- `cupy.spsolve` documents that `cupyx` `splu` factorizes on the CPU.

### Fixed

- `nvmath.direct_solver(scheme="CSC")` wrote coefficient updates in CSC order
  into a CSR matrix, which is wrong for nonsymmetric matrices.
- `nvmath.direct_solver` on the GPU failed every `COEFFICIENTS_CHANGED` solve
  because resetting the GPU operand discarded the cuDSS plan.
- `hybrid()` refactorized after a GMRES failure with `COEFFICIENTS_CHANGED`
  even when the sparsity pattern had changed.
- `hybrid()` on the GPU failed on every call because `cupyx` gmres does not
  accept `callback_type="legacy"`.
- `scipy.gmres` with a callback counted `maxiter` in inner iterations instead
  of restart cycles (SciPy's `legacy` callback type).
- `precond.direct()` refactorized a new preconditioner with
  `COEFFICIENTS_CHANGED` instead of `STRUCTURE_CHANGED`.
- Failed solves were silent unless `debug=True`.

## [0.2.0] - 2026-08-06

### Added

- `record_stats=True` on linear solvers — opt-in update of `solver.stats`
  (counters, timing, `last_info`). Relative residual (`last_residual_norm`,
  extra SpMV) requires both `record_stats` and `debug`.
- `umfpack(..., index_dtype=...)` — `'auto'` / `'int32'` / `'di'` prefer UMFPACK
  `di`; `'int64'` / `'dl'` force `dl` (mirrors native `useLongIndices`).

### Changed

- Solver stats and residual SpMV are no longer updated on every solve (was a
  fixed overhead even when unused). `debug` alone only re-raises exceptions.
- `umfpack()` default `scheme` is `'CSC'` (UMFPACK-native layout).
- Docs: prefer CSC for UMFPACK; prefer CSR for nvmath/cuDSS; drop always-`dl`
  wording.

### Fixed

- SciPy matrix build avoids pointless extra copies while keeping owned C arrays
  for UMFPACK.
- Inner direct solvers (e.g. `precond.direct(umfpack())`) marshal buffers in
  the inner solver's `scheme` instead of always CSR.

## [0.1.3] - 2026-06-08

### Changed

- User guide: focus on `ops.system` / `ops.eigen`; simplified `to_openseespy()`,
  preconditioners (built-in vs custom), and solver-object pages.
- API overview: scipy/cupy/nvmath compatibility examples; cross-links to module
  and constructor reference pages.
- Docs terminology: **solver constructor** instead of factory; Development overview
  with GitHub repository, issues, and pull-request links.
- RTD navigation: Get Started, User Guide, API Reference, and Development tabs.

## [0.1.2] - 2026-06-08

### Added

- Brick-bar benchmark plot script (`examples/plot_brick_bar_benchmark.py`) with live
  progress, incremental CSV writes, and `--append` resume.
- Published static and eigen benchmark assets under `docs/assets/`.
- Contributor guide: [Adding a solver](adding-a-solver.md).
- Tutorial FEM theory section and MathJax support in the docs site.

### Changed

- Rewrote the [Examples](examples.md) page with mesh tables, analysis snippets,
  and timing plots (mesh sweeps through factor 11, ~91k equations).
- Normalized backend naming in solver docstrings and PyPI description.
- Docs navigation: merged API reference, workflow diagram, removed
  `recommended-solvers.md`; refreshed preconditioner guidance.
- Static benchmark: dropped native OpenSees `UmfPack` from comparisons.
- Examples: `RCM` numberer for direct solvers; benchmark scripts skip completed
  mesh/solver pairs when appending results.

### Fixed

- Example smoke tests: isolate `sys.argv` so pytest flags are not passed to
  `brick_bar.py` / `brick_bar_eigen.py` argparse.

## [0.1.1] - 2026-06-05

### Added

- `hybrid(direct=...)` — frozen direct factorization reused as a GMRES preconditioner.
- CUDA optional extras `[cuda12]` and `[cuda13]` (matching `cupy` + nvmath GPU wheels).
- Documentation: full install walkthrough, verify checklist (`pytest` / examples / benchmarks), serial OpenSeesPy note.

### Changed

- Install hints and error messages recommend `[cuda12]` / `[cuda13]` instead of generic `[cupy]`.
- PyPI metadata: description, classifiers, and project URLs.
- LOBPCG example scripts use a larger mesh and are manual-only (not pytest smoke tests).

## [0.1.0] - 2026-06-03

### Added

- Per-backend namespaces `openseespy_solvers.scipy` and `openseespy_solvers.cupy`.
- SciPy solvers: `spsolve`, `cg`, `gmres`, `eigsh`, `lobpcg`.
- CuPy solvers: `spsolve`, `cg`, `gmres`, `lobpcg` (no generalized `eigsh`).
- `.to_opensees()` helper for OpenSeesPy `PythonSparse` system and eigen commands.
- `formAp` matrix-vector product on all linear solvers.
- `copy.copy` support via `__copy__` / `__deepcopy__` for OpenSees SOE cloning.
- Exposed solver arrays: `A`, `b`, `x` (linear) and `K`, `M` (eigen).
- Preconditioner factories: `M=` accepts a ready operator or callable `M(A)`.
- Built-in SciPy preconditioners in `openseespy_solvers.scipy.precond` (`jacobi`, `ilu`).
- Solver statistics via `.stats`.
- Examples and tests that do not require OpenSeesPy.
