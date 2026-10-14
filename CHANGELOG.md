# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `scheme="COO"` for linear solvers (OpenSees' COO entries are already in CSR
  order, so no reordering is needed).
- `nvmath.direct_solver(matrix_type="symmetric" | "spd")`: LDL^T / Cholesky in
  cuDSS. These solvers ask OpenSees for lower-triangle storage
  (`matrix_type="SYMMETRIC_LOWER"`), which halves the values copied to the GPU
  on OpenSees builds that support it. Older builds ignore the key and send the
  full matrix; the solver detects that per solve.
- Lower-triangle buffers from OpenSees are accepted by every solver (expanded
  once per pattern for backends that need the full matrix).
- `SolverWarning`: failed solves always report the exception text.
- End-to-end tests that run OpenSeesPy Newton loops (CSR/CSC/COO, nonsymmetric
  tangent, hybrid fallback, cuDSS); GPU tests are marked `gpu`.

### Changed

- Each backend declares the compressed format it factorizes, and the mapping
  from OpenSees' layout is computed once per sparsity pattern. SuperLU and
  UMFPACK no longer convert CSR to CSC on every factorization (UMFPACK shares
  the values array), and `spsolve()` / `cupy.spsolve()` ask OpenSees for CSC by
  default, like `umfpack()`.
- `UNCHANGED` solves no longer read or cast the O(nnz) value buffers.
- GPU solvers reuse preallocated device buffers for the matrix values and the
  right-hand side, keep cuDSS operand references across solves, and copy the
  solution straight into the OpenSees buffer.
- `EigenSolver.solve` returns `0`/`-1` like `LinearSolver.solve` instead of
  raising; `debug=True` still raises.
- `hybrid()` applies the frozen factorization directly, without re-marshalling
  the matrix or copying vectors through the host on each GMRES iteration.
- `cupy.spsolve` is documented as factorizing on the CPU (cupyx `splu` copies
  the matrix to the host); use `nvmath.direct_solver` for GPU factorization.
- `inspect.signature` lookups for iterative solvers are cached.
- The `[opensees]` extra requires `openseespy>=3.8.0`.

### Fixed

- `nvmath.direct_solver(scheme="CSC")`: coefficient updates were written into the
  converted CSR matrix in CSC order, which is wrong for nonsymmetric matrices.
- `nvmath.direct_solver(device="gpu")`: every `COEFFICIENTS_CHANGED` solve failed
  (returned -1) because resetting the GPU operand discarded the cuDSS plan.
- `hybrid()`: the fallback refactorization after a GMRES failure passed
  `COEFFICIENTS_CHANGED` for what could be a new sparsity pattern, so inner
  solvers could factorize an outdated matrix or reuse a stale symbolic analysis.
- `hybrid()` on the GPU: GMRES failed on every call because cupy does not accept
  `callback_type="legacy"`.
- `precond.direct()`: the first refactorization of each new preconditioner uses
  `STRUCTURE_CHANGED`, for the same reason as the `hybrid()` fix.
- Failed solves were silent unless `debug=True`; they now emit `SolverWarning`.

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
