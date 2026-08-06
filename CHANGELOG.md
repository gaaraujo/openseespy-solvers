# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
