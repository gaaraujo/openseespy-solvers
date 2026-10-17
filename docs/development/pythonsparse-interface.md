# PythonSparse interface

OpenSeesPy passes sparse matrix data to solver objects as memoryviews. Solvers
in this package read those buffers with NumPy, build a backend sparse matrix,
call the numerical routine, and write the results in place.

## Linear solve

OpenSees calls `solver.solve(**kwargs)`.

**Keyword arguments**

`index_ptr`, `indices`, `values`
: CSR/CSC sparse structure and coefficients.

`row`, `col`
: COO structure (`scheme='COO'`) instead of `index_ptr`/`indices`. Entries are
  ordered by row, then column, so they are also in CSR order.

`rhs`, `x`
: Right-hand side and solution buffers. `x` is overwritten.

`num_eqn`, `nnz`
: Matrix order and number of nonzeros.

`matrix_status`
: One of `'STRUCTURE_CHANGED'`, `'COEFFICIENTS_CHANGED'`, `'UNCHANGED'`.

`storage_scheme`
: `'CSR'`, `'CSC'` or `'COO'`. Default is `'CSR'`.

`matrix_type`
: Only when the configuration dict requested it (see below):
  `'SYMMETRIC_LOWER'` means the buffers hold only the lower triangle
  (row >= column). Builds without this option ignore the key and send the full
  matrix without `matrix_type`; solvers in this package handle both.

**Optional dictionary keys**

!!! warning "Requires a future OpenSees release"
    `matrix_type` and `typed` are not in the current `openseespy` wheel on PyPI
    (3.8.0). That wheel ignores both keys, sends the full matrix, and passes
    raw-byte memoryviews. The solvers in this package check on every solve
    whether `matrix_type` was sent and read the buffers with an explicit
    dtype, so they give the same results on 3.8.0 and on newer builds. A
    custom solver that relies on `typed` should check `memoryview.format`
    (`'B'` for raw bytes on 3.8.0). To use these keys now, see
    [Building OpenSees with the upcoming features](#opensees-branch).

`'matrix_type': 'SYMMETRIC_LOWER'`
: Store and send only the lower triangle of a symmetric tangent (about half
  the values of the full matrix). Requested automatically by `pardiso(matrix_type=...)` and
  `nvmath.direct_solver(matrix_type=...)` with `'symmetric'` or `'spd'`.

`'typed': True`
: Memoryviews with format `'i'`/`'d'` instead of raw bytes, so
  `np.asarray(values)` works directly. Off by default (it changes `len()` and
  `memoryview.cast()` on the views).

**Returns**

`0` (or `None`) if the solve succeeded; a negative integer otherwise. Failures
emit a `SolverWarning` with the exception text; set `debug=True` on the solver
to raise the exception instead.

## Eigen solve

OpenSees calls `solver.solve(**kwargs)` for eigen analysis.

Additional keyword arguments: `k_values`, `m_values`, `eigenvalues`,
`eigenvectors`, `num_modes`, `find_smallest`.

Eigen solvers return `0` on success and `-1` on failure (with a
`SolverWarning`), like linear solvers; `debug=True` raises instead.

## formAp

Linear solvers implement `formAp(**kwargs)` to compute `Ap = A @ p` without a
full solve. OpenSees supplies read-only `p` and writable `Ap`.

## Matrix status

OpenSees indicates how the system matrix changed since the previous solve:

`STRUCTURE_CHANGED`
: Rebuild the sparse matrix (new index structure).

`COEFFICIENTS_CHANGED`
: Update coefficients in place; sparsity pattern unchanged.

`UNCHANGED`
: Reuse the cached matrix. Direct solvers also reuse the LU factorization.

## Serial OpenSeesPy and parallelism {#parallelism}

With the standard (serial) OpenSeesPy build, the whole analysis runs in one
process: state determination, element and nodal assembly, constraint handling,
convergence tests, and the `PythonSparse` call with the assembled stiffness
(and mass, for eigen analysis).

![OpenSeesPy assembles on the CPU; PythonSparse receives buffers; GPU backends solve on device](../assets/pythonsparse-gpu-workflow.png)

The diagram shows one step of a nonlinear static or transient analysis with a
GPU backend:

1. OpenSees (C++, CPU) runs the Newton loop: element state determination,
   assembly of the residual `r` and the effective tangent `K_T*`, and the
   displacement update after convergence.
2. The Python solver object (CPU) reads `r` and `K_T*` from the OpenSees
   memoryviews, calls the backend, and writes `Δu` back.
3. On the GPU (optional), the backend copies `r` and `K_T*` to the device,
   solves `K_T* Δu = r`, and copies `Δu` back to the host. The scipy, UMFPACK
   and PARDISO solvers do this step on the CPU.

Eigen analysis follows the same split: OpenSees assembles `K` and `M` on the
CPU, and the solver object runs ARPACK, LOBPCG or a shift-invert iteration on
the CPU or the GPU, depending on the constructor.

!!! warning "Requires a future OpenSees release"
    `DistributedPythonSparse` is not in the current `openseespy` wheel on PyPI
    (3.8.0), which is a serial build. It requires an OpenSeesPy built with MPI
    (parallel interpreters) from a future OpenSees release. On the 3.8.0 wheel,
    `ops.system('DistributedPythonSparse', ...)` and
    `ops.eigen('DistributedPythonSparse', ...)` raise `OpenSeesError`, and
    OpenSees prints `unknown system type DistributedPythonSparse` (or
    `eigen - unknown option specified`) to stderr. See
    [Building OpenSees with the upcoming features](#opensees-branch).

**Partitioned (MPI) models.** OpenSeesPy builds with parallel interpreters
provide `system('DistributedPythonSparse', solver.to_openseespy())` and
`eigen('DistributedPythonSparse', n, ...)`. Every rank runs the same script and
builds the same solver object. Worker ranks send their part of the right-hand
side and their matrix contributions to rank 0, rank 0 calls the solver, and the
solution is sent back to the workers. Solver constructors in this package
allocate no GPU memory before the first solve, so GPU solvers on worker ranks
never touch the device. `'PythonSparse'` remains serial on every build, so
scripts that run independent models on each rank are unaffected.

**Where the work runs**

| Layer | Behavior |
|-------|----------|
| OpenSees model pipeline | One CPU process, or partitioned across MPI ranks with `DistributedPythonSparse` (solve on rank 0) |
| scipy backends ([`spsolve`](../api/scipy.md#openseespy_solvers.scipy.spsolve), [`pardiso`](../api/scipy.md#openseespy_solvers.scipy.pardiso), [`cg`](../api/scipy.md#openseespy_solvers.scipy.cg), [`eigsh`](../api/scipy.md#openseespy_solvers.scipy.eigsh), ...) | CPU; SciPy, UMFPACK and MKL may use multiple threads |
| cupy and nvmath backends | GPU factorization, iterative solve or eigen iteration; assembly remains on the CPU in OpenSees, with host-device transfers inside the solver |

A GPU backend therefore accelerates the sparse solve only, not assembly or
state determination.

## Building OpenSees with the upcoming features {#opensees-branch}

The `openseespy-solvers` branch of
[gaaraujo/OpenSees](https://github.com/gaaraujo/OpenSees/tree/openseespy-solvers)
contains the OpenSees changes that `matrix_type`, `typed` and
`DistributedPythonSparse` need, on top of a recent OpenSees `master`. In
general, compiling OpenSeesPy from that branch gives access to these features
before an OpenSees release includes them. The branch has not been merged into
OpenSees, so it has had less testing than a release, and its behavior and
interface may change in the final version.

- Serial OpenSeesPy: build the `OpenSeesPy` target as for any OpenSees build.
- MPI OpenSeesPy (`DistributedPythonSparse`): configure with
  `-DOPENSEESPY_MPI=ON`, which also requires MPI and METIS 5 (set
  `METIS5_ROOT` if CMake does not find it). The option is off by default and
  changes no other target.

Copy the built library to `opensees.pyd` (Windows) or `opensees.so` (Linux),
put its directory on `sys.path`, and use `import opensees as ops` in place of
`import openseespy.opensees as ops`. The end-to-end tests of this package
(`tests/test_opensees_integration.py`) use such a build when
`OPENSEESPY_PATH` is set to that directory.

## See also

- [User guide](../user-guide/index.md) — `ops.system` / `ops.eigen` usage for end users
- [Solver objects](../user-guide/solver-objects.md)
- [Adding a solver](../adding-a-solver.md)
- [OpenSees PythonSparse documentation](https://opensees.github.io/OpenSeesDocumentation/user/manual/analysis/system/PythonSparse.html)
