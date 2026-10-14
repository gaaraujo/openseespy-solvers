# PythonSparse interface

OpenSeesPy passes sparse matrix data to solver objects as memoryviews. Solvers
in this package wrap those buffers with NumPy, assemble a backend sparse matrix,
invoke the numerical routine, and write results in place.

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

**Optional dictionary keys** (newer OpenSees builds; older builds ignore them)

`'matrix_type': 'SYMMETRIC_LOWER'`
: Store and send only the lower triangle of a symmetric tangent (halves the
  values buffer). Requested automatically by `pardiso(matrix_type=...)` and
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

These solvers work with **serial OpenSeesPy** — the standard build where the full
analysis pipeline runs on **one process**. That includes state determination, element
and nodal assembly, constraint handling, convergence tests, and the call into
`PythonSparse` with the assembled stiffness (and mass, for eigen).

![OpenSeesPy assembles on the CPU; PythonSparse receives buffers; GPU backends solve on device](../assets/pythonsparse-gpu-workflow.png)

The diagram above is a typical **nonlinear static/transient** step with a GPU direct or
iterative backend:

1. **OpenSees (CPU, C++)** — Newton loop: element state determination, assemble residual
   `r` and effective tangent `K_T*`, update displacements when converged.
2. **Python solver (CPU, Python)** — `PythonSparse` callback reads `r` and `K_T*`
   from OpenSees memoryviews, orchestrates the backend, writes `Δu` back.
3. **GPU (optional)** — copy `r` and `K_T*` host→device, solve
   `K_T* Δu = r`, copy `Δu` device→host. scipy/UMFPACK paths stay on the CPU instead
   of the third column.

Eigen analysis follows the same split: OpenSees assembles **K** and **M** on the CPU; the
solver object runs ARPACK, LOBPCG, or shift-invert on CPU and/or GPU depending on the
solver constructor you choose.

**Partitioned (MPI) models.** OpenSeesPy builds with parallel interpreters
provide `system('DistributedPythonSparse', solver.to_openseespy())` and
`eigen('DistributedPythonSparse', n, ...)`. Every rank runs the same script and
builds the same solver object; worker ranks send their part of the RHS and
their matrix contributions to rank 0, only rank 0 calls the solver, and the
solution is sent back. Nothing else changes in the script. Constructors in this
package allocate nothing on the GPU until the first solve, so worker ranks never
touch the device. `'PythonSparse'` itself stays serial on every build, so
scripts that run independent models per rank are unaffected. No PyPI wheel
includes the MPI build yet.

**Where parallelism exists today**

| Layer | Behavior |
|-------|----------|
| OpenSees model pipeline | Serial (single CPU process), or partitioned across MPI ranks with `DistributedPythonSparse` (solve on rank 0) |
| scipy backends ([`spsolve`](../api/scipy.md#openseespy_solvers.scipy.spsolve), [`cg`](../api/scipy.md#openseespy_solvers.scipy.cg), [`eigsh`](../api/scipy.md#openseespy_solvers.scipy.eigsh), …) | CPU linear algebra; may use multithreaded BLAS/LAPACK inside scipy/UMFPACK |
| cupy / nvmath backends | **GPU** factorization, iterative solve, or eigen iteration; matrix assembly still happens on the CPU in OpenSees, with host↔device transfers inside the solver |

So a GPU setup speeds up the **sparse solve step**, not domain decomposition or
multi-rank OpenSees analysis.

## See also

- [User guide](../user-guide/index.md) — `ops.system` / `ops.eigen` usage for end users
- [Solver objects](../user-guide/solver-objects.md)
- [Adding a solver](../adding-a-solver.md)
- [OpenSees PythonSparse documentation](https://opensees.github.io/OpenSeesDocumentation/user/manual/analysis/system/PythonSparse.html)
