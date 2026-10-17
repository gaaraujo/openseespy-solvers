# Adding a solver

This page is a short checklist for adding a new solver to the package. Submit
changes as a [pull request on GitHub](https://github.com/gaaraujo/openseespy-solvers/pulls).
For how OpenSees calls solvers at runtime, see the
[PythonSparse interface](development/pythonsparse-interface.md).

## Before you start

Pick the solve type and the module to edit:

| Solve type | Base class | Typical module |
|------------|------------|----------------|
| Linear `Ax = b` | `LinearSolver` | `scipy`, `cupy`, or `nvmath` |
| Eigen `K φ = λ M φ` | `EigenSolver` | `scipy` or `cupy` |

Copy an existing solver that is closest to yours:

| Kind | Copy from |
|------|-----------|
| CPU iterative (`cg`, `gmres`) | `scipy/__init__.py` → `_CG` |
| CPU direct | `scipy/__init__.py` → `_SpSolve` or `_Umfpack` |
| GPU iterative | `cupy/__init__.py` |
| GPU direct (nvmath) | `nvmath/__init__.py` → `_DirectSolver` |
| Eigen (ARPACK) | `scipy/__init__.py` → `_Eigsh` |

Shared OpenSees plumbing (buffers, layout mapping, caching, stats) lives in
`_base.py`. A new solver implements the backend hooks and one solve method, and
sets two class attributes when the defaults do not fit: `_formats`, the
compressed formats the backend factorizes (for example `("CSC",)`), and
`_accepts_lower`, whether it can use the lower triangle of a symmetric matrix
directly.

## Steps

### 1. Add a private solver class

In the right `__init__.py`, add a class that mixes in the backend and the base:

```python
class _MySolver(ScipyMixin, LinearSolver):
    ...
```

Implement:

- **`_solve_system`** (linear) or **`_solve_eigen`** (eigen) — call the numerical library.
- **`__init__`** — store options and build **`self._params`** with every constructor
  argument. OpenSees uses `copy.copy(solver)`; `_params` must be enough to recreate
  the instance.

For **direct** solvers, reuse work when OpenSees sends `matrix_status='UNCHANGED'`.
Refresh on `'COEFFICIENTS_CHANGED'`; rebuild on `'STRUCTURE_CHANGED'`. See `_SpSolve`
for a minimal example.

For **iterative** solvers with a preconditioner, users pass `M=precond.jacobi` (or
similar). Built-in preconditioners live in `scipy/precond.py` or `cupy/precond.py`.

### 2. Add a public solver constructor

Add a function that returns your class and append its name to `__all__`:

```python
def my_solver(*, scheme=None, writable="none", debug=False, dtype=np.float64) -> _MySolver:
    """Configure ... for OpenSees PythonSparse."""
    return _MySolver(scheme=scheme or "CSR", writable=writable, debug=debug, dtype=dtype)
```

Use the docstring fragments in `_docstrings.py` (`_OPENSEES_LINEAR`, `_LINEAR_RETURNS`,
etc.) so constructor docs stay consistent.

Match the underlying library’s keyword names where you can. Do **not** expose `A`, `b`,
`K`, or `M` — OpenSees supplies those at solve time.

### 3. Optional dependency (only if needed)

If the solver needs an extra package:

1. Add an optional extra in `pyproject.toml`.
2. Lazy-import inside a `_import_*()` helper (see `scipy/_base.py` for umfpack).
3. Raise a clear `ImportError` with an install hint when the package is missing.

The module should import without the extra installed; only calling the constructor should
require it.

### 4. Tests

In `tests/test_solvers.py` (or `tests/test_eigen.py` for eigen):

- Build fake OpenSees kwargs with `csr_linear_kwargs()` or `csr_eigen_kwargs()` from
  `tests/conftest.py`.
- Solve a small known system; assert status `0` and correct `x` (or eigenvalues).
- For direct solvers, add a `matrix_status` caching test (see `test_matrix_status_caching`).

Use `pytest.importorskip(...)` when the backend is optional.

### 5. Example script

Add `examples/solvers/<backend>_<name>.py` using `_brick_common.py` (see
`examples/solvers/scipy_spsolve.py`). End with `Passed!` so `tests/test_examples.py`
can smoke-test it.

### 6. Docs

- Add a row to the table in `docs/api/index.md` (and `README.md` if it is a notable
  default).
- Note the change in `CHANGELOG.md`.

API pages under `docs/api/` are generated from docstrings; a good constructor docstring is
usually enough.

## Done checklist

- [ ] Constructor in `__all__`
- [ ] `self._params` set in `__init__`
- [ ] Direct solver respects `matrix_status`
- [ ] Unit test with synthetic OpenSees kwargs
- [ ] Example script prints `Passed!`
- [ ] `pytest` passes

## References

- Base classes and hooks: `src/openseespy_solvers/_base.py`
- CPU backend mixin: `src/openseespy_solvers/scipy/_base.py`
- GPU backend mixin: `src/openseespy_solvers/cupy/_base.py`
- Synthetic test kwargs: `tests/conftest.py`
- OpenSees buffer contract: [PythonSparse interface](development/pythonsparse-interface.md)
