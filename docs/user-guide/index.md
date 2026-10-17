# User guide

OpenSeesPy can delegate linear and eigen solves to a **Python solver** through the
[`PythonSparse` system and eigen commands](https://opensees.github.io/OpenSeesDocumentation/user/manual/analysis/system/PythonSparse.html).
OpenSees assembles the sparse matrices and right-hand side, then calls methods on
your solver object (`solve`, and for linear solvers `formAp`).

`openseespy-solvers` implements that callback protocol for [`scipy`](../api/scipy.md),
[`cupy`](../api/cupy.md), and [`nvmath`](../api/nvmath.md) solvers. Application code:

1. Call a solver constructor ([`spsolve`](../api/scipy.md#openseespy_solvers.scipy.spsolve), [`eigsh`](../api/scipy.md#openseespy_solvers.scipy.eigsh), [`direct_solver`](../api/nvmath.md#openseespy_solvers.nvmath.direct_solver), …).
2. Pass `solver.to_openseespy()` to OpenSeesPy.

```python
import openseespy.opensees as ops
from openseespy_solvers.scipy import spsolve

solver = spsolve()
ops.system("PythonSparse", solver.to_openseespy())
# … numberer, constraints, integrator, analysis …
ops.analyze(num_steps)
```

```python
from openseespy_solvers.scipy import eigsh

eig_solver = eigsh()
lam = ops.eigen("PythonSparse", num_modes, eig_solver.to_openseespy())
```

The pages below cover solver configuration, preconditioners, and the data
available after a solve.

## to_openseespy()

`solver.to_openseespy()` returns a dict. Pass it to OpenSeesPy unchanged in
`ops.system("PythonSparse", …)` and `ops.eigen("PythonSparse", …)`. Configure
the solver when you create it ([`spsolve()`](../api/scipy.md#openseespy_solvers.scipy.spsolve), [`cg(rtol=1e-8)`](../api/scipy.md#openseespy_solvers.scipy.cg), …), not by editing
that dict.

Use `copy.copy(solver)` to obtain a fresh instance with the same configuration
but empty internal state. OpenSees may clone solvers when copying a system of
equations.

## See also

- [Solver objects](solver-objects.md) — attributes, `stats`, `dtype`, `copy.copy`
- [Preconditioners](preconditioners.md) — `M=` for iterative solvers
- [Tutorial](../getting-started.md) — full static and eigen workflows
- [PythonSparse interface](../development/pythonsparse-interface.md) — buffers,
  `matrix_status`, and parallelism (for contributors and advanced users)
