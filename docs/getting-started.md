# Tutorial

This page shows how to wire solvers from this package into OpenSeesPy. Model-building
details are omitted; see the [OpenSeesPy documentation](https://openseespydoc.readthedocs.io/)
and [examples](examples.md). For install steps, see [Installation](installation.md).

## Linear analysis

The finite element method discretizes a continuous structural model into nodes and
elements. After assembly, the equations of motion are

$$
\mathbf{M} \ddot{\mathbf{u}} + \mathbf{C} \dot{\mathbf{u}} + \mathbf{p}_r(\mathbf{u}) = \mathbf{p}_f(t),
$$

where $\mathbf{M}$ is the mass matrix, $\mathbf{C}$ is the damping matrix, $\mathbf{u}$ is
the vector of nodal displacements (and rotations), $\mathbf{p}_r(\mathbf{u})$ collects
resisting forces (which may be nonlinear in $\mathbf{u}$), and $\mathbf{p}_f(t)$ is the
applied load vector. For a **static** analysis the inertial term
$\mathbf{M} \ddot{\mathbf{u}}$ and the damping term $\mathbf{C} \dot{\mathbf{u}}$ are absent.

OpenSees advances the solution in time with an **incremental integrator**. At
each step the nonlinear equilibrium equations are linearized, typically in a Newton
iteration, resulting in the linear system
$$
\mathbf{A} \mathbf{x} = \mathbf{b},
$$

where $\mathbf{A}$ is a matrix, and $\mathbf{x}$ and $\mathbf{b}$ are vectors. What
they represent depends on the integrator and algorithm.

OpenSees assembles $\mathbf{A}$ and $\mathbf{b}$ from the model and passes them to a
**linear solver** selected with the [`system`](https://opensees.github.io/OpenSeesDocumentation/user/manual/analysis/system.html) command.

A linear `solver` object from this library can be used in OpenSeesPy with the following syntax:

```python
ops.system("PythonSparse", solver.to_openseespy())
```

### CPU direct solver

[`scipy.spsolve`](api/scipy.md#openseespy_solvers.scipy.spsolve) (SuperLU) needs no optional packages:

```python
import openseespy.opensees as ops
from openseespy_solvers.scipy import spsolve

solver = spsolve()
ops.system("PythonSparse", solver.to_openseespy())
```

Two other CPU direct solvers need optional libraries:
[`scipy.umfpack`](api/scipy.md#openseespy_solvers.scipy.umfpack)
([UMFPACK install](installation.md#umfpack)) and
[`scipy.pardiso`](api/scipy.md#openseespy_solvers.scipy.pardiso)
([PARDISO install](installation.md#pardiso)). [Examples](examples.md) compares
`umfpack` with `spsolve` on one model family.

### GPU direct solver

If you have an NVIDIA GPU and the matching optional wheels ([GPU install](installation.md#gpu)),
use [`nvmath.direct_solver`](api/nvmath.md#openseespy_solvers.nvmath.direct_solver):

```python
import openseespy.opensees as ops
from openseespy_solvers.nvmath import direct_solver

solver = direct_solver()
ops.system("PythonSparse", solver.to_openseespy())
```

When a full factorization is too expensive, try iterative solvers (`cg`, `gmres`) with a
[preconditioner](user-guide/preconditioners.md), or [`hybrid`](api/hybrid.md) to reuse a
direct factorization as a GMRES preconditioner. See the [API overview](api/index.md) for all
solver constructors.

## Generalized eigenvalue analysis

Modal analysis solves the **generalized eigenvalue problem**:

$$
\left( \mathbf{K} - \lambda \mathbf{M} \right) \mathbf{\Phi} = \mathbf{0}.
$$

Here $\lambda$ is an eigenvalue, $\mathbf{\Phi}$ is the corresponding eigenvector, $\mathbf{K}$
is the stiffness matrix, and $\mathbf{M}$ is the mass matrix. The eigenvalue is the square
of the natural angular frequency, $\omega = \sqrt{\lambda}$ (rad/s); $\mathbf{\Phi}$ gives
the mode shape.

An eigen `solver` object from this library can be used in OpenSeesPy with the following syntax:

```python
lam = ops.eigen("PythonSparse", num_modes, eig_solver.to_openseespy())
```

### CPU eigen solver (`scipy.eigsh`)

On the CPU, use [`scipy.eigsh`](api/scipy.md#openseespy_solvers.scipy.eigsh) (ARPACK):

```python
import openseespy.opensees as ops
from openseespy_solvers.scipy import eigsh

eig_solver = eigsh()
num_modes = 5
lam = ops.eigen("PythonSparse", num_modes, eig_solver.to_openseespy())
```

### GPU eigen solver (`cupy.eigsh`)

On the GPU, use [`cupy.eigsh`](api/cupy.md#openseespy_solvers.cupy.eigsh):

```python
import openseespy.opensees as ops
from openseespy_solvers.cupy import eigsh

eig_solver = eigsh()
num_modes = 5
lam = ops.eigen("PythonSparse", num_modes, eig_solver.to_openseespy())
```

