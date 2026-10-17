# openseespy-solvers

`scipy`-style sparse linear and eigen solvers for OpenSeesPy
[`PythonSparse`](https://opensees.github.io/OpenSeesDocumentation/user/manual/analysis/system/PythonSparse.html).

`openseespy-solvers` wraps numerical backends such as `scipy.sparse.linalg`, Intel MKL PARDISO,
`cupyx.scipy.sparse.linalg`, and NVIDIA `nvmath.sparse` as solver objects that OpenSeesPy can call directly. OpenSeesPy
assembles the stiffness, mass, and right-hand-side arrays; the solver object performs
the sparse solve and writes the result back to OpenSeesPy.

Documentation: [openseespy-solvers.readthedocs.io](https://openseespy-solvers.readthedocs.io/)

## Installation

```bash
python -m pip install openseespy-solvers
```

The base install provides NumPy and `scipy.sparse.linalg` CPU solvers. It requires Python 3.12 or newer.

OpenSeesPy is optional so the package can be used and tested without forcing an OpenSees
install. If OpenSeesPy is not already in your environment, install the extra:

```bash
python -m pip install "openseespy-solvers[opensees]"
```

Optional backends:

```bash
# CPU UMFPACK direct solver
python -m pip install "openseespy-solvers[umfpack]"

# CPU PARDISO direct solver (needs only the MKL runtime)
python -m pip install mkl

# NVIDIA GPU backends: choose the CUDA generation reported by nvidia-smi
python -m pip install "openseespy-solvers[cuda13]"   # or [cuda12]
```

On Windows, install UMFPACK with conda-forge instead of pip:

```bash
conda install -c conda-forge scikit-umfpack
```

See the [installation guide](https://openseespy-solvers.readthedocs.io/en/latest/installation/)
for platform notes, CUDA 12.x alternatives, and verification steps.

## Quick Example

```python
import openseespy.opensees as ops
from openseespy_solvers.scipy import spsolve

solver = spsolve()

# after defining the OpenSeesPy model:
ops.system("PythonSparse", solver.to_openseespy())
```

For eigen analysis:

```python
from openseespy_solvers.scipy import eigsh

eig_solver = eigsh()
eigenvalues = ops.eigen("PythonSparse", 5, eig_solver.to_openseespy())
```

## Default Solvers

| Analysis | CPU | NVIDIA GPU |
|----------|-----|-------------|
| Static or transient linear solve | `scipy.spsolve`; `scipy.umfpack` for larger CPU systems | `nvmath.direct_solver` |
| Generalized eigen solve | `scipy.eigsh` | `cupy.eigsh` |

The tutorial and API pages cover the other solvers and the optional backends:
[openseespy-solvers.readthedocs.io](https://openseespy-solvers.readthedocs.io/).

## Modules

| Module | Provides |
|--------|----------|
| `openseespy_solvers.scipy` | CPU solvers: `spsolve`, `umfpack`, `pardiso`, `cg`, `gmres`, `eigsh`, `lobpcg` |
| `openseespy_solvers.scipy.precond` | CPU preconditioners: `jacobi`, `ilu`, `direct` |
| `openseespy_solvers.cupy` | GPU solvers: `spsolve`, `cg`, `gmres`, `eigsh`, `lobpcg` |
| `openseespy_solvers.cupy.precond` | GPU preconditioners: `jacobi`, `ilu`, `direct` |
| `openseespy_solvers.nvmath` | GPU direct sparse solver: `direct_solver` |
| `openseespy_solvers.hybrid` | Direct factorization reused as a GMRES preconditioner |

Factory signatures follow the corresponding `scipy`/`cupy` functions where possible. The
matrix and right-hand side are supplied by OpenSeesPy at solve time.

## Examples and Development

The pip wheel contains the library. Examples, benchmarks, and tests live in the source
repository:

```bash
git clone https://github.com/gaaraujo/openseespy-solvers.git
cd openseespy-solvers
python -m pip install -e ".[dev,opensees]"
pytest
```

Then run a smoke example:

```bash
cd examples
python solvers/scipy_spsolve.py
python solvers/scipy_eigsh.py
```

## Project Links

- Documentation: [openseespy-solvers.readthedocs.io](https://openseespy-solvers.readthedocs.io/)
- Source: [github.com/gaaraujo/openseespy-solvers](https://github.com/gaaraujo/openseespy-solvers)
- Issues: [github.com/gaaraujo/openseespy-solvers/issues](https://github.com/gaaraujo/openseespy-solvers/issues)

## License

BSD 3-Clause. See [LICENSE](LICENSE).
