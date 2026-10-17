# Installation

Use a fresh Python environment, install `openseespy-solvers`, then add only the optional
backends you need.

## Requirements

| Component | Requirement |
|-----------|-------------|
| Python | 3.12 or newer |
| NumPy | 1.26 or newer |
| scipy | 1.12 or newer |
| OpenSeesPy | 3.8.0 or newer, to run OpenSees analyses and the examples |

`openseespy-solvers` does not install OpenSeesPy by default. This keeps the base package
small and lets users add OpenSeesPy only when they need the OpenSees integration.

## Quick Install

```bash
python -m pip install openseespy-solvers
```

If OpenSeesPy is not already installed in the environment:

```bash
python -m pip install "openseespy-solvers[opensees]"
```

## Fresh Environment {#full-setup}

Conda:

```bash
conda create -n openseespy-solvers python=3.12
conda activate openseespy-solvers
python -m pip install "openseespy-solvers[opensees]"
```

venv:

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install "openseespy-solvers[opensees]"
```

## Optional Backends

### UMFPACK {#umfpack}

UMFPACK enables the `openseespy_solvers.scipy.umfpack` CPU direct solver.

Linux or macOS:

```bash
python -m pip install "openseespy-solvers[umfpack]"
```

Windows:

```bash
conda install -c conda-forge scikit-umfpack
```

If pip builds from source on Linux, you may also need SuiteSparse from your system package
manager, for example `libsuitesparse-dev` on Debian or Ubuntu.

### PARDISO {#pardiso}

`openseespy_solvers.scipy.pardiso` calls Intel MKL PARDISO through `ctypes`
and needs only the MKL runtime library (`mkl_rt`). Install it from PyPI or
conda:

```bash
python -m pip install mkl
# or
conda install mkl
```

An Intel oneAPI installation also works. To use a specific library, set
`OPENSEESPY_SOLVERS_MKL_RT` to its full path. If `mkl_rt` cannot be loaded,
`pardiso()` raises `BackendNotAvailableError` at construction.

An OpenSeesPy that you build yourself against Intel oneAPI MKL with Intel
threading loads oneAPI's OpenMP runtime (`libiomp5md.dll` on Windows). If
`pardiso()` then loads `mkl_rt` from the `mkl` wheel, the process holds two
copies of that runtime and aborts with `OMP: Error #15`. In that case, set
`OPENSEESPY_SOLVERS_MKL_RT` to the oneAPI library that the build uses, for
example `C:\Program Files (x86)\Intel\oneAPI\mkl\latest\bin\mkl_rt.2.dll`,
and put the oneAPI `bin` directory on `PATH`. The `openseespy` wheel on PyPI
does not have this conflict.

### NVIDIA GPU {#gpu}

GPU solvers require an NVIDIA driver and CUDA-compatible wheels. Start by checking the CUDA
generation reported by the driver:

```bash
nvidia-smi
```

For CUDA 13.x:

```bash
python -m pip install "openseespy-solvers[cuda13]"
```

For CUDA 12.x:

```bash
python -m pip install "openseespy-solvers[cuda12]"
```

These extras install `cupy` and `nvmath-python` 1.0 or newer, which uses cuDSS
0.8, for the selected CUDA generation. Keep `cupy` and nvmath on the same CUDA
generation. The generic `cupy` extra builds `cupy` from source; use it only if
you need that.

## Development Install

Clone the repository when you want examples, benchmarks, tests, or documentation sources:

```bash
git clone https://github.com/gaaraujo/openseespy-solvers.git
cd openseespy-solvers
python -m pip install -e ".[dev,opensees]"
```

Add optional backends from the sections above as needed.

## Verify {#verify}

From the repository root:

```bash
python -m pip check
python -m pytest
```

Run two small OpenSeesPy examples:

```bash
cd examples
python solvers/scipy_spsolve.py
python solvers/scipy_eigsh.py
```

GPU and UMFPACK tests run only when the corresponding optional packages are installed;
otherwise they are skipped.

For timing comparisons against native OpenSees solvers:

```bash
python brick_bar.py
python brick_bar_eigen.py
```

## Troubleshooting

### OpenSeesPy Parallel Builds

The `openseespy` wheel on PyPI is a serial build, and GPU backends accelerate
only the sparse solve; assembly remains in one process. Partitioned models
need `DistributedPythonSparse`, which requires an MPI build of OpenSeesPy from a
future OpenSees release. See
[PythonSparse and parallelism](development/pythonsparse-interface.md#parallelism)
and, to build it before that release,
[Building OpenSees with the upcoming features](development/pythonsparse-interface.md#opensees-branch).

### UMFPACK Fails to Install on Windows

Use conda-forge:

```bash
conda install -c conda-forge scikit-umfpack
```

The PyPI package is distributed as source only and needs a compiler and SuiteSparse.

### cupy Cannot Find CUDA Libraries

Install the `cupy` wheel with bundled runtime libraries:

```bash
python -m pip install "cupy-cuda13x[ctk]"   # or cupy-cuda12x[ctk]
```

Also confirm that `nvidia-smi` works and that `cupy` and nvmath use the same CUDA generation.

## Build Documentation Locally

```bash
python -m pip install -e ".[docs]"
mkdocs serve
```

Open `http://127.0.0.1:8000`.
