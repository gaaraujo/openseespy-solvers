"""Shared NumPy-style docstring fragments for public factory functions."""

_OPENSEES_LINEAR = """
scheme : {'CSR', 'CSC'}, optional
    Sparse storage scheme for :meth:`~openseespy_solvers._base.BaseOpenSeesSolver.to_openseespy`.
    Default is ``'CSR'``.
writable : str or list of str, optional
    Writable buffers declared to OpenSees. Default is ``'none'``.
debug : bool, optional
    If ``True``, exceptions raised during :meth:`~openseespy_solvers._base.LinearSolver.solve`
    or :meth:`~openseespy_solvers._base.LinearSolver.formAp` are re-raised. Otherwise a
    negative status code is returned. Default is ``False``.
record_stats : bool, optional
    If ``True``, update ``solver.stats`` after each solve (counters, timing,
    ``last_info``). Relative residual (``last_residual_norm``, extra SpMV) is
    computed only when *both* ``record_stats`` and ``debug`` are ``True``.
    Default is ``False``.
dtype : dtype or str, optional
    Floating-point precision for the numerical solve (``float32`` or ``float64``).
    OpenSees buffer I/O remains ``float64``; values are cast at the boundary.
    Default is ``float64``.
"""

_OPENSEES_EIGEN = """
scheme : {'CSR', 'CSC'}, optional
    Sparse storage scheme for :meth:`~openseespy_solvers._base.BaseOpenSeesSolver.to_openseespy`.
    Default is ``'CSR'``.
debug : bool, optional
    If ``True``, exceptions raised during
    :meth:`~openseespy_solvers._base.EigenSolver.solve` are re-raised. Default is ``False``.
dtype : dtype or str, optional
    Floating-point precision for the eigen solve (``float32`` or ``float64``).
    OpenSees buffer I/O remains ``float64``. Default is ``float64``.
"""

_LINEAR_RETURNS = """
Returns
-------
solver : LinearSolver
    Configured linear solver. Pass ``solver.to_openseespy()`` to
    ``ops.system('PythonSparse', ...)``.
"""

_EIGEN_RETURNS = """
Returns
-------
solver : EigenSolver
    Configured eigen solver. Pass ``solver.to_openseespy()`` to
    ``ops.eigen('PythonSparse', num_modes, ...)``.
"""

_LINEAR_NOTES = """
Notes
-----
OpenSees assembles the sparse system matrix and right-hand side and calls
:meth:`~openseespy_solvers._base.LinearSolver.solve` with buffer memoryviews.
The solution is written in place to the ``x`` buffer. After each solve,
``solver.A``, ``solver.b``, and ``solver.x`` refer to the cached matrix and
vectors from the last call.
"""

_EIGEN_NOTES = """
Notes
-----
OpenSees assembles ``K`` and ``M`` and calls
:meth:`~openseespy_solvers._base.EigenSolver.solve`. Eigenvalues and
eigenvectors are written in place to the output buffers. After each solve,
``solver.K`` and ``solver.M`` refer to the cached matrices from the last call.
"""
