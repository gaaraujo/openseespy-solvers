"""Shared helpers for OpenSees-style synthetic kwargs."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def csr_linear_kwargs(
    A: sp.spmatrix | np.ndarray,
    b: np.ndarray,
    *,
    x: np.ndarray | None = None,
    matrix_status: str = "STRUCTURE_CHANGED",
) -> dict:
    """Build kwargs mimicking OpenSees PythonSparse linear ``solve``."""
    mat = sp.csr_matrix(A)
    if x is None:
        x = np.zeros(mat.shape[0], dtype=np.float64)
    rhs = np.asarray(b, dtype=np.float64)
    return {
        "index_ptr": memoryview(mat.indptr.astype(np.int32)),
        "indices": memoryview(mat.indices.astype(np.int32)),
        "values": memoryview(mat.data.astype(np.float64)),
        "rhs": memoryview(rhs),
        "x": memoryview(x),
        "num_eqn": mat.shape[0],
        "nnz": mat.nnz,
        "matrix_status": matrix_status,
        "storage_scheme": "CSR",
    }


def csc_linear_kwargs(
    A: sp.spmatrix | np.ndarray,
    b: np.ndarray,
    *,
    x: np.ndarray | None = None,
    matrix_status: str = "STRUCTURE_CHANGED",
) -> dict:
    """Build kwargs mimicking OpenSees PythonSparse linear ``solve`` (CSC)."""
    mat = sp.csc_matrix(A)
    if x is None:
        x = np.zeros(mat.shape[0], dtype=np.float64)
    rhs = np.asarray(b, dtype=np.float64)
    return {
        "index_ptr": memoryview(mat.indptr.astype(np.int32)),
        "indices": memoryview(mat.indices.astype(np.int32)),
        "values": memoryview(mat.data.astype(np.float64)),
        "rhs": memoryview(rhs),
        "x": memoryview(x),
        "num_eqn": mat.shape[0],
        "nnz": mat.nnz,
        "matrix_status": matrix_status,
        "storage_scheme": "CSC",
    }


def csr_eigen_kwargs(
    K: sp.spmatrix | np.ndarray,
    M: sp.spmatrix | np.ndarray,
    *,
    num_modes: int,
    find_smallest: bool = True,
    matrix_status: str = "STRUCTURE_CHANGED",
) -> dict:
    """Build kwargs mimicking OpenSees PythonSparse eigen ``solve``."""
    k_mat = sp.csr_matrix(K)
    m_mat = sp.csr_matrix(M)
    n = k_mat.shape[0]
    eigenvalues = np.zeros(num_modes, dtype=np.float64)
    eigenvectors = np.zeros(num_modes * n, dtype=np.float64)
    return {
        "index_ptr": memoryview(k_mat.indptr.astype(np.int32)),
        "indices": memoryview(k_mat.indices.astype(np.int32)),
        "k_values": memoryview(k_mat.data.astype(np.float64)),
        "m_values": memoryview(m_mat.data.astype(np.float64)),
        "eigenvalues": memoryview(eigenvalues),
        "eigenvectors": memoryview(eigenvectors),
        "num_eqn": n,
        "nnz": k_mat.nnz,
        "matrix_status": matrix_status,
        "storage_scheme": "CSR",
        "num_modes": num_modes,
        "find_smallest": find_smallest,
    }


def csc_eigen_kwargs(
    K: sp.spmatrix | np.ndarray,
    M: sp.spmatrix | np.ndarray,
    *,
    num_modes: int,
    find_smallest: bool = True,
    matrix_status: str = "STRUCTURE_CHANGED",
) -> dict:
    """Build kwargs mimicking OpenSees PythonSparse eigen ``solve`` (CSC).

    Useful for ``eigsh(scheme='CSC')``: SciPy's shift-invert LU (SuperLU /
    UMFPACK) prefers CSC, so OpenSees can assemble ``K``/``M`` in that layout.
    """
    k_mat = sp.csc_matrix(K)
    m_mat = sp.csc_matrix(M)
    n = k_mat.shape[0]
    eigenvalues = np.zeros(num_modes, dtype=np.float64)
    eigenvectors = np.zeros(num_modes * n, dtype=np.float64)
    return {
        "index_ptr": memoryview(k_mat.indptr.astype(np.int32)),
        "indices": memoryview(k_mat.indices.astype(np.int32)),
        "k_values": memoryview(k_mat.data.astype(np.float64)),
        "m_values": memoryview(m_mat.data.astype(np.float64)),
        "eigenvalues": memoryview(eigenvalues),
        "eigenvectors": memoryview(eigenvectors),
        "num_eqn": n,
        "nnz": k_mat.nnz,
        "matrix_status": matrix_status,
        "storage_scheme": "CSC",
        "num_modes": num_modes,
        "find_smallest": find_smallest,
    }


def form_ap_kwargs(
    A: sp.spmatrix | np.ndarray,
    p: np.ndarray,
    *,
    matrix_status: str = "STRUCTURE_CHANGED",
) -> dict:
    """Build kwargs mimicking OpenSees PythonSparse ``formAp``."""
    mat = sp.csr_matrix(A)
    ap = np.zeros(mat.shape[0], dtype=np.float64)
    base = csr_linear_kwargs(mat, np.zeros(mat.shape[0]), matrix_status=matrix_status)
    base["p"] = memoryview(np.asarray(p, dtype=np.float64))
    base["Ap"] = memoryview(ap)
    return base


# --- minimal 3-D brick bar (kip, in, sec) for OpenSeesPy solver tests ---
_BAR_LENGTH = 10.0
_BAR_HEIGHT = 2.0
_BAR_THICKNESS = 1.0
_ELASTIC_MODULUS = 29_000.0
_POISSON_RATIO = 0.3
_STEEL_DENSITY = 0.284e-3 / 386.4
_YIELD_STRESS = 50.0


def build_brick_bar(ops, nx, ny, nz):
    """Build a small cantilever brick bar and return its far-corner node tag."""
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 3)
    ops.nDMaterial(
        "ElasticIsotropic", 1, _ELASTIC_MODULUS, _POISSON_RATIO, _STEEL_DENSITY
    )
    ops.block3D(
        nx, ny, nz, 1, 1, "stdBrick", 1,
        1, 0.0, -_BAR_THICKNESS / 2.0, -_BAR_HEIGHT / 2.0,
        2, _BAR_LENGTH, -_BAR_THICKNESS / 2.0, -_BAR_HEIGHT / 2.0,
        3, _BAR_LENGTH, _BAR_THICKNESS / 2.0, -_BAR_HEIGHT / 2.0,
        4, 0.0, _BAR_THICKNESS / 2.0, -_BAR_HEIGHT / 2.0,
        5, 0.0, -_BAR_THICKNESS / 2.0, _BAR_HEIGHT / 2.0,
        6, _BAR_LENGTH, -_BAR_THICKNESS / 2.0, _BAR_HEIGHT / 2.0,
        7, _BAR_LENGTH, _BAR_THICKNESS / 2.0, _BAR_HEIGHT / 2.0,
        8, 0.0, _BAR_THICKNESS / 2.0, _BAR_HEIGHT / 2.0,
    )
    ops.fixX(0.0, 1, 1, 1)
    far_corner = (_BAR_LENGTH, _BAR_THICKNESS / 2.0, _BAR_HEIGHT / 2.0)
    for node in ops.getNodeTags():
        coord = [ops.nodeCoord(node, i) for i in (1, 2, 3)]
        if np.allclose(coord, far_corner, atol=1e-9):
            return node
    return None


def apply_face_load(ops):
    """Apply a downward load over the free-end face of the brick bar."""
    total = 1.25 * _YIELD_STRESS * (_BAR_THICKNESS * _BAR_HEIGHT**2) / (6 * _BAR_LENGTH)
    ops.timeSeries("Trig", 1, 0.0, 6.0, 4.0, "-factor", 1.0)
    ops.pattern("Plain", 1, 1)
    far_nodes = [
        n
        for n in ops.getNodeTags()
        if np.isclose(ops.nodeCoord(n, 1), _BAR_LENGTH, atol=1e-9)
    ]
    load = total / len(far_nodes)
    for node in far_nodes:
        ops.load(node, 0.0, 0.0, -load)
