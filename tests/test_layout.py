"""Unit tests for mapping OpenSees sparse layouts onto backend layouts."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from openseespy_solvers._sparse import plan_layout

rng = np.random.default_rng(0)
N = 7
_dense = rng.standard_normal((N, N)) * (rng.random((N, N)) < 0.4) + np.diag(np.arange(1, N + 1.0))
NONSYM = sp.csr_matrix(_dense)
SYM = sp.csr_matrix(_dense + _dense.T)


def opensees_kwargs(A, scheme, lower=False):
    """Buffers in the order OpenSees would send them."""
    A = sp.tril(A) if lower else A
    if scheme == "CSR":
        m = sp.csr_matrix(A)
        m.sort_indices()
        kw = {"index_ptr": memoryview(m.indptr.astype(np.int32)),
              "indices": memoryview(m.indices.astype(np.int32)), "data": m.data}
    elif scheme == "CSC":
        m = sp.csc_matrix(A)
        m.sort_indices()
        kw = {"index_ptr": memoryview(m.indptr.astype(np.int32)),
              "indices": memoryview(m.indices.astype(np.int32)), "data": m.data}
    else:  # COO sorted by row, then column (as SparsePythonCOOLinSOE stores it)
        m = sp.csr_matrix(A)
        m.sort_indices()
        coo = m.tocoo()
        kw = {"row": memoryview(coo.row.astype(np.int32)),
              "col": memoryview(coo.col.astype(np.int32)), "data": coo.data}
    kw.update(num_eqn=N, nnz=kw["data"].size, storage_scheme=scheme)
    if lower:
        kw["matrix_type"] = "SYMMETRIC_LOWER"
    return kw


def rebuild(layout, values):
    cls = sp.csr_matrix if layout.fmt == "CSR" else sp.csc_matrix
    return cls((layout.reorder(values), layout.indices, layout.indptr), shape=layout.shape)


@pytest.mark.parametrize("scheme", ["CSR", "CSC", "COO"])
@pytest.mark.parametrize("formats", [("CSR",), ("CSC",), ("CSR", "CSC"), ("CSC", "CSR")])
def test_general_layouts_round_trip(scheme, formats) -> None:
    kw = opensees_kwargs(NONSYM, scheme)
    layout = plan_layout(kw, formats=formats)
    assert layout.fmt in formats
    np.testing.assert_array_equal(rebuild(layout, kw["data"]).toarray(), NONSYM.toarray())
    if scheme in formats:
        assert layout.fmt == scheme and layout.perm is None
    if scheme == "COO" and formats[0] == "CSR":
        assert layout.perm is None  # sorted COO is already CSR order


@pytest.mark.parametrize("scheme", ["CSR", "CSC", "COO"])
@pytest.mark.parametrize("formats", [("CSR",), ("CSC",)])
def test_lower_storage_expands_for_general_backends(scheme, formats) -> None:
    kw = opensees_kwargs(SYM, scheme, lower=True)
    layout = plan_layout(kw, formats=formats, accepts_lower=False)
    assert not layout.lower
    np.testing.assert_array_equal(rebuild(layout, kw["data"]).toarray(), SYM.toarray())


@pytest.mark.parametrize("scheme", ["CSR", "CSC", "COO"])
def test_lower_storage_kept_for_symmetric_backends(scheme) -> None:
    kw = opensees_kwargs(SYM, scheme, lower=True)
    layout = plan_layout(kw, formats=("CSR",), accepts_lower=True)
    assert layout.lower
    np.testing.assert_array_equal(rebuild(layout, kw["data"]).toarray(), sp.tril(SYM).toarray())


def test_coefficient_update_uses_same_permutation() -> None:
    kw = opensees_kwargs(NONSYM, "CSC")
    layout = plan_layout(kw, formats=("CSR",))
    assert layout.perm is not None
    new = sp.csc_matrix(NONSYM * 3.0 + sp.csr_matrix(NONSYM.astype(bool)) * 0.5)
    new.sort_indices()
    np.testing.assert_allclose(rebuild(layout, new.data).toarray(), new.toarray())
