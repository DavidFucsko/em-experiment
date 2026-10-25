import numpy as np
from scipy import sparse

from src.preprocessing.code_vectors import all_but_the_top, cooccurrence, train_glove

# 4 records x 4 codes: codes 0 and 1 occur together in 3 records, code 2 in 3 records but never with 0 or 1, code 3 in one
RECORDS = sparse.csr_matrix(np.array([[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 1], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]]))


def test_cooccurrence_counts_each_pair_once_and_drops_rare_codes():
    kept, i, j, count = cooccurrence(RECORDS, min_records=3)
    assert kept.tolist() == [0, 1, 2]
    assert list(zip(i, j, count)) == [(0, 1, 3)]


def test_glove_gives_one_vector_per_code():
    vectors = train_glove(3, np.array([0, 0]), np.array([1, 2]), np.array([3.0, 1.0]), dim=8, epochs=2)
    assert vectors.shape == (3, 8)


def test_all_but_the_top_returns_unit_rows_without_the_common_direction():
    vectors = np.random.default_rng(0).normal(size=(50, 6)) + np.array([10, 0, 0, 0, 0, 0])
    processed = all_but_the_top(vectors, n_top=1)
    assert np.allclose(np.linalg.norm(processed, axis=1), 1)
    assert abs((processed @ processed.T)[np.triu_indices(50, 1)].mean()) < 0.1

