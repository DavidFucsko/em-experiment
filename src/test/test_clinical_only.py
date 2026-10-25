import numpy as np
import pytest
from scipy import sparse

from src.experiments.clinical_only import rank_true_partners
from src.utils.exact_similarity import ExactSimilarity


def identity_similarity(n: int) -> ExactSimilarity:
    """Every history record shares codes only with its own later record."""
    matrix = sparse.identity(n, format="csr")
    return ExactSimilarity(matrix, matrix)


def test_without_a_filter_every_patient_is_found_first_when_only_the_true_partner_shares_codes():
    result = rank_true_partners(identity_similarity(4), {"none": None})["none"].metrics()
    assert result["passing"] == 1.0 and result["candidates"] == 4 and result["top-1 all"] == 1.0


def test_a_filter_that_removes_the_true_partner_is_a_miss_overall_but_not_among_the_passing():
    history_keys, later_keys = np.array(list("aabb")), np.array(list("aabc"))  # patient 3's partner is filtered out
    result = rank_true_partners(identity_similarity(4), {"filter": (history_keys, later_keys)})["filter"].metrics()
    assert result["passing"] == 0.75 and result["candidates"] == pytest.approx(1.5)
    assert result["top-1 all"] == 0.75 and result["top-1 passing"] == 1.0


def test_a_tie_with_the_true_partner_counts_against_it():
    tied = sparse.csr_matrix(np.array([[1.0, 1.0], [0.0, 1.0]]))
    similarity = ExactSimilarity(sparse.identity(2, format="csr"), tied)  # record 0 scores 1.0 against both later records
    assert rank_true_partners(similarity, {"none": None})["none"].metrics()["top-1 all"] == 0.5
