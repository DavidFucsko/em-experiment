import numpy as np
import pandas as pd
import pytest

from src.utils.clinical_codes import PERCENTILES
from src.utils.embedded_similarity import EmbeddedSimilarity

# patient 0: history A, later B; patient 1: history C, later C; patient 2: no history codes
HISTORY = [(0, "dx", "icd9", "A"), (1, "dx", "icd9", "C")]
LATER = [(0, "dx", "icd9", "B"), (1, "dx", "icd9", "C"), (2, "dx", "icd9", "C")]
TOKENS = np.array(["dx:icd9:A", "dx:icd9:B", "dx:icd9:C"])
VECTORS = np.array([[1, 0], [0.6, 0.8], [0, 1]], dtype=np.float32)   # cos(A, B) = 0.6, cos(B, C) = 0.8, cos(A, C) = 0


def similarity(tmp_path, tokens=TOKENS, vectors=VECTORS):
    for side, rows in (("history", HISTORY), ("index_admission", LATER)):
        pd.DataFrame(rows, columns=["subject_id", "modality", "vocabulary", "code"]).to_parquet(tmp_path / f"{side}.parquet")
    return EmbeddedSimilarity.from_vectors([0, 1, 2], ("dx",), tokens, vectors, tmp_path)


def test_best_match_of_single_codes_is_their_cosine(tmp_path):
    scores = similarity(tmp_path).scores(np.array([0, 1, 0, 1]), np.array([0, 1, 1, 0]))
    assert scores == pytest.approx([0.6, 1.0, 0.0, 0.8], abs=1e-6)


def test_a_code_without_a_vector_matches_only_itself(tmp_path):
    scores = similarity(tmp_path, TOKENS[:2], VECTORS[:2]).scores(np.array([1, 1]), np.array([1, 0]))
    assert scores == pytest.approx([1.0, 0.0])


def test_pairs_are_scored_in_their_given_order_and_an_empty_side_scores_zero(tmp_path):
    scores = similarity(tmp_path).scores(np.array([2, 1, 0]), np.array([1, 1, 0]))
    assert scores == pytest.approx([0.0, 1.0, 0.6], abs=1e-6)


def test_cuts_ignore_pairs_with_an_empty_side(tmp_path):
    cuts = similarity(tmp_path).cuts(np.array([0, 1, 2]), np.array([0, 1, 1]))
    assert len(cuts) == len(PERCENTILES) and cuts[0] == pytest.approx(0.8, abs=1e-6)  # BM 0.6 and 1.0 -> median 0.8


def test_scores_of_a_pair_set_seen_before_come_from_the_cache(tmp_path):
    embedded = similarity(tmp_path)
    first = embedded.scores(np.array([0, 1]), np.array([0, 1]))
    assert embedded.scores(np.array([0, 1]), np.array([0, 1])) is first
    assert embedded.scores(np.array([1, 0]), np.array([1, 0])) is not first
