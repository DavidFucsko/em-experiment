import numpy as np
import pandas as pd
import pytest

from src.utils.clinical_codes import CATEGORIES, PERCENTILES, ClinicalField
from src.utils.exact_similarity import SHARED, exact_similarities

# patient 1 and 2 share the common code A; only patient 1 has the rare code B; patient 3 has no medications in its history
HISTORY = [(1, "dx", "icd9", "A"), (1, "dx", "icd9", "B"), (2, "dx", "icd9", "A"), (2, "dx", "icd9", "C"), (3, "dx", "icd9", "Z"),
           (1, "homemed", "gsn", "M"), (2, "homemed", "gsn", "M"),
           (1, "px", "icd9", "P"), (2, "px", "icd9", "Q"), (3, "px", "icd9", "Q"),
           (1, "drg", "drg_hcfa", "D1"), (2, "drg", "drg_hcfa", "D2"), (3, "drg", "drg_hcfa", "D2")]
LATER = [(1, "dx", "icd9", "B"), (2, "dx", "icd9", "A"), (3, "dx", "icd9", "Z"),
         (1, "homemed", "gsn", "M"), (2, "homemed", "gsn", "M"), (3, "homemed", "gsn", "M"),
         (1, "px", "icd9", "P"), (2, "px", "icd9", "Q"), (3, "px", "icd9", "Q"),
         (1, "drg", "drg_hcfa", "D1"), (2, "drg", "drg_hcfa", "D2"), (3, "drg", "drg_hcfa", "D2")]


@pytest.fixture
def similarities(tmp_path):
    for side, rows in (("history", HISTORY), ("index_admission", LATER)):
        pd.DataFrame(rows, columns=["subject_id", "modality", "vocabulary", "code"]).to_parquet(tmp_path / f"{side}.parquet")
    return exact_similarities([1, 2, 3], tmp_path)


def test_identical_code_sets_score_one_and_disjoint_ones_zero(similarities):
    diagnoses = similarities["diagnoses"]
    assert diagnoses.scores(np.array([2]), np.array([2])) == pytest.approx([1.0])
    assert diagnoses.scores(np.array([0]), np.array([2])) == pytest.approx([0.0])


def test_a_shared_rare_code_counts_for_more_than_a_shared_common_one(similarities):
    rare_shared, common_shared = similarities["diagnoses"].scores(np.array([0, 0]), np.array([0, 1]))
    assert rare_shared > common_shared > 0


def test_missing_marks_pairs_where_a_side_has_no_code_in_the_category(similarities):
    assert similarities["medications"].missing(np.array([2, 0]), np.array([2, 0])).tolist() == [True, False]


def test_levels_are_missing_none_shared_below_cut_or_above_cut(similarities):
    i, j = np.array([0, 2, 0]), np.array([0, 2, 2])
    assert ClinicalField("diagnoses", similarities["diagnoses"], (SHARED, 0.5)).levels(i, j).tolist() == [3, 3, 1]
    assert ClinicalField("medications", similarities["medications"], (SHARED, 0.5)).levels(i, j).tolist() == [3, 0, 3]


def test_cuts_put_no_shared_code_first_and_use_only_the_positive_similarities(similarities):
    field = ClinicalField.cut_on("diagnoses", similarities["diagnoses"], np.array([0, 0, 1]), np.array([0, 2, 1]))
    assert len(field.cut_points) == 1 + len(PERCENTILES) and field.cut_points[0] == SHARED and field.cut_points[1] > 0
    assert len(field.level_names) == 2 + len(field.cut_points)


def test_there_is_one_similarity_per_category_and_one_over_all_codes(similarities):
    assert set(similarities) == {*CATEGORIES, "all"}
