import numpy as np
import pandas as pd

from src.utils.fellegi_sunter import FellegiSunter
from src.utils.runs import Scores, load_links, save_links

PAIRS = pd.DataFrame({"history_subject_id": [10, 10, 20], "later_subject_id": [10, 20, 20],
                      "sex": [2, 1, 2], "clinical": [30, 8, 22], "probability": [0.9, 0.1, 0.6]})
LEVEL_NAMES = {"sex": ["missing", "different", "equal"], "clinical": [f"level {k}" for k in range(36)]}
MODEL = FellegiSunter([np.full(3, 1 / 3), np.full(36, 1 / 36)], [np.full(3, 1 / 3), np.full(36, 1 / 36)], 0.5)


def test_saved_scores_load_back_with_their_pairs_level_names_and_model(tmp_path):
    Scores(PAIRS, LEVEL_NAMES, MODEL).save("D+C_no_names_natural", tmp_path)
    loaded = Scores.load("D+C_no_names_natural", tmp_path)
    pd.testing.assert_frame_equal(loaded.pairs, PAIRS)
    assert loaded.fields == ["sex", "clinical"] and loaded.level_names == LEVEL_NAMES
    assert loaded.levels().tolist() == [[2, 30], [1, 8], [2, 22]]
    assert all(np.allclose(a, b) for a, b in zip(loaded.model.m + loaded.model.u, MODEL.m + MODEL.u))


def test_links_are_stored_apart_from_the_scores(tmp_path):
    links = PAIRS[["history_subject_id", "later_subject_id"]].assign(linked=[True, False, True], assigned=[True, False, False])
    save_links(links, "D_full_hard", tmp_path)
    pd.testing.assert_frame_equal(load_links("D_full_hard", tmp_path), links)


def test_pairs_are_located_by_subject_id_in_the_sorted_patients():
    i, j = Scores(PAIRS, LEVEL_NAMES, MODEL).positions(np.array([10, 20]))
    assert i.tolist() == [0, 0, 1] and j.tolist() == [0, 1, 1]
