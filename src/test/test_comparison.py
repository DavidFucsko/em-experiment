import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler

from src.utils.comparison import closeness_levels, exact_levels, graded_levels, text_levels


def test_graded_levels_count_the_cuts_reached_and_zero_means_missing():
    levels = graded_levels(np.array([0.1, 0.5, 0.9, 0.99]), (0.4, 0.8), np.array([False, False, False, True]))
    assert levels.tolist() == [1, 2, 3, 0]


def test_text_levels_use_the_similarity_and_flag_missing_values():
    left = pd.Series(["ANNA", "ANNA", "ANNA", None])
    right = pd.Series(["ANNA", "ANNE", "ZOLTAN", "ANNA"])
    assert text_levels(left, right, JaroWinkler.similarity, (0.85, 0.94)).tolist() == [3, 2, 1, 0]


def test_closeness_levels_rank_a_smaller_gap_higher():
    left, right = pd.Series([1980, 1980, 1980, None]), pd.Series([1980, 1981, 1990, 1980])
    assert closeness_levels(left, right, (-1, 0)).tolist() == [3, 2, 1, 0]


def test_exact_levels_distinguish_equal_different_and_missing():
    assert exact_levels(pd.Series(["a", "a", None]), pd.Series(["a", "b", "a"])).tolist() == [2, 1, 0]
