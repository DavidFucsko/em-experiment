"""Comparison levels of one field: 0 = missing on either side, then 1, 2, ... by increasing similarity."""
import numpy as np
import pandas as pd
from rapidfuzz import process


def graded_levels(score: np.ndarray, cuts: tuple, missing: np.ndarray) -> np.ndarray:
    """0 where `missing`, otherwise 1 + the number of ascending `cuts` the score reaches."""
    reached_cuts = (np.asarray(score)[:, None] >= np.asarray(cuts)).sum(axis=1)
    levels = 1 + reached_cuts
    return np.where(missing, 0, levels).astype(np.int8)


def _missing(left: pd.Series, right: pd.Series) -> np.ndarray:
    return (left.isna() | right.isna()).to_numpy()


def text_levels(left: pd.Series, right: pd.Series, scorer, cuts: tuple) -> np.ndarray:
    """Levels from a rapidfuzz string similarity in [0, 1]."""
    left_text = left.fillna("").tolist()
    right_text = right.fillna("").tolist()
    score = process.cpdist(left_text, right_text, scorer=scorer, workers=-1)
    return graded_levels(score, cuts, _missing(left, right))


def closeness_levels(left: pd.Series, right: pd.Series, cuts: tuple) -> np.ndarray:
    """Levels from the negative absolute difference of two numbers, so a smaller gap is a higher level."""
    score = -(left - right).abs().fillna(0).to_numpy()
    return graded_levels(score, cuts, _missing(left, right))


def exact_levels(left: pd.Series, right: pd.Series) -> np.ndarray:
    """1 = different, 2 = equal."""
    equal = (left == right).to_numpy()
    return graded_levels(equal, (1,), _missing(left, right))
