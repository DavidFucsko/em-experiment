"""Union blocking: a pair is a candidate if the two records agree on every column of at least one rule."""
import numpy as np
import pandas as pd


def blocking_pairs(history: pd.DataFrame, later: pd.DataFrame, rules: list[list[str]]) -> tuple[np.ndarray, np.ndarray]:
    """Sorted unique (i, j) row positions. A row with a missing key value never matches on that rule."""
    n_later = len(later)
    keys = []
    for columns in rules:
        left = history[columns].dropna().reset_index(names="i")
        right = later[columns].dropna().reset_index(names="j")
        matches = left.merge(right, on=columns)
        keys.append(pair_keys(matches["i"].to_numpy(np.int64), matches["j"].to_numpy(np.int64), n_later))
    return merge_pairs(keys, n_later)


def merge_pairs(keys: list[np.ndarray], n_later: int) -> tuple[np.ndarray, np.ndarray]:
    """Sorted unique (i, j) from pair keys i * n_later + j."""
    unique_keys = np.unique(np.concatenate(keys))
    return unique_keys // n_later, unique_keys % n_later


def pair_keys(i: np.ndarray, j: np.ndarray, n_later: int) -> np.ndarray:
    return i.astype(np.int64) * n_later + j
