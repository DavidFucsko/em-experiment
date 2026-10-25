"""Exact-code clinical similarity: IDF-weighted cosine between two patients' code sets."""
from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import normalize

from src.config import SPLIT
from src.utils.clinical_codes import ALL_MODALITIES, BANDS, CATEGORIES, PERCENTILES, CodeSets, idf_weighted_codes

CHUNK = 200_000             # pairs scored at a time
SHARED = 1e-9               # a similarity above this means at least one shared code


@dataclass(frozen=True)
class ExactSimilarity(CodeSets):
    """Both matrices are IDF-weighted and L2-normalised."""
    LEVEL_NAMES = ("missing", "no shared code") + BANDS

    @classmethod
    def build(cls, subject_ids, modalities, split_dir=SPLIT) -> "ExactSimilarity":
        _, history, later = idf_weighted_codes(subject_ids, modalities, split_dir)
        return cls(normalize(history), normalize(later))

    def scores(self, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        """Cosine similarity of history record i and later record j, for every pair."""
        parts = []
        for start in range(0, len(i), CHUNK):
            history = self.history[i[start:start + CHUNK]]
            later = self.later[j[start:start + CHUNK]]
            parts.append(np.asarray(history.multiply(later).sum(axis=1)).ravel())
        if not parts:
            return np.empty(0)
        return np.concatenate(parts)

    def cuts(self, i: np.ndarray, j: np.ndarray) -> tuple[float, ...]:
        """"Any shared code", then the PERCENTILES of the positive similarities. Pairs without a shared code
        all score 0, so they get a level of their own instead of a percentile."""
        scores = self.scores(i, j)
        percentiles = np.percentile(scores[scores > SHARED], PERCENTILES)
        return (SHARED,) + tuple(percentiles)


def exact_similarities(subject_ids, split_dir=SPLIT) -> dict[str, ExactSimilarity]:
    """One ExactSimilarity per category, plus "all" over every code."""
    groups = dict(CATEGORIES)
    groups["all"] = ALL_MODALITIES
    similarities = {}
    for name, modalities in groups.items():
        similarities[name] = ExactSimilarity.build(subject_ids, modalities, split_dir)
    return similarities
