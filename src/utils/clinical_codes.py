"""Shared parts of the exact (C) and embedded (E) clinical similarity: the code sets of both sides, their IDF weights
and the clinical field of the comparison vector."""
from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd
from scipy import sparse

from src.config import SPLIT
from src.utils.comparison import graded_levels

CATEGORIES = {
    "diagnoses": ("dx",),
    "medications": ("homemed", "edrx", "icurx"),
    "procedures": ("px", "icupx", "hcpcs"),
}
ALL_MODALITIES = CATEGORIES["diagnoses"] + CATEGORIES["medications"] + CATEGORIES["procedures"]
HISTORY = "history"
LATER = "index_admission"
PERCENTILES = (50, 90, 99)  # cut points of the candidate pairs' similarity, the same for C and E
TOKEN = "modality || ':' || vocabulary || ':' || code"  # SQL: the one-string name of a code


def _band_names(percentiles: tuple[int, ...]) -> tuple[str, ...]:
    names = [f"< p{percentiles[0]}"]
    for low, high in zip(percentiles, percentiles[1:]):
        names.append(f"p{low}-p{high}")
    names.append(f">= p{percentiles[-1]}")
    return tuple(names)


BANDS = _band_names(PERCENTILES)


def code_sets(side: str, subject_ids, modalities, split_dir) -> pd.DataFrame:
    """(row, token) of every distinct code of the patients, where row is the position in subject_ids."""
    modality_list = ", ".join(f"'{modality}'" for modality in modalities)
    side_file = (split_dir / f"{side}.parquet").as_posix()
    codes = duckdb.sql(f"""
        SELECT DISTINCT subject_id, {TOKEN} AS token
        FROM '{side_file}' WHERE modality IN ({modality_list})
    """).df()
    codes["row"] = pd.Index(subject_ids).get_indexer(codes["subject_id"])
    return codes[codes["row"] >= 0]


def _binary_matrix(codes: pd.DataFrame, tokens: pd.Index, n_patients: int) -> sparse.csr_matrix:
    values = np.ones(len(codes))
    columns = tokens.get_indexer(codes["token"])
    return sparse.csr_matrix((values, (codes["row"], columns)), shape=(n_patients, len(tokens)))


def idf_weighted_codes(subject_ids, modalities, split_dir=SPLIT) -> tuple[pd.Index, sparse.csr_matrix, sparse.csr_matrix]:
    """(tokens, history, later): patients x tokens matrices, every present code weighted by its IDF."""
    history_codes = code_sets(HISTORY, subject_ids, modalities, split_dir)
    later_codes = code_sets(LATER, subject_ids, modalities, split_dir)
    tokens = pd.Index(pd.concat([history_codes["token"], later_codes["token"]]).unique())
    n = len(subject_ids)
    history = _binary_matrix(history_codes, tokens, n)
    later = _binary_matrix(later_codes, tokens, n)

    # IDF = log((1 + n) / (1 + document frequency)) + 1, document frequency counted over the history records
    document_frequency = np.asarray(history.sum(axis=0)).ravel()
    idf = np.log((1 + n) / (1 + document_frequency)) + 1
    return tokens, history.multiply(idf).tocsr(), later.multiply(idf).tocsr()


def _has_codes(matrix: sparse.csr_matrix) -> np.ndarray:
    return np.diff(matrix.indptr) > 0


@dataclass(frozen=True)
class CodeSets:
    """Base class of the exact and embedded similarity. Rows are patients (in the order of subject_ids), columns codes."""
    history: sparse.csr_matrix
    later: sparse.csr_matrix
    LEVEL_NAMES = ()

    def scores(self, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def cuts(self, i: np.ndarray, j: np.ndarray) -> tuple[float, ...]:
        """Ascending cut points of the scores, taken from the unlabelled pairs (i, j)."""
        raise NotImplementedError

    def missing(self, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        """True where either record has no code in this category."""
        return ~(_has_codes(self.history)[i] & _has_codes(self.later)[j])


@dataclass(frozen=True, eq=False)
class ClinicalField:
    """One category of the comparison vector: 0 missing, then 1 + the number of cut points the similarity reaches."""
    name: str
    similarity: CodeSets
    cut_points: tuple[float, ...]

    @classmethod
    def cut_on(cls, name: str, similarity: CodeSets, i: np.ndarray, j: np.ndarray) -> "ClinicalField":
        """The cut points come from the candidate pairs (i, j), which carry no labels."""
        return cls(name, similarity, similarity.cuts(i, j))

    @property
    def level_names(self) -> tuple[str, ...]:
        return self.similarity.LEVEL_NAMES

    def levels(self, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        scores = self.similarity.scores(i, j)
        return graded_levels(scores, self.cut_points, self.similarity.missing(i, j))
