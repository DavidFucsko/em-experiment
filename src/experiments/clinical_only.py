"""Experiment 3: how well do the clinical codes alone identify a patient? As in MedLink, there is no demographic field
and no Fellegi-Sunter model: the later records are ranked by IDF cosine, after an optional prefilter on administrative
attributes (plan R-cal)."""
from collections.abc import Iterator
from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd

from src.config import SPLIT
from src.utils.exact_similarity import ExactSimilarity, exact_similarities
from src.utils.experiment_data import load_patient_ids

FILTERS = {"none": [], "gender": ["gender"], "gender + race": ["gender", "race"],
           "MedLink-style": ["gender", "insurance", "language", "marital_status", "race"]}
KS = (1, 5)
CHUNK = 500


def _side_keys(attributes: pd.DataFrame, side: str, patient_ids: np.ndarray, fields: list[str]) -> np.ndarray:
    side_rows = attributes[attributes["side"] == side].set_index("subject_id").reindex(patient_ids)
    return side_rows[fields].fillna("NA").astype(str).agg("|".join, axis=1).to_numpy()


def attribute_keys(patient_ids: np.ndarray, fields: list[str]) -> tuple[np.ndarray, np.ndarray] | None:
    """One key string per patient for the history side and the index side; None if there is no filter."""
    if not fields:
        return None
    attributes_file = (SPLIT / "attributes.parquet").as_posix()
    patients_file = (SPLIT / "patients.parquet").as_posix()
    attributes = duckdb.sql(f"""
        SELECT a.subject_id, a.side, p.gender, a.insurance, a.language, a.marital_status, a.race
        FROM '{attributes_file}' a JOIN '{patients_file}' p USING (subject_id)
    """).df()
    return _side_keys(attributes, "history", patient_ids, fields), _side_keys(attributes, "index", patient_ids, fields)


def score_chunks(similarity: ExactSimilarity) -> Iterator[tuple[int, np.ndarray]]:
    """(first history record, CHUNK history records x all later records): the n^2 scores, never all in memory."""
    for start in range(0, similarity.history.shape[0], CHUNK):
        yield start, (similarity.history[start:start + CHUNK] @ similarity.later.T).toarray()


@dataclass(frozen=True)
class Retrieval:
    """Per history record, under one filter."""
    rank: np.ndarray        # of the true partner among the later records that pass (1 = first)
    passing: np.ndarray     # the true partner passes the filter
    candidates: np.ndarray  # later records that pass

    def metrics(self, ks: tuple[int, ...] = KS) -> dict[str, float]:
        """Top-k accuracy over all patients (a filtered-out partner is a miss) and, as MedLink reports it, among the
        patients whose true pair passes."""
        metrics = {"passing": self.passing.mean(), "candidates": self.candidates.mean()}
        hits = {}
        for k in ks:
            hits[k] = self.passing & (self.rank <= k)
        for k in ks:
            metrics[f"top-{k} all"] = hits[k].mean()
        for k in ks:
            metrics[f"top-{k} passing"] = hits[k].sum() / max(self.passing.sum(), 1)
        return metrics


def rank_true_partners(similarity: ExactSimilarity, keys: dict) -> dict[str, Retrieval]:
    """filter name -> where each true partner ranks among the later records that pass the filter (a tie counts against it)."""
    n = similarity.history.shape[0]
    retrieval = {}
    for name in keys:
        retrieval[name] = Retrieval(np.empty(n, int), np.empty(n, bool), np.empty(n, int))

    for start, scores in score_chunks(similarity):
        rows = np.arange(len(scores))
        true_columns = start + rows
        window = slice(start, start + len(scores))
        for name, side_keys in keys.items():
            if side_keys is None:
                allowed = np.ones(scores.shape, bool)
            else:
                history_keys, later_keys = side_keys
                allowed = history_keys[window, None] == later_keys[None, :]
            masked = np.where(allowed, scores, -np.inf)
            true_scores = masked[rows, true_columns]
            retrieval[name].rank[window] = (masked >= true_scores[:, None]).sum(axis=1)
            retrieval[name].passing[window] = allowed[rows, true_columns]
            retrieval[name].candidates[window] = allowed.sum(axis=1)
    return retrieval


def main() -> None:
    patients = load_patient_ids()
    similarities = exact_similarities(patients)
    keys = {}
    for name, fields in FILTERS.items():
        keys[name] = attribute_keys(patients, fields)

    print(f"{len(patients):,} patients; no demographic field is used for scoring\n")
    print(f"{'codes':12s} {'prefilter':14s} {'true pair passes':>16s} {'candidates':>10s} | "
          f"{'top-1':>6s} {'top-5':>6s} (all patients) | {'top-1':>6s} {'top-5':>6s} (among passing)")
    for codes, similarity in similarities.items():
        for name, retrieval in rank_true_partners(similarity, keys).items():
            row = retrieval.metrics()
            print(f"{codes:12s} {name:14s} {row['passing']:16.3f} {row['candidates']:10.0f} | {row['top-1 all']:6.3f} "
                  f"{row['top-5 all']:6.3f}                  | {row['top-1 passing']:6.3f} {row['top-5 passing']:6.3f}")


if __name__ == "__main__":
    main()
