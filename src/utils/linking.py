"""From match probabilities to links: E2 links every candidate pair with p >= 0.5, P3 also gives each record
at most one partner (optimal one-to-one assignment)."""
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

from src.utils.runs import PAIR, Scores

NO_PAIR_COST = 1e6
THRESHOLD = 0.5            # a pair with at least this match probability is linked


def one_to_one(i: np.ndarray, j: np.ndarray, p: np.ndarray, n: int, threshold: float = THRESHOLD) -> np.ndarray:
    """Optimal one-to-one assignment on -log p (scipy's sparse Hungarian); a pair is linked if assigned and p >= threshold.
    Every record also gets a dummy partner at NO_PAIR_COST, so a record without a real partner can stay unassigned."""
    # the +1 keeps every weight positive (a zero would read as no edge), and adds the same to every record's cost
    cost = 1 - np.log(np.clip(p, 1e-12, 1))
    real = sparse.csr_matrix((cost, (i, j)), shape=(n, n))
    with_dummies = sparse.hstack([real, NO_PAIR_COST * sparse.eye(n)]).tocsr()
    rows, columns = min_weight_full_bipartite_matching(with_dummies)
    partner = np.empty(n, dtype=int)
    partner[rows] = columns
    return (partner[i] == j) & (p >= threshold)


def link(scores: Scores, subject_ids) -> pd.DataFrame:
    """One row per candidate pair, in the order of scores.pairs: history_subject_id, later_subject_id, linked
    (p >= 0.5, E2), assigned (linked after one-to-one assignment, E2+P3)."""
    i, j = scores.positions(subject_ids)
    p = scores.pairs["probability"].to_numpy()
    return scores.pairs[PAIR].assign(linked=p >= THRESHOLD, assigned=one_to_one(i, j, p, len(subject_ids)))
