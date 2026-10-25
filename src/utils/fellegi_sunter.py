"""Fellegi-Sunter (1969) with m, u and the match share estimated by EM over all record pairs, without labels (Jaro 1989;
Enamorado, Fifield & Imai 2019). The fields are independent given the class (match / non-match)."""
from dataclasses import dataclass

import numpy as np

from src.utils.priors import DEFAULT_PRIOR, Prior

EPSILON = 1e-12        # keeps log() finite where a level has no pairs in a class
M_START = 0.9          # starting m of full agreement (the highest level), the rest spread over the other levels
SHARE_START = 0.1      # starting match share


@dataclass(frozen=True)
class FellegiSunter:
    m: list[np.ndarray]  # per field: P(level | match)
    u: list[np.ndarray]  # per field: P(level | non-match)
    match_share: float   # P(match) among the pairs the model scores

    def match_probability(self, levels: np.ndarray) -> np.ndarray:
        """sigmoid(logit(match share) + the sum of log(m / u) over the fields, at each pair's level)."""
        share = np.clip(self.match_share, EPSILON, 1 - EPSILON)
        log_odds = np.log(share / (1 - share))
        for field in range(len(self.m)):
            level = levels[:, field]
            log_odds = log_odds + np.log(self.m[field][level] + EPSILON) - np.log(self.u[field][level] + EPSILON)
        return 1 / (1 + np.exp(-log_odds))


def level_counts(vectors: np.ndarray, weights: np.ndarray, n_levels: list[int]) -> list[np.ndarray]:
    """Per field the weighted number of comparison vectors at each level."""
    counts = []
    for field, k in enumerate(n_levels):
        counts.append(np.bincount(vectors[:, field], weights, minlength=k))
    return counts


def _start_m(k: int) -> np.ndarray:
    start = np.full(k, (1 - M_START) / max(k - 1, 1))
    start[-1] = M_START
    return start / start.sum()


def _largest_change(new: FellegiSunter, old: FellegiSunter) -> float:
    change = abs(new.match_share - old.match_share)
    for new_values, old_values in zip(new.m + new.u, old.m + old.u):
        change = max(change, np.abs(new_values - old_values).max())
    return change


def fit(vectors: np.ndarray, counts: np.ndarray, n_levels: list[int], max_share: float = 1.0,
        prior: str = DEFAULT_PRIOR, iterations: int = 5_000, tolerance: float = 1e-8) -> FellegiSunter:
    """EM on the distinct comparison vectors (vectors x fields) and how many pairs have each. Starts from m = M_START at
    full agreement, u = the level shares of all pairs (nearly all are non-matches) and the match share SHARE_START."""
    counts = np.asarray(counts, float)
    total = counts.sum()
    shares = []
    for level in level_counts(vectors, counts, n_levels):
        shares.append(level / total)
    smoothing = Prior.of(prior, shares)
    model = FellegiSunter([_start_m(k) for k in n_levels], shares, min(SHARE_START, max_share))

    for _ in range(iterations):
        # E step: the match probability of every comparison vector
        p = model.match_probability(vectors)
        # M step: expected level counts among the matches and the non-matches. The match share is their mean, cut at
        # max_share (Winkler's 1993 bound: the log-likelihood is concave in the share, so the bounded maximum is the cut mean)
        matches = level_counts(vectors, p * counts, n_levels)
        non_matches = level_counts(vectors, (1 - p) * counts, n_levels)
        match_share = min((p * counts).sum() / total, max_share)
        # smoothing: the prior turns the counts into m and u
        new = FellegiSunter(smoothing.probabilities(matches), smoothing.probabilities(non_matches), match_share)

        # the match share is tiny (about 1 / n), so EM runs until no parameter moves by more than `tolerance`
        change = _largest_change(new, model)
        model = new
        if change < tolerance:
            break
    return model
