"""How good the links are. Candidate pair (i, j) is history record i and later record j, the true pair is
j == i. Every metric takes an optional `multiplicity` per history record: 0/1 selects a subgroup, counts a bootstrap draw."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

from src.config import DEVICE, SEED
from src.utils.linking import one_to_one
from src.utils.runs import Scores

MIN_PRECISION = 0.99       # automatic linking must keep this precision (upper threshold of the review zone)
MAX_RECALL_LOSS = 0.01     # automatic rejection may lose this share of the true pairs (lower threshold)
MAX_FALSE_LINKS = 0.001    # link_review_or_new: false automatic links tolerated (Mayo: "0 to .1%", ONC 2014)

# zones of a candidate pair (Fellegi & Sunter's A3, A2, A1)
REJECTED = 0
REVIEW = 1
AUTOMATIC = 2
ZONES = ("rejected", "review", "automatic")


def review_zones(i: np.ndarray, j: np.ndarray, p: np.ndarray, min_precision: float = MIN_PRECISION,
                 max_recall_loss: float = MAX_RECALL_LOSS) -> np.ndarray:
    """The zone of every candidate pair. AUTOMATIC: from the top of the ranking down to the last rank where precision is
    still >= min_precision. REJECTED: below the rank that keeps all but max_recall_loss of the true pairs. REVIEW: in
    between (|A2|). The thresholds are set with the labels."""
    is_true = i == j
    order = np.lexsort((is_true, -p))  # by p descending, a tie counts against the true pair
    found = np.cumsum(is_true[order])
    precision = found / np.arange(1, len(p) + 1)
    precise_ranks = np.flatnonzero(precision >= min_precision)
    upper = precise_ranks[-1] if len(precise_ranks) else -1
    lower = np.searchsorted(found, np.ceil((1 - max_recall_loss) * found[-1]))

    rank = np.arange(len(p))
    zone_by_rank = np.full(len(p), REJECTED)
    zone_by_rank[rank <= lower] = REVIEW
    zone_by_rank[rank <= upper] = AUTOMATIC
    zones = np.empty(len(p), int)
    zones[order] = zone_by_rank
    return zones


def automatic_links(i: np.ndarray, j: np.ndarray, p: np.ndarray) -> np.ndarray:
    """The pairs linked automatically at 99 % precision."""
    return review_zones(i, j, p) == AUTOMATIC


def top1_hits(i: np.ndarray, j: np.ndarray, p: np.ndarray, n: int) -> np.ndarray:
    """Per history record 1 if its true partner is its highest-scored candidate (MedLink's top-1 accuracy). A tie counts
    against the true pair, a record without candidates is a miss."""
    is_true = i == j
    order = np.lexsort((is_true, -p, i))  # by record, then by p descending
    sorted_records = i[order]
    is_best = np.concatenate(([True], sorted_records[1:] != sorted_records[:-1]))
    hits = np.zeros(n)
    hits[sorted_records[is_best]] = is_true[order][is_best]
    return hits


@dataclass(frozen=True)
class LinkLabels:
    """The pairs that are linked or true (no other pair changes precision, recall or F1) as scikit-learn labels. A true
    pair that is not a candidate is added as a missed row."""
    record: np.ndarray
    is_match: np.ndarray
    linked: np.ndarray

    @classmethod
    def from_pairs(cls, i: np.ndarray, j: np.ndarray, linked: np.ndarray, n: int) -> "LinkLabels":
        correct = i == j
        shown = linked | correct
        missing = np.setdiff1d(np.arange(n), i[correct])
        record = np.concatenate((i[shown], missing))
        is_match = np.concatenate((correct[shown], np.ones(len(missing), bool)))
        was_linked = np.concatenate((linked[shown], np.zeros(len(missing), bool)))
        return cls(record, is_match, was_linked)

    def _weight(self, multiplicity: np.ndarray | None) -> np.ndarray | None:
        if multiplicity is None:
            return None
        return multiplicity[self.record]

    def precision(self, multiplicity: np.ndarray | None = None) -> float:
        return precision_score(self.is_match, self.linked, sample_weight=self._weight(multiplicity), zero_division=0)

    def recall(self, multiplicity: np.ndarray | None = None) -> float:
        return recall_score(self.is_match, self.linked, sample_weight=self._weight(multiplicity), zero_division=0)

    def f1(self, multiplicity: np.ndarray | None = None) -> float:
        return f1_score(self.is_match, self.linked, sample_weight=self._weight(multiplicity), zero_division=0)


@dataclass(frozen=True)
class Ranking:
    """Candidate pairs by decreasing match probability (a tie counts against the true pair), on the device, so every
    bootstrap draw is a few vector operations. Needs no probability threshold."""
    record: torch.Tensor
    is_match: torch.Tensor

    @classmethod
    def from_pairs(cls, i: np.ndarray, j: np.ndarray, p: np.ndarray) -> "Ranking":
        order = np.lexsort((i == j, -p))
        record = torch.as_tensor(i[order], device=DEVICE)
        is_match = torch.as_tensor((i == j)[order], device=DEVICE)
        return cls(record, is_match)

    def summary(self, multiplicity: np.ndarray) -> tuple[float, float]:
        """(recall at 99 % precision, |A2|). |A2| is the number of pairs between the automatic threshold and the lower
        one, below which at most 1 % of the true pairs would be lost."""
        weight = torch.as_tensor(multiplicity, dtype=torch.float64, device=DEVICE)[self.record]
        found = torch.cumsum(weight * self.is_match, dim=0)
        pairs = torch.cumsum(weight, dim=0)
        precision = torch.where(pairs > 0, found / pairs, 0.0)

        # last rank with precision >= MIN_PRECISION, -1 if there is none
        ranks = torch.arange(len(pairs), device=DEVICE)
        auto = torch.where((precision >= MIN_PRECISION) & (pairs > 0), ranks, -1).max().item()
        reached = torch.searchsorted(found, torch.ceil((1 - MAX_RECALL_LOSS) * found[-1:])).item()

        recall = 0.0
        auto_pairs = 0
        if auto >= 0:
            recall = found[auto].item() / multiplicity.sum()
            auto_pairs = pairs[auto].item()
        review_pairs = max(pairs[reached].item() - auto_pairs, 0)
        return recall, review_pairs


@dataclass(frozen=True)
class Evaluation:
    """The result of one method on the candidate pairs, kept per record so that a subgroup or a bootstrap draw is cheap."""
    n: int
    top1: np.ndarray
    ranking: Ranking
    threshold_links: LinkLabels
    one_to_one_links: LinkLabels

    @classmethod
    def of(cls, i: np.ndarray, j: np.ndarray, p: np.ndarray, linked: np.ndarray, assigned: np.ndarray,
           n: int) -> "Evaluation":
        """p: match probabilities; linked, assigned: the pairs linked at the threshold and after one-to-one assignment."""
        return cls(n, top1_hits(i, j, p, n), Ranking.from_pairs(i, j, p),
                   LinkLabels.from_pairs(i, j, linked, n), LinkLabels.from_pairs(i, j, assigned, n))

    def metrics(self, multiplicity: np.ndarray | None = None) -> dict[str, float]:
        weight = np.ones(self.n) if multiplicity is None else multiplicity
        recall, review_pairs = self.ranking.summary(weight)
        return {
            "top-1": (weight * self.top1).sum() / weight.sum(),
            "|A2|": review_pairs,
            "R@99%P": recall,
            "F1 (P3)": self.one_to_one_links.f1(weight),
            "precision (P3)": self.one_to_one_links.precision(weight),
            "precision (0.5)": self.threshold_links.precision(weight),
            "recall (0.5)": self.threshold_links.recall(weight),
        }


def _metric_differences(new: Evaluation, base: Evaluation, multiplicity: np.ndarray) -> np.ndarray:
    new_metrics = new.metrics(multiplicity)
    base_metrics = base.metrics(multiplicity)
    return np.array([new_metrics[metric] - base_metrics[metric] for metric in new_metrics])


def bootstrap_difference(new: Evaluation, base: Evaluation, records: np.ndarray, draws: int = 500,
                         seed: int = SEED) -> dict[str, tuple[float, float, float]]:
    """metric -> (new - base on `records`, lower and upper end of the 95 % interval), resampling `records` with replacement."""
    rng = np.random.default_rng(seed)
    resampled = []
    for _ in range(draws):
        draw = rng.choice(records, len(records))
        multiplicity = np.bincount(draw, minlength=new.n).astype(float)
        resampled.append(_metric_differences(new, base, multiplicity))
    low, high = np.quantile(resampled, [0.025, 0.975], axis=0)

    indicator = np.zeros(new.n)
    indicator[records] = 1
    point = _metric_differences(new, base, indicator)
    result = {}
    for k, metric in enumerate(new.metrics()):
        result[metric] = (point[k], low[k], high[k])
    return result


def evaluate(scores: Scores, links: pd.DataFrame, subject_ids) -> Evaluation:
    """The evaluation of a stored run, from its probabilities and its links."""
    i, j = scores.positions(subject_ids)
    p = scores.pairs["probability"].to_numpy()
    return Evaluation.of(i, j, p, links["linked"].to_numpy(), links["assigned"].to_numpy(), len(subject_ids))


@dataclass(frozen=True)
class RecordDecisions:
    """What a master patient index does with each incoming (history) record (Morris et al. 2014, ONC): link it to its
    assigned partner automatically, send the pair to clerical review, or create a new record."""
    decision: np.ndarray        # AUTOMATIC, REVIEW or REJECTED (new record)
    correct: np.ndarray         # the assigned partner is the true one
    has_partner: np.ndarray     # the true partner is among the candidates
    upper: float                # lowest p linked automatically
    lower: float                # lowest p sent to review

    def summary(self) -> dict[str, int]:
        automatic = self.decision == AUTOMATIC
        review = self.decision == REVIEW
        new = self.decision == REJECTED
        return {
            "automatic links": int(automatic.sum()),
            "false automatic links": int((automatic & ~self.correct).sum()),
            "review tasks": int(review.sum()),
            "new records": int(new.sum()),
            "missed (partner among the candidates)": int((new & self.has_partner).sum()),
            "missed (partner not a candidate)": int((~self.has_partner).sum()),
        }


def _lowest_p(p: np.ndarray, zones: np.ndarray, zone: int) -> float:
    """The lowest p in `zone` or above, inf if the zone is empty."""
    in_zone = zones >= zone
    if not in_zone.any():
        return np.inf
    return float(p[in_zone].min())


def link_review_or_new(i: np.ndarray, j: np.ndarray, p: np.ndarray, n: int, max_false_links: float = MAX_FALSE_LINKS,
                       max_missed: float = MAX_RECALL_LOSS) -> RecordDecisions:
    """One-to-one assignment of all candidate pairs (no 0.5 threshold), then Fellegi & Sunter's two thresholds on the
    assigned pairs, set by error bounds (Winkler 2002). A record without an assigned pair becomes a new record."""
    assigned = one_to_one(i, j, p, n, threshold=0.0)
    assigned_i = i[assigned]
    assigned_j = j[assigned]
    assigned_p = p[assigned]
    zones = review_zones(assigned_i, assigned_j, assigned_p, min_precision=1 - max_false_links, max_recall_loss=max_missed)

    decision = np.zeros(n, int)
    correct = np.zeros(n, bool)
    has_partner = np.zeros(n, bool)
    decision[assigned_i] = zones
    correct[assigned_i] = assigned_i == assigned_j
    has_partner[i[i == j]] = True
    return RecordDecisions(decision, correct, has_partner,
                           _lowest_p(assigned_p, zones, AUTOMATIC), _lowest_p(assigned_p, zones, REVIEW))
