import numpy as np
import pytest
from sklearn.metrics import precision_recall_curve

from src.utils.evaluation import (
    MIN_PRECISION,
    Evaluation,
    LinkLabels,
    Ranking,
    automatic_links,
    bootstrap_difference,
    link_review_or_new,
    review_zones,
    top1_hits,
)
from src.utils.linking import THRESHOLD, one_to_one


def test_links_give_precision_recall_and_f1_with_true_pairs_that_are_not_candidates_as_misses():
    i, j = np.array([0, 0, 1, 2]), np.array([0, 1, 1, 0])
    links = LinkLabels.from_pairs(i, j, np.array([True, True, True, False]), n=3)
    assert links.precision() == pytest.approx(2 / 3)
    assert links.recall() == pytest.approx(2 / 3)
    assert links.f1() == pytest.approx(2 * 2 / (2 * 2 + 1 + 1))


def test_a_multiplicity_selects_a_subgroup_or_weights_records():
    links = LinkLabels(record=np.arange(4), is_match=np.ones(4, bool), linked=np.array([True, False, True, False]))
    assert links.recall(np.array([1.0, 0.0, 1.0, 0.0])) == 1.0
    assert links.recall(np.array([1.0, 3.0, 1.0, 3.0])) == pytest.approx(2 / 8)


def test_top1_hits_count_records_whose_true_partner_scores_highest_and_ties_and_gaps_as_misses():
    i = np.array([0, 0, 1, 1, 2, 2])
    j = np.array([0, 1, 1, 0, 2, 0])
    p = np.array([0.9, 0.5, 0.4, 0.4, 0.7, 0.2])  # record 0 hit, record 1 tied (miss), record 2 hit, record 3 has no candidate
    assert top1_hits(i, j, p, n=4).tolist() == [1, 0, 1, 0]


# ten records; record r's true pair is (r, r), a wrong pair is (r, r + 100); record 9's true pair is not a candidate
RANKED_RECORDS = [0, 1, 2, 3, 4, 5, 5, 6, 7, 8, 0, 1, 2, 3, 4]
RANKED_IS_MATCH = [True] * 5 + [False, True, True, True, True] + [False] * 5


def ranking() -> Ranking:
    i = np.array(RANKED_RECORDS)
    j = np.where(RANKED_IS_MATCH, i, i + 100)
    return Ranking.from_pairs(i, j, np.linspace(0.99, 0.1, len(i)))


def test_recall_at_99_precision_and_the_review_zone_on_a_ranked_list():
    recall, zone = ranking().summary(np.ones(10))
    assert recall == pytest.approx(5 / 10)  # the first 5 pairs are all true; the 6th is wrong, precision falls below 99 %
    assert zone == 10 - 5                   # 9 of 9 true pairs need 10 ranks, 5 are linked automatically


def test_a_subgroup_is_ranked_on_its_own_pairs_only():
    subgroup = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    recall, zone = ranking().summary(subgroup)
    assert recall == pytest.approx(5 / 6) and zone == 0


def test_a_tie_between_a_true_and_a_false_pair_counts_against_the_true_pair():
    ranked = Ranking.from_pairs(np.array([0, 0]), np.array([0, 100]), np.array([0.5, 0.5]))
    assert ranked.summary(np.ones(1))[0] == 0.0


def test_bootstrap_difference_is_positive_with_an_interval_above_zero_for_a_clearly_better_method():
    n = 200
    i = np.repeat(np.arange(n), 2)
    j = np.where(np.tile([True, False], n), i, (i + 1) % n)
    better = np.where(i == j, 0.9, 0.1)
    worse = np.where((i == j) & (i % 2 == 0), 0.9, np.where(i == j, 0.2, 0.3))  # every second true pair loses to its decoy
    new, base = (Evaluation.of(i, j, p, p >= THRESHOLD, one_to_one(i, j, p, n), n) for p in (better, worse))
    difference, low, high = bootstrap_difference(new, base, np.arange(n), draws=100)["top-1"]
    assert difference == pytest.approx(0.5) and 0 < low <= difference <= high


@pytest.mark.parametrize("weighted", [False, True])
def test_recall_at_99_precision_agrees_with_scikit_learns_precision_recall_curve(weighted):
    rng = np.random.default_rng(0)
    n = 400
    wrong_i = rng.integers(0, n, 3000)
    i = np.r_[np.arange(n - 20), wrong_i]                              # records n - 20 .. n - 1 have no true pair among the candidates
    j = np.r_[np.arange(n - 20), (wrong_i + rng.integers(1, n, 3000)) % n]   # a wrong pair never has j == i
    is_match = i == j
    p = np.where(is_match, rng.uniform(0.4, 1, len(i)), rng.uniform(0, 0.7, len(i)))
    weight = rng.integers(0, 4, n).astype(float) if weighted else np.ones(n)
    precision, recall, _ = precision_recall_curve(np.r_[is_match, np.ones(20, bool)], np.r_[p, np.full(20, -1.0)],
                                                  sample_weight=np.r_[weight[i], weight[n - 20:]])
    assert Ranking.from_pairs(i, j, p).summary(weight)[0] == pytest.approx(recall[precision >= MIN_PRECISION].max())


def test_automatic_links_are_the_pairs_counted_by_recall_at_99_precision():
    rng = np.random.default_rng(1)
    n = 300
    wrong = rng.integers(0, n, 2000)
    i = np.r_[np.arange(n), wrong]
    j = np.r_[np.arange(n), (wrong + rng.integers(1, n, 2000)) % n]
    p = np.where(i == j, rng.uniform(0.3, 1, len(i)), rng.uniform(0, 0.8, len(i)))
    linked = automatic_links(i, j, p)
    assert (i == j)[linked].mean() >= MIN_PRECISION
    assert (linked & (i == j)).sum() / n == pytest.approx(Ranking.from_pairs(i, j, p).summary(np.ones(n))[0])


def test_the_review_zone_has_as_many_pairs_as_A2():
    rng = np.random.default_rng(2)
    n = 300
    wrong = rng.integers(0, n, 2000)
    i = np.r_[np.arange(n), wrong]
    j = np.r_[np.arange(n), (wrong + rng.integers(1, n, 2000)) % n]
    p = np.where(i == j, rng.uniform(0.3, 1, len(i)), rng.uniform(0, 0.8, len(i)))
    zones = review_zones(i, j, p)
    assert (zones == 1).sum() == Ranking.from_pairs(i, j, p).summary(np.ones(n))[1]
    assert (zones[i == j] == 0).sum() <= 0.01 * (i == j).sum()        # at most 1 % of the true pairs rejected


def test_an_incoming_record_is_linked_reviewed_or_new_by_its_assigned_pair():
    # record 1 also scores high with record 0's partner, which the assignment leaves to record 0; record 3's only
    # candidate is a stranger; records 3, 4, 7 have no true candidate, 4 and 7 none at all
    i = np.array([0, 1, 1, 2, 3, 5, 6])
    j = np.array([0, 1, 0, 2, 4, 5, 6])
    p = np.array([0.99, 0.995, 0.993, 0.40, 0.90, 0.996, 0.997])
    decisions = link_review_or_new(i, j, p, n=8, max_false_links=0.1, max_missed=0.0)
    assert (decisions.upper, decisions.lower) == (0.99, 0.40)     # the assigned pairs from 0.997 down to 0.99 are all true
    assert decisions.decision.tolist() == [2, 2, 1, 1, 0, 2, 2, 0]
    assert decisions.summary()["false automatic links"] == 0 and decisions.summary()["missed (partner not a candidate)"] == 3