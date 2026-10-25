import numpy as np
import pytest

from src.utils.priors import WINKLER_DELTA, Prior

SHARES = [np.array([0.7, 0.2, 0.1])]           # one field: its levels' share among all pairs
COUNTS = [np.array([0.0, 30.0, 70.0])]         # e.g. the expected matches per level: none at level 0


def test_without_a_prior_the_probabilities_are_the_count_shares():
    assert Prior.of("none", SHARES).probabilities(COUNTS)[0].tolist() == [0.0, 0.3, 0.7]


def test_winkler_adds_a_small_delta_to_every_level():
    probabilities = Prior.of("winkler", SHARES).probabilities(COUNTS)[0]
    assert probabilities[0] == pytest.approx(WINKLER_DELTA / (100 + 3 * WINKLER_DELTA))
    assert probabilities.sum() == pytest.approx(1.0)


def test_flat_is_laplace_smoothing_with_one_per_level():
    assert Prior.of("flat", SHARES).probabilities(COUNTS)[0] == pytest.approx([1 / 103, 31 / 103, 71 / 103])


def test_centred_spreads_one_pseudo_count_by_the_shares_among_all_pairs():
    assert Prior.of("centred", SHARES).probabilities(COUNTS)[0] == pytest.approx([0.7 / 101, 30.2 / 101, 70.1 / 101])
