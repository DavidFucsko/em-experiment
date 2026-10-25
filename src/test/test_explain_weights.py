import numpy as np

from src.experiments.explain_weights import change_summary, changes, contributions, outcomes, prior_weight
from src.utils.fellegi_sunter import FellegiSunter

MODEL = FellegiSunter([np.array([0.1, 0.2, 0.7]), np.array([0.05, 0.95])],
                      [np.array([0.5, 0.4, 0.1]), np.array([0.5, 0.5])], 0.01)


def test_prior_plus_contributions_give_the_models_probability():
    levels = np.array([[0, 0], [1, 1], [2, 1], [2, 0]])
    log2_odds = prior_weight(MODEL) + contributions(MODEL, levels).sum(axis=1)
    assert np.allclose(1 / (1 + 2 ** -log2_odds), MODEL.match_probability(levels))


def test_outcomes_after_assignment():
    i, j = np.array([0, 0, 1, 2]), np.array([0, 1, 1, 0])
    kinds = outcomes(i, j, np.array([True, False, False, True]))
    assert kinds["true positive"].tolist() == [True, False, False, False]
    assert kinds["false positive"].tolist() == [False, False, False, True]
    assert kinds["false negative"].tolist() == [False, False, True, False]


def test_changes_against_the_base_model():
    true = np.array([True, True, False, False])
    groups = changes(true, linked=np.array([True, False, True, False]), base_linked=np.array([False, True, False, True]))
    assert [mask.tolist().index(True) for mask in groups.values()] == [0, 1, 2, 3]


def test_the_summary_counts_right_and_wrong_changes_and_their_ratio():
    groups = {"true pair gained": np.ones(6, bool), "true pair lost": np.ones(2, bool),
              "false link added": np.zeros(2, bool), "false link removed": np.ones(3, bool)}
    assert change_summary(groups).startswith("11 decisions changed: 9 right") and change_summary(groups).endswith("= 4.50")
