import numpy as np
import pandas as pd

from src.utils.fellegi_sunter import FellegiSunter
from src.utils.linking import link, one_to_one
from src.utils.runs import Scores


def test_one_to_one_assigns_globally_so_a_record_with_an_alternative_gives_way():
    i, j = np.array([0, 1, 1]), np.array([0, 0, 1])
    p = np.array([0.9, 0.95, 0.6])
    assert one_to_one(i, j, p, n=2).tolist() == [True, False, True]


def test_one_to_one_gives_a_contested_record_to_the_higher_probability_when_there_is_no_alternative():
    assert one_to_one(np.array([0, 1]), np.array([0, 0]), np.array([0.9, 0.95]), n=2).tolist() == [False, True]


def test_one_to_one_does_not_link_below_the_threshold():
    assert one_to_one(np.array([0]), np.array([0]), np.array([0.3]), n=1).tolist() == [False]


def test_link_keeps_the_pairs_order_and_gives_both_decisions():
    pairs = pd.DataFrame({"history_subject_id": [10, 20, 20], "later_subject_id": [10, 10, 20], "probability": [0.9, 0.95, 0.6]})
    links = link(Scores(pairs, {}, FellegiSunter([], [], 0.5)), np.array([10, 20]))
    assert links[["history_subject_id", "later_subject_id"]].equals(pairs[["history_subject_id", "later_subject_id"]])
    assert links.linked.tolist() == [True, True, True] and links.assigned.tolist() == [True, False, True]
