import pandas as pd

from src.utils.blocking import blocking_pairs


def test_a_pair_is_a_candidate_if_any_rule_agrees_and_each_pair_appears_once():
    history = pd.DataFrame({"name": ["ann", "bob", None], "zip": ["1", "2", "3"]})
    later = pd.DataFrame({"name": ["ann", "cat", None], "zip": ["9", "2", "3"]})
    i, j = blocking_pairs(history, later, [["name"], ["zip"], ["name", "zip"]])
    assert list(zip(i, j)) == [(0, 0), (1, 1), (2, 2)]


def test_missing_key_values_never_match():
    frame = pd.DataFrame({"name": [None, None]})
    i, j = blocking_pairs(frame, frame, [["name"]])
    assert len(i) == 0
