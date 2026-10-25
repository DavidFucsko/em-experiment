import pandas as pd

from src.preprocessing.voter_candidates import MIMIC_MAX_AGE, REFERENCE_YEAR, load_candidates


def pairs_file(tmp_path, voters):
    """voters: (ncid, sex_a, sex_b, age at the older snapshot, n_changed_fields)."""
    path = tmp_path / "pairs.parquet"
    pd.DataFrame([(ncid, sex_a, sex_b, REFERENCE_YEAR - age, changed) for ncid, sex_a, sex_b, age, changed in voters],
                 columns=["ncid", "sex_a", "sex_b", "birth_year_a", "n_changed_fields"]).to_parquet(path)
    return path


def test_candidates_drop_unstable_or_unknown_sex_and_cap_age_like_mimic(tmp_path):
    path = pairs_file(tmp_path, [
        ("keep", "F", "F", 40, 0),
        ("sex changed", "F", "M", 40, 0),
        ("sex unknown", "U", "U", 40, 0),
        ("many changes", "M", "M", 40, 4),
        ("very old", "F", "F", 97, 1),
    ])
    candidates = load_candidates(path).set_index("ncid")
    assert set(candidates.index) == {"keep", "many changes", "very old"}
    assert candidates.loc["many changes", "n_changed_fields"] == 4
    assert candidates.loc["very old", "age"] == MIMIC_MAX_AGE


def test_candidates_are_a_random_subset_of_each_sex_and_age_cell_that_keeps_the_mix_of_changes(tmp_path):
    voters = [(f"a{i}", "F", "F", 50, int(i % 4 == 0)) for i in range(2_000)] + [(f"b{i}", "M", "M", 50, 0) for i in range(5)]
    candidates = load_candidates(pairs_file(tmp_path, voters), cell_cap=400)
    assert candidates.groupby("sex").size().to_dict() == {"F": 400, "M": 5}
    assert 0.2 < candidates[candidates.sex == "F"].n_changed_fields.mean() < 0.3  # a quarter of the cell changed


def test_min_changes_keeps_only_the_heavily_changed_voters(tmp_path):
    path = pairs_file(tmp_path, [("none", "F", "F", 40, 0), ("two", "F", "F", 40, 2), ("three", "F", "F", 40, 3), ("five", "M", "M", 50, 5)])
    assert set(load_candidates(path, min_changes=3).ncid) == {"three", "five"}


def test_candidate_choice_is_deterministic_for_a_seed(tmp_path):
    path = pairs_file(tmp_path, [(f"p{i}", "M", "M", 30, 0) for i in range(50)])
    ncids = lambda seed: sorted(load_candidates(path, cell_cap=5, seed=seed).ncid)
    assert ncids(1) == ncids(1)
    assert ncids(1) != ncids(2)
