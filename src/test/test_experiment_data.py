import pandas as pd

from src.preprocessing.identity_assignment import assignment_file
from src.utils.experiment_data import load_demographic_records, race_group, side_fields


def test_side_fields_strips_the_suffix_of_one_side_only():
    assert side_fields({"city_a": "X", "city_b": "Y", "ncid": "N"}, "_a") == {"city": "X"}


def test_race_is_grouped_by_its_first_word_and_unknown_is_missing():
    race = pd.Series(["WHITE - RUSSIAN", "BLACK/CAPE VERDEAN", "PORTUGUESE", "UNKNOWN", None])
    assert race_group(race).fillna("missing").tolist() == ["WHITE", "BLACK", "OTHER", "missing", "missing"]


def test_demographic_records_follow_the_order_of_the_requested_patients_with_race_per_side(tmp_path):
    identity = {f"{field}_{side}": ["X", "Y"] for field in ("first_name", "last_name", "zip_code") for side in "ab"}
    pd.DataFrame({"ncid": ["N1", "N2"], "city_a": ["A1", "A2"], "city_b": ["B1", "B2"], "birth_year_a": [1950, 1960],
                  "birth_year_b": [1950, 1960], **identity, "n_changed_fields": [0, 1]}).to_parquet(tmp_path / "pairs.parquet")
    pd.DataFrame({"subject_id": [10, 20], "ncid": ["N1", "N2"], "age_gap": [0, 1]}
                 ).to_parquet(assignment_file("hard", "cond", tmp_path))
    pd.DataFrame({"subject_id": [10, 10, 20], "side": ["history", "index", "index"],
                  "race": ["WHITE", "WHITE - RUSSIAN", "ASIAN"]}).to_parquet(tmp_path / "attributes.parquet")
    records = load_demographic_records("hard", "cond", [20, 10], tmp_path / "pairs.parquet", tmp_path,
                                       tmp_path / "attributes.parquet")
    assert records.history.city.tolist() == ["A2", "A1"]
    assert records.later.city.tolist() == ["B2", "B1"]
    assert records.history.race.fillna("missing").tolist() == ["missing", "WHITE"] and records.later.race.tolist() == ["ASIAN", "WHITE"]
    assert records.history.age_band.tolist() == [1960, 1950]
    assert records.n_changed_fields.tolist() == [1, 0]


def test_patient_subsets_are_nested_random_and_sorted(tmp_path, monkeypatch):
    pd.DataFrame({"subject_id": range(100, 200)}).to_parquet(tmp_path / "patients.parquet")
    monkeypatch.setattr("src.utils.experiment_data.SPLIT", tmp_path)
    from src.utils.experiment_data import load_patient_ids
    small, large, every = load_patient_ids(18), load_patient_ids(40), load_patient_ids(None)
    assert len(every) == 100 and set(small) <= set(large) <= set(every)
    assert list(small) == sorted(small) and list(small) != list(range(100, 118))
