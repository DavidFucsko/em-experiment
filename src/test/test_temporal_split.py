import duckdb
import pytest

from src.preprocessing.temporal_split import build_temporal_split


@pytest.fixture
def split_dir(mimic_dirs, tmp_path):
    return build_temporal_split(mimic_dirs["hosp"], mimic_dirs["icu"], mimic_dirs["ed"], tmp_path / "split")


def read(split_dir, side, columns):
    return set(duckdb.sql(f"SELECT DISTINCT {columns} FROM '{(split_dir / f'{side}.parquet').as_posix()}' "
                          f"WHERE subject_id = 1").fetchall())


def events(split_dir, side):
    """(modality, vocabulary, code) of patient 1 on one side."""
    return read(split_dir, side, "modality, vocabulary, code")


def test_only_patients_with_earlier_admission_are_eligible(split_dir):
    patients = duckdb.sql(f"SELECT * FROM '{(split_dir / 'patients.parquet').as_posix()}'").fetchall()
    assert patients == [(1, "F", 50)]


def test_history_holds_earlier_admission_and_earlier_ed_stays(split_dir):
    assert events(split_dir, "history") == {
        ("dx", "icd9", "250"), ("drg", "drg_hcfa", "057"), ("dx", "icd10", "R07"), ("homemed", "gsn", "000555"),
    }


def test_index_admission_holds_index_record_and_ignores_later_admissions(split_dir):
    assert events(split_dir, "index_admission") == {
        ("dx", "icd10", "I10"), ("px", "icd10", "0BH17EZ"), ("hcpcs", "hcpcs", "99285"),
        ("org", "organism", "STAPH AUREUS"), ("svc", "service", "MED"), ("icurx", "mimic_item", "5"),
        ("icupx", "mimic_item", "7"), ("homemed", "gsn", "000777"), ("edrx", "gsn", "000888"),
    }


def test_ed_arrival_holds_only_the_index_ed_stay(split_dir):
    assert events(split_dir, "ed_arrival") == {
        ("dx", "icd10", "I10"), ("homemed", "gsn", "000777"), ("edrx", "gsn", "000888"),
    }


def test_codes_keep_leading_zeros(split_dir):
    codes = {code for _, _, code in events(split_dir, "history") | events(split_dir, "index_admission")}
    assert {"057", "000555", "000777", "000888"} <= codes


def test_visits_number_the_whole_timeline_and_merge_an_admission_with_its_ed_stay(split_dir):
    assert read(split_dir, "history", "modality, vocabulary, code, visit") == {
        ("dx", "icd10", "R07", 1), ("homemed", "gsn", "000555", 1),                      # ED stay without admission
        ("dx", "icd9", "250", 2), ("drg", "drg_hcfa", "057", 2), ("dx", "icd10", "R07", 2),  # admission 10 + ED stay 101
    }
    assert {visit for (visit,) in read(split_dir, "index_admission", "visit")} == {3}
    assert {visit for (visit,) in read(split_dir, "ed_arrival", "visit")} == {3}


def test_attributes_come_from_the_last_history_admission_and_the_index_admission(split_dir):
    rows = duckdb.sql(f"SELECT side, insurance, marital_status, race FROM '{(split_dir / 'attributes.parquet').as_posix()}' "
                      f"ORDER BY side").fetchall()
    assert rows == [("history", "Medicaid", "SINGLE", "WHITE"), ("index", "Medicare", "MARRIED", "WHITE")]


def test_visit_time_is_the_earliest_start_of_the_visit(split_dir):
    visit_times = (read(split_dir, "history", "visit, strftime(visit_time, '%Y-%m-%d %H:%M')")
                   | read(split_dir, "index_admission", "visit, strftime(visit_time, '%Y-%m-%d %H:%M')"))
    assert dict(visit_times) == {1: "2099-12-30 10:00", 2: "2099-12-31 20:00", 3: "2100-12-31 20:00"}
