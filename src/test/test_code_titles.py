import duckdb
import pytest

from src.preprocessing.code_titles import build_code_titles


@pytest.fixture
def titles(mimic_dirs, tmp_path):
    out_file = build_code_titles(mimic_dirs["hosp"], mimic_dirs["icu"], mimic_dirs["ed"], tmp_path / "titles.parquet")
    rows = duckdb.sql(f"SELECT modality, vocabulary, code, title FROM '{out_file.as_posix()}'").fetchall()
    return {(modality, vocabulary, code): title for modality, vocabulary, code, title in rows}


def test_icd_titles_are_keyed_by_version_and_modality(titles):
    assert titles[("dx", "icd9", "250")] == "Diabetes mellitus"
    assert titles[("dx", "icd10", "I10")] == "Essential hypertension"
    assert titles[("px", "icd10", "0BH17EZ")] == "Insertion of airway"
    assert ("px", "icd10", "I10") not in titles


def test_drg_hcpcs_and_icu_item_titles(titles):
    assert titles[("drg", "drg_hcfa", "057")] == "DEGENERATIVE NERVOUS SYSTEM"
    assert titles[("hcpcs", "hcpcs", "99285")] == "Emergency visit"      # empty long description -> short one
    assert titles[("icurx", "mimic_item", "5")] == "NaCl 0.9%"
    assert titles[("icupx", "mimic_item", "7")] == "Invasive Ventilation"


def test_gsn_title_is_the_most_frequent_name_per_source_and_keeps_leading_zeros(titles):
    assert titles[("homemed", "gsn", "000777")] == "Aspirin"
    assert titles[("edrx", "gsn", "000777")] == "Aspirin"                 # "Aspirin EC" is rarer
    assert titles[("edrx", "gsn", "000888")] == "Heparin"


def test_codes_that_are_already_words_have_no_title_row(titles):
    assert {modality for modality, _, _ in titles} == {"dx", "px", "drg", "hcpcs", "icurx", "icupx", "homemed", "edrx"}
