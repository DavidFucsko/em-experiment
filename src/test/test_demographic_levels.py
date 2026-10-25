import numpy as np
import pandas as pd

from src.utils.demographic_levels import DEMOGRAPHIC_FIELDS, add_derived_fields, demographic_fields
from src.utils.fields import ComparisonVector


def record(**changes):
    base = dict(first_name="ANNA", middle_name="LEE", last_name="KING", address="12 OAK ST", birth_year=1976,
                sex="F", city="GRAHAM", zip_code="27249", phone="5551234", race="WHITE")
    return {**base, **changes}


def vector(history: dict, later: dict, fields=DEMOGRAPHIC_FIELDS) -> ComparisonVector:
    return ComparisonVector(demographic_fields(*(add_derived_fields(pd.DataFrame([side])) for side in (history, later)), fields))


def levels_of(history: dict, later: dict) -> dict:
    row = vector(history, later).levels(np.array([0]), np.array([0]))[0]
    return dict(zip(DEMOGRAPHIC_FIELDS, row.tolist()))


def test_identical_records_have_full_agreement_on_every_field():
    levels = levels_of(record(), record())
    assert levels == {"first_name": 3, "middle_name": 3, "last_name": 3, "address": 3, "birth_year": 3, "sex": 2,
                      "city": 2, "zip_code": 2, "phone": 2, "race": 2, "first_initial": 2, "last_initial": 2,
                      "zip3": 2, "age_band": 2}


def test_a_move_lowers_address_city_and_zip_and_a_missing_phone_is_level_zero():
    levels = levels_of(record(), record(address="905 JOE COLLINS RD", city="MEBANE", zip_code="27302", phone=None))
    assert (levels["address"], levels["city"], levels["zip_code"], levels["phone"]) == (1, 1, 1, 0)


def test_birth_year_off_by_one_and_a_name_typo_get_the_middle_level():
    levels = levels_of(record(), record(birth_year=1977, last_name="KIMG"))
    assert levels["birth_year"] == 2
    assert levels["last_name"] == 2  # KING vs KIMG: Jaro-Winkler between 0.85 and 0.94


def test_a_subset_of_fields_gives_those_columns_only_and_levels_stay_within_their_names():
    subset = vector(record(), record(last_name="KIMG", phone=None), ("last_name", "phone"))
    assert subset.levels(np.array([0]), np.array([0])).tolist() == [[2, 0]]
    assert subset.n_levels == [4, 3]
    every = vector(record(), record())
    assert all(every.levels(np.array([0]), np.array([0]))[0] < every.n_levels)


def test_a_move_within_the_zip3_area_and_a_year_within_the_age_band_still_agree_on_the_coarse_fields():
    levels = levels_of(record(), record(zip_code="27258", birth_year=1977, first_name="ANN"))
    assert (levels["zip_code"], levels["zip3"], levels["birth_year"], levels["age_band"]) == (1, 2, 2, 2)
    assert levels["first_initial"] == 2
