"""What every experiment starts from: the eligible patients and each one's two demographic records."""
from collections.abc import Mapping
from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd

from src.config import SEED, SIZE, SPLIT, VOTER
from src.preprocessing.identity_assignment import assignment_file
from src.preprocessing.voter_pairs import PAIRS_FILE
from src.utils.demographic_levels import add_derived_fields

RACE_GROUPS = ("WHITE", "BLACK", "HISPANIC", "ASIAN")  # MIMIC's race values start with one of these or are OTHER
RACE_UNKNOWN = ("UNKNOWN", "PATIENT DECLINED TO ANSWER", "UNABLE TO OBTAIN")


def side_fields(record: Mapping, suffix: str) -> dict:
    """The fields of one side of a voter pair: {"city_a": ...} -> {"city": ...}."""
    fields = {}
    for key, value in record.items():
        if key.endswith(suffix):
            fields[key.removesuffix(suffix)] = value
    return fields


def race_group(race: pd.Series) -> pd.Series:
    """MIMIC's race in five groups: the one of RACE_GROUPS it starts with, otherwise OTHER; unknown is missing."""
    pattern = f"^({'|'.join(RACE_GROUPS)})"
    group = race.str.extract(pattern, expand=False).fillna("OTHER")
    is_known = race.notna() & ~race.isin(RACE_UNKNOWN)
    return group.where(is_known)


def load_patient_ids(size: int | None = SIZE) -> np.ndarray:
    """The eligible patients, sorted: all of them, or the first `size` of one seeded shuffle (EM_SIZE in config.py), so
    a smaller subset is part of every larger one."""
    patients = np.sort(pd.read_parquet(SPLIT / "patients.parquet")["subject_id"].to_numpy())
    if size is None:
        return patients
    shuffled = np.random.default_rng(SEED).permutation(patients)
    return np.sort(shuffled[:size])


@dataclass(frozen=True)
class DemographicRecords:
    """Row k of `history` (voter, older snapshot) and of `later` (newer snapshot) belong to patient k."""
    history: pd.DataFrame
    later: pd.DataFrame
    n_changed_fields: np.ndarray  # of the seven identity fields, between the two snapshots


def _side_records(rows: pd.DataFrame, suffix: str) -> pd.DataFrame:
    side = pd.DataFrame(side_fields(rows, suffix))
    side = side.assign(race=race_group(side["race"]))
    return add_derived_fields(side)


def load_demographic_records(pool: str, variant: str, subject_ids, pairs_file=PAIRS_FILE, assignment_dir=VOTER,
                             attributes_file=SPLIT / "attributes.parquet") -> DemographicRecords:
    """The voter identity assigned from `pool` ("natural" or "hard") in `variant` ("cond" or "indep") to each patient,
    in the order of `subject_ids`. The race comes from the patient's last history admission and the index admission."""
    wanted = pd.DataFrame({"subject_id": np.asarray(subject_ids), "position": range(len(subject_ids))})
    assignments = assignment_file(pool, variant, assignment_dir).as_posix()
    attributes = attributes_file.as_posix()
    rows = duckdb.sql(f"""
        SELECT pairs.*, history.race AS race_a, later.race AS race_b
        FROM '{assignments}' assignment
        JOIN wanted USING (subject_id) JOIN '{pairs_file.as_posix()}' pairs USING (ncid)
        LEFT JOIN '{attributes}' history ON history.subject_id = wanted.subject_id AND history.side = 'history'
        LEFT JOIN '{attributes}' later ON later.subject_id = wanted.subject_id AND later.side = 'index'
        ORDER BY position
    """).df()
    history = _side_records(rows, "_a")
    later = _side_records(rows, "_b")
    return DemographicRecords(history, later, rows["n_changed_fields"].to_numpy())
