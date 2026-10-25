"""Candidate voters for the identity assignment: a random subset of each (sex, age) cell of the linked persons, so the
real mix of changes in the cell is kept and memory stays small."""
import duckdb
import pandas as pd

from src.config import SEED, VOTER_SNAPSHOTS

MIMIC_MAX_AGE = 91                                # MIMIC reports every patient over 89 as 91
REFERENCE_YEAR = int(VOTER_SNAPSHOTS[0][:4])      # voter age at the older snapshot, closer to MIMIC's 2008-2022
CELL_CAP = 2_000                                  # voters kept per (sex, age) cell, far above a cell's demand


def load_candidates(pairs_file, min_changes: int = 0, cell_cap: int = CELL_CAP,
                    seed: int = SEED) -> pd.DataFrame:
    """ncid, sex, age, n_changed_fields of the voters with at least min_changes changed fields and the same F/M sex in
    both snapshots. Sorted by ncid, so the seeded shuffle of the assignment does not depend on the query plan."""
    return duckdb.sql(f"""
        SELECT ncid, sex_a AS sex, least({REFERENCE_YEAR} - birth_year_a, {MIMIC_MAX_AGE}) AS age, n_changed_fields
        FROM '{pairs_file.as_posix()}' WHERE sex_a = sex_b AND sex_a IN ('F', 'M') AND n_changed_fields >= {min_changes}
        QUALIFY row_number() OVER (PARTITION BY sex, age ORDER BY hash(ncid, {seed})) <= {cell_cap}
        ORDER BY ncid
    """).df()
