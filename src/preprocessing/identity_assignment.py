"""Gives every MIMIC patient a real voter identity: the same person's fields in two snapshots 10 years apart. V-cond
matches on the patient's sex and age, V-indep shuffles them first; the "hard" pool draws only voters with 3+ changes."""
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import COHORT, SEED, SPLIT, VOTER
from src.preprocessing.voter_candidates import load_candidates
from src.preprocessing.voter_pairs import PAIRS_FILE

MAX_AGE_GAP = 3                                   # years between a patient's anchor_age and the voter's age
VARIANTS = {"cond": False, "indep": True}         # variant name -> shuffle (sex, age) before matching
POOLS = {"natural": 0, "hard": 3}                 # pool name -> minimum number of changed fields of the voters drawn


def assignment_file(pool: str, variant: str, directory: Path = VOTER) -> Path:
    """The voters assigned to the current cohort's patients."""
    return directory / f"assignment_{COHORT}_{pool}_{variant}.parquet"


def _take_nearest(available: dict, sex: str, age: int) -> tuple[int, int]:
    """(row of the nearest-age unused voter of this sex, age gap)."""
    for gap in range(MAX_AGE_GAP + 1):
        candidate_ages = [age] if gap == 0 else [age - gap, age + gap]
        for candidate_age in candidate_ages:
            voters = available.get((sex, candidate_age))
            if voters:
                return voters.pop(), gap
    raise LookupError(f"no unused voter within {MAX_AGE_GAP} years for sex={sex}, age={age}")


def assign_identities(patients: pd.DataFrame, candidates: pd.DataFrame, independent: bool,
                      seed: int = SEED) -> pd.DataFrame:
    """One row per patient: subject_id, ncid, age_gap (years between the patient's age and the voter's)."""
    rng = np.random.default_rng(seed)
    patients = patients.iloc[rng.permutation(len(patients))].reset_index(drop=True)  # random serving order
    matched_on = patients[["gender", "anchor_age"]]
    if independent:
        matched_on = matched_on.iloc[rng.permutation(len(patients))]

    candidates = candidates.iloc[rng.permutation(len(candidates))].reset_index(drop=True)
    available = defaultdict(list)  # (sex, age) -> rows of the unused voters
    for row, (sex, age) in enumerate(zip(candidates["sex"], candidates["age"])):
        available[(sex, age)].append(row)

    rows = []
    gaps = []
    for sex, age in zip(matched_on["gender"], matched_on["anchor_age"]):
        row, gap = _take_nearest(available, sex, age)
        rows.append(row)
        gaps.append(gap)
    result = pd.DataFrame({"subject_id": patients["subject_id"], "ncid": candidates["ncid"].iloc[rows].to_numpy(),
                           "age_gap": gaps})
    return result.sort_values("subject_id").reset_index(drop=True)


def main() -> None:
    patients = pd.read_parquet(SPLIT / "patients.parquet")
    for pool, min_changes in POOLS.items():
        candidates = load_candidates(PAIRS_FILE, min_changes)
        for variant, independent in VARIANTS.items():
            assignment = assign_identities(patients, candidates, independent)
            assignment.to_parquet(assignment_file(pool, variant), index=False)
            print(f"cohort {COHORT}, {pool} V-{variant}: {len(assignment):,} patients, "
                  f"age gap mean {assignment['age_gap'].mean():.3f} max {assignment['age_gap'].max()}")


if __name__ == "__main__":
    main()
