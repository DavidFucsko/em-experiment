import pandas as pd
import pytest

from src.preprocessing.identity_assignment import MAX_AGE_GAP, assign_identities

N_PATIENTS = 20


def candidates_frame(ages=range(20, 61), voters_per_cell=3):
    rows = [(f"{sex}_{age}_{i}", sex, age, i % 2) for sex in "FM" for age in ages for i in range(voters_per_cell)]
    return pd.DataFrame(rows, columns=["ncid", "sex", "age", "n_changed_fields"])


@pytest.fixture
def patients():
    return pd.DataFrame({"subject_id": range(1, N_PATIENTS + 1), "gender": list("FM") * (N_PATIENTS // 2),
                         "anchor_age": range(25, 25 + N_PATIENTS)})


def merged(assignment, patients, candidates):
    voters = candidates.set_index("ncid")
    joined = assignment.merge(patients, on="subject_id")
    return joined.assign(voter_sex=joined.ncid.map(voters.sex), voter_age=joined.ncid.map(voters.age))


def test_cond_assignment_matches_sex_and_age_and_uses_each_voter_once(patients):
    candidates = candidates_frame()
    assignment = assign_identities(patients, candidates, independent=False)
    joined = merged(assignment, patients, candidates)
    assert len(assignment) == N_PATIENTS and assignment.ncid.is_unique
    assert (joined.voter_sex == joined.gender).all()
    assert (joined.voter_age - joined.anchor_age).abs().max() <= MAX_AGE_GAP
    assert assignment.age_gap.max() <= MAX_AGE_GAP


def test_the_nearest_age_is_used_when_the_exact_age_has_no_voter_left(patients):
    sparse = candidates_frame(ages=[24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46], voters_per_cell=3)
    assignment = assign_identities(patients, sparse, independent=False)
    assert assignment.age_gap.max() <= 1 and (assignment.age_gap > 0).any()


def test_indep_assignment_keeps_the_demographic_mix_but_breaks_the_link_to_each_patient(patients):
    candidates = candidates_frame()
    joined = merged(assign_identities(patients, candidates, independent=True), patients, candidates)
    assert sorted(zip(joined.voter_sex, joined.voter_age)) == sorted(zip(patients.gender, patients.anchor_age))
    assert (joined.voter_age != joined.anchor_age).any()


def test_assignment_is_deterministic_for_a_seed(patients):
    candidates = candidates_frame()
    first = assign_identities(patients, candidates, independent=False, seed=1)
    pd.testing.assert_frame_equal(first, assign_identities(patients, candidates, independent=False, seed=1))
    assert not first.equals(assign_identities(patients, candidates, independent=False, seed=2))


def test_missing_voters_raise_instead_of_silently_dropping_patients(patients):
    with pytest.raises(LookupError):
        assign_identities(patients, candidates_frame(ages=[60], voters_per_cell=1), independent=False)
