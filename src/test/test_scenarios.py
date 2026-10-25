import numpy as np
import pandas as pd

from src.experiments.scenarios import SCENARIOS
from src.utils import fellegi_sunter
from src.utils.demographic_levels import COMPARISONS, add_derived_fields, demographic_fields
from src.utils.experiment_data import DemographicRecords
from src.utils.fields import ComparisonVector
from src.utils.runs import Scores


def synthetic_records(n=400, seed=0) -> DemographicRecords:
    rng = np.random.default_rng(seed)
    history = pd.DataFrame({
        "first_name": rng.choice([f"FIRST{k}" for k in range(60)], n), "middle_name": rng.choice([f"MID{k}" for k in range(40)], n),
        "last_name": rng.choice([f"LAST{k}" for k in range(150)], n), "address": [f"{k} MAIN ST" for k in rng.integers(1, 9999, n)],
        "birth_year": rng.integers(1930, 2000, n), "sex": rng.choice(["F", "M"], n), "city": rng.choice([f"CITY{k}" for k in range(25)], n),
        "zip_code": rng.choice([f"27{k:03d}" for k in range(40)], n), "phone": [f"{k:07d}" for k in rng.integers(0, 9_999_999, n)],
    })
    later = history.copy()
    for column, new_value in {"address": "1 NEW RD", "zip_code": "27999", "city": "NEWTOWN", "phone": "9999999"}.items():
        later.loc[rng.random(n) < 0.15, column] = new_value  # fields change independently: what the model assumes
    return DemographicRecords(add_derived_fields(history), add_derived_fields(later), np.zeros(n, int))


def test_the_unsupervised_model_finds_nearly_all_true_pairs_with_high_probability_and_few_false_ones():
    records, scenario = synthetic_records(), SCENARIOS["full"]
    ids = np.arange(len(records.history))
    vector = ComparisonVector(demographic_fields(records.history, records.later, scenario.fields))
    candidates = scenario.candidates(records)
    model = fellegi_sunter.fit(*vector.count_comparison_vectors(len(ids), len(ids)), vector.n_levels)
    levels = vector.levels(*candidates)
    scores = Scores.of(model, vector, candidates, levels, ids)
    (i, j), p = scores.positions(ids), scores.pairs.probability.to_numpy()
    true = i == j
    assert true.sum() == len(records.history)
    assert (p[true] >= 0.5).mean() > 0.9
    assert (p[~true] >= 0.5).sum() <= 0.01 * (p[true] >= 0.5).sum()


def test_the_bound_allows_one_partner_per_patient_of_the_smaller_side():
    from src.experiments.scenarios import max_match_share
    assert max_match_share(4, 10) == 4 / 40 and max_match_share(10, 10) == 1 / 10


def test_every_scenario_compares_and_blocks_on_fields_the_records_have():
    columns = set(synthetic_records().history.columns) | {"race"}
    for scenario in SCENARIOS.values():
        assert set(scenario.fields) <= set(COMPARISONS) and {field for rule in scenario.blocking for field in rule} <= columns
