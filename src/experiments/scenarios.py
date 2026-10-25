"""The identifier-availability scenarios: which demographic fields are available and how the candidate
pairs are formed from them."""
from dataclasses import dataclass

import numpy as np

from src.utils.blocking import blocking_pairs
from src.utils.experiment_data import DemographicRecords


@dataclass(frozen=True)
class Scenario:
    fields: tuple[str, ...]
    blocking: list[list[str]]

    def candidates(self, records: DemographicRecords) -> tuple[np.ndarray, np.ndarray]:
        """(i, j) of the candidate pairs: the union of the blocking rules."""
        return blocking_pairs(records.history, records.later, self.blocking)


def max_match_share(n_history: int, n_later: int) -> float:
    """The largest possible share of matches among all pairs if no patient has two partners (Jaro 1989)."""
    return min(n_history, n_later) / (n_history * n_later)


# Fellegi-Sunter counts every field as independent, so no scenario has both city and zip_code (two strangers in one ZIP
# almost always share the city, which would double-count the location). Ordered from the most to the least identifying.
SCENARIOS = {
    "full": Scenario(("first_name", "last_name", "birth_year", "address", "phone"),  # MedLink's five identifiers
                     [["first_name"], ["last_name"], ["phone"], ["birth_year", "last_initial"]]),
    "core_id": Scenario(("first_name", "last_name", "birth_year", "sex"),  # a master patient index, no contact data
                        [["first_name"], ["last_name"], ["birth_year", "last_initial"]]),
    "initials": Scenario(("first_initial", "last_initial", "birth_year", "sex", "zip_code"),  # registries, drug safety reports
                         [["first_initial", "last_initial"], ["birth_year", "sex"]]),
    "no_names": Scenario(("birth_year", "sex", "zip_code"), [["birth_year", "sex"]]),  # HIPAA Limited Data Set
    "safe_harbor": Scenario(("birth_year", "sex", "zip3"), [["birth_year", "sex"]]),  # HIPAA Safe Harbor
    # fully de-identified, no geography (like MIMIC itself); race is compared, not blocked on: it is missing or differs
    # between the two admissions of 4 % of the patients
    "sex_race_age": Scenario(("sex", "race", "age_band"), [["sex", "age_band"]]),
}
