"""The demographic fields of the comparison vector D. Every field has its own comparison, a scenario picks
the fields."""
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler, Levenshtein

from src.utils.comparison import closeness_levels, exact_levels, text_levels

AGE_BAND = 5  # years of birth per band, with fixed boundaries so the band is the same in both records

NAME_FIELDS = ("first_name", "middle_name", "last_name")
DERIVED_FIELDS = ("first_initial", "last_initial", "zip3", "age_band")
EXACT_FIELDS = ("sex", "city", "zip_code", "phone", "race") + DERIVED_FIELDS

NAME_CUTS = (0.85, 0.94)       # Jaro-Winkler: below 0.85 / from 0.85 / from 0.94
ADDRESS_CUTS = (0.7, 0.9)      # normalised Levenshtein similarity
BIRTH_YEAR_CUTS = (-1, 0)      # -|difference|: more than a year apart / one year / same


def add_derived_fields(frame: pd.DataFrame) -> pd.DataFrame:
    """Coarser versions of the fields, for the scenarios with fewer identifiers and for the blocking."""
    return frame.assign(
        first_initial=frame["first_name"].str[:1],
        last_initial=frame["last_name"].str[:1],
        zip3=frame["zip_code"].str[:3],
        age_band=frame["birth_year"] // AGE_BAND * AGE_BAND,
    )


@dataclass(frozen=True)
class Comparison:
    compare: Callable[[pd.Series, pd.Series], np.ndarray]
    level_names: tuple[str, ...]  # missing first, then one more than the cut points


def _cut_names(cuts: tuple[float, float]) -> tuple[str, ...]:
    low, high = cuts
    return "missing", f"< {low}", f"{low}-{high}", f">= {high}"


def _compare_names(left: pd.Series, right: pd.Series) -> np.ndarray:
    return text_levels(left, right, scorer=JaroWinkler.similarity, cuts=NAME_CUTS)


def _compare_addresses(left: pd.Series, right: pd.Series) -> np.ndarray:
    return text_levels(left, right, scorer=Levenshtein.normalized_similarity, cuts=ADDRESS_CUTS)


def _compare_birth_years(left: pd.Series, right: pd.Series) -> np.ndarray:
    return closeness_levels(left, right, cuts=BIRTH_YEAR_CUTS)


def _build_comparisons() -> dict[str, Comparison]:
    comparisons = {}
    for field in NAME_FIELDS:
        comparisons[field] = Comparison(_compare_names, _cut_names(NAME_CUTS))
    comparisons["address"] = Comparison(_compare_addresses, _cut_names(ADDRESS_CUTS))
    comparisons["birth_year"] = Comparison(_compare_birth_years, ("missing", "> 1 year apart", "1 year apart", "same"))
    for field in EXACT_FIELDS:
        comparisons[field] = Comparison(exact_levels, ("missing", "different", "equal"))
    return comparisons


COMPARISONS = _build_comparisons()
DEMOGRAPHIC_FIELDS = tuple(COMPARISONS)


@dataclass(frozen=True, eq=False)
class DemographicField:
    """One column of the demographic records; row k of history and of later is patient k."""
    name: str
    history: pd.DataFrame
    later: pd.DataFrame

    @property
    def level_names(self) -> tuple[str, ...]:
        return COMPARISONS[self.name].level_names

    def levels(self, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        left = self.history[self.name].iloc[i].reset_index(drop=True)
        right = self.later[self.name].iloc[j].reset_index(drop=True)
        return COMPARISONS[self.name].compare(left, right)


def demographic_fields(history: pd.DataFrame, later: pd.DataFrame,
                       names: tuple[str, ...] = DEMOGRAPHIC_FIELDS) -> list[DemographicField]:
    return [DemographicField(name, history, later) for name in names]
