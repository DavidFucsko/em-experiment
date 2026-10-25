"""Stored linkage runs, per cohort, model, scenario and pool, so any run can be compared or explained without refitting.
Files: <run>.scores.parquet (pairs, levels, probability), <run>.scores.json (level names, m, u) and <run>.links.parquet."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DATA, RUN_LABEL
from src.utils.fellegi_sunter import FellegiSunter
from src.utils.fields import ComparisonVector

RUNS = DATA / f"matches_{RUN_LABEL}"
PAIR = ["history_subject_id", "later_subject_id"]


def run_name(model: str, scenario: str, pool: str) -> str:
    return f"{model}_{scenario}_{pool}"


@dataclass(frozen=True)
class Scores:
    """pairs: history_subject_id, later_subject_id, one level column per field of the comparison vector, probability."""
    pairs: pd.DataFrame
    level_names: dict[str, list[str]]   # field -> the names of its levels, lowest first
    model: FellegiSunter

    @classmethod
    def of(cls, model: FellegiSunter, vector: ComparisonVector, pairs: tuple[np.ndarray, np.ndarray],
           levels: np.ndarray, subject_ids) -> "Scores":
        """pairs: (i, j) positions in subject_ids; levels: vector.levels(i, j)."""
        i, j = pairs
        ids = np.asarray(subject_ids)
        columns = {"history_subject_id": ids[i], "later_subject_id": ids[j]}
        for k, field in enumerate(vector.names):
            columns[field] = levels[:, k].astype(np.int16)
        columns["probability"] = model.match_probability(levels)

        level_names = {}
        for field, names in vector.level_names.items():
            level_names[field] = list(names)
        return cls(pd.DataFrame(columns), level_names, model)

    @property
    def fields(self) -> list[str]:
        return list(self.level_names)

    def levels(self) -> np.ndarray:
        """Pairs x fields, the input of the model."""
        return self.pairs[self.fields].to_numpy()

    def positions(self, subject_ids) -> tuple[np.ndarray, np.ndarray]:
        """(i, j): each pair's history and later record as a position in subject_ids."""
        index = pd.Index(subject_ids)
        return index.get_indexer(self.pairs["history_subject_id"]), index.get_indexer(self.pairs["later_subject_id"])

    def save(self, name: str, directory: Path = RUNS) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        scores_file = directory / f"{name}.scores.parquet"
        self.pairs.to_parquet(scores_file, index=False)
        meta = {
            "level_names": self.level_names,
            "m": [m.tolist() for m in self.model.m],
            "u": [u.tolist() for u in self.model.u],
            "match_share": self.model.match_share,
        }
        (directory / f"{name}.scores.json").write_text(json.dumps(meta, indent=1))
        return scores_file

    @classmethod
    def load(cls, name: str, directory: Path = RUNS) -> "Scores":
        meta = json.loads((directory / f"{name}.scores.json").read_text())
        m = [np.array(values) for values in meta["m"]]
        u = [np.array(values) for values in meta["u"]]
        model = FellegiSunter(m, u, meta["match_share"])
        return cls(pd.read_parquet(directory / f"{name}.scores.parquet"), meta["level_names"], model)


def save_links(links: pd.DataFrame, name: str, directory: Path = RUNS) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    links_file = directory / f"{name}.links.parquet"
    links.to_parquet(links_file, index=False)
    return links_file


def load_links(name: str, directory: Path = RUNS) -> pd.DataFrame:
    return pd.read_parquet(directory / f"{name}.links.parquet")
