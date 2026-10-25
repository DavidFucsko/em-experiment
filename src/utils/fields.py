"""The comparison vector (Fellegi & Sunter's gamma): the level of a pair on every field. Level 0 is always
"missing on either side" and a higher level is a more similar pair."""
from dataclasses import dataclass
from itertools import product
from typing import Protocol

import numpy as np

BLOCK = 6_000_000      # pairs compared at a time when the comparison vectors of all pairs are counted


class Field(Protocol):
    name: str
    level_names: tuple[str, ...]

    def levels(self, i: np.ndarray, j: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True, eq=False)
class CompositeField:
    """One field whose levels are all combinations of its parts' levels (Tromp et al. 2008), so
    the parts may depend on each other. Full agreement on every part is the highest level."""
    name: str
    parts: list[Field]

    @property
    def level_names(self) -> tuple[str, ...]:
        part_level_names = [part.level_names for part in self.parts]
        names = []
        for combination in product(*part_level_names):
            labels = []
            for part, level in zip(self.parts, combination):
                labels.append(f"{part.name} {level}")
            names.append(" / ".join(labels))
        return tuple(names)

    def levels(self, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        part_levels = [part.levels(i, j) for part in self.parts]
        part_sizes = [len(part.level_names) for part in self.parts]
        return np.ravel_multi_index(part_levels, part_sizes)  # the first part is the most significant


@dataclass(frozen=True, eq=False)
class ComparisonVector:
    fields: list[Field]

    @property
    def names(self) -> list[str]:
        return [field.name for field in self.fields]

    @property
    def level_names(self) -> dict[str, tuple[str, ...]]:
        names = {}
        for field in self.fields:
            names[field.name] = field.level_names
        return names

    @property
    def n_levels(self) -> list[int]:
        return [len(field.level_names) for field in self.fields]

    def levels(self, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        """Level matrix, pairs x fields."""
        return np.column_stack([field.levels(i, j) for field in self.fields])

    def count_comparison_vectors(self, n_history: int, n_later: int) -> tuple[np.ndarray, np.ndarray]:
        """(vectors, counts): how many of all n_history x n_later pairs have each comparison vector. EM needs only these
        counts. The pairs are compared BLOCK at a time, so they are never all in memory."""
        n_levels = self.n_levels
        counts = np.zeros(np.prod(n_levels), np.int64)
        rows_per_block = max(1, BLOCK // n_later)
        for start in range(0, n_history, rows_per_block):
            history = np.arange(start, min(start + rows_per_block, n_history))
            i = np.repeat(history, n_later)
            j = np.tile(np.arange(n_later), len(history))
            vector_ids = np.ravel_multi_index(self.levels(i, j).T, n_levels)
            counts += np.bincount(vector_ids, minlength=len(counts))
        present = np.flatnonzero(counts)
        vectors = np.column_stack(np.unravel_index(present, n_levels))
        return vectors, counts[present]
