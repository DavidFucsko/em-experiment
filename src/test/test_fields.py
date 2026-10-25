from dataclasses import dataclass

import numpy as np

from src.utils.fields import ComparisonVector, CompositeField


@dataclass(frozen=True)
class Fixed:
    """A field whose level of pair k is levels[k]."""
    name: str
    level_names: tuple[str, ...]
    values: tuple[int, ...]

    def levels(self, i, j):
        return np.array(self.values)[i]


def test_composite_levels_give_every_combination_its_own_level_with_full_agreement_highest():
    levels = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [3, 3, 3, 3]])   # pairs x parts
    parts = [Fixed(f"f{k}", ("a", "b", "c", "d"), tuple(column)) for k, column in enumerate(levels.T)]
    pairs = np.arange(len(levels))
    assert CompositeField("all", parts).levels(pairs, pairs).tolist() == [0, 1, 64, 255]


def test_composite_names_follow_composite_levels_with_the_first_part_most_significant():
    composite = CompositeField("clinical", [Fixed("dx", ("missing", "low", "high"), (2,)),
                                            Fixed("px", ("missing", "equal"), (0,))])
    level = composite.levels(np.array([0]), np.array([0]))[0]
    assert len(composite.level_names) == 6 and composite.level_names[level] == "dx high / px missing"


def test_the_comparison_vector_stacks_its_fields_in_order():
    vector = ComparisonVector([Fixed("sex", ("missing", "different", "equal"), (2, 1)),
                               Fixed("dx", ("missing", "low", "high"), (0, 2))])
    assert vector.names == ["sex", "dx"] and vector.n_levels == [3, 3]
    assert vector.levels(np.array([0, 1]), np.array([0, 1])).tolist() == [[2, 0], [1, 2]]


@dataclass(frozen=True)
class Same:
    """Level 1 where history and later record are the same patient, else 0."""
    name: str = "same"
    level_names: tuple[str, ...] = ("different", "same")

    def levels(self, i, j):
        return (i == j).astype(int)


def test_every_pair_of_the_two_sides_is_counted_once(monkeypatch):
    monkeypatch.setattr("src.utils.fields.BLOCK", 7)   # several blocks of whole history records
    vectors, counts = ComparisonVector([Same()]).count_comparison_vectors(5, 4)
    assert dict(zip(vectors[:, 0].tolist(), counts.tolist())) == {0: 16, 1: 4}
