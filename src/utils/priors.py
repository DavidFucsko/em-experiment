"""The prior of m and u: pseudo-counts added to EM's level counts, so a level unseen in a class does not get a
probability of 0 (Winkler 2002, eq. 6)."""
from dataclasses import dataclass

import numpy as np

# winkler  WINKLER_DELTA added to every level, for m and u alike (Winkler 2002, after Larsen 1996); the default
# none     no pseudo-counts: maximum likelihood
# flat     one pseudo-count per level: Laplace smoothing (Nigam et al. 2000)
# centred  one pseudo-count per field, spread over its levels by their share among all pairs; trial runs only
PRIORS = ("winkler", "none", "flat", "centred")
DEFAULT_PRIOR = "winkler"
WINKLER_DELTA = 0.001      # the largest value Winkler (2002) uses


def _pseudo_counts(name: str, shares: np.ndarray) -> np.ndarray:
    if name == "winkler":
        return np.full_like(shares, WINKLER_DELTA)
    if name == "none":
        return np.zeros_like(shares)
    if name == "flat":
        return np.ones_like(shares)
    if name == "centred":
        return shares
    raise ValueError(f"unknown prior '{name}', expected one of {PRIORS}")


@dataclass(frozen=True)
class Prior:
    name: str
    pseudo_counts: list[np.ndarray]     # per field: added to the count of each level

    @classmethod
    def of(cls, name: str, level_shares: list[np.ndarray]) -> "Prior":
        """level_shares: per field the share of all pairs at each level (only the centred prior uses it)."""
        return cls(name, [_pseudo_counts(name, shares) for shares in level_shares])

    def probabilities(self, level_counts: list[np.ndarray]) -> list[np.ndarray]:
        """Per field P(level): the level counts with the pseudo-counts added, as shares."""
        probabilities = []
        for counts, pseudo in zip(level_counts, self.pseudo_counts):
            probabilities.append((counts + pseudo) / (counts.sum() + pseudo.sum()))
        return probabilities
