import numpy as np

from src.utils.fellegi_sunter import fit as fit_counted

N_LEVELS = [3, 2, 3]
M = [np.array([0.05, 0.15, 0.80]), np.array([0.10, 0.90]), np.array([0.02, 0.08, 0.90])]
U = [np.array([0.70, 0.25, 0.05]), np.array([0.85, 0.15]), np.array([0.80, 0.18, 0.02])]
MATCH_SHARE = 0.05


def fit(levels, n_levels):
    return fit_counted(*np.unique(levels, axis=0, return_counts=True), n_levels)


def draw(rng, n_pairs, match_share=MATCH_SHARE):
    """Level matrix of pairs drawn from the model, and which of them are matches."""
    is_match = rng.random(n_pairs) < match_share
    levels = np.column_stack([np.where(is_match, rng.choice(len(m), n_pairs, p=m), rng.choice(len(u), n_pairs, p=u))
                              for m, u in zip(M, U)])
    return levels, is_match


def test_em_recovers_m_u_and_the_match_share_from_unlabelled_pairs():
    levels, _ = draw(np.random.default_rng(7), 100_000)
    model = fit(levels, N_LEVELS)
    assert abs(model.match_share - MATCH_SHARE) < 0.01
    assert all(np.allclose(fitted, true, atol=0.03) for fitted, true in zip(model.m, M))
    assert all(np.allclose(fitted, true, atol=0.01) for fitted, true in zip(model.u, U))


def test_probability_ranks_true_pairs_above_the_rest_and_rises_with_agreement():
    levels, is_match = draw(np.random.default_rng(9), 60_000)
    model = fit(levels, N_LEVELS)
    p = model.match_probability(levels)
    assert p[is_match].mean() > 0.8 > 0.1 > p[~is_match].mean()
    assert model.match_probability(np.array([[2, 1, 2]]))[0] > 0.99 > 0.01 > model.match_probability(np.array([[0, 0, 0]]))[0]


def test_a_small_match_share_as_among_all_pairs_is_recovered():
    rng = np.random.default_rng(11)
    is_match = rng.random(2_000_000) < 0.001
    fields = M * 2, U * 2                              # six independent fields, so the classes are well identified
    levels = np.column_stack([np.where(is_match, rng.choice(len(m), len(is_match), p=m), rng.choice(len(u), len(is_match), p=u))
                              for m, u in zip(*fields)])
    model = fit(levels, N_LEVELS * 2)
    assert abs(model.match_share - is_match.mean()) < 0.0001
    assert all(np.allclose(fitted, true, atol=0.03) for fitted, true in zip(model.m, fields[0]))


def test_the_match_share_never_exceeds_its_bound():
    levels, _ = draw(np.random.default_rng(12), 60_000)
    assert fit(levels, N_LEVELS).match_share > 0.04
    bounded = fit_counted(*np.unique(levels, axis=0, return_counts=True), N_LEVELS, max_share=0.01)
    assert bounded.match_share == 0.01


def test_a_prior_bounds_the_weight_of_a_level_never_seen_among_the_matches():
    rng = np.random.default_rng(13)
    levels, is_match = draw(rng, 200_000, match_share=0.01)
    levels[~is_match & (rng.random(len(levels)) < 0.2), 0] = 0     # level 0 of the first field: only among non-matches
    levels[is_match, 0] = np.where(levels[is_match, 0] == 0, 1, levels[is_match, 0])
    weight = lambda model: np.log2(model.m[0][0] + 1e-12) - np.log2(model.u[0][0] + 1e-12)
    vectors, counts = np.unique(levels, axis=0, return_counts=True)
    plain, flat, winkler, centred = (fit_counted(vectors, counts, N_LEVELS, prior=p) for p in ("none", "flat", "winkler", "centred"))
    assert weight(plain) < -25 < weight(winkler) < -15 < weight(flat) < 0 and -15 < weight(centred) < 0
    assert abs(flat.match_share - 0.01) < 0.002 and abs(centred.match_share - 0.01) < 0.002
