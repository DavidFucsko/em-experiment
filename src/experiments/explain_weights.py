"""What a stored run weighs and why it links: the weight log2(m / u) of every level, the mean
contribution of each field, example waterfalls and, for a clinical model, the decisions it changes against D. Only
levels and weights are shown, never the content of a record (PhysioNet DUA)."""
import argparse

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import ROOT, RUN_LABEL, SEED
from src.experiments.scenarios import SCENARIOS
from src.preprocessing.identity_assignment import POOLS
from src.utils.evaluation import AUTOMATIC, ZONES, review_zones
from src.utils.experiment_data import load_patient_ids
from src.utils.fellegi_sunter import EPSILON, FellegiSunter
from src.utils.priors import DEFAULT_PRIOR, PRIORS
from src.utils.runs import PAIR, Scores, load_links, run_name

EXAMPLES = 3           # pairs shown per outcome
SHOWN_LEVELS = 10      # levels of a large field shown on the console, by m and by u


def weights(model: FellegiSunter) -> list[np.ndarray]:
    """Per field the match weight log2(m / u) of every level (with the same EPSILON as the model's probabilities)."""
    return [np.log2(m + EPSILON) - np.log2(u + EPSILON) for m, u in zip(model.m, model.u)]


def prior_weight(model: FellegiSunter) -> float:
    share = np.clip(model.match_share, EPSILON, 1 - EPSILON)
    return float(np.log2(share / (1 - share)))


def contributions(model: FellegiSunter, levels: np.ndarray) -> np.ndarray:
    """Pairs x fields: the weight of each pair's level in each field. prior_weight + the row sum = the pair's log2 odds."""
    columns = []
    for field, weight in enumerate(weights(model)):
        columns.append(weight[levels[:, field]])
    return np.column_stack(columns)


def outcomes(i: np.ndarray, j: np.ndarray, linked: np.ndarray) -> dict[str, np.ndarray]:
    """Outcome -> the candidate pairs with it after the assignment (a true pair outside the candidates is in none)."""
    true = i == j
    return {
        "true positive": linked & true,
        "false positive": linked & ~true,
        "false negative": ~linked & true,
    }


def level_table(fields: list[str], names: list[list[str]], model: FellegiSunter, levels: np.ndarray,
                true: np.ndarray) -> pd.DataFrame:
    """One row per level of every field: m, u, weight, and how many candidate pairs and true pairs are at it."""
    field_weights = weights(model)
    rows = []
    for field, name in enumerate(fields):
        m = model.m[field]
        u = model.u[field]
        pairs = np.bincount(levels[:, field], minlength=len(m))
        true_pairs = np.bincount(levels[true, field], minlength=len(m))
        for level in range(len(m)):
            rows.append({"field": name, "level": level, "label": names[field][level], "m": m[level], "u": u[level],
                         "weight": field_weights[field][level], "pairs": pairs[level], "true pairs": true_pairs[level]})
    return pd.DataFrame(rows)


def changes(true: np.ndarray, linked: np.ndarray, base_linked: np.ndarray) -> dict[str, np.ndarray]:
    """Change -> the candidate pairs whose decision after the assignment differs from the base model's in that way."""
    return {
        "true pair gained": true & linked & ~base_linked,
        "true pair lost": true & ~linked & base_linked,
        "false link added": ~true & linked & ~base_linked,
        "false link removed": ~true & ~linked & base_linked,
    }


def change_summary(groups: dict[str, np.ndarray]) -> str:
    """How many decisions changed in the right direction (true pair gained, false link removed) and in the wrong one
    (true pair lost, false link added), and the ratio of the two."""
    count = {}
    for kind, mask in groups.items():
        count[kind] = int(mask.sum())
    right = count["true pair gained"] + count["false link removed"]
    wrong = count["true pair lost"] + count["false link added"]
    ratio = f"{right / wrong:.2f}" if wrong else "inf"
    return (f"{right + wrong:,} decisions changed: {right:,} right ({count['true pair gained']:,} true pairs gained, "
            f"{count['false link removed']:,} false links removed), {wrong:,} wrong ({count['true pair lost']:,} true "
            f"pairs lost, {count['false link added']:,} false links added); right / wrong = {ratio}")


def zone_moves(true: np.ndarray, zones: np.ndarray, base_zones: np.ndarray) -> pd.DataFrame:
    """Rows: the base model's zone, columns: this model's zone; each cell "pairs (true pairs)"."""
    order = list(range(len(ZONES) - 1, -1, -1))  # automatic, review, rejected
    cells = []
    for base_zone in order:
        row = []
        for zone in order:
            moved = (base_zones == base_zone) & (zones == zone)
            row.append(f"{moved.sum():,} ({(true & moved).sum():,})")
        cells.append(row)
    return pd.DataFrame(cells, index=[f"D {ZONES[zone]}" for zone in order], columns=[ZONES[zone] for zone in order])


def waterfall(axis, prior: float, contribution: np.ndarray, labels: list[str], title: str) -> None:
    """Splink's waterfall: the prior, then each field's weight added, then the total log2 odds."""
    steps = np.concatenate(([prior], contribution))
    starts = np.concatenate(([0], np.cumsum(steps)[:-1]))
    total = steps.sum()
    colors = ["#999999"] + ["#2a9d8f" if weight >= 0 else "#e76f51" for weight in contribution]
    axis.bar(range(len(steps)), steps, bottom=starts, color=colors)
    axis.bar(len(steps), total, color="#264653")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(range(len(steps) + 1), ["prior"] + labels + ["total"], fontsize=7)
    axis.set_ylabel("log2 odds")
    axis.set_title(f"{title}: p = {1 / (1 + 2 ** -total):.3f}", fontsize=9)


def _print_levels(frame: pd.DataFrame) -> None:
    print(frame.drop(columns="field").to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def _prior_suffix(model_name: str) -> str:
    """The run-name suffix of a non-default prior of m and u, e.g. "-flat", or "" for the default."""
    for name in PRIORS:
        if name != DEFAULT_PRIOR and model_name.endswith(f"-{name}"):
            return f"-{name}"
    return ""


def explain(model_name: str, scenario_name: str, pool: str) -> None:
    run = run_name(model_name, scenario_name, pool)
    scores = Scores.load(run)
    links = load_links(run)
    patients = load_patient_ids()
    n = len(patients)
    i, j = scores.positions(patients)
    levels = scores.levels()
    fields = scores.fields
    model = scores.model
    names = list(scores.level_names.values())
    contribution = contributions(model, levels)
    prior = prior_weight(model)
    p = 1 / (1 + 2 ** -(prior + contribution.sum(axis=1)))
    kinds = outcomes(i, j, links["assigned"].to_numpy())
    stem = f"{RUN_LABEL}_{scenario_name}_{pool}_{model_name}"
    print(f"\ncohort {RUN_LABEL}, {pool} pool, scenario {scenario_name}, model {model_name} | {len(i):,} candidate pairs, "
          f"prior log2 odds {prior:+.2f}")

    # 1. m, u and weight of every level; a field with many levels shows only the largest m and u on the console
    table = level_table(fields, names, model, levels, i == j)
    table.to_csv(ROOT / "results" / f"weights_{stem}.csv", index=False)
    for field in fields:
        rows = table[table["field"] == field]
        if len(rows) > 2 * SHOWN_LEVELS:
            print(f"\n  {field} ({len(rows)} levels, all in results/weights_{stem}.csv): the {SHOWN_LEVELS} with the largest m")
            _print_levels(rows.nlargest(SHOWN_LEVELS, "m"))
            print(f"  and the {SHOWN_LEVELS} with the largest u")
            _print_levels(rows.nlargest(SHOWN_LEVELS, "u"))
        else:
            print(f"\n  {field}")
            _print_levels(rows)

    # 2. mean contribution of each field per outcome
    print("\n  mean contribution (log2) after one-to-one assignment")
    columns = {}
    for kind, mask in kinds.items():
        if mask.any():
            log2_odds = prior + contribution[mask].sum(axis=1).mean()
            columns[kind] = [mask.sum()] + list(contribution[mask].mean(axis=0)) + [log2_odds]
    means = pd.DataFrame(columns, index=["pairs"] + fields + ["log2 odds"]).T
    print(means.to_string(formatters={"pairs": "{:,.0f}".format}, float_format=lambda x: f"{x:+.2f}"))
    print(f"  true pairs outside the candidates: {n - (i == j).sum():,}")

    # 3. example pairs, field by field
    rng = np.random.default_rng(SEED)

    def describe(pair: int) -> str:
        parts = []
        for f, field in enumerate(fields):
            parts.append(f"{field} {names[f][levels[pair, f]]} {contribution[pair, f]:+.2f}")
        return " | ".join(parts)

    def bar_labels(pair: int) -> list[str]:
        labels = []
        for f, field in enumerate(fields):
            labels.append(f"{field}\n{names[f][levels[pair, f]]}".replace(" / ", "\n"))
        return labels

    def examples(groups: dict[str, np.ndarray], file: str, base_p: np.ndarray | None = None) -> None:
        """A few random pairs of each group on the console, the first of each drawn as a waterfall into `file`."""
        figure, axes = plt.subplots(len(groups), 1, figsize=(8, 3 * len(groups)))
        for axis, (kind, mask) in zip(axes, groups.items()):
            chosen = []
            if mask.any():
                chosen = rng.choice(np.flatnonzero(mask), min(EXAMPLES, mask.sum()), replace=False)
            print(f"\n  {kind} examples" + (f" ({mask.sum():,} pairs)" if base_p is not None else ""))
            for pair in chosen:
                base = f"D p {base_p[pair]:.3f} -> " if base_p is not None else ""
                print(f"    {base}prior {prior:+.2f} | {describe(pair)} | p {p[pair]:.3f}")
            if len(chosen):
                waterfall(axis, prior, contribution[chosen[0]], bar_labels(chosen[0]), kind)
            else:
                axis.set_axis_off()
        figure.tight_layout()
        figure.savefig(ROOT / "results" / file, dpi=200)
        plt.close(figure)

    examples(kinds, f"waterfall_{stem}.png")

    # 4. a model with clinical fields against the D run with the same prior of m and u
    if "+" not in model_name:
        return
    base_run = run_name("D" + _prior_suffix(model_name), scenario_name, pool)
    base = Scores.load(base_run)
    base_links = load_links(base_run)
    if not base.pairs[PAIR].equals(scores.pairs[PAIR]):
        raise ValueError(f"D and {model_name} were not run on the same candidate pairs ({scenario_name}, {pool})")
    base_p = base.pairs["probability"].to_numpy()
    changed = changes(i == j, links["assigned"].to_numpy(), base_links["assigned"].to_numpy())
    zones = review_zones(i, j, p)
    base_zones = review_zones(i, j, base_p)
    automatic_changed = changes(i == j, zones == AUTOMATIC, base_zones == AUTOMATIC)
    print("\n  decisions changed against D")
    print(f"    one-to-one assignment, p >= 0.5: {change_summary(changed)}")
    print(f"    automatic links at 99 % precision: {change_summary(automatic_changed)}")
    print(f"\n  zones: D (rows) against {model_name} (columns), pairs (true pairs); review = |A2|")
    print(zone_moves(i == j, zones, base_zones).to_string())
    examples(changed, f"waterfall_{stem}_changes.png", base_p)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="D+C", help="the stored run to explain, e.g. D, D+C, D+E-stat, D+E-ktx")
    parser.add_argument("--scenario", nargs="+", choices=list(SCENARIOS), default=list(SCENARIOS))
    parser.add_argument("--pool", choices=list(POOLS), default="natural", help="voters with any / 3+ changed fields")
    args = parser.parse_args()
    for name in args.scenario:
        explain(args.model, name, args.pool)


if __name__ == "__main__":
    main()
