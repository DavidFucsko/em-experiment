"""What the experiment scripts print: the result of one run, or the difference between two runs on the same pairs.
F1 at 0.5 is fragile (records sharing a comparison vector flip together); top-1, |A2| and R@99%P are not."""
import numpy as np

from src.preprocessing.identity_assignment import POOLS
from src.utils.evaluation import Evaluation, bootstrap_difference
from src.utils.experiment_data import DemographicRecords

S2_MIN_CHANGES = POOLS["hard"]  # the hard subgroup: records with at least this many changed fields
MAX_CHANGES_SHOWN = 5           # records with this many changed fields or more form the last group


def print_summary(label: str, records: DemographicRecords, i: np.ndarray, j: np.ndarray, evaluation: Evaluation) -> None:
    """One run: top-1, |A2|, recall at 99 % precision, and precision, recall, F1 (also by number of changed fields)
    with a plain 0.5 threshold (E2) and after one-to-one assignment (E2+P3)."""
    n = len(records.history)
    metrics = evaluation.metrics()
    print(f"\n{label}: {n:,} patients | {len(i):,} candidate pairs, true pairs covered {(i == j).sum() / n:.2%} "
          f"| top-1 accuracy {metrics['top-1']:.3f} | |A2| {metrics['|A2|']:,.0f} "
          f"| recall at 99 % precision {metrics['R@99%P']:.3f}")

    changes = np.minimum(records.n_changed_fields, MAX_CHANGES_SHOWN)
    groups = [(changes == k).astype(float) for k in range(MAX_CHANGES_SHOWN + 1)]
    group_sizes = " / ".join(str(int(group.sum())) for group in groups)
    print(f"{'':7s} {'P':>6s} {'R':>6s} {'F1':>6s} | F1 with 0 / 1 / 2 / 3 / 4 / 5+ changed fields (records: {group_sizes})")
    for method, links in {"E2": evaluation.threshold_links, "E2+P3": evaluation.one_to_one_links}.items():
        group_f1 = []
        for group in groups:
            group_f1.append(f"{links.f1(group):.3f}" if group.sum() else "-")
        print(f"{method:7s} {links.precision():6.3f} {links.recall():6.3f} {links.f1():6.3f} | {' / '.join(group_f1)}")


def _format(metric: str, value: float, sign: str = "") -> str:
    if metric == "|A2|":
        return f"{value:{sign},.0f}"
    return f"{value:{sign}.3f}"


def print_comparison(title: str, base: Evaluation, new: Evaluation, records: np.ndarray, names: tuple[str, str]) -> None:
    """Every metric on `records` for the two runs, with the difference and its 95 % patient-level bootstrap interval."""
    weight = np.zeros(base.n)
    weight[records] = 1
    before = base.metrics(weight)
    after = new.metrics(weight)
    intervals = bootstrap_difference(new, base, records)
    print(f"  {title} ({len(records):,} records)\n    {'':16s} {names[0]:>9s} {names[1]:>9s}   difference [95 % CI]")
    for metric, (difference, low, high) in intervals.items():
        print(f"    {metric:16s} {_format(metric, before[metric]):>9s} {_format(metric, after[metric]):>9s}   "
              f"{_format(metric, difference, '+')} [{_format(metric, low, '+')}, {_format(metric, high, '+')}]")
