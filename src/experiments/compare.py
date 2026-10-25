"""Compares two stored runs on the same patients and candidate pairs: every metric, the difference
and its 95 % patient-level bootstrap interval, for all records and for the hard subgroup S2."""
import argparse

import numpy as np

from src.config import RUN_LABEL
from src.experiments.report import S2_MIN_CHANGES, print_comparison
from src.experiments.scenarios import SCENARIOS
from src.preprocessing.identity_assignment import POOLS
from src.utils.evaluation import evaluate
from src.utils.experiment_data import load_demographic_records, load_patient_ids
from src.utils.runs import PAIR, Scores, load_links, run_name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="the stored run compared against, e.g. D or D+C")
    parser.add_argument("--new", required=True, help="the stored run that may improve on it, e.g. D+C or D+E-stat")
    parser.add_argument("--scenario", nargs="+", choices=list(SCENARIOS), default=list(SCENARIOS))
    parser.add_argument("--pool", choices=list(POOLS), default="natural", help="voters with any / 3+ changed fields")
    args = parser.parse_args()

    patients = load_patient_ids()
    records = load_demographic_records(args.pool, "cond", patients)
    names = (args.base, args.new)
    for name in args.scenario:
        base_run = run_name(args.base, name, args.pool)
        new_run = run_name(args.new, name, args.pool)
        base_scores = Scores.load(base_run)
        new_scores = Scores.load(new_run)
        if not base_scores.pairs[PAIR].equals(new_scores.pairs[PAIR]):
            raise ValueError(f"{args.base} and {args.new} were not run on the same candidate pairs ({name}, {args.pool})")
        base = evaluate(base_scores, load_links(base_run), patients)
        new = evaluate(new_scores, load_links(new_run), patients)

        print(f"\ncohort {RUN_LABEL}, {args.pool} pool, scenario {name}: {args.base} -> {args.new} | "
              f"{len(base_scores.pairs):,} candidate pairs")
        print_comparison("all records", base, new, np.arange(len(patients)), names)
        hard_records = np.flatnonzero(records.n_changed_fields >= S2_MIN_CHANGES)
        print_comparison(f"S2: {S2_MIN_CHANGES}+ changed fields", base, new, hard_records, names)


if __name__ == "__main__":
    main()
