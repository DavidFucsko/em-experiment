"""How each stored model would run in a master patient index (Morris et al. 2014, ONC): every incoming record is linked
automatically, sent to clerical review, or created as a new record. The thresholds allow at most 0.1 % false automatic
links and 1 % missed correct assignments, set with the labels as if tuned on a reviewed sample."""
import argparse

from src.config import RUN_LABEL
from src.experiments.scenarios import SCENARIOS
from src.preprocessing.identity_assignment import POOLS
from src.utils.evaluation import MAX_FALSE_LINKS, MAX_RECALL_LOSS, link_review_or_new
from src.utils.experiment_data import load_patient_ids
from src.utils.runs import Scores, run_name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["D", "D+C", "D+E-stat", "D+E-ktx"])
    parser.add_argument("--scenario", nargs="+", choices=list(SCENARIOS), default=list(SCENARIOS))
    parser.add_argument("--pool", choices=list(POOLS), default="natural", help="voters with any / 3+ changed fields")
    args = parser.parse_args()

    patients = load_patient_ids()
    for name in args.scenario:
        print(f"\ncohort {RUN_LABEL}, {args.pool} pool, scenario {name}: {len(patients):,} incoming records | at most "
              f"{MAX_FALSE_LINKS:.1%} false automatic links, at most {MAX_RECALL_LOSS:.0%} of the correct assignments missed")
        print(f"  {'model':9s} {'automatic links':>16s} {'of them false':>14s} {'review tasks':>13s} {'new records':>12s} "
              f"{'missed':>7s} {'no candidate':>13s} {'upper p':>8s} {'lower p':>8s}")
        for model in args.models:
            scores = Scores.load(run_name(model, name, args.pool))
            i, j = scores.positions(patients)
            decisions = link_review_or_new(i, j, scores.pairs["probability"].to_numpy(), len(patients))
            row = decisions.summary()
            print(f"  {model:9s} {row['automatic links']:>16,} {row['false automatic links']:>14,} {row['review tasks']:>13,} "
                  f"{row['new records']:>12,} {row['missed (partner among the candidates)']:>7,} "
                  f"{row['missed (partner not a candidate)']:>13,} {decisions.upper:>8.3f} {decisions.lower:>8.3f}")
        print("  missed: new records although the partner was a candidate (duplicates); no candidate: the partner was not "
              "among the candidates at all")


if __name__ == "__main__":
    main()
