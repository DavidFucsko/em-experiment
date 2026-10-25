"""Model D: the demographic-only Fellegi-Sunter baseline, per scenario of available identifiers."""
import argparse

from src.config import RUN_LABEL
from src.experiments.report import print_summary
from src.experiments.scenarios import SCENARIOS, max_match_share
from src.preprocessing.identity_assignment import POOLS
from src.utils import fellegi_sunter
from src.utils.demographic_levels import demographic_fields
from src.utils.evaluation import evaluate
from src.utils.experiment_data import load_demographic_records, load_patient_ids
from src.utils.fields import ComparisonVector
from src.utils.linking import link
from src.utils.priors import DEFAULT_PRIOR, PRIORS
from src.utils.runs import Scores, run_name, save_links


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", nargs="+", choices=list(SCENARIOS), default=list(SCENARIOS))
    parser.add_argument("--pool", choices=list(POOLS), default="natural", help="voters with any / 3+ changed fields")
    parser.add_argument("--prior", choices=PRIORS, default=DEFAULT_PRIOR,
                        help="pseudo-counts of m and u (priors.py); a non-default one is added to the run name")
    args = parser.parse_args()
    run_model = "D"
    if args.prior != DEFAULT_PRIOR:
        run_model += f"-{args.prior}"

    patients = load_patient_ids()
    n = len(patients)
    records = load_demographic_records(args.pool, "cond", patients)
    for name in args.scenario:
        scenario = SCENARIOS[name]
        run = run_name(run_model, name, args.pool)
        i, j = scenario.candidates(records)
        vector = ComparisonVector(demographic_fields(records.history, records.later, scenario.fields))

        # EM over all n x n pairs, without labels; the match share is at most min(n_A, n_B) / (n_A n_B)
        vectors, counts = vector.count_comparison_vectors(n, n)
        model = fellegi_sunter.fit(vectors, counts, vector.n_levels, prior=args.prior, max_share=max_match_share(n, n))

        scores = Scores.of(model, vector, (i, j), vector.levels(i, j), patients)
        scores.save(run)
        links = link(scores, patients)
        save_links(links, run)
        print_summary(f"cohort {RUN_LABEL}, {args.pool} {name}, model {run_model}", records, i, j,
                      evaluate(scores, links, patients))


if __name__ == "__main__":
    main()
