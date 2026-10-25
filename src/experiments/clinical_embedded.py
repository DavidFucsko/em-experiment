"""Model D+E: the same run as D+C with the exact-code similarity replaced by the best match of embedded codes, from
GloVe (stat) or Bio_ClinicalBERT title vectors (ktx)."""
import argparse

from src.config import RUN_LABEL
from src.experiments.report import print_summary
from src.experiments.scenarios import SCENARIOS, max_match_share
from src.preprocessing.code_vectors import KTX_FILE, STAT_FILE
from src.preprocessing.identity_assignment import POOLS
from src.utils import fellegi_sunter
from src.utils.clinical_codes import ClinicalField
from src.utils.demographic_levels import demographic_fields
from src.utils.embedded_similarity import embedded_similarities
from src.utils.evaluation import evaluate
from src.utils.experiment_data import load_demographic_records, load_patient_ids
from src.utils.fields import ComparisonVector, CompositeField
from src.utils.linking import link
from src.utils.priors import DEFAULT_PRIOR, PRIORS
from src.utils.runs import Scores, run_name, save_links

VECTOR_FILES = {"stat": STAT_FILE, "ktx": KTX_FILE}
VARIANTS = ("naive", "B")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vectors", choices=list(VECTOR_FILES), required=True, help="GloVe (stat) or Bio_ClinicalBERT (ktx)")
    parser.add_argument("--scenario", nargs="+", choices=list(SCENARIOS), default=list(SCENARIOS))
    parser.add_argument("--pool", choices=list(POOLS), default="natural", help="voters with any / 3+ changed fields")
    parser.add_argument("--prior", choices=PRIORS, default=DEFAULT_PRIOR,
                        help="pseudo-counts of m and u (priors.py); a non-default one is added to the run name")
    parser.add_argument("--variant", choices=VARIANTS, default="B", help="how the clinical fields enter the model")
    args = parser.parse_args()
    run_model = f"D+E-{args.vectors}"
    if args.variant != "B":
        run_model += f"-{args.variant}"
    if args.prior != DEFAULT_PRIOR:
        run_model += f"-{args.prior}"

    patients = load_patient_ids()
    n = len(patients)
    records = load_demographic_records(args.pool, "cond", patients)
    codes = embedded_similarities(patients, VECTOR_FILES[args.vectors])
    for name in args.scenario:
        scenario = SCENARIOS[name]
        run = run_name(run_model, name, args.pool)
        i, j = scenario.candidates(records)

        # the clinical fields are cut at percentiles of the candidate pairs' similarity
        clinical = [ClinicalField.cut_on(category, similarity, i, j) for category, similarity in codes.items()]
        fields = demographic_fields(records.history, records.later, scenario.fields)
        if args.variant == "B":
            fields.append(CompositeField("clinical", clinical))
        else:
            fields.extend(clinical)
        vector = ComparisonVector(fields)

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
