"""Experiment 1b: the demographic baseline fitted by fastLink (Enamorado, Fifield & Imai 2019), the reference
implementation, as an independent check of our Fellegi-Sunter. Needs R with fastLink; the D run must exist first."""
import argparse
import glob
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RUN_LABEL
from src.experiments.report import print_summary
from src.experiments.scenarios import SCENARIOS, max_match_share
from src.preprocessing.identity_assignment import POOLS
from src.utils.demographic_levels import ADDRESS_CUTS, BIRTH_YEAR_CUTS, EXACT_FIELDS, NAME_CUTS, NAME_FIELDS
from src.utils.evaluation import Evaluation
from src.utils.experiment_data import load_demographic_records, load_patient_ids
from src.utils.linking import THRESHOLD, one_to_one
from src.utils.runs import Scores, run_name

R_SCRIPT = Path(__file__).with_name("fastlink_em.R")
PRIOR_WEIGHT = 0.99   # fastLink's w.lambda, between 0 (no prior) and 1


def rscript() -> str:
    """Rscript on the PATH, else the newest R in the default Windows location."""
    on_path = shutil.which("Rscript")
    if on_path:
        return on_path
    installed = sorted(glob.glob("C:/Program Files/R/R-*/bin/Rscript.exe"))
    if not installed:
        raise FileNotFoundError("Rscript not found: install R and the fastLink package")
    return installed[-1]


def field_spec(field: str) -> str:
    """name:kind:cut_a:cut_p, our cut points in fastLink's terms (agree from cut_a, partial from cut_p)."""
    if field in NAME_FIELDS:
        return f"{field}:jw:{NAME_CUTS[1]}:{NAME_CUTS[0]}"
    if field == "address":
        return f"{field}:lv:{ADDRESS_CUTS[1]}:{ADDRESS_CUTS[0]}"
    if field == "birth_year":   # ours: same / at most 1 year apart; fastLink: |difference| <= cut_a / <= cut_p
        return f"{field}:number:{-BIRTH_YEAR_CUTS[1] + 0.5}:{-BIRTH_YEAR_CUTS[0] + 0.5}"
    return f"{field}:exact:0:0"


def fastlink_probabilities(records, fields: tuple[str, ...], scores: Scores, prior_share: float) -> np.ndarray:
    """fastLink's match share, m and u applied to our levels of the pairs in `scores`, so both estimates are compared on
    the same pairs. fastLink cannot bound the match share, so it gets a Beta prior with our bound as its mean."""
    with tempfile.TemporaryDirectory() as directory:
        history_file = Path(directory) / "history.csv"
        later_file = Path(directory) / "later.csv"
        records.history[list(fields)].to_csv(history_file, index=False)
        records.later[list(fields)].to_csv(later_file, index=False)
        field_specs = [field_spec(field) for field in fields]
        command = [rscript(), str(R_SCRIPT), directory, str(prior_share), str(PRIOR_WEIGHT)] + field_specs
        run = subprocess.run(command, capture_output=True, text=True)
        if run.returncode:
            raise RuntimeError(f"fastLink failed (exit {run.returncode:#x}):\n{run.stdout[-3000:]}\n{run.stderr[-3000:]}")
        print(run.stdout.strip().splitlines()[-1])
        parameters = pd.read_csv(Path(directory) / "parameters.csv")

    share = parameters.loc[parameters["field"] == "match_share", "m"].item()
    log_odds = np.full(len(scores.pairs), np.log(share / (1 - share)))
    for field in fields:
        estimates = parameters[parameters["field"] == field].set_index("gamma")
        weight = np.log(estimates["m"]) - np.log(estimates["u"])
        levels = scores.pairs[field].to_numpy()
        # our levels in fastLink's coding: graded 1 / 2 / 3 -> 0 / 1 / 2, exact 1 / 2 -> 0 / 2
        gamma = (levels - 1) * (2 if field in EXACT_FIELDS else 1)
        # a missing value (our level 0) is no evidence, as in fastLink (missing at random)
        log_odds += np.where(levels == 0, 0.0, weight.reindex(gamma).fillna(0).to_numpy())
    return 1 / (1 + np.exp(-log_odds))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", nargs="+", choices=list(SCENARIOS), default=list(SCENARIOS))
    parser.add_argument("--pool", choices=list(POOLS), default="natural", help="voters with any / 3+ changed fields")
    args = parser.parse_args()

    patients = load_patient_ids()
    records = load_demographic_records(args.pool, "cond", patients)
    n = len(patients)
    for name in args.scenario:
        ours = Scores.load(run_name("D", name, args.pool))
        i, j = ours.positions(patients)
        p_ours = ours.pairs["probability"].to_numpy()
        p_fastlink = fastlink_probabilities(records, SCENARIOS[name].fields, ours, max_match_share(n, n))
        evaluation = Evaluation.of(i, j, p_fastlink, p_fastlink >= THRESHOLD, one_to_one(i, j, p_fastlink, n), n)
        print_summary(f"cohort {RUN_LABEL}, {args.pool} {name}, fastLink", records, i, j, evaluation)

        ours_linked = p_ours >= THRESHOLD
        theirs_linked = p_fastlink >= THRESHOLD
        print(f"agreement at {THRESHOLD}: both link {(ours_linked & theirs_linked).sum():,} | only ours "
              f"{(ours_linked & ~theirs_linked).sum():,} | only fastLink {(~ours_linked & theirs_linked).sum():,} "
              f"| match share ours {ours.model.match_share:.3e}")


if __name__ == "__main__":
    main()
