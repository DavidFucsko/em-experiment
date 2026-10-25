"""Print a few full records from every data source of the experiment (run after the preprocessing steps).

    uv run python -m src.test.show_samples [--n 2] [--seed 42] [--variant cond|indep] [--pool hard|natural] [--no-titles]

1. voter pairs      same person in the 2016 and 2026 snapshots, n per number of changed fields (0, 1, 2, 3+)
2. MIMIC patients   demographics + clinical codes of the three sides, with their visits
3. assigned records the experiment's records: voter identity (A = 2016, B = 2026) + clinical codes per side
The output holds MIMIC rows and real voter identities: keep it out of commits, issues and chats (PhysioNet DUA).
"""
import argparse
import sys

import duckdb

from src.config import SEED, SPLIT
from src.preprocessing.code_titles import CODE_TITLES_FILE
from src.preprocessing.identity_assignment import POOLS, VARIANTS, assignment_file
from src.preprocessing.voter_pairs import PAIRS_FILE
from src.utils.experiment_data import side_fields

MAX_CHANGES_SHOWN = 3  # voter pairs are sampled per number of changed fields, 3 meaning 3 or more


def parquet(path) -> str:
    return f"'{path.as_posix()}'"


def side_record(side: str, subject_id: int, titles: bool) -> dict[str, str]:
    """The patient's visits on one side, then the distinct codes per modality and vocabulary."""
    side_file = parquet(SPLIT / f"{side}.parquet")
    n_visits, first, last, first_time, last_time = duckdb.sql(f"""
        SELECT count(DISTINCT visit), min(visit), max(visit), strftime(min(visit_time), '%Y-%m-%d %H:%M'),
               strftime(max(visit_time), '%Y-%m-%d %H:%M')
        FROM {side_file} WHERE subject_id = {subject_id}
    """).fetchone()
    shown = "code || coalesce(' (' || title || ')', '')" if titles else "code"
    join = f"LEFT JOIN {parquet(CODE_TITLES_FILE)} USING (modality, vocabulary, code)" if titles else ""
    rows = duckdb.sql(f"""
        SELECT modality, vocabulary, list(DISTINCT {shown} ORDER BY {shown}) FROM {side_file} {join}
        WHERE subject_id = {subject_id} GROUP BY modality, vocabulary ORDER BY modality, vocabulary
    """).fetchall()
    visits = f"{n_visits} (numbers {first}..{last}; starts {first_time} .. {last_time})"
    return {"visits": visits} | {f"{modality} / {vocabulary} ({len(codes)})": ", ".join(codes)
                                 for modality, vocabulary, codes in rows}


def print_fields(fields: dict, indent: int = 2) -> None:
    width = max(map(len, fields))
    for key, value in fields.items():
        print(f"{' ' * indent}{key:<{width}} : {value}")


def show_voter_pairs(n: int, seed: int) -> None:
    print("\n=== 1. VOTER PAIRS (NC voter snapshots 2016 -> 2026, linked on ncid) ===")
    records = duckdb.sql(f"""
        SELECT * FROM {parquet(PAIRS_FILE)}
        QUALIFY row_number() OVER (PARTITION BY least(n_changed_fields, {MAX_CHANGES_SHOWN}) ORDER BY hash(ncid, {seed})) <= {n}
        ORDER BY n_changed_fields, ncid
    """).df().to_dict("records")
    for record in records:
        print(f"\n--- ncid {record['ncid']} | {record['n_changed_fields']} changed fields ---")
        for label, suffix in (("2016 (A)", "_a"), ("2026 (B)", "_b")):
            print(f"  [{label}]")
            print_fields(side_fields(record, suffix), indent=4)


def show_mimic_patients(n: int, seed: int, titles: bool) -> None:
    print("\n=== 2. MIMIC PATIENTS (temporal split) ===")
    records = duckdb.sql(f"""
        SELECT * FROM {parquet(SPLIT / 'patients.parquet')} ORDER BY hash(subject_id, {seed}) LIMIT {n}
    """).df().to_dict("records")
    for record in records:
        print(f"\n--- subject_id {record['subject_id']} ---")
        print_fields({"gender": record["gender"], "anchor_age": record["anchor_age"]})
        for side in ("history", "index_admission", "ed_arrival"):
            print(f"  [{side}]")
            print_fields(side_record(side, record["subject_id"], titles), indent=4)


def show_assigned(n: int, seed: int, pool: str, variant: str, titles: bool) -> None:
    print(f"\n=== 3. ASSIGNED RECORDS ({pool} pool, V-{variant}): what the experiment links ===")
    records = duckdb.sql(f"""
        SELECT assignment.*, patients.gender, patients.anchor_age, pairs.* EXCLUDE (ncid)
        FROM {parquet(assignment_file(pool, variant))} assignment
        JOIN {parquet(SPLIT / 'patients.parquet')} patients USING (subject_id)
        JOIN {parquet(PAIRS_FILE)} pairs USING (ncid)
        ORDER BY hash(subject_id, {seed}) LIMIT {n}
    """).df().to_dict("records")
    for record in records:
        print(f"\n--- subject_id {record['subject_id']} -> ncid {record['ncid']} ---")
        print_fields({key: record[key] for key in ("gender", "anchor_age", "age_gap", "n_changed_fields")})
        for label, suffix, side in (("A: history + voter 2016", "_a", "history"),
                                    ("B: index admission + voter 2026", "_b", "index_admission")):
            print(f"  [{label}]")
            print_fields(side_fields(record, suffix) | side_record(side, record["subject_id"], titles), indent=4)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console default cannot print every voter name
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=2, help="records per source (per damage stratum for voter pairs)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--variant", choices=list(VARIANTS), default="cond")
    parser.add_argument("--pool", choices=list(POOLS), default="hard")
    parser.add_argument("--titles", action=argparse.BooleanOptionalAction, default=True,
                        help="show each code with its title (default on; --no-titles for bare codes)")
    args = parser.parse_args()
    show_voter_pairs(args.n, args.seed)
    show_mimic_patients(args.n, args.seed, args.titles)
    show_assigned(args.n, args.seed, args.pool, args.variant, args.titles)


if __name__ == "__main__":
    main()
