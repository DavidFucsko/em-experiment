"""NCVR statistics for the thesis (aggregates only): the whole linked population and the voters assigned to each cohort
and pool. Run from em-experiment: uv run python ncvr_stats.py"""
import duckdb

from src.config import VOTER

FIELDS = ("first_name", "middle_name", "last_name", "address", "city", "zip_code", "phone")
PAIRS = f"'{(VOTER / 'voter_pairs.parquet').as_posix()}'"
db = duckdb.connect()


def _one_row(query: str) -> tuple:
    return db.execute(query).fetchone()


def describe(label: str, source: str) -> None:
    n = _one_row(f"SELECT count(*) FROM {source}")[0]
    print(f"\n=== {label}: {n:,} persons")

    changed_columns = ", ".join(f"avg(({field}_a IS DISTINCT FROM {field}_b)::int)" for field in FIELDS)
    missing_columns = ", ".join(f"avg(({field}_a IS NULL OR {field}_a = '')::int)" for field in FIELDS)
    changed = _one_row(f"SELECT {changed_columns} FROM {source}")
    missing = _one_row(f"SELECT {missing_columns} FROM {source}")
    print(f"{'field':<12} {'changed':>8} {'missing 2016':>13}")
    for field, changed_share, missing_share in zip(FIELDS, changed, missing):
        print(f"{field:<12} {100 * changed_share:>7.1f}% {100 * missing_share:>12.1f}%")

    by_changes = db.execute(f"SELECT least(n_changed_fields, 4), count(*) / {n} FROM {source} GROUP BY 1 ORDER BY 1")
    for k, share in by_changes.fetchall():
        n_changes = ">= 4" if k == 4 else k
        print(f"{n_changes} changed fields: {100 * share:.1f}%")

    sex = db.execute(f"SELECT sex_b, count(*) / {n} FROM {source} GROUP BY 1 ORDER BY 2 DESC").fetchall()
    print("sex (2026):", ", ".join(f"{value}: {100 * share:.1f}%" for value, share in sex))
    print("birth year p10/p50/p90:", _one_row(f"SELECT quantile_cont(birth_year_b, [0.1, 0.5, 0.9]) FROM {source}")[0])
    year_differs = _one_row(f"SELECT avg((birth_year_a IS DISTINCT FROM birth_year_b)::int) FROM {source}")[0]
    print("birth year differs 2016 vs 2026:", f"{100 * year_differs:.1f}%")


describe("NCVR, linked 2016-2026 population", PAIRS)
for cohort in ("k1", "ked"):
    for pool in ("natural", "hard"):
        assignment = VOTER / f"assignment_{cohort}_{pool}_cond.parquet"
        if assignment.exists():
            describe(f"assigned voters, {cohort}, {pool}",
                     f"(SELECT p.* FROM {PAIRS} p JOIN '{assignment.as_posix()}' a USING (ncid))")
