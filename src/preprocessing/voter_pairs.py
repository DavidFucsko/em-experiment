"""Links two cleaned voter snapshots on ncid: the same person's identity fields 10 years apart."""
from pathlib import Path

import duckdb

from src.config import ROOT, VOTER, VOTER_SNAPSHOTS
from src.preprocessing.voter_snapshot import clean_snapshot

IDENTITY_FIELDS = ("first_name", "middle_name", "last_name", "address", "city", "zip_code", "phone", "sex",
                   "birth_year")
# birth_year is derived from age (+-1) and sex is filtered to be stable, so neither counts as a change
CHANGE_FIELDS = ("first_name", "middle_name", "last_name", "address", "city", "zip_code", "phone")
MAX_BIRTH_YEAR_GAP = 1  # larger gaps mean the ncid was reused for a different person

PAIRS_FILE = VOTER / "voter_pairs.parquet"


def raw_snapshot_file(snapshot: str) -> Path:
    return ROOT / f"VR_Snapshot_{snapshot}" / f"VR_Snapshot_{snapshot}.txt"


def clean_snapshot_file(snapshot: str) -> Path:
    return VOTER / f"snapshot_{snapshot}.parquet"


def link_snapshots(older_file: Path, newer_file: Path, out_file: Path) -> Path:
    """One row per person present in both snapshots: ncid, <field>_a (older), <field>_b (newer), n_changed_fields."""
    sides = ", ".join(f"a.{field} AS {field}_a, b.{field} AS {field}_b" for field in IDENTITY_FIELDS)
    changed = " + ".join(f"CAST(a.{field} IS DISTINCT FROM b.{field} AS INTEGER)" for field in CHANGE_FIELDS)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    duckdb.execute(f"""
        COPY (
            SELECT ncid, {sides}, {changed} AS n_changed_fields
            FROM '{older_file.as_posix()}' a JOIN '{newer_file.as_posix()}' b USING (ncid)
            WHERE abs(a.birth_year - b.birth_year) <= {MAX_BIRTH_YEAR_GAP}
        ) TO '{out_file.as_posix()}' (FORMAT parquet)
    """)
    return out_file


def main() -> None:
    cleaned = []
    for snapshot in VOTER_SNAPSHOTS:
        target = clean_snapshot_file(snapshot)
        if not target.exists():
            clean_snapshot(raw_snapshot_file(snapshot), target)
        cleaned.append(target)
    older, newer = cleaned
    link_snapshots(older, newer, PAIRS_FILE)
    changed_counts = duckdb.sql(f"""
        SELECT n_changed_fields, count(*) AS persons
        FROM '{PAIRS_FILE.as_posix()}' GROUP BY ALL ORDER BY ALL
    """).fetchall()
    for n_changed_fields, persons in changed_counts:
        print(f"{n_changed_fields} changed fields: {persons:,} persons")


if __name__ == "__main__":
    main()
