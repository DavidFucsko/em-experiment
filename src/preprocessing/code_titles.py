"""Titles of the clinical codes, one row per (modality, vocabulary, code), for display and for the text embeddings.
A code that is already a word (organism, service) has no row; a GSN with several drug names keeps the most frequent."""
from pathlib import Path

import duckdb

from src.config import MIMIC_ED, MIMIC_HOSP, MIMIC_ICU, SPLIT

CODE_TITLES_FILE = SPLIT / "code_titles.parquet"

# (modality, vocabulary expression, code column, title expression, silo, table)
TITLE_SOURCES = (
    ("dx", "'icd' || icd_version", "icd_code", "long_title", "hosp", "d_icd_diagnoses"),
    ("px", "'icd' || icd_version", "icd_code", "long_title", "hosp", "d_icd_procedures"),
    ("drg", "'drg_' || lower(drg_type)", "drg_code", "description", "hosp", "drgcodes"),
    ("hcpcs", "'hcpcs'", "code", "coalesce(long_description, short_description)", "hosp", "d_hcpcs"),
    ("icurx", "'mimic_item'", "itemid", "label", "icu", "d_items"),
    ("icupx", "'mimic_item'", "itemid", "label", "icu", "d_items"),
    ("homemed", "'gsn'", "gsn", "name", "ed", "medrecon"),
    ("edrx", "'gsn'", "gsn", "name", "ed", "pyxis"),
)


def build_code_titles(hosp_dir: Path, icu_dir: Path, ed_dir: Path, out_file: Path) -> Path:
    directories = {"hosp": hosp_dir, "icu": icu_dir, "ed": ed_dir}
    selects = []
    for modality, vocabulary, code, title, silo, table in TITLE_SOURCES:
        table_file = (directories[silo] / table).as_posix()
        selects.append(f"SELECT '{modality}' AS modality, {vocabulary} AS vocabulary, {code} AS code, {title} AS title "
                       f"FROM read_csv('{table_file}.csv.gz', all_varchar = true)")
    sources = " UNION ALL ".join(selects)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    duckdb.execute(f"""
        COPY (
            SELECT modality, vocabulary, code, mode(title) AS title FROM ({sources})
            WHERE code IS NOT NULL AND title IS NOT NULL GROUP BY modality, vocabulary, code ORDER BY ALL
        ) TO '{out_file.as_posix()}' (FORMAT parquet)
    """)
    return out_file


def main() -> None:
    build_code_titles(MIMIC_HOSP, MIMIC_ICU, MIMIC_ED, CODE_TITLES_FILE)
    selects = []
    for side in ("history", "index_admission", "ed_arrival"):
        selects.append(f"SELECT DISTINCT modality, vocabulary, code FROM '{(SPLIT / f'{side}.parquet').as_posix()}'")
    sides = " UNION ".join(selects)
    coverage = duckdb.sql(f"""
        SELECT modality, vocabulary, count(*) AS codes, count(title) AS with_title
        FROM ({sides}) LEFT JOIN '{CODE_TITLES_FILE.as_posix()}' USING (modality, vocabulary, code)
        GROUP BY ALL ORDER BY ALL
    """).fetchall()
    for modality, vocabulary, codes, with_title in coverage:
        print(f"{modality:<8} {vocabulary:<11} {with_title:>7,} of {codes:>7,} distinct codes have a title")


if __name__ == "__main__":
    main()
