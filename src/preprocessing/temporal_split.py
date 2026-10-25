"""Temporal split: each patient's earlier record (history) against their later index record. A clinical
event is (modality, vocabulary, code) with the code as MIMIC stores it, plus the visit it belongs to."""
from pathlib import Path

import duckdb

from src.config import COHORT, MIMIC_ED, MIMIC_HOSP, MIMIC_ICU, SPLIT

HOSP_TABLES = ("admissions", "patients", "diagnoses_icd", "procedures_icd", "drgcodes", "hcpcsevents",
               "microbiologyevents", "services")
ED_TABLES = ("edstays", "diagnosis", "medrecon", "pyxis")
ICU_EVENT_TABLES = ("inputevents", "procedureevents")  # large; reduced to distinct (hadm_id, itemid) of the cohort

# (modality, vocabulary expression, code column, source table)
ADMISSION_SOURCES = (
    ("dx", "'icd' || icd_version", "icd_code", "diagnoses_icd"),
    ("px", "'icd' || icd_version", "icd_code", "procedures_icd"),
    ("drg", "'drg_' || lower(drg_type)", "drg_code", "drgcodes"),
    ("hcpcs", "'hcpcs'", "hcpcs_cd", "hcpcsevents"),
    ("org", "'organism'", "org_name", "microbiologyevents"),
    ("svc", "'service'", "curr_service", "services"),
    ("icurx", "'mimic_item'", "itemid", "inputevents"),
    ("icupx", "'mimic_item'", "itemid", "procedureevents"),
)
ED_SOURCES = (
    ("dx", "'icd' || icd_version", "icd_code", "diagnosis"),
    ("homemed", "'gsn'", "gsn", "medrecon"),
    ("edrx", "'gsn'", "gsn", "pyxis"),
)
PLACEHOLDER_CODES = ("", "0")  # MIMIC's "unknown" values

# side -> (admission table, ED stay table); None = the side has no admission part
#   history          admissions before the index admission + ED stays that ended before the index ED visit began
#   index_admission  everything recorded for the index admission and its ED stay(s)
#   ed_arrival       only the index ED stay
SIDES = {
    "history": ("history_adm", "history_ed"),
    "index_admission": ("index_adm", "index_ed"),
    "ed_arrival": (None, "index_ed"),
}


def _code_columns() -> dict[str, str]:
    """table -> its code column, read as text so that type inference does not drop leading zeros."""
    columns = {}
    for _, _, code, table in ADMISSION_SOURCES + ED_SOURCES:
        columns[table] = code
    return columns


CODE_COLUMNS = _code_columns()


def _csv(directory: Path, table: str) -> str:
    code_column = CODE_COLUMNS.get(table)
    types = f", types={{'{code_column}': 'VARCHAR'}}" if code_column else ""
    return f"read_csv_auto('{(directory / table).as_posix()}.csv.gz'{types})"


def _events(sources: tuple, cohort_table: str, key: str, visit_key: str) -> str:
    """(subject_id, visit_key, modality, vocabulary, code) of every event in `sources` of a row of cohort_table."""
    selects = []
    for modality, vocabulary, code, table in sources:
        selects.append(f"SELECT c.subject_id, {visit_key} AS visit_key, '{modality}' AS modality, "
                       f"{vocabulary} AS vocabulary, CAST({code} AS VARCHAR) AS code "
                       f"FROM {cohort_table} c JOIN {table} USING ({key})")
    return " UNION ALL ".join(selects)


def _select_admissions(db: duckdb.DuckDBPyConnection, require_icu: bool) -> None:
    icu_stay = "AND hadm_id IN (SELECT hadm_id FROM icustays)" if require_icu else ""
    db.sql(f"""CREATE TABLE index_adm AS
               SELECT subject_id, arg_max(hadm_id, admittime) AS hadm_id, max(admittime) AS admittime FROM admissions
               WHERE hadm_id IN (SELECT hadm_id FROM edstays WHERE hadm_id IS NOT NULL) {icu_stay}
               GROUP BY 1""")
    db.sql("""CREATE TABLE history_adm AS
              SELECT a.subject_id, a.hadm_id, a.admittime FROM admissions a JOIN index_adm i USING (subject_id)
              WHERE a.admittime < i.admittime""")
    db.sql("DELETE FROM index_adm WHERE subject_id NOT IN (SELECT subject_id FROM history_adm)")
    db.sql("""CREATE TABLE index_ed AS
              SELECT e.subject_id, e.stay_id, e.intime, e.hadm_id AS visit_key
              FROM edstays e JOIN index_adm USING (hadm_id)""")
    db.sql("""CREATE TABLE history_ed AS
              SELECT e.subject_id, e.stay_id, e.intime, coalesce(e.hadm_id, -e.stay_id) AS visit_key
              FROM edstays e
              JOIN (SELECT subject_id, min(intime) AS first_index_ed FROM index_ed GROUP BY 1) USING (subject_id)
              WHERE e.outtime < first_index_ed""")


def _admission_attributes(db: duckdb.DuckDBPyConnection) -> None:
    """Administrative attributes of each patient's last history admission and of the index admission."""
    db.sql("""CREATE TABLE attributes AS
              SELECT subject_id, 'history' AS side, insurance, language, marital_status, race FROM (
                  SELECT a.subject_id, a.insurance, a.language, a.marital_status, a.race,
                         row_number() OVER (PARTITION BY a.subject_id ORDER BY a.admittime DESC) AS recency
                  FROM admissions a JOIN history_adm USING (hadm_id)) WHERE recency = 1
              UNION ALL
              SELECT a.subject_id, 'index', a.insurance, a.language, a.marital_status, a.race
              FROM admissions a JOIN index_adm USING (hadm_id)""")


def _number_visits(db: duckdb.DuckDBPyConnection) -> None:
    """visits(subject_id, visit_key, visit, visit_time), numbered 1, 2, ... over the whole timeline. An admission and
    the ED stay that led to it share a visit_key, an ED stay without admission is its own visit."""
    db.sql("""CREATE TABLE visits AS
              SELECT subject_id, visit_key, visit_time,
                     row_number() OVER (PARTITION BY subject_id ORDER BY visit_time, visit_key) AS visit
              FROM (SELECT subject_id, visit_key, min(visit_time) AS visit_time FROM (
                        SELECT subject_id, hadm_id AS visit_key, admittime AS visit_time FROM history_adm
                        UNION ALL SELECT subject_id, hadm_id, admittime FROM index_adm
                        UNION ALL SELECT subject_id, visit_key, intime FROM history_ed
                        UNION ALL SELECT subject_id, visit_key, intime FROM index_ed)
                    GROUP BY subject_id, visit_key)""")


def _reduce_icu_events(db: duckdb.DuckDBPyConnection, icu_dir: Path) -> None:
    for table in ICU_EVENT_TABLES:
        db.sql(f"""CREATE TABLE {table} AS SELECT DISTINCT hadm_id, itemid
                   FROM read_csv_auto('{(icu_dir / table).as_posix()}.csv.gz',
                                      types={{'hadm_id': 'BIGINT', 'itemid': 'VARCHAR'}})
                   WHERE hadm_id IN (SELECT hadm_id FROM history_adm UNION SELECT hadm_id FROM index_adm)""")


def _side_events(admission_table: str | None, ed_table: str) -> str:
    parts = [_events(ED_SOURCES, ed_table, "stay_id", "c.visit_key")]
    if admission_table:
        parts.append(_events(ADMISSION_SOURCES, admission_table, "hadm_id", "c.hadm_id"))
    return " UNION ALL ".join(parts)


def build_temporal_split(hosp_dir: Path, icu_dir: Path, ed_dir: Path, out_dir: Path, require_icu: bool = True) -> Path:
    """Writes the sides, patients and attributes of the eligible patients (an earlier admission and at least one event
    on every side) to out_dir."""
    db = duckdb.connect()
    sources = [(hosp_dir, HOSP_TABLES), (ed_dir, ED_TABLES), (icu_dir, ("icustays",))]
    for directory, tables in sources:
        for table in tables:
            db.sql(f"CREATE VIEW {table} AS SELECT * FROM {_csv(directory, table)}")
    _select_admissions(db, require_icu)
    _admission_attributes(db)
    _number_visits(db)
    _reduce_icu_events(db, icu_dir)

    for side, (admission_table, ed_table) in SIDES.items():
        db.sql(f"""CREATE TABLE {side} AS
                   SELECT DISTINCT e.subject_id, e.modality, e.vocabulary, e.code, v.visit, v.visit_time
                   FROM ({_side_events(admission_table, ed_table)}) e JOIN visits v USING (subject_id, visit_key)
                   WHERE e.code IS NOT NULL AND e.code NOT IN {PLACEHOLDER_CODES}""")
    on_every_side = " INTERSECT ".join(f"SELECT subject_id FROM {side}" for side in SIDES)
    db.sql(f"CREATE TABLE eligible AS {on_every_side}")

    out_dir.mkdir(parents=True, exist_ok=True)
    exports = {}
    for side in SIDES:
        exports[side] = f"SELECT * FROM {side}"
    exports["patients"] = "SELECT subject_id, gender, anchor_age FROM patients"
    exports["attributes"] = "SELECT * FROM attributes"
    for name, query in exports.items():
        db.sql(f"""COPY ({query} WHERE subject_id IN (SELECT subject_id FROM eligible)
                   ORDER BY subject_id) TO '{(out_dir / f'{name}.parquet').as_posix()}' (FORMAT parquet)""")
    return out_dir


def main() -> None:
    out_dir = build_temporal_split(MIMIC_HOSP, MIMIC_ICU, MIMIC_ED, SPLIT, require_icu=COHORT == "k1")
    n_eligible = duckdb.sql(f"SELECT count(*) FROM '{(out_dir / 'patients.parquet').as_posix()}'").fetchone()[0]
    print("eligible patients:", n_eligible)
    for side in SIDES:
        patients, events, visits = duckdb.sql(f"""
            SELECT count(*), median(events), median(visits) FROM (
                SELECT count(*) AS events, count(DISTINCT visit) AS visits
                FROM '{(out_dir / f'{side}.parquet').as_posix()}' GROUP BY subject_id)
        """).fetchone()
        print(f"{side}: {patients:,} patients, median {events:.0f} (code, visit) events and {visits:.0f} visits per patient")


if __name__ == "__main__":
    main()
