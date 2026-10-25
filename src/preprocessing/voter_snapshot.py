"""Raw NC voter snapshot (UTF-16 TSV) -> one clean row per voter (ncid) as Parquet."""
from pathlib import Path

import duckdb

# A voter who moved county keeps an old removed record next to the new one, so the most alive one is kept. Remaining
# ties are broken by county and registration number, so the choice never depends on row order.
STATUS_PRIORITY = ["A", "I", "S", "R", "D"]
ADDRESS_PARTS = ("house_num", "street_dir", "street_name", "street_type_cd", "unit_num")
UNRANKED_STATUS = len(STATUS_PRIORITY)


def _text(column: str) -> str:
    return f"nullif(trim({column}), '')"


def clean_snapshot(raw_file: Path, out_file: Path) -> Path:
    """birth_year is snapshot year - age, so it is off by one for some voters (there is no birth date in the file).
    phone is the 7-digit number only: the area code is missing for a third of the voters."""
    address = ", ".join(_text(part) for part in ADDRESS_PARTS)
    status_rank = f"coalesce(list_position({STATUS_PRIORITY}, trim(status_cd)), {UNRANKED_STATUS})"
    registered_on = "coalesce(try_strptime(trim(registr_dt), '%Y-%m-%d'), try_strptime(trim(registr_dt), '%m/%d/%Y'))"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    duckdb.execute(f"""
        COPY (
            SELECT ncid, first_name, middle_name, last_name, address, city, zip_code, phone, sex, birth_year
            FROM (
                SELECT
                    trim(ncid) AS ncid,
                    {_text("first_name")} AS first_name,
                    {_text("midl_name")} AS middle_name,
                    {_text("last_name")} AS last_name,
                    nullif(concat_ws(' ', {address}), '') AS address,
                    {_text("res_city_desc")} AS city,
                    {_text("zip_code")} AS zip_code,
                    {_text("phone_num")} AS phone,
                    {_text("sex_code")} AS sex,
                    year(CAST(trim(snapshot_dt) AS DATE)) - try_cast(age AS INTEGER) AS birth_year,
                    {status_rank} AS status_rank,
                    {registered_on} AS registered_on,
                    trim(county_id) AS county_id,
                    trim(voter_reg_num) AS voter_reg_num
                FROM read_csv('{raw_file.as_posix()}', encoding = 'utf-16', delim = '\t', header = true,
                              quote = '', escape = '', all_varchar = true, strict_mode = false)
                WHERE trim(ncid) <> ''
            )
            QUALIFY row_number() OVER (
                PARTITION BY ncid
                ORDER BY status_rank, registered_on DESC NULLS LAST, county_id, voter_reg_num) = 1
        ) TO '{out_file.as_posix()}' (FORMAT parquet)
    """)
    return out_file
