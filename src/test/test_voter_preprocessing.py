import duckdb

from src.preprocessing.voter_pairs import link_snapshots
from src.preprocessing.voter_snapshot import clean_snapshot

RAW_COLUMNS = ("snapshot_dt", "county_id", "voter_reg_num", "ncid", "status_cd", "first_name", "midl_name", "last_name", "house_num", "street_dir",
               "street_name", "street_type_cd", "unit_num", "res_city_desc", "zip_code", "area_cd", "phone_num",
               "sex_code", "age", "registr_dt")


def voter(ncid, **fields):
    defaults = dict(snapshot_dt="2016-03-15", county_id="1", voter_reg_num="1", status_cd="A", first_name="ANNA", midl_name="LEE", last_name="KING",
                    house_num="12", street_dir="", street_name="OAK", street_type_cd="ST", unit_num="",
                    res_city_desc="GRAHAM", zip_code="27249", area_cd="", phone_num="5551234", sex_code="F",
                    age="40", registr_dt="2001-05-04")
    return dict(defaults, ncid=ncid, **fields)


def write_raw(path, voters):
    lines = ["\t".join(RAW_COLUMNS)] + ["\t".join(row[column] for column in RAW_COLUMNS) for row in voters]
    path.write_text("\n".join(lines) + "\n", encoding="utf-16")
    return path


def read_rows(path, columns="*"):
    return duckdb.sql(f"SELECT {columns} FROM '{path.as_posix()}' ORDER BY ncid").fetchall()


def link(tmp_path, older_voters, newer_voters):
    newer_voters = [dict(row, snapshot_dt="2026-03-03") for row in newer_voters]
    return link_snapshots(
        clean_snapshot(write_raw(tmp_path / "older.txt", older_voters), tmp_path / "a.parquet"),
        clean_snapshot(write_raw(tmp_path / "newer.txt", newer_voters), tmp_path / "b.parquet"),
        tmp_path / "pairs.parquet",
    )


def test_clean_snapshot_trims_builds_address_and_derives_birth_year(tmp_path):
    raw = write_raw(tmp_path / "raw.txt", [
        voter("AA1  ", first_name=" ANNA ", street_dir="N", unit_num="4B", age="40", sex_code="F "),
    ])
    out = clean_snapshot(raw, tmp_path / "clean.parquet")
    assert read_rows(out) == [("AA1", "ANNA", "LEE", "KING", "12 N OAK ST 4B", "GRAHAM", "27249", "5551234", "F", 1976)]


def test_clean_snapshot_leaves_missing_phone_and_middle_name_empty_and_ignores_area_code(tmp_path):
    raw = write_raw(tmp_path / "raw.txt", [
        voter("AA1", midl_name="", phone_num="", area_cd="336"),
        voter("AA2", area_cd="336"),
    ])
    out = clean_snapshot(raw, tmp_path / "clean.parquet")
    assert read_rows(out, "ncid, middle_name, phone") == [("AA1", None, None), ("AA2", "LEE", "5551234")]


def test_clean_snapshot_keeps_most_alive_record_per_voter(tmp_path):
    raw = write_raw(tmp_path / "raw.txt", [
        voter("AA1", status_cd="R", zip_code="28000", registr_dt="1994-08-19"),
        voter("AA1", status_cd="A ", zip_code="28107", registr_dt="2011-10-18"),
        voter("AA2", status_cd="R", zip_code="11111", registr_dt="01/20/1984"),
        voter("AA2", status_cd="R", zip_code="22222", registr_dt="10/22/2011"),
    ])
    out = clean_snapshot(raw, tmp_path / "clean.parquet")
    assert read_rows(out, "ncid, zip_code") == [("AA1", "28107"), ("AA2", "22222")]


def test_clean_snapshot_breaks_ties_the_same_way_whatever_the_row_order(tmp_path):
    tied = [voter("AA1", voter_reg_num="7", zip_code="27001"), voter("AA1", voter_reg_num="9", zip_code="27009")]
    picks = []
    for rows in (tied, tied[::-1]):
        raw = write_raw(tmp_path / "raw.txt", rows)
        picks.append(read_rows(clean_snapshot(raw, tmp_path / "clean.parquet"), "zip_code"))
    assert picks[0] == picks[1]


def test_link_snapshots_pairs_same_person_and_counts_changed_fields(tmp_path):
    paired = link(
        tmp_path,
        [voter("AA1"), voter("AA2"), voter("AA3"), voter("AA4")],
        [voter("AA1", last_name="SMITH", zip_code="28001", age="50"), voter("AA2", age="50"),
         voter("AA3", age="75"), voter("AA9", age="50")],
    )
    assert read_rows(paired, "ncid, last_name_a, last_name_b, n_changed_fields") == [
        ("AA1", "KING", "SMITH", 2), ("AA2", "KING", "KING", 0),
    ]


def test_link_snapshots_counts_middle_name_city_and_phone_changes_but_not_a_new_area_code(tmp_path):
    older = [voter("AA1"), voter("AA2", phone_num=""), voter("AA3"), voter("AA4"), voter("AA5")]
    newer = [voter("AA1", age="50", phone_num="5559999"),                    # phone changed
             voter("AA2", age="50", phone_num=""),                           # missing on both sides: no change
             voter("AA3", age="50", phone_num=""),                           # phone dropped: a change
             voter("AA4", age="50", res_city_desc="DURHAM", midl_name="L"),  # city and middle name changed
             voter("AA5", age="50", area_cd="336")]                          # area code is not part of phone
    paired = link(tmp_path, older, newer)
    assert read_rows(paired, "ncid, n_changed_fields") == [("AA1", 1), ("AA2", 0), ("AA3", 1), ("AA4", 2), ("AA5", 0)]


def test_link_snapshots_tolerates_one_year_age_rounding(tmp_path):
    paired = link(tmp_path, [voter("AA1", age="56")], [voter("AA1", age="65")])
    assert [row[0] for row in read_rows(paired)] == ["AA1"]
