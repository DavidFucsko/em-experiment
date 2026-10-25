import gzip

import pytest

# header, rows of every file of a tiny fake MIMIC; patient 1 is the only eligible one
MIMIC_FIXTURE = {
    "hosp": {
        "admissions": ("subject_id,hadm_id,admittime,dischtime,insurance,language,marital_status,race", [
            (1, 10, "2100-01-01 08:00:00", "2100-01-05 08:00:00", "Medicaid", "ENGLISH", "SINGLE", "WHITE"),   # history (ED, no ICU)
            (1, 20, "2101-01-01 08:00:00", "2101-01-09 08:00:00", "Medicare", "ENGLISH", "MARRIED", "WHITE"),  # index: last with ED and ICU
            (1, 25, "2102-01-01 08:00:00", "2102-01-03 08:00:00", "Other", "?", "WIDOWED", "WHITE"),         # later, no ED/ICU: on neither side
            (2, 30, "2100-06-01 08:00:00", "2100-06-05 08:00:00", "Medicare", "ENGLISH", "SINGLE", "ASIAN"),  # only admission: no history
        ]),
        "patients": ("subject_id,gender,anchor_age", [(1, "F", 50), (2, "M", 60)]),
        "diagnoses_icd": ("hadm_id,icd_version,icd_code", [(10, 9, "250"), (10, 10, "R07"), (20, 10, "I10"),
                                                          (25, 10, "Z99"), (30, 10, "J18")]),
        "procedures_icd": ("hadm_id,icd_version,icd_code", [(20, 10, "0BH17EZ")]),
        "drgcodes": ("hadm_id,drg_type,drg_code,description", [(10, "HCFA", "057", "DEGENERATIVE NERVOUS SYSTEM")]),
        "hcpcsevents": ("hadm_id,hcpcs_cd", [(20, "99285")]),
        "microbiologyevents": ("hadm_id,org_name", [(20, "STAPH AUREUS")]),
        "services": ("hadm_id,curr_service", [(20, "MED")]),
        "d_icd_diagnoses": ("icd_code,icd_version,long_title", [
            ("250", 9, "Diabetes mellitus"), ("I10", 10, "Essential hypertension"), ("R07", 10, "Pain in chest"),
        ]),
        "d_icd_procedures": ("icd_code,icd_version,long_title", [("0BH17EZ", 10, "Insertion of airway")]),
        "d_hcpcs": ("code,category,long_description,short_description", [("99285", 2, "", "Emergency visit")]),
    },
    "icu": {
        "icustays": ("hadm_id", [(20,), (30,)]),
        "inputevents": ("hadm_id,itemid", [(20, 5), (25, 6)]),
        "procedureevents": ("hadm_id,itemid", [(20, 7)]),
        "d_items": ("itemid,label", [(5, "NaCl 0.9%"), (7, "Invasive Ventilation")]),
    },
    "ed": {
        "edstays": ("subject_id,stay_id,hadm_id,intime,outtime", [
            (1, 100, "", "2099-12-30 10:00:00", "2099-12-31 10:00:00"),   # earlier ED visit, not admitted
            (1, 101, 10, "2099-12-31 20:00:00", "2100-01-01 07:00:00"),
            (1, 102, 20, "2100-12-31 20:00:00", "2101-01-01 07:00:00"),   # index ED stay
            (2, 103, 30, "2100-05-31 20:00:00", "2100-06-01 07:00:00"),
        ]),
        "diagnosis": ("stay_id,icd_version,icd_code", [(100, 10, "R07"), (102, 10, "I10")]),
        "medrecon": ("stay_id,name,gsn", [(100, "Metoprolol", "000555"), (102, "Aspirin", "000777"),
                                          (103, "Other", "000001")]),
        "pyxis": ("stay_id,name,gsn", [(102, "Heparin", "000888"), (103, "Aspirin", "000777"),
                                       (103, "Aspirin", "000777"), (103, "Aspirin EC", "000777")]),
    },
}


@pytest.fixture
def mimic_dirs(tmp_path):
    """Directories (hosp, icu, ed) of a tiny fake MIMIC written as gzipped CSVs."""
    directories = {}
    for silo, tables in MIMIC_FIXTURE.items():
        directories[silo] = tmp_path / silo
        directories[silo].mkdir()
        for table, (header, rows) in tables.items():
            lines = [header] + [",".join(str(value) for value in row) for row in rows]
            with gzip.open(directories[silo] / f"{table}.csv.gz", "wt") as file:
                file.write("\n".join(lines) + "\n")
    return directories
