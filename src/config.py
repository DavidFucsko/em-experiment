"""Data paths and global settings of the experiments."""
import os
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

MIMIC_HOSP = ROOT / "mimiciv" / "3.1" / "hosp"
MIMIC_ICU = ROOT / "mimiciv" / "3.1" / "icu"
MIMIC_ED = ROOT / "mimic-iv-ed-2.2" / "ed"

# Cohort, set with EM_COHORT:
#   k1   index = the patient's last admission with an ED stay and an ICU stay
#   ked  index = the patient's last admission with an ED stay (no ICU requirement)
COHORTS = ("k1", "ked")
COHORT = os.environ.get("EM_COHORT", "k1")
if COHORT not in COHORTS:
    raise ValueError(f"EM_COHORT must be one of {COHORTS}, not '{COHORT}'")

# EM_SIZE runs the experiments on a random subset of the cohort (K-ED has 2.3 billion pairs). The subsets are nested
# and stored under RUN_LABEL. The preprocessing always covers the whole cohort.
_size_setting = os.environ.get("EM_SIZE", "")
SIZE = int(_size_setting) if _size_setting else None
RUN_LABEL = COHORT if SIZE is None else f"{COHORT}_{SIZE}"

DATA = ROOT / "data"
SPLIT = DATA / f"split_{COHORT}"   # temporal split: clinical events per side, patients, attributes, code titles
VOTER = DATA / "voter"             # cleaned voter snapshots, linked persons, voters assigned to patients

VOTER_SNAPSHOTS = ("20160315", "20260303")  # older (side A), newer (side B); raw files at ROOT/VR_Snapshot_<date>/

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
