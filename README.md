# EM experiment

This project reproduces the demographic + clinical record-linkage experiments described in the code under `src/`.

The repository expects restricted external data to be linked into the project root before running preprocessing or experiments.

## Data layout

The code reads these paths, relative to the repository root:

- `mimiciv/3.1/hosp/`
- `mimiciv/3.1/icu/`
- `mimic-iv-ed-2.2/ed/`
- `VR_Snapshot_20160315/VR_Snapshot_20160315.txt`
- `VR_Snapshot_20260303/VR_Snapshot_20260303.txt`
- `data/` for derived data and experiment outputs

The MIMIC files are restricted and must be downloaded under a PhysioNet DUA. The voter snapshot files are the demographic source data used for the linkage experiment. The project does not include either set in git; `.gitignore` excludes them intentionally.

### Linking the restricted and demographic data

If you have the data locally, link them into this repo instead of copying them.

On Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force -Path .\mimiciv\3.1 | Out-Null
New-Item -ItemType SymbolicLink -Path .\mimiciv\3.1\hosp -Target "C:\path\to\mimiciv\3.1\hosp"
New-Item -ItemType SymbolicLink -Path .\mimiciv\3.1\icu -Target "C:\path\to\mimiciv\3.1\icu"
New-Item -ItemType SymbolicLink -Path .\mimic-iv-ed-2.2 -Target "C:\path\to\mimic-iv-ed-2.2"
New-Item -ItemType SymbolicLink -Path .\VR_Snapshot_20160315 -Target "C:\path\to\VR_Snapshot_20160315"
New-Item -ItemType SymbolicLink -Path .\VR_Snapshot_20260303 -Target "C:\path\to\VR_Snapshot_20260303"
```

Or use `mklink /D` from `cmd.exe` if you prefer junctions/symlinks:

```cmd
mklink /D "C:\path\to\em-experiment\mimiciv\3.1\hosp" "C:\path\to\mimiciv\3.1\hosp"
mklink /D "C:\path\to\em-experiment\mimiciv\3.1\icu" "C:\path\to\mimiciv\3.1\icu"
mklink /D "C:\path\to\em-experiment\mimic-iv-ed-2.2" "C:\path\to\mimic-iv-ed-2.2"
mklink /D "C:\path\to\em-experiment\VR_Snapshot_20160315" "C:\path\to\VR_Snapshot_20160315"
mklink /D "C:\path\to\em-experiment\VR_Snapshot_20260303" "C:\path\to\VR_Snapshot_20260303"
```

The repo also expects a local `data/` directory for generated parquet/duckdb outputs. Create it if needed:

```powershell
New-Item -ItemType Directory -Force -Path .\data | Out-Null
```

## Install dependencies

This project targets Python 3.12 (`pyproject.toml`). The recommended setup is `uv`:

```powershell
uv sync --group dev
```

If you do not use `uv`, install from the project root with:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

## Run the experiments

Set the cohort before running any experiment. Supported values are `k1` and `ked`:

```powershell
$env:EM_COHORT = "k1"
```

Optional: run only a subset for faster debugging:

```powershell
$env:EM_SIZE = "18000"
```

Run a baseline experiment:

```powershell
uv run python -m src.experiments.demographic_baseline
```

Run a clinical-only baseline:

```powershell
uv run python -m src.experiments.clinical_only
```

Compare two stored runs on the same candidate pairs:

```powershell
uv run python -m src.experiments.compare --base D --new D+C --pool natural
```

Run the test suite:

```powershell
uv run pytest
```

## Notes

- The project writes derived outputs under `data/` and generated results under `results/`.
- Do not commit the restricted MIMIC directories or any downloaded credentialed data.
- If the data are missing or the expected folders are not linked, preprocessing and experiments will fail with path errors.
