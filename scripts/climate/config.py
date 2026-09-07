"""Shared config for the HadUK-Grid climate pipeline (download + merge stages).

Paths resolve against PROJECT_ROOT so scripts run from anywhere.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths (relative to project root)
# --------------------------------------------------------------------------
# config.py lives at <root>/scripts/climate/config.py -> parents[2] == <root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Cleaned UKBMS analysis table (grain: SITE_ID x SPECIES_NAME x YEAR).
CLEAN_TABLE = PROJECT_ROOT / "output" / "ukbms_site_species_year_clean.parquet"

DATA_DIR       = PROJECT_ROOT / "data"
HADUK_DIR      = DATA_DIR / "haduk"                       # per-var netCDF live here
MANIFEST_CSV   = HADUK_DIR / "manifest.csv"               # download manifest
SITE_DAILY_DIR = DATA_DIR / "site_daily_climate"         # cached per-site daily series (parquet, partitioned by year)
FEATURE_TABLE  = DATA_DIR / "feature_table.parquet"       # final merged site x species x year table
REPORT_MD      = PROJECT_ROOT / "climate_merge_report.md"

# --------------------------------------------------------------------------
# HadUK-Grid dataset selection
# --------------------------------------------------------------------------
HADUK_VERSION = "v1.3.1.ceda"
RESOLUTION    = "5km"
VARIABLES     = ["tasmax", "tasmin", "rainfall"]          # daily
FREQ          = "day"

# One base URL per variable (the dated version sub-folder is DISCOVERED at runtime).
DAP_HOST = "https://dap.ceda.ac.uk"
BASE_URLS = {
    var: f"{DAP_HOST}/badc/ukmo-hadobs/data/insitu/MOHC/HadOBS/HadUK-Grid/"
         f"{HADUK_VERSION}/{RESOLUTION}/{var}/{FREQ}/"
    for var in VARIABLES
}

# Filename pattern: {var}_hadukgrid_uk_5km_day_YYYYMMDD-YYYYMMDD.nc (one file per month)
FILENAME_RE = r"{var}_hadukgrid_uk_{res}_day_(\d{{8}})-(\d{{8}})\.nc"

# Optional manual override if directory listing is blocked: map var -> "v20YYMMDD"
# (leave empty to auto-discover). Can also be supplied via env HADUK_VDIR_<VAR>.
VERSION_DIR_OVERRIDE: dict[str, str] = {}

# --------------------------------------------------------------------------
# Temporal filter
# --------------------------------------------------------------------------
YEAR_MIN = 1976
YEAR_MAX = 2024

# --------------------------------------------------------------------------
# Expected netCDF structure (confirmed against a real file before use)
# --------------------------------------------------------------------------
# BNG projected coordinates on the HadUK-Grid 5km product.
X_COORD = "projection_x_coordinate"
Y_COORD = "projection_y_coordinate"
# Data variable name inside each file == the variable folder name (tasmax/tasmin/rainfall).

# --------------------------------------------------------------------------
# Growing-degree-day / feature parameters
# --------------------------------------------------------------------------
T_BASE = 5.0          # deg C base temperature for GDD = max(0, Tmean - T_BASE)
GDD_START_DOY = 1     # accumulate GDD from this day-of-year

# British National Grid plausibility bounds (EPSG:27700), metres.
# Used to identify sites whose coordinates are NOT true BNG (NI Irish Grid,
# Channel Islands / local grids, negative northings, etc.).
BNG_EASTING_MIN, BNG_EASTING_MAX = 0, 700_000
BNG_NORTHING_MIN, BNG_NORTHING_MAX = 0, 1_300_000

# --------------------------------------------------------------------------
# Networking politeness / robustness
# --------------------------------------------------------------------------
REQUEST_TIMEOUT = 120        # seconds per request
MAX_RETRIES = 4
BACKOFF_BASE = 2.0           # seconds; exponential backoff base
POLITE_DELAY = 0.3           # seconds between file downloads


def get_token() -> str:
    """Read the CEDA bearer token from CEDA_TOKEN (username/password never used)."""
    token = os.environ.get("CEDA_TOKEN", "").strip()
    if not token:
        raise SystemExit(
            "ERROR: CEDA_TOKEN is not set.\n"
            "  Generate a token at https://services.ceda.ac.uk/ and export it:\n"
            "    export CEDA_TOKEN='<your-token>'\n"
            "  Then re-run. The pipeline never uses your username/password."
        )
    return token


def auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {get_token()}"}
