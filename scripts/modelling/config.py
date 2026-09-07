"""
Shared config for the phenology modelling phase (target = FIRSTDAY).

Paths resolve against PROJECT_ROOT so any module runs from anywhere.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

# config.py lives at <root>/scripts/modelling/config.py -> parents[2] == <root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------
FEATURE_TABLE  = PROJECT_ROOT / "data" / "feature_table.parquet"
SITE_DAILY_DIR = PROJECT_ROOT / "data" / "site_daily_climate"   # per-year parquet, FULL calendar year
NAO_CSV        = PROJECT_ROOT / "data" / "nao_djfm.csv"          # reproducible raw index (year, nao_djfm)

# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------
OUTPUT_DIR        = PROJECT_ROOT / "output"
MODELLING_FRAME   = OUTPUT_DIR / "modelling_frame.parquet"       # frame + fixed split assignment (reused by all models)
PREDICTIONS_PATH  = OUTPUT_DIR / "model_predictions.parquet"     # long: model,regime,keys,y_true,y_pred
METRICS_PATH      = OUTPUT_DIR / "model_metrics.csv"             # tidy metrics table
GDD_CALIB_PATH    = OUTPUT_DIR / "gdd_species_calibration.csv"   # per-species S* (and T_BASE)
REPORT_MD         = OUTPUT_DIR / "modelling_report.md"

# --------------------------------------------------------------------------
# Target & FIRSTDAY convention
# --------------------------------------------------------------------------
# FIRSTDAY is "days after 1 April" (can be negative). It is NOT a raw day-of-year.
# To compare against a GDD crossing day we convert to true DOY:
#     DOY = april1_doy(year) + FIRSTDAY        (april1_doy = 91 non-leap, 92 leap)
TARGET = "FIRSTDAY"


def april1_doy(year: int) -> int:
    """Day-of-year of 1 April for `year` (91 non-leap, 92 leap)."""
    return date(int(year), 4, 1).timetuple().tm_yday


def april1_doy_vec(years):
    """Vectorised april1_doy over an int array of years (91 non-leap, 92 leap)."""
    import numpy as np
    y = np.asarray(years).astype(int)
    leap = (y % 4 == 0) & ((y % 100 != 0) | (y % 400 == 0))
    return 91 + leap.astype(int)


def firstday_to_doy(firstday, year):
    return april1_doy(year) + firstday


def doy_to_firstday(doy, year):
    return doy - april1_doy(year)


# --------------------------------------------------------------------------
# Modelling frame filters
# --------------------------------------------------------------------------
# Climate/daily cache begin in 1976; earlier survey years carry no predictors.
YEAR_MIN = 1976
YEAR_MAX = 2024
# A valid FIRSTDAY is present and its implied DOY is a real calendar day.
DOY_MIN, DOY_MAX = 1, 366

# --------------------------------------------------------------------------
# Evaluation regimes
# --------------------------------------------------------------------------
# (a) Temporal hold-out: train YEAR <= TEMPORAL_CUTOFF, test YEAR > TEMPORAL_CUTOFF.
#     Default puts the last ~10 years in test (train 1976-2014, test 2015-2024).
TEMPORAL_CUTOFF = 2014
# (b) Grouped k-fold CV, groups = SITE_ID (a site never spans train & test).
CV_GROUP_COL = "SITE_ID"
CV_K = 5
CV_SEED = 42

# --------------------------------------------------------------------------
# GDD process baseline
# --------------------------------------------------------------------------
T_BASE = 5.0            # deg C base temperature; daily GDD = max(0, Tmean - T_BASE)
GDD_START_DOY = 1       # accumulate cumulative GDD from this day-of-year
FIT_TBASE = False       # if True, also fit T_BASE per species (2-parameter search)
TBASE_GRID = [0.0, 2.5, 5.0, 7.5, 10.0]     # candidate base temps when FIT_TBASE
SSTAR_N_CANDIDATES = 250                     # grid size for critical-sum S* search
# A candidate S* is only accepted if at least this fraction of a species' train
# events actually reach it (avoids degenerate "never crosses" optima).
SSTAR_MIN_CROSS_FRAC = 0.80
# Species need at least this many train events to get their own calibration.
GDD_MIN_TRAIN_EVENTS = 30

# --------------------------------------------------------------------------
# NAO source (Hurrell station-based DJFM, NCAR Climate Data Guide)
# --------------------------------------------------------------------------
NAO_URL = ("https://climatedataguide.ucar.edu/sites/default/files/"
           "2023-07/nao_station_djfm.txt")
NAO_CITATION = ("NAO Index Data provided by the Climate Analysis Section, NCAR, "
                "Boulder, USA, Hurrell (2003). Updated regularly.")
NAO_MISSING = -999.0

# --------------------------------------------------------------------------
# 2024 NAO gap fix (Part 1): NOAA CPC monthly NAO, rescaled onto the Hurrell scale
# --------------------------------------------------------------------------
# CPC monthly NAO index (updated to present). We build DJFM ourselves and rescale
# onto the Hurrell station scale via the overlap regression (the two products use
# DIFFERENT normalisations — never concatenate raw).
CPC_NAO_URL = ("https://www.cpc.ncep.noaa.gov/products/precip/CWlink/pna/"
               "norm.nao.monthly.b5001.current.ascii")
CPC_NAO_CSV = PROJECT_ROOT / "data" / "nao_djfm_cpc.csv"        # reproducible CPC DJFM series
CPC_CITATION = ("NAO monthly index, NOAA/NWS Climate Prediction Center, "
                "https://www.cpc.ncep.noaa.gov/ (accessed for the 2024 DJFM value).")
# Toggle: if True, mixed-effects NAO models DROP flight-year 2024 instead of using the
# CPC-rescaled value. The analysis is run BOTH ways and the report states if it matters.
NAO_EXCLUDE_2024 = False

# --------------------------------------------------------------------------
# Mixed-effects (RQ1) outputs & parameters
# --------------------------------------------------------------------------
VOLTINISM_CSV      = PROJECT_ROOT / "data" / "species_voltinism.csv"
MODELS_DIR         = OUTPUT_DIR / "models"                       # diagnostic figs + coef CSVs
ME_REPORT_MD       = OUTPUT_DIR / "mixedeffects_report.md"
ME_COEF_CSV        = OUTPUT_DIR / "models" / "mixedeffects_coefficients.csv"
ME_PRED_PATH       = OUTPUT_DIR / "models" / "mixedeffects_predictions.parquet"
# Voltinism factor: primary = 3-level (ref univoltine); sensitivity = binary uni vs multi.
VOLT_REF = "univoltine"
VOLT_LEVELS = ["univoltine", "multivoltine", "variable"]
VOLT_DROP = {"MANUAL_REVIEW"}          # the Thymelicus lineola/sylvestris aggregate
# MixedLM is attempted first; if it exceeds this wall-time (s) or fails to converge, the
# report leads with the FE-OLS + cluster-robust route (both are always reported).
MIXEDLM_MAXITER = 100
MIXEDLM_TIME_BUDGET_S = 600

# --------------------------------------------------------------------------
# Machine-learning rung (RQ2 / RQ3) — leakage-safe feature families
# --------------------------------------------------------------------------
ML_REPORT_MD  = OUTPUT_DIR / "ml_report.md"
ML_SEED       = 42

# Predictor families (all pre-flight: within Dec(y-1) -> 31 May(y)). Grouped so the NAO
# block can be toggled off (RQ3) and so SHAP importance can be summed within a family.
_MONTHS = ["M12P", "M01", "M02", "M03", "M04", "M05"]
_SEASONS = ["DJF", "MAM"]
ML_FEATURES = {
    "temp": ([f"{p}_{m}" for m in _MONTHS for p in ("TMAX", "TMIN", "TMEAN")]
             + [f"{p}_{s}" for s in _SEASONS for p in ("TMAX", "TMIN", "TMEAN")]
             + ["GDD"]),
    "rain": [f"RAIN_{m}" for m in _MONTHS] + [f"RAIN_{s}" for s in _SEASONS],
    "nao":  ["NAO_DJFM", "NAO_DJFM_LAG1"],           # toggleable block for RQ3
    "coords": ["EASTING", "NORTHING", "LENGTH_M"],   # spatial attrs (no site dummies)
}
ML_SPECIES_COL = "SPECIES_NAME"                       # encoded per-model (ordinal / one-hot)

# Columns that must NEVER enter X (leakage: outcomes / derived / abundance / raw year).
ML_FORBIDDEN = (["FIRSTDAY", "LASTDAY", "PEAKDAY", "PEAKCOUNT", "MEAN_FLIGHT_DATE",
                 "FLIGHTPERIOD_SD", "FLIGHTPERIOD_RANGE", "DOY", "YEAR",
                 "SITE_INDEX", "SITE_INDEX_AMBIGUOUS_CONFLICT",
                 "SITE_INDEX_INSUFFICIENT_MONITORING"]
                + ["N_YRS_SURVEYED", "FIRST_YEAR_SURVEYED", "LAST_YEAR_SURVEYED"])
# (TREND_* and NATIONAL_* are also forbidden — matched by prefix in the assertion.)
ML_FORBIDDEN_PREFIXES = ("TREND_", "NATIONAL_", "FLAG_")

# HistGradientBoosting (primary) — handles NaN natively, native categorical for species.
HGB_PARAMS = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=63,
                  min_samples_leaf=100, l2_regularization=1.0,
                  early_stopping=True, validation_fraction=0.1,
                  n_iter_no_change=20, random_state=ML_SEED)
# RandomForest (comparator) — needs complete X (train-only median impute). Modest config.
RF_PARAMS = dict(n_estimators=200, min_samples_leaf=25, max_features=0.5,
                 max_samples=0.5, n_jobs=-1, random_state=ML_SEED)

SHAP_SAMPLE = 15_000            # test-set rows for TreeSHAP (tractability)
SHAP_BG_SAMPLE = 1_000         # background sample for the explainer

EXPECTED_ROWS = 648_788   # feature_table row count that merges must preserve
FRAME_ROWS = 620_711      # modelling_frame row count the ML models must preserve
