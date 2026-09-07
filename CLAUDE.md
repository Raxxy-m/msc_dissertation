# CLAUDE.md

Guidance for working in this repository. This is a **dissertation data-science project** (Python) analysing UK butterfly phenology and abundance against climate.

## What this project is

We combine five **UKBMS 2024** downloads (UK Butterfly Monitoring Scheme) into one cleaned analysis table, then join **HadUK-Grid** climate data to produce a modelling feature table. The scientific question is how winter/spring climate relates to butterfly phenology (flight timing) and abundance.

The core grain of every derived table is **one row per `SITE_ID` × `SPECIES_NAME` × `YEAR`** (648,788 rows).

## Environment

- Python venv at `.venv/` (Python 3.11). Use `.venv/bin/python`.
- Key libs: pandas, numpy, pyarrow, xarray, netCDF4, dask, requests.
- Not a git repository.

## Folder structure

```
Dissertation/
├── phenology/           ┐
├── site indices/        │  Raw UKBMS 2024 downloads. Each is an RO-Crate:
├── site location/       │  data/<name>2024.csv + readme.html + ro-crate-metadata.json
├── speciesTrends/       │  + supporting-documents/ (source docx with column definitions)
├── collated indices/    ┘
│
├── scripts/
│   ├── clean_pipeline.py        # Stage 1: merge the 5 UKBMS CSVs -> cleaned table + QC report
│   ├── climate/
│   │   ├── config.py            # Shared CONFIG: paths, HadUK-Grid selection, GDD params
│   │   ├── download_haduk.py    # Stage 2a: download HadUK-Grid netCDF from CEDA
│   │   └── merge_climate.py     # Stage 2b: extract per-site climate, build features, join
│   ├── modelling/               # Stage 3: FIRSTDAY modelling (target = first appearance)
│   │   ├── config.py            # CONFIG: splits, GDD/NAO params, FIRSTDAY<->DOY helpers
│   │   ├── add_nao.py           # Part 1: join winter DJFM NAO onto feature_table
│   │   ├── harness.py           # Part 2: modelling frame + fixed splits + naive baselines
│   │   ├── gdd_baseline.py      # Part 3: GDD process baseline (per-species critical sum S*)
│   │   ├── report.py            # emits output/modelling_report.md from artefacts
│   │   ├── fix_nao_2024.py      # RQ1 Part 1: fill 2024 NAO (CPC-rescaled) + NAO_SOURCE
│   │   ├── mixed_effects.py     # RQ1 Part 2/3: NAO x voltinism mixed-effects models
│   │   ├── me_report.py         # emits output/mixedeffects_report.md
│   │   ├── ml_baseline.py       # RQ2/RQ3: HGB + RF (leakage-safe) via harness + with/without NAO
│   │   ├── ml_report.py         # emits output/ml_report.md (comparison ladder, SHAP)
│   │   ├── report_figures.py    # Stage 5: curated body/appendix figures -> output/figures/*.{png,pdf}
│   │   └── run_all.py           # orchestrates Stage 3 + RQ1 + RQ2/RQ3 rungs end-to-end
│   ├── eda.py                   # Stage 4a: EDA of feature_table (read-only) -> output/eda/*.png
│   ├── voltinism.py             # Stage 4b: species voltinism lookup from Cook traits -> data/species_voltinism.csv
│   └── eda_report.py            # emits output/eda_report.md from EDA + voltinism artefacts
│
├── traits/                      # Cook et al. (2021, 2024 update) traits RO-Crate (user-downloaded)
│   └── data/ecological_traits_2024.csv   # two-row header; one-hot voltinism columns
│
├── data/
│   ├── haduk/                   # Downloaded HadUK-Grid netCDF (tasmax/tasmin/rainfall) + manifest.csv
│   ├── site_daily_climate/      # Cached per-site daily climate, parquet partitioned by year=YYYY
│   └── feature_table.parquet    # FINAL modelling table (90 cols): clean UKBMS + climate features
│
├── output/
│   ├── ukbms_site_species_year_clean.parquet   # Stage 1 output — cleaned UKBMS table (57 cols)
│   ├── ukbms_site_species_year_clean.csv       # same, CSV
│   ├── data_dictionary.md      # Column-by-column dictionary for the clean table
│   └── qc_report.md            # Every QC check, flag, and filter decision from Stage 1
│
├── climate_merge_report.md     # QC summary for the climate merge (Stage 2)
└── exploration.ipynb           # EDA notebook (survey coverage, phenology, abundance, maps)
```

## The pipeline (run order)

1. **`scripts/clean_pipeline.py`** — merges the five UKBMS downloads into
   `output/ukbms_site_species_year_clean.parquet` and writes `output/qc_report.md`.
   Run from anywhere: `.venv/bin/python scripts/clean_pipeline.py`. All tunable knobs
   live in the `CONFIG` dict at the top. Note: `phenology` and `site_location` CSVs are
   **cp1252-encoded** (Windows apostrophes), not UTF-8 — handled in CONFIG.
2. **`scripts/climate/download_haduk.py`** — downloads HadUK-Grid v1.3.1 5km daily
   netCDF (tasmax, tasmin, rainfall) into `data/haduk/`. Requires a CEDA OAuth2 Bearer
   token in env var **`CEDA_TOKEN`** (never username/password). Idempotent via
   `data/haduk/manifest.csv`.
3. **`scripts/climate/merge_climate.py`** — snaps each site to the nearest HadUK-Grid
   cell, extracts daily series (cached in `data/site_daily_climate/`), builds monthly /
   seasonal / GDD features, and joins them onto the clean table to produce
   `data/feature_table.parquet`.
4. **`scripts/modelling/run_all.py`** — Stage 3 modelling of `FIRSTDAY`. Adds winter DJFM
   NAO (`NAO_DJFM`, `NAO_DJFM_LAG1`), builds a shared eval harness with two regimes
   (temporal hold-out train≤2014/test>2014; site-grouped 5-fold CV), naive baselines
   (species mean, site×species mean, persistence), and a GDD process baseline (per-species
   critical-sum S* calibrated on train MAE). Writes `output/modelling_report.md`,
   `output/model_metrics.csv`, `output/model_predictions.parquet`. Run modules individually
   with `.venv/bin/python -m scripts.modelling.<mod>`.
5. **`scripts/eda.py`** + **`scripts/eda_report.py`** — Stage 4a EDA of `feature_table`
   (READ ONLY; never modifies it). 9 figures → `output/eda/*.png`, report →
   `output/eda_report.md`. Covers survey coverage, records/species, a BNG site map (the
   non-GB mislocation gotcha is visible), FIRSTDAY distributions, missingness, phenology
   trend (≈−1.7 days/decade), FIRSTDAY↔spring-temp, climate multicollinearity, and winter
   NAO relationships.
6. **`scripts/voltinism.py`** — Stage 4b per-species voltinism lookup (univoltine /
   multivoltine / variable) from Cook et al. traits (`traits/data/ecological_traits_2024.csv`)
   → `data/species_voltinism.csv`. 59/60 classified; `Thymelicus lineola/sylvestris`
   (recorder aggregate) left MANUAL_REVIEW. Middleton-Welling cross-check optional (drop the
   xlsx in `data/`). Needed for RQ1 and the mixed-effects rung.
7. **`scripts/modelling/{fix_nao_2024,mixed_effects,me_report}.py`** — Stage 3b RQ1
   mixed-effects rung. `fix_nao_2024` fills the 2024 winter-NAO gap by rescaling NOAA CPC
   DJFM onto the Hurrell scale via the overlap regression (adds `NAO_SOURCE`; toggle to
   exclude 2024 instead). `mixed_effects` fits `FIRSTDAY ~ NAO_DJFM×VOLTINISM + YEAR_c + species FE + (1|SITE_ID)` via MixedLM (ICC) **and** FE-OLS with year-clustered SEs (the
   honest inference route, since NAO varies only annually) **and** a two-stage cross-check;
   plus climate-only / both / binary specs, VIFs, diagnostics, and shared-harness eval.
   → `output/mixedeffects_report.md`, `output/models/*`.
8. **`scripts/modelling/{ml_baseline,ml_report}.py`** — Stage 3c RQ2/RQ3 ML rung.
   HistGradientBoosting (primary, native NaN, one-hot species for reliable TreeSHAP) +
   RandomForest (comparator, train-only median impute) via the shared harness. Strict
   leakage guard (no outcomes/abundance/TREND_*/NATIONAL_*/raw YEAR). Each fit with & without
   the NAO block (RQ3). → `output/ml_report.md`, SHAP figs in `output/models/`. Appends
   ml_hgb/ml_rf (+ no-NAO) to the shared metrics/predictions.
9. **`scripts/modelling/report_figures.py`** — Stage 5 curated dissertation figures (READ
   ONLY on all artefacts; never recomputes the pipeline except to re-derive ME by-class
   slopes / MixedLM ICC for exact CIs). 9 methodology + results figures →
   `output/figures/*.{png@300dpi,pdf}` + catalogue `output/report_figures_report.md` (BODY
   vs APPENDIX). Every number computed from artefacts; Okabe-Ito palette; matches EDA style.
   Copy `output/figures/*` into the LaTeX `figures/` folder.

Note: `FIRSTDAY` is "days after 1 April" (not a raw day-of-year); it can be negative.
RQ1 finding: NAO–phenology slope is negative but weak/non-significant over 1976–2024 with
honest year-clustered SEs, no voltinism difference; NAO's effect flips sign once spring
temperature is included (it acts *through* local temperature — sets up RQ3).
RQ2/RQ3 finding: ML (HGB) is best out-of-sample (temporal MAE ~21.2, grouped-CV ~20.3-20.7),
beats climatology and GDD, and does NOT inherit GDD's warm-year bias (2024 bias −4 d vs −26 d).
Removing NAO changes MAE by hundredths of a day → NAO adds no predictive signal beyond local
temperature (RQ3, both ML and mixed-effects agree). SHAP: species ~69%, temperature ~18%, NAO ~2%.

## Key domain facts & gotchas

- **Coordinates**: of 3,974 sites, only **3,496 are usable BNG (EPSG:27700)**. 368 have
  no coordinates. Northern Ireland uses the **Irish Grid**, whose values fall *inside* the
  numeric BNG range — a bounds check alone lets them silently mis-locate onto GB cells.
  Exclude non-GB sites by `COUNTRY` (Northern Ireland, Channel Islands, Isle of Man);
  Channel Islands have negative northings. 110 non-BNG sites are excluded from the climate join.
- **Leakage window**: climate predictors use ONLY 1 Dec (year−1) through 31 May (year) —
  the winter/spring preceding the flight period. No post-flight climate. Monthly features are
  suffixed `_M12P` (prev Dec), `_M01`..`_M05`; seasonal `_DJF`/`_MAM`; plus `GDD`.
  Caveat: very early species may fly inside the spring window (documented future refinement).
- **GDD**: sum of max(0, Tmean − `T_BASE`) from `GDD_START_DOY` through 31 May. Defaults
  `T_BASE=5.0`, `GDD_START_DOY=1`.
- **Sentinels**: `SITE_INDEX = -2` means insufficient monitoring (flagged, not a real value).
- **`TREND_*`** columns are species-level trend stats (from speciesTrends); **`NATIONAL_*`**
  are national collated indices (from collatedindices). See each download's
  `supporting-documents/` docx for authoritative definitions.

## Conventions

- Paths in scripts resolve against a computed project root (`Path(__file__).parents[...]`),
  so scripts run correctly from any working directory. Keep this pattern.
- Centralise tunable parameters in the `CONFIG` dict (clean_pipeline) / `config.py` (climate)
  rather than hardcoding inline.
- Every stage writes a Markdown QC/report file — keep them current when changing logic.
- Prefer parquet for derived tables; CSV mirrors are for external/manual inspection only.
