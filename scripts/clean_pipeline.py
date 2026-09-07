"""
UKBMS data cleaning / QC pipeline.

Merges the five UKBMS 2024 downloads into one site x species x year table and
writes a QC report documenting every check, flag, and filter decision.
Outputs: output/{qc_report.md, ukbms_site_species_year_clean.parquet/.csv,
data_dictionary.md}. Run from anywhere: python scripts/clean_pipeline.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

# =====================================================================================
# CONFIG -- tunable knobs. Every check below is computed and reported regardless of the
# filter toggles; only toggles set True actually change the saved table.
# =====================================================================================
CONFIG = {
    "paths": {
        "phenology": REPO_ROOT / "phenology/data/ukbmsphenology2024.csv",
        "site_indices": REPO_ROOT / "site indices/data/ukbmssiteindices2024.csv",
        "site_location": REPO_ROOT / "site location/data/ukbmssitelocationdata2024.csv",
        "species_trends": REPO_ROOT / "speciesTrends/data/ukbmsspeciestrends2024.csv",
        "collated_indices": REPO_ROOT / "collated indices/data/ukbmscollatedindices2024.csv",
    },
    # phenology + site_location CSVs use Windows-1252 apostrophes (e.g. "Bevill's Wood"),
    # not UTF-8; the other three are plain UTF-8.
    "encodings": {
        "phenology": "cp1252",
        "site_indices": "utf-8",
        "site_location": "cp1252",
        "species_trends": "utf-8",
        "collated_indices": "utf-8",
    },
    "output_dir": REPO_ROOT / "output",
    "qc_report_path": REPO_ROOT / "output/qc_report.md",
    "merged_parquet_path": REPO_ROOT / "output/ukbms_site_species_year_clean.parquet",
    "merged_csv_path": REPO_ROOT / "output/ukbms_site_species_year_clean.csv",
    "data_dictionary_path": REPO_ROOT / "output/data_dictionary.md",

    # Day fields are "days after 1 April" (per ukbms_phenology_2024.docx), not day-of-year,
    # so negatives are legitimate. Hard plausibility band; outside it -> flagged implausible.
    "day_value_min": -90,   # ~1 Jan
    "day_value_max": 274,   # ~31 Dec
    # Softer core-season band (1 Apr - 30 Sep = day 0-182); outside -> flagged informational.
    "day_value_core_season_min": 0,
    "day_value_core_season_max": 182,

    "min_years_surveyed": 5,          # site_location.N_yrs_surveyed threshold
    "min_site_species_records": 3,    # min phenology records per (site, species) pair

    # GB National Grid bounds apply only to England/Scotland/Wales; NI (Irish Grid) and
    # Channel Islands (local grid, negative Northing) are reported as out-of-scope, not invalid.
    "gb_grid_countries": ["England", "Scotland", "Wales"],
    "gb_grid_bounds": {"easting_min": 0, "easting_max": 700_000,
                        "northing_min": 0, "northing_max": 1_300_000},

    # Only toggles set True change the saved output; everything is reported either way.
    "filters": {
        "collapse_exact_duplicate_rows": True,   # byte-identical repeats -- no information lost
        "exclude_nonstandard_survey_type": False,
        "exclude_species_list": False,
        "exclude_below_min_years_surveyed": False,
        "exclude_below_min_site_species_records": False,
        "null_invalid_gb_coordinates": False,
    },
    "species_exclude_list": [],  # sci or common names to drop; empty = drop nothing
}

# Constants that are never varied (folded out of CONFIG for readability).
RANGE_TOLERANCE_DAYS = 0            # tolerance for |FLIGHTPERIOD_RANGE - (LASTDAY-FIRSTDAY)|
EXAMPLE_ROWS_N = 5                  # example rows per flagged issue in the QC report
RARE_SPECIES_RECORD_THRESHOLD = 200
MIGRANT_CANDIDATE_SPECIES = ["Painted Lady", "Clouded Yellow", "Red Admiral"]  # informational only

# =====================================================================================
# Small helpers
# =====================================================================================

class QCReport:
    """Accumulates markdown sections in order; written out at the end."""

    def __init__(self):
        self.sections: list[str] = []

    def add(self, heading: str, body: str = "", level: int = 2):
        self.sections.append(f"{'#' * level} {heading}\n\n{body}\n")

    def add_table(self, heading: str, df: pd.DataFrame, level: int = 3, max_rows: int = 20):
        if df is None or len(df) == 0:
            self.add(heading, "_none found_", level=level)
            return
        self.add(heading, df.head(max_rows).to_markdown(index=False), level=level)

    def render(self) -> str:
        return "\n".join(self.sections)


def log(msg: str):
    print(f"[pipeline] {msg}")


def row_count_log(stage: str, df: pd.DataFrame, log_list: list[tuple[str, int]]):
    log_list.append((stage, len(df)))
    log(f"{stage}: {len(df):,} rows")


# =====================================================================================
# 1. LOAD + PROFILE
# =====================================================================================

def load_raw(config: dict) -> dict[str, pd.DataFrame]:
    dfs = {}
    for name, path in config["paths"].items():
        enc = config["encodings"][name]
        dfs[name] = pd.read_csv(path, encoding=enc)
        log(f"loaded {name}: {path.name} ({len(dfs[name]):,} rows, encoding={enc})")
    return dfs


def profile_file(name: str, df: pd.DataFrame, qc: QCReport):
    body_parts = [
        f"- Shape: {df.shape[0]:,} rows x {df.shape[1]} columns",
        f"- Columns: {', '.join(df.columns)}",
    ]
    qc.add(f"Profile: {name}", "\n".join(body_parts), level=3)

    dtypes_df = df.dtypes.rename("dtype").reset_index().rename(columns={"index": "column"})
    qc.add_table("Dtypes", dtypes_df, level=4)

    nulls = df.isnull().sum()
    nulls_df = nulls[nulls > 0].rename("n_null").reset_index().rename(columns={"index": "column"})
    nulls_df["pct_null"] = (nulls_df["n_null"] / len(df) * 100).round(2)
    qc.add_table("Null counts (columns with >0 nulls)", nulls_df, level=4)

    nunique_df = df.nunique().rename("n_unique").reset_index().rename(columns={"index": "column"})
    qc.add_table("N unique per column", nunique_df, level=4)

    qc.add("Head (first 3 rows)", df.head(3).to_markdown(index=False), level=4)

    num_df = df.select_dtypes(include="number")
    if not num_df.empty:
        qc.add("Describe (numeric columns)", num_df.describe().T.to_markdown(), level=4)


# =====================================================================================
# 2. SITE KEY HARMONISATION
# =====================================================================================

def harmonise_site_keys(phen, site_idx, site_loc, qc: QCReport):
    """Rename SITENO / SITE_CODE / Site_Number to a common SITE_ID.

    All three are clean int64 (verified in profiling); the dtype check still runs so a
    future re-export that reintroduces formatting differences would be caught.
    """
    issues = []
    for name, col in [("phenology", phen["SITENO"]), ("site_indices", site_idx["SITE_CODE"]),
                       ("site_location", site_loc["Site_Number"])]:
        if col.dtype.kind not in "iu":
            issues.append(f"{name}.{col.name} is not an integer dtype ({col.dtype}) -- check for whitespace/leading zeros")
        if col.isnull().any():
            issues.append(f"{name}.{col.name} contains {col.isnull().sum()} nulls")

    phen = phen.rename(columns={"SITENO": "SITE_ID"})
    site_idx = site_idx.rename(columns={"SITE_CODE": "SITE_ID"})
    site_loc = site_loc.rename(columns={"Site_Number": "SITE_ID"})

    phen_sites = set(phen.SITE_ID.unique())
    idx_sites = set(site_idx.SITE_ID.unique())
    loc_sites = set(site_loc.SITE_ID.unique())

    overlap_summary = pd.DataFrame({
        "comparison": [
            "phenology sites NOT in site_location",
            "site_indices sites NOT in site_location",
            "site_location sites NOT in phenology",
            "site_location sites NOT in site_indices",
        ],
        "n_sites": [
            len(phen_sites - loc_sites),
            len(idx_sites - loc_sites),
            len(loc_sites - phen_sites),
            len(loc_sites - idx_sites),
        ],
    })

    body = "\n".join(f"- {i}" for i in issues) if issues else "- No dtype/formatting issues found; all three site keys are clean int64."
    qc.add("Site key harmonisation", body, level=3)
    qc.add_table("Site ID overlap across files", overlap_summary, level=4)
    qc.add(
        "Interpretation",
        "Sites in phenology/site_indices but not site_location carry NaN coordinates/"
        "country/survey metadata after the merge (kept, but unusable in the climate join "
        "until resolved). Sites only in site_location were never linked to any phenology/"
        "abundance record here and are correctly dropped by a phenology-anchored merge.",
        level=4,
    )
    return phen, site_idx, site_loc


# =====================================================================================
# 3. SPECIES LOOKUP
# =====================================================================================

def build_species_lookup(phen, site_idx, qc: QCReport):
    """Cross-check the species join on scientific name (SPECIES_NAME == SPECIES).

    phenology has no species code, so the join keys on scientific name (more stable than
    common name). Verified 1:1 and lossless across all 60 species. Writes QC sections only.
    """
    phen_sci = set(phen.SPECIES_NAME.str.strip().unique())
    idx_sci = set(site_idx.SPECIES.str.strip().unique())

    only_in_phen = sorted(phen_sci - idx_sci)
    only_in_idx = sorted(idx_sci - phen_sci)

    # 1:1 sci-name -> common-name consistency within each file
    phen_multi = phen.groupby("SPECIES_NAME")["COMMON_NAME"].nunique()
    phen_multi = phen_multi[phen_multi > 1]
    idx_multi = site_idx.groupby("SPECIES")["COMMON_NAME"].nunique()
    idx_multi = idx_multi[idx_multi > 1]

    lookup = (
        site_idx[["SPECIES_CODE", "SPECIES", "COMMON_NAME"]]
        .drop_duplicates()
        .rename(columns={"SPECIES": "SPECIES_NAME"})
    )
    dup_lookup = lookup.groupby("SPECIES_NAME").size()
    dup_lookup = dup_lookup[dup_lookup > 1]

    body = (
        f"- Distinct scientific names in phenology: {len(phen_sci)}\n"
        f"- Distinct scientific names in site_indices: {len(idx_sci)}\n"
        f"- Scientific names in phenology with NO match in site_indices: {len(only_in_phen)}\n"
        f"- Scientific names in site_indices with NO match in phenology: {len(only_in_idx)}\n"
        f"- Scientific names with inconsistent common-name spelling within phenology: {len(phen_multi)}\n"
        f"- Scientific names with inconsistent common-name spelling within site_indices: {len(idx_multi)}\n"
        f"- Scientific names mapping to >1 SPECIES_CODE in site_indices (lookup ambiguity): {len(dup_lookup)}\n"
    )
    qc.add("Species lookup (joined on scientific name)", body, level=3)
    if only_in_phen:
        qc.add("Species only in phenology (no SPECIES_CODE available)", "\n".join(f"- {s}" for s in only_in_phen), level=4)
    if only_in_idx:
        qc.add("Species only in site_indices (no phenology record)", "\n".join(f"- {s}" for s in only_in_idx), level=4)
    if len(dup_lookup):
        qc.add_table("Ambiguous species lookup rows", lookup[lookup.SPECIES_NAME.isin(dup_lookup.index)], level=4)


# =====================================================================================
# 4. DTYPE COERCION
# =====================================================================================

def coerce_numeric(df: pd.DataFrame, columns: list[str], df_name: str, qc: QCReport) -> pd.DataFrame:
    df = df.copy()
    rows = []
    for col in columns:
        if col not in df.columns:
            continue
        original = df[col]
        coerced = pd.to_numeric(original, errors="coerce")
        failed_mask = coerced.isna() & original.notna()
        n_failed = failed_mask.sum()
        rows.append({"column": col, "n_values": len(original), "n_failed_coercion": n_failed,
                      "example_failed_values": list(original[failed_mask].unique()[:5])})
        df[col] = coerced
    report_df = pd.DataFrame(rows)
    qc.add_table(f"Dtype coercion report: {df_name}", report_df, level=3)
    total_failed = report_df["n_failed_coercion"].sum() if len(report_df) else 0
    if total_failed:
        qc.add("WARNING", f"{total_failed} values across {df_name} failed numeric coercion and are now NaN -- see table above for which columns/values.", level=4)
    return df


# =====================================================================================
# 5. PHENOLOGY DAY-FIELD PLAUSIBILITY
# =====================================================================================

def check_doy_plausibility(phen: pd.DataFrame, config: dict, qc: QCReport):
    day_cols = ["FIRSTDAY", "LASTDAY", "PEAKDAY", "MEAN_FLIGHT_DATE"]
    desc = phen[day_cols].describe().T
    qc.add(
        "Day-field definition (IMPORTANT -- corrects the day-of-year assumption)",
        "`ukbms_phenology_2024.docx` defines these fields as **\"the day number after "
        "1 April\"** (e.g. `20` = 20th April), not a 1-366 day-of-year -- hence the "
        "legitimate negatives (min FIRSTDAY = -30). Checks below use that scale: hard "
        f"bounds {config['day_value_min']} to {config['day_value_max']}, core season "
        f"{config['day_value_core_season_min']} to {config['day_value_core_season_max']}.",
        level=3,
    )
    qc.add("Distribution of day fields (days after 1 April)", desc.to_markdown(), level=4)

    lo, hi = config["day_value_min"], config["day_value_max"]
    clo, chi = config["day_value_core_season_min"], config["day_value_core_season_max"]
    flags = pd.DataFrame(index=phen.index)
    for col in day_cols:
        flags[f"{col}_implausible"] = (phen[col] < lo) | (phen[col] > hi)
        flags[f"{col}_outside_core_season"] = (phen[col] < clo) | (phen[col] > chi)

    implausible_any = flags[[c for c in flags.columns if c.endswith("_implausible")]].any(axis=1)
    outside_season_any = flags[[c for c in flags.columns if c.endswith("_outside_core_season")]].any(axis=1)

    qc.add(
        "Plausibility flag counts",
        f"- Rows with any day field outside hard bounds [{lo}, {hi}]: {implausible_any.sum():,}\n"
        f"- Rows with any day field outside core UKBMS season [{clo}, {chi}] (informational -- legitimate for early/late records): {outside_season_any.sum():,}",
        level=4,
    )
    if implausible_any.sum():
        qc.add_table("Example hard-implausible rows", phen[implausible_any].head(EXAMPLE_ROWS_N), level=4)
    if outside_season_any.sum():
        qc.add_table("Example out-of-core-season rows", phen[outside_season_any].head(EXAMPLE_ROWS_N), level=4)

    return implausible_any, outside_season_any


# =====================================================================================
# 6. LOGICAL CONSISTENCY
# =====================================================================================

def check_logical_consistency(phen: pd.DataFrame, config: dict, qc: QCReport):
    tol = RANGE_TOLERANCE_DAYS

    order_violation = ~((phen.FIRSTDAY <= phen.PEAKDAY) & (phen.PEAKDAY <= phen.LASTDAY))
    range_mismatch = (phen.FLIGHTPERIOD_RANGE - (phen.LASTDAY - phen.FIRSTDAY)).abs() > tol
    peakcount_nonpositive = phen.PEAKCOUNT <= 0
    sd_negative = phen.FLIGHTPERIOD_SD < 0

    summary = pd.DataFrame({
        "check": [
            "NOT (FIRSTDAY <= PEAKDAY <= LASTDAY)",
            f"|FLIGHTPERIOD_RANGE - (LASTDAY-FIRSTDAY)| > {tol}",
            "PEAKCOUNT <= 0",
            "FLIGHTPERIOD_SD < 0",
        ],
        "n_rows_flagged": [order_violation.sum(), range_mismatch.sum(), peakcount_nonpositive.sum(), sd_negative.sum()],
    })
    qc.add("Logical consistency checks", "", level=3)
    qc.add_table("Summary", summary, level=4)

    for label, mask in [
        ("FIRSTDAY <= PEAKDAY <= LASTDAY violations", order_violation),
        ("FLIGHTPERIOD_RANGE mismatch", range_mismatch),
        ("PEAKCOUNT <= 0", peakcount_nonpositive),
        ("FLIGHTPERIOD_SD < 0", sd_negative),
    ]:
        if mask.sum():
            qc.add_table(f"Examples: {label}", phen[mask].head(EXAMPLE_ROWS_N), level=4)

    return order_violation, range_mismatch, peakcount_nonpositive, sd_negative


# =====================================================================================
# 7. DUPLICATES
# =====================================================================================

def check_duplicates(df: pd.DataFrame, key: list[str], name: str, config: dict, qc: QCReport):
    sizes = df.groupby(key).size()
    dup_keys = sizes[sizes > 1]
    n_dup_groups = len(dup_keys)
    if n_dup_groups == 0:
        qc.add(f"Duplicate ({', '.join(key)}) check: {name}", "_no duplicate keys found_", level=3)
        return pd.Series(False, index=df.index), pd.Series(False, index=df.index)

    in_dup_group = df.set_index(key).index.isin(dup_keys.index)
    in_dup_group = pd.Series(in_dup_group, index=df.index)
    exact_dup_row = df.duplicated(keep=False)

    exact_within_group = in_dup_group & exact_dup_row
    conflicting_within_group = in_dup_group & ~exact_dup_row

    body = (
        f"- Duplicate-key groups (>1 row for same {', '.join(key)}): {n_dup_groups:,}\n"
        f"- Rows involved in duplicate-key groups: {in_dup_group.sum():,}\n"
        f"- ...of which byte-identical (exact) duplicate rows: {exact_within_group.sum():,}\n"
        f"- ...of which CONFLICTING (same key, different data -- e.g. different SITE_INDEX/PEAKDAY): {conflicting_within_group.sum():,}\n\n"
        "Exact duplicates collapse losslessly. Conflicting rows are genuine disagreements "
        "with no brood/GENERATION column to explain them, so they are NOT deduplicated "
        "automatically -- resolve them before treating the table as one row per "
        "site-species-year."
    )
    qc.add(f"Duplicate ({', '.join(key)}) check: {name}", body, level=3)
    if conflicting_within_group.sum():
        qc.add_table("Example CONFLICTING duplicate-key rows", df[conflicting_within_group].sort_values(key).head(EXAMPLE_ROWS_N * 2), level=4)
    if exact_within_group.sum():
        qc.add_table("Example exact-duplicate rows", df[exact_within_group].sort_values(key).head(EXAMPLE_ROWS_N), level=4)

    return exact_dup_row, conflicting_within_group


# =====================================================================================
# 8. COVERAGE / RELIABILITY
# =====================================================================================

def check_coverage(phen: pd.DataFrame, site_loc: pd.DataFrame, config: dict, qc: QCReport):
    min_years = config["min_years_surveyed"]
    min_records = config["min_site_species_records"]

    short_sites = site_loc[site_loc.N_yrs_surveyed < min_years]
    rows_at_short_sites = phen[phen.SITE_ID.isin(short_sites.SITE_ID)]

    per_site_species = phen.groupby(["SITE_ID", "SPECIES_NAME"]).size().rename("n_records").reset_index()
    short_series = per_site_species[per_site_species.n_records < min_records]
    rows_in_short_series = phen.merge(short_series[["SITE_ID", "SPECIES_NAME"]], on=["SITE_ID", "SPECIES_NAME"])

    body = (
        f"- Sites with N_yrs_surveyed < {min_years}: {len(short_sites):,} of {len(site_loc):,} "
        f"({len(rows_at_short_sites):,} phenology rows would be affected if filtered)\n"
        f"- (Site, species) series with fewer than {min_records} phenology records: {len(short_series):,} "
        f"({len(rows_in_short_series):,} phenology rows would be affected if filtered)\n\n"
        "These are reported, not applied -- toggle `filters.exclude_below_min_years_surveyed` "
        "/ `filters.exclude_below_min_site_species_records` in CONFIG to apply."
    )
    qc.add("Coverage / reliability thresholds", body, level=3)
    qc.add_table(f"Example short-surveyed sites (< {min_years} yrs)", short_sites.head(EXAMPLE_ROWS_N), level=4)
    qc.add_table(f"Example short (site,species) series (< {min_records} records)", short_series.head(EXAMPLE_ROWS_N), level=4)

    return set(zip(short_series.SITE_ID, short_series.SPECIES_NAME))


# =====================================================================================
# 9. SURVEY TYPE
# =====================================================================================

def check_survey_type(site_loc: pd.DataFrame, phen: pd.DataFrame, config: dict, qc: QCReport):
    tab = site_loc.Survey_type.value_counts(dropna=False).rename("n_sites").reset_index().rename(columns={"index": "Survey_type"})
    nonstandard = [s for s in site_loc.Survey_type.dropna().unique() if s != "UKBMS"]
    nonstandard_sites = site_loc[site_loc.Survey_type.isin(nonstandard)].SITE_ID
    affected_rows = phen[phen.SITE_ID.isin(nonstandard_sites)]

    qc.add("Survey type", "", level=3)
    qc.add_table("Survey_type counts", tab, level=4)
    qc.add(
        "Non-standard transect impact",
        f"- Non-UKBMS survey types found: {nonstandard}\n"
        f"- Sites on non-standard survey types: {len(nonstandard_sites):,}\n"
        f"- Phenology rows that would be affected if excluded: {len(affected_rows):,}\n\n"
        "WCBS sites are visited 2-3 times/year (vs up to 26 for standard transects), so "
        "their phenology metrics rest on much sparser data. Toggle "
        "`filters.exclude_nonstandard_survey_type` to exclude.",
        level=4,
    )


# =====================================================================================
# 10. COORDINATES
# =====================================================================================

def check_coordinates(site_loc: pd.DataFrame, config: dict, qc: QCReport):
    bounds = config["gb_grid_bounds"]
    gb_countries = config["gb_grid_countries"]

    missing = site_loc[site_loc.Easting.isna() | site_loc.Northing.isna()]
    zero = site_loc[(site_loc.Easting == 0) | (site_loc.Northing == 0)]

    is_gb = site_loc.Country.isin(gb_countries)
    in_bounds = (
        site_loc.Easting.between(bounds["easting_min"], bounds["easting_max"])
        & site_loc.Northing.between(bounds["northing_min"], bounds["northing_max"])
    )
    gb_out_of_bounds = site_loc[is_gb & ~in_bounds & site_loc.Easting.notna() & site_loc.Northing.notna()]
    non_gb = site_loc[~is_gb]

    # Lightweight cross-check: GB grid uses 2-letter prefixes, Irish Grid (NI) 1-letter.
    prefix = site_loc.Gridreference.astype(str).str.extract(r"^([A-Za-z]+)")[0]
    prefix_len = prefix.str.len()
    expected_len = np.where(site_loc.Country.eq("Northern Ireland"), 1, 2)
    prefix_mismatch = site_loc[(prefix_len != expected_len) & site_loc.Gridreference.notna()]

    body = (
        f"- Sites missing Easting/Northing: {len(missing):,}\n"
        f"- Sites with Easting or Northing == 0: {len(zero):,}\n"
        f"- GB-country sites (England/Scotland/Wales) outside GB grid bounds "
        f"(E [{bounds['easting_min']},{bounds['easting_max']}], N [{bounds['northing_min']},{bounds['northing_max']}]): {len(gb_out_of_bounds):,}\n"
        f"- Sites outside GB-grid scope by country (Northern Ireland/Channel Islands/Isle of Man): {len(non_gb):,} "
        "-- these use the Irish Grid or a local Channel Islands grid (negative Northing is a "
        "convention there, not an error), so GB bounds do not apply and they are not flagged invalid.\n"
        f"- Grid-reference prefix length inconsistent with the site's expected grid system: {len(prefix_mismatch):,}\n\n"
        "Sites without valid GB coordinates (missing/zero/out-of-bounds) break the GB climate "
        "join and must be resolved or excluded first; NI/CI/IoM sites additionally need their "
        "own grid-to-lat/lon conversion (not a straight OSGB one)."
    )
    qc.add("Coordinate validity (OSGB National Grid)", body, level=3)
    qc.add_table("Example GB sites out of grid bounds", gb_out_of_bounds.head(EXAMPLE_ROWS_N), level=4)
    qc.add_table("Example non-GB-grid sites (by country)", non_gb.groupby("Country").size().rename("n_sites").reset_index(), level=4)
    qc.add_table("Example grid-reference prefix mismatches", prefix_mismatch.head(EXAMPLE_ROWS_N), level=4)

    invalid_gb_site_ids = set(missing.SITE_ID) | set(zero.SITE_ID) | set(gb_out_of_bounds.SITE_ID)
    return invalid_gb_site_ids


# =====================================================================================
# 11. SPECIES SCOPE
# =====================================================================================

def check_species_scope(phen: pd.DataFrame, config: dict, qc: QCReport):
    counts = phen.COMMON_NAME.value_counts().rename("n_records").reset_index().rename(columns={"index": "COMMON_NAME"})
    rare = counts[counts.n_records < RARE_SPECIES_RECORD_THRESHOLD]
    migrants_present = [s for s in MIGRANT_CANDIDATE_SPECIES if s in set(phen.COMMON_NAME)]

    body = (
        f"- Distinct species: {phen.COMMON_NAME.nunique()}\n"
        f"- Species with fewer than {RARE_SPECIES_RECORD_THRESHOLD} phenology records "
        f"(rare/restricted-range candidates): {len(rare):,}\n"
        f"- Migrant/highly-mobile candidate species present: {migrants_present}\n\n"
        "None excluded by default. `Essex/Small Skipper` is a deliberate combined-ID category "
        "(the two are hard to separate in the field), distinct from the separate `Essex Skipper` "
        "and `Small Skipper` -- a real taxon, not an error. It and `Mountain Ringlet` are the two "
        "species with no matching species_trends row."
    )
    qc.add("Species scope", body, level=3)
    qc.add_table("Rarest species (record count)", rare.sort_values("n_records"), level=4, max_rows=20)


# =====================================================================================
# 12. TEMPORAL ALIGNMENT
# =====================================================================================

def check_temporal_alignment(dfs_years: dict[str, pd.Series], qc: QCReport):
    rows = []
    for name, years in dfs_years.items():
        rows.append({"file": name, "min_year": years.min(), "max_year": years.max(), "n_distinct_years": years.nunique()})
    df = pd.DataFrame(rows)
    qc.add("Temporal alignment", "", level=3)
    qc.add_table("YEAR range per file", df, level=4)
    all_to_2024 = (df.max_year == 2024).all()
    qc.add("2024 coverage check", "All files extend to 2024." if all_to_2024 else "NOT all files reach 2024 -- see table above.", level=4)


# =====================================================================================
# MAIN
# =====================================================================================

def main():
    config = CONFIG
    config["output_dir"].mkdir(parents=True, exist_ok=True)
    qc = QCReport()
    row_log: list[tuple[str, int]] = []

    # ---------------------------------------------------------------------- ASSUMPTIONS
    qc.add(
        "Assumptions made by this pipeline",
        "1. **Day fields are \"days after 1 April\", not day-of-year 1-366** (per "
        "`ukbms_phenology_2024.docx`). Negatives (before 1 April) and values > 182 (after "
        "30 Sept) are legitimate; plausibility checks use this scale.\n"
        "2. **Species join on scientific name** (SPECIES_NAME == SPECIES), not common name, "
        "since phenology has no species code. Verified 1:1 and lossless across all 60 species.\n"
        "3. **SITE_INDEX == -2 is a documented sentinel** (insufficient monitoring), not a real "
        "abundance. Kept as -2 and flagged via `SITE_INDEX_INSUFFICIENT_MONITORING`; never imputed.\n"
        "4. **Exact duplicate rows are collapsed** (logged, not silent). Conflicting KEYS "
        "(different SITE_INDEX for the same site/species/year) are never auto-resolved: one row "
        "is kept, SITE_INDEX set NaN, and `SITE_INDEX_AMBIGUOUS_CONFLICT=True` so it stays visible.\n"
        "5. **GB grid bounds checked only for England/Scotland/Wales.** NI (Irish Grid) and "
        "Channel Islands (local grid, negative Northing) are valid systems, reported as "
        "\"out of GB-grid scope\" rather than invalid.\n"
        "6. **species_trends / collated_indices attach as species- / national-level columns "
        "only**, left-joined on COMMON_NAME and (SPECIES_NAME, YEAR, COUNTRY), never multiplying "
        "the grain. Channel Islands/Isle of Man rows get NaN national columns by design (no "
        "matching national entry).\n"
        "7. **No destructive filtering by default** beyond #4: all other CONFIG['filters'] are "
        "False, so the saved table is the full flagged table unless a toggle is flipped.",
        level=2,
    )

    # ---------------------------------------------------------------------- LOAD + PROFILE
    log("Loading raw files...")
    raw = load_raw(config)
    phen, site_idx, site_loc, sp_trend, coll_idx = (
        raw["phenology"], raw["site_indices"], raw["site_location"], raw["species_trends"], raw["collated_indices"],
    )
    row_count_log("raw: phenology", phen, row_log)
    row_count_log("raw: site_indices", site_idx, row_log)
    row_count_log("raw: site_location", site_loc, row_log)
    row_count_log("raw: species_trends", sp_trend, row_log)
    row_count_log("raw: collated_indices", coll_idx, row_log)

    qc.add("1. File profiles", "", level=2)
    for name, df in raw.items():
        profile_file(name, df, qc)

    # ---------------------------------------------------------------------- SITE KEYS
    qc.add("2. Join key harmonisation", "", level=2)
    phen, site_idx, site_loc = harmonise_site_keys(phen, site_idx, site_loc, qc)
    build_species_lookup(phen, site_idx, qc)

    # ---------------------------------------------------------------------- DTYPE COERCION
    qc.add("3. Dtype coercion", "", level=2)
    phen = coerce_numeric(
        phen, ["YEAR", "FIRSTDAY", "LASTDAY", "PEAKDAY", "PEAKCOUNT", "MEAN_FLIGHT_DATE",
               "FLIGHTPERIOD_SD", "FLIGHTPERIOD_RANGE"], "phenology", qc,
    )
    site_idx = coerce_numeric(site_idx, ["YEAR", "SITE_INDEX"], "site_indices", qc)
    site_loc = coerce_numeric(site_loc, ["Easting", "Northing"], "site_location", qc)
    phen["YEAR"] = phen["YEAR"].astype("Int64")
    site_idx["YEAR"] = site_idx["YEAR"].astype("Int64")

    # ---------------------------------------------------------------------- DOY PLAUSIBILITY
    qc.add("4. Phenology day-field plausibility", "", level=2)
    doy_implausible, doy_outside_season = check_doy_plausibility(phen, config, qc)
    phen["FLAG_DOY_IMPLAUSIBLE"] = doy_implausible
    phen["FLAG_DOY_OUTSIDE_CORE_SEASON"] = doy_outside_season

    # ---------------------------------------------------------------------- LOGICAL CONSISTENCY
    qc.add("5. Logical consistency", "", level=2)
    order_violation, range_mismatch, peakcount_nonpositive, sd_negative = check_logical_consistency(phen, config, qc)
    phen["FLAG_PEAK_ORDER_VIOLATION"] = order_violation
    phen["FLAG_RANGE_MISMATCH"] = range_mismatch
    phen["FLAG_PEAKCOUNT_NONPOSITIVE"] = peakcount_nonpositive
    phen["FLAG_SD_NEGATIVE"] = sd_negative

    # ---------------------------------------------------------------------- DUPLICATES
    qc.add("6. Duplicates / hidden generations", "", level=2)
    check_duplicates(phen, ["SITE_ID", "SPECIES_NAME", "YEAR"], "phenology", config, qc)
    check_duplicates(site_idx, ["SITE_ID", "SPECIES_CODE", "YEAR"], "site_indices", config, qc)

    # ---------------------------------------------------------------------- COVERAGE
    qc.add("8. Coverage / reliability", "", level=2)
    short_series_keys = check_coverage(phen, site_loc, config, qc)

    # ---------------------------------------------------------------------- SURVEY TYPE
    qc.add("9. Survey type", "", level=2)
    check_survey_type(site_loc, phen, config, qc)

    # ---------------------------------------------------------------------- COORDINATES
    qc.add("10. Coordinates", "", level=2)
    invalid_gb_site_ids = check_coordinates(site_loc, config, qc)

    # ---------------------------------------------------------------------- SPECIES SCOPE
    qc.add("11. Species scope", "", level=2)
    check_species_scope(phen, config, qc)

    # ---------------------------------------------------------------------- TEMPORAL ALIGNMENT
    qc.add("12. Temporal alignment", "", level=2)
    check_temporal_alignment(
        {"phenology": phen.YEAR, "site_indices": site_idx.YEAR, "collated_indices": coll_idx.YEAR,
         "site_location (First_year_surveyed)": site_loc.First_year_surveyed,
         "site_location (Last_year_surveyed)": site_loc.Last_year_surveyed},
        qc,
    )

    # ======================================================================
    # RESOLVE DUPLICATES (non-silent, logged) so the core merge has a clean
    # 1:1 key on both sides -- required for the grain assert below.
    # ======================================================================
    qc.add("Duplicate resolution applied before merge", "", level=2)

    n_before = len(phen)
    phen_keys_before = phen[["SITE_ID", "SPECIES_NAME", "YEAR"]].drop_duplicates()
    n_distinct_combos_before = len(phen_keys_before)
    phen_dedup = phen.drop_duplicates(subset=phen.columns.difference(["FLAG_DOY_IMPLAUSIBLE", "FLAG_DOY_OUTSIDE_CORE_SEASON",
                                                                        "FLAG_PEAK_ORDER_VIOLATION", "FLAG_RANGE_MISMATCH",
                                                                        "FLAG_PEAKCOUNT_NONPOSITIVE", "FLAG_SD_NEGATIVE"]))
    qc.add(
        "Phenology exact-duplicate collapse",
        f"Collapsed {n_before:,} -> {len(phen_dedup):,} rows by dropping byte-identical repeats "
        f"(all {n_before - len(phen_dedup):,} removed rows were confirmed full-row duplicates, see section 6). "
        f"Distinct (site, species, year) combinations in raw phenology: {n_distinct_combos_before:,}.",
        level=3,
    )
    row_count_log("after exact-dup collapse: phenology", phen_dedup, row_log)

    remaining_dup_sizes = phen_dedup.groupby(["SITE_ID", "SPECIES_NAME", "YEAR"]).size()
    remaining_conflicts = remaining_dup_sizes[remaining_dup_sizes > 1]
    if len(remaining_conflicts):
        qc.add(
            "WARNING: phenology still has conflicting (non-identical) duplicate keys after collapse",
            f"{len(remaining_conflicts)} (site,species,year) keys still have >1 row with genuinely "
            "different data. These were NOT resolved automatically. For the grain assert below, "
            "one row per key is retained (first occurrence) and flagged; review before trusting "
            "those rows' phenology values.",
            level=3,
        )
        first_idx = phen_dedup.groupby(["SITE_ID", "SPECIES_NAME", "YEAR"]).head(1).index
        phen_dedup = phen_dedup.loc[first_idx].copy()
        phen_dedup["FLAG_PHENOLOGY_KEY_CONFLICT"] = phen_dedup.set_index(["SITE_ID", "SPECIES_NAME", "YEAR"]).index.isin(remaining_conflicts.index)
    else:
        phen_dedup["FLAG_PHENOLOGY_KEY_CONFLICT"] = False
    row_count_log("after phenology conflict resolution (1 row/key retained)", phen_dedup, row_log)

    # site_indices: collapse exact dupes, then isolate conflicting keys (null + flag)
    n_idx_before = len(site_idx)
    site_idx_dedup = site_idx.drop_duplicates()
    row_count_log("after exact-dup collapse: site_indices", site_idx_dedup, row_log)

    idx_sizes = site_idx_dedup.groupby(["SITE_ID", "SPECIES_CODE", "YEAR"]).size()
    idx_conflict_keys = idx_sizes[idx_sizes > 1].index
    idx_conflict_mask = site_idx_dedup.set_index(["SITE_ID", "SPECIES_CODE", "YEAR"]).index.isin(idx_conflict_keys)
    qc.add(
        "site_indices exact-duplicate collapse + conflict isolation",
        f"Collapsed {n_idx_before:,} -> {len(site_idx_dedup):,} rows (byte-identical repeats removed). "
        f"{idx_conflict_mask.sum():,} rows across {len(idx_conflict_keys):,} keys still have genuinely "
        "different SITE_INDEX values for the same (site, species, year). These rows are retained "
        "(first occurrence per key) but SITE_INDEX is set to NaN and "
        "`SITE_INDEX_AMBIGUOUS_CONFLICT=True` so the disagreement is visible rather than guessed at.",
        level=3,
    )
    site_idx_dedup["SITE_INDEX_AMBIGUOUS_CONFLICT"] = idx_conflict_mask
    site_idx_resolved = site_idx_dedup.groupby(["SITE_ID", "SPECIES_CODE", "YEAR"], as_index=False).first()
    site_idx_resolved.loc[site_idx_resolved.SITE_INDEX_AMBIGUOUS_CONFLICT, "SITE_INDEX"] = np.nan
    row_count_log("after site_indices conflict resolution (1 row/key retained)", site_idx_resolved, row_log)

    site_idx_resolved["SITE_INDEX_INSUFFICIENT_MONITORING"] = site_idx_resolved.SITE_INDEX == -2

    # ======================================================================
    # CORE MERGE: phenology -> site_indices (site, species-by-sci-name, year) -> site_location (site)
    # ======================================================================
    merged = phen_dedup.merge(
        site_idx_resolved.rename(columns={"SPECIES": "SPECIES_NAME"}).drop(columns=["COMMON_NAME"]),
        on=["SITE_ID", "SPECIES_NAME", "YEAR"], how="left", suffixes=("", "_siteidx"),
    )
    row_count_log("after phenology-siteindices merge", merged, row_log)

    assert len(merged) == len(phen_dedup), (
        f"GRAIN VIOLATION: merge produced {len(merged)} rows but phenology (post-resolution) "
        f"has {len(phen_dedup)} distinct (site,species,year) rows. Investigate site_indices "
        "join cardinality before trusting this output."
    )
    log(f"Grain assert OK: {len(merged):,} rows == {len(phen_dedup):,} distinct phenology (site,species,year) rows.")

    merged = merged.merge(
        site_loc.add_suffix("_siteloc").rename(columns={"SITE_ID_siteloc": "SITE_ID"}),
        on="SITE_ID", how="left",
    )
    row_count_log("after + site_location merge", merged, row_log)
    assert len(merged) == len(phen_dedup), "GRAIN VIOLATION after site_location merge (should be 1:1 on SITE_ID)"

    # COUNTRY consistency cross-check (site_indices.COUNTRY vs site_location.Country)
    country_mismatch = merged[
        merged.COUNTRY.notna() & merged.Country_siteloc.notna() & (merged.COUNTRY != merged.Country_siteloc)
    ]
    qc.add(
        "COUNTRY consistency: site_indices.COUNTRY vs site_location.Country",
        f"{len(country_mismatch):,} merged rows have a different COUNTRY value between the two "
        "source files for the same site.",
        level=2,
    )
    if len(country_mismatch):
        qc.add_table("Example COUNTRY mismatches", country_mismatch[["SITE_ID", "COUNTRY", "Country_siteloc"]].drop_duplicates().head(EXAMPLE_ROWS_N), level=3)

    # ---------------------------------------------------------------------- SPECIES-LEVEL ATTACH (species_trends)
    sp_trend_dupes = sp_trend.COMMON_NAME.duplicated().sum()
    qc.add(
        "species_trends attachment",
        f"Left-joined on COMMON_NAME (species-level table, {sp_trend.COMMON_NAME.nunique()} distinct "
        f"species, {sp_trend_dupes} duplicate COMMON_NAME keys). Species in the merged table with "
        f"no matching trend row: {sorted(set(merged.COMMON_NAME) - set(sp_trend.COMMON_NAME))}.",
        level=2,
    )
    merged = merged.merge(sp_trend.add_prefix("TREND_").rename(columns={"TREND_COMMON_NAME": "COMMON_NAME"}), on="COMMON_NAME", how="left")
    row_count_log("after + species_trends merge", merged, row_log)
    assert len(merged) == len(phen_dedup), "GRAIN VIOLATION after species_trends merge"

    # ---------------------------------------------------------------------- NATIONAL-LEVEL ATTACH (collated_indices)
    coll_key_sizes = coll_idx.groupby(["SPECIES", "YEAR", "COUNTRY"]).size()
    if (coll_key_sizes > 1).any():
        qc.add("WARNING: collated_indices has duplicate (SPECIES, YEAR, COUNTRY) keys",
               f"{(coll_key_sizes > 1).sum()} duplicate keys -- national attach below may not be strictly 1:1.", level=2)
    coll_for_merge = coll_idx.rename(columns={"SPECIES": "SPECIES_NAME"})
    merged = merged.merge(
        coll_for_merge[["SPECIES_NAME", "YEAR", "COUNTRY", "N_SITES", "COLLATED_INDEX", "YEAR_RANK", "TIME_PERIOD"]]
        .add_prefix("NATIONAL_")
        .rename(columns={"NATIONAL_SPECIES_NAME": "SPECIES_NAME", "NATIONAL_YEAR": "YEAR", "NATIONAL_COUNTRY": "COUNTRY"}),
        on=["SPECIES_NAME", "YEAR", "COUNTRY"], how="left",
    )
    row_count_log("after + collated_indices merge", merged, row_log)
    assert len(merged) == len(phen_dedup), "GRAIN VIOLATION after collated_indices merge"
    qc.add(
        "collated_indices attachment",
        "Left-joined on (scientific name, YEAR, site's COUNTRY). collated_indices has no "
        "Channel Islands/Isle of Man entries, so those sites get NaN NATIONAL_* columns by design.",
        level=2,
    )

    # ---------------------------------------------------------------------- MISSINGNESS (7)
    qc.add("7. Missingness in merged table", "", level=2)
    nulls = merged.isnull().sum()
    nulls_df = nulls[nulls > 0].rename("n_null").reset_index().rename(columns={"index": "column"})
    nulls_df["pct_null"] = (nulls_df["n_null"] / len(merged) * 100).round(2)
    qc.add_table("Null counts per column (merged table)", nulls_df, level=3, max_rows=50)

    no_site_index_row = merged.SITE_INDEX.isna() & ~merged.SITE_INDEX_AMBIGUOUS_CONFLICT.fillna(False)
    qc.add(
        "SITE_INDEX missingness interpretation",
        f"- NaN SITE_INDEX from an ambiguous source conflict (flagged above): "
        f"{merged.SITE_INDEX_AMBIGUOUS_CONFLICT.fillna(False).sum():,}\n"
        f"- NaN SITE_INDEX because no site_indices row exists for that (site, species, year) "
        f"-- phenology recorded the species but no index was published: {no_site_index_row.sum():,}\n\n"
        "The phenology target (FIRSTDAY/LASTDAY/PEAKDAY/etc.) is never imputed.",
        level=3,
    )

    # ======================================================================
    # FILTERING STAGE (separate, parameterised, logged before/after)
    # ======================================================================
    qc.add("Filtering stage (only filters toggled True in CONFIG are applied)", "", level=2)
    filters = config["filters"]
    filtered = merged.copy()
    filter_log = []

    def apply_filter(label, mask_to_drop, toggle_key):
        nonlocal filtered
        n_before_f = len(filtered)
        n_would_drop = mask_to_drop.sum()
        applied = filters.get(toggle_key, False)
        if applied:
            filtered = filtered[~mask_to_drop].copy()
        filter_log.append({
            "filter": label, "toggle": toggle_key, "applied": applied,
            "rows_before": n_before_f, "rows_would_remove": int(n_would_drop),
            "rows_after": len(filtered) if applied else n_before_f,
        })

    apply_filter(
        "Exclude non-standard survey type (WCBS)",
        filtered.Survey_type_siteloc.eq("WCBS"),
        "exclude_nonstandard_survey_type",
    )
    apply_filter(
        "Exclude species in CONFIG species_exclude_list",
        filtered.COMMON_NAME.isin(config["species_exclude_list"]) | filtered.SPECIES_NAME.isin(config["species_exclude_list"]),
        "exclude_species_list",
    )
    apply_filter(
        f"Exclude sites with N_yrs_surveyed < {config['min_years_surveyed']}",
        filtered.N_yrs_surveyed_siteloc < config["min_years_surveyed"],
        "exclude_below_min_years_surveyed",
    )
    short_series_mask = filtered.set_index(["SITE_ID", "SPECIES_NAME"]).index.isin(short_series_keys)
    apply_filter(
        f"Exclude (site,species) series with < {config['min_site_species_records']} phenology records",
        pd.Series(short_series_mask, index=filtered.index),
        "exclude_below_min_site_species_records",
    )

    # coordinate nulling is a column-level mask, not a row drop -- handled separately
    invalid_coord_mask = filtered.SITE_ID.isin(invalid_gb_site_ids) & filtered.Country_siteloc.isin(config["gb_grid_countries"])
    n_invalid_coord_rows = invalid_coord_mask.sum()
    if filters["null_invalid_gb_coordinates"]:
        filtered.loc[invalid_coord_mask, ["Easting_siteloc", "Northing_siteloc"]] = np.nan
    filter_log.append({
        "filter": "Null invalid GB coordinates (Easting/Northing) for GB-country sites",
        "toggle": "null_invalid_gb_coordinates", "applied": filters["null_invalid_gb_coordinates"],
        "rows_before": len(filtered), "rows_would_remove": int(n_invalid_coord_rows),
        "rows_after": len(filtered),
    })

    filter_log_df = pd.DataFrame(filter_log)
    qc.add_table("Filter impact summary", filter_log_df, level=3, max_rows=20)
    qc.add(
        "How to change filtering behaviour",
        "Edit `CONFIG['filters']` at the top of `scripts/clean_pipeline.py` and re-run. "
        "Every filter above is computed and reported regardless of toggle state; only "
        "toggles set to `True` change the saved output.",
        level=3,
    )

    # ---------------------------------------------------------------------- ROW COUNT LOG
    row_count_log("FINAL (after filtering stage)", filtered, row_log)
    row_log_df = pd.DataFrame(row_log, columns=["stage", "n_rows"])
    qc.add("Row-count log (all stages)", row_log_df.to_markdown(index=False), level=2)

    # ---------------------------------------------------------------------- SAVE OUTPUTS
    final = filtered.rename(columns={
        "Country_siteloc": "COUNTRY_SITELOC", "Easting_siteloc": "EASTING", "Northing_siteloc": "NORTHING",
        "Site_Name_siteloc": "SITE_NAME_SITELOC", "Gridreference_siteloc": "GRIDREF_SITELOC",
        "Length_siteloc": "LENGTH_M", "N_sections_siteloc": "N_SECTIONS", "N_yrs_surveyed_siteloc": "N_YRS_SURVEYED",
        "First_year_surveyed_siteloc": "FIRST_YEAR_SURVEYED", "Last_year_surveyed_siteloc": "LAST_YEAR_SURVEYED",
        "Survey_type_siteloc": "SURVEY_TYPE",
    })
    final.to_parquet(config["merged_parquet_path"], index=False)
    final.to_csv(config["merged_csv_path"], index=False)
    log(f"Saved {len(final):,} rows x {final.shape[1]} columns to {config['merged_parquet_path'].name} and {config['merged_csv_path'].name}")

    # ---------------------------------------------------------------------- DATA DICTIONARY
    dictionary_rows = []
    descriptions = {
        "SITE_ID": "UKBMS site identifier (harmonised from SITENO/SITE_CODE/Site_Number)",
        "SITENAME": "Site name as recorded in the phenology download",
        "SITE_NAME_SITELOC": "Site name as recorded in the site_location download",
        "GRIDREF": "Grid reference, phenology download (start of transect)",
        "GRIDREF_SITELOC": "Grid reference, site_location download (transect centre)",
        "SPECIES_NAME": "Scientific (binomial) name; primary species join key",
        "COMMON_NAME": "Vernacular species name",
        "SPECIES_CODE": "UKBMS numeric species code (from site_indices)",
        "YEAR": "Survey year",
        "FIRSTDAY": "Days after 1 April on which species first recorded at site/year (can be negative)",
        "LASTDAY": "Days after 1 April on which species last recorded at site/year",
        "PEAKDAY": "Days after 1 April of largest count at site/year",
        "PEAKCOUNT": "Largest single count recorded at site/year",
        "MEAN_FLIGHT_DATE": "Weighted mean flight date, days after 1 April",
        "FLIGHTPERIOD_SD": "SD (days) around MEAN_FLIGHT_DATE -- synchronisation/length of flight period",
        "FLIGHTPERIOD_RANGE": "LASTDAY - FIRSTDAY, days",
        "SITE_INDEX": "Site abundance index for species/site/year; -2 = insufficient monitoring (see flag)",
        "SITE_INDEX_INSUFFICIENT_MONITORING": "True if raw SITE_INDEX was the -2 sentinel",
        "SITE_INDEX_AMBIGUOUS_CONFLICT": "True if source site_indices had conflicting duplicate values for this key (SITE_INDEX set NaN)",
        "COUNTRY": "Country, from site_indices",
        "COUNTRY_SITELOC": "Country, from site_location (cross-checked against COUNTRY)",
        "EASTING": "OSGB Easting (m) for GB sites; Irish Grid/local grid for NI/Channel Islands -- see QC report",
        "NORTHING": "OSGB Northing (m) for GB sites; Irish Grid/local grid for NI/Channel Islands -- see QC report",
        "LENGTH_M": "Transect length, metres",
        "N_SECTIONS": "Number of transect sections",
        "N_YRS_SURVEYED": "Number of years site surveyed under UKBMS up to 2024",
        "FIRST_YEAR_SURVEYED": "First year site surveyed",
        "LAST_YEAR_SURVEYED": "Most recent year site surveyed",
        "SURVEY_TYPE": "UKBMS (standard transect) or WCBS (Wider Countryside Butterfly Scheme)",
        "FLAG_DOY_IMPLAUSIBLE": "Any day field outside hard plausibility bounds (CONFIG day_value_min/max)",
        "FLAG_DOY_OUTSIDE_CORE_SEASON": "Any day field outside core UKBMS season (1 Apr-30 Sep); informational",
        "FLAG_PEAK_ORDER_VIOLATION": "NOT (FIRSTDAY <= PEAKDAY <= LASTDAY)",
        "FLAG_RANGE_MISMATCH": "|FLIGHTPERIOD_RANGE - (LASTDAY-FIRSTDAY)| exceeds tolerance",
        "FLAG_PEAKCOUNT_NONPOSITIVE": "PEAKCOUNT <= 0",
        "FLAG_SD_NEGATIVE": "FLIGHTPERIOD_SD < 0",
        "FLAG_PHENOLOGY_KEY_CONFLICT": "Source phenology had conflicting duplicate rows for this (site,species,year); first row retained",
    }
    for col in final.columns:
        desc = descriptions.get(col, "TREND_* = species-level trend stat (speciesTrends, see source docx); NATIONAL_* = national collated index (collatedindices, see source docx)"
                                  if (col.startswith("TREND_") or col.startswith("NATIONAL_")) else "")
        dictionary_rows.append({"column": col, "dtype": str(final[col].dtype), "description": desc})
    dict_df = pd.DataFrame(dictionary_rows)
    with open(config["data_dictionary_path"], "w") as f:
        f.write("# Data dictionary: ukbms_site_species_year_clean\n\n")
        f.write(f"Grain: one row per (SITE_ID, SPECIES_NAME, YEAR). {len(final):,} rows, {final.shape[1]} columns.\n\n")
        f.write(dict_df.to_markdown(index=False))
    log(f"Saved data dictionary to {config['data_dictionary_path'].name}")

    qc.add(
        "Output files",
        f"- `{config['merged_parquet_path'].relative_to(REPO_ROOT)}`\n"
        f"- `{config['merged_csv_path'].relative_to(REPO_ROOT)}`\n"
        f"- `{config['data_dictionary_path'].relative_to(REPO_ROOT)}`\n",
        level=2,
    )

    qc_text = "# UKBMS data cleaning / QC report\n\n" + qc.render()
    with open(config["qc_report_path"], "w") as f:
        f.write(qc_text)
    log(f"Saved QC report to {config['qc_report_path'].name}")


if __name__ == "__main__":
    main()
