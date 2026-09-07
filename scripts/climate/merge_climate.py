#!/usr/bin/env python
"""
First working merge of HadUK-Grid daily climate onto the cleaned UKBMS table.

Pipeline (idempotent, memory-bounded, no reprojection — both sides are EPSG:27700):

  1. Distinct sites (SITE_ID + Easting/Northing) from the cleaned table.
  2. Classify coordinates: true BNG vs NI/Channel-Islands/local grids vs missing.
  3. Confirm coordinate/variable names by opening one netCDF file.
  4. Year-by-year: open that year's monthly files (open_mfdataset, lazy/chunked),
     vectorised NEAREST pointwise selection at each site's x/y, cache daily series.
  5. QC: all-NaN (sea/edge) sites, max site-to-cell distance, match counts.
  6. Build a leakage-safe site x year feature set (antecedent winter DJF + spring
     MAM window; monthly means/totals; GDD accumulation).
  7. Merge onto the site x species x year table; assert row count unchanged.

Run:  .venv/bin/python -m scripts.climate.merge_climate
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.climate import config as C
else:
    from . import config as C

# ==========================================================================
# CONFIG block (feature window; other settings imported from config.py)
# ==========================================================================
# Leakage-safe antecedent window for spring/summer flight events:
#   climate from 1 December (year-1) through 31 May (year) — i.e. the winter
#   and spring that PRECEDE the flight period, never climate after it.
ANTE_WINTER_MONTHS = [12, 1, 2]     # DJF (Dec is from the PREVIOUS calendar year)
ANTE_SPRING_MONTHS = [3, 4, 5]      # MAM
ANTE_END_MONTH = 5                  # last month included (May)
# GDD accumulates from GDD_START_DOY (config) through the end of the window.
SNAP_TO_LAND = False                # if True, all-NaN sites snap to nearest valid land cell

# Countries whose UKBMS coordinates are NOT EPSG:27700 and must be excluded from a
# BNG join. Critically, Northern Ireland uses the Irish Grid whose easting/northing
# numbers fall INSIDE the BNG numeric range, so a bounds check alone lets them
# silently mis-locate onto GB cells — hence an explicit country exclusion.
NON_BNG_COUNTRIES = {"Northern Ireland", "Channel Islands", "Isle of Man"}

# Runtime structure (confirmed by confirm_structure(); may override config defaults)
STRUCT = {"x": C.X_COORD, "y": C.Y_COORD, "time": "time"}


# ==========================================================================
# 1-2. Sites
# ==========================================================================
def load_sites() -> pd.DataFrame:
    df = pd.read_parquet(C.CLEAN_TABLE, columns=["SITE_ID", "EASTING", "NORTHING", "COUNTRY"])
    sites = (df.dropna(subset=["SITE_ID"])
               .drop_duplicates("SITE_ID")
               .reset_index(drop=True))
    e, n = sites.EASTING, sites.NORTHING
    sites["has_coords"] = e.notna() & n.notna()
    in_bounds = (e.between(C.BNG_EASTING_MIN, C.BNG_EASTING_MAX)
                 & n.between(C.BNG_NORTHING_MIN, C.BNG_NORTHING_MAX))
    gb_grid = ~sites["COUNTRY"].isin(NON_BNG_COUNTRIES)   # NaN country -> treated as GB, bounds decide
    sites["is_bng"] = sites["has_coords"] & in_bounds & gb_grid
    return sites


# ==========================================================================
# 3. Confirm netCDF structure
# ==========================================================================
def _first_file(var: str) -> Path | None:
    d = C.HADUK_DIR / var
    files = sorted(d.glob(f"{var}_hadukgrid_uk_{C.RESOLUTION}_day_*.nc")) if d.exists() else []
    return files[0] if files else None


def confirm_structure() -> None:
    import xarray as xr
    f = _first_file(C.VARIABLES[0])
    if f is None:
        raise SystemExit(
            f"ERROR: no netCDF files under {C.HADUK_DIR}. Run download_haduk.py first "
            "(after setting CEDA_TOKEN)."
        )
    with xr.open_dataset(f) as ds:
        print(f"Confirming structure from {f.name}")
        print("  data_vars:", list(ds.data_vars))
        print("  coords   :", list(ds.coords))
        for key, cand in (("x", C.X_COORD), ("y", C.Y_COORD)):
            if cand in ds.coords or cand in ds.variables:
                STRUCT[key] = cand
            else:
                raise SystemExit(f"ERROR: expected coordinate '{cand}' not found in {f.name}.")
        if "time" not in ds.dims:
            # find the unlimited/time-like dim
            for d in ds.dims:
                if "time" in d.lower():
                    STRUCT["time"] = d
        if C.VARIABLES[0] not in ds.variables:
            raise SystemExit(f"ERROR: data var '{C.VARIABLES[0]}' not in {f.name}.")
        print(f"  using x={STRUCT['x']} y={STRUCT['y']} time={STRUCT['time']}")


# ==========================================================================
# 4. Year-by-year extraction (cached)
# ==========================================================================
def _year_files(var: str, year: int) -> list[Path]:
    d = C.HADUK_DIR / var
    return sorted(d.glob(f"{var}_hadukgrid_uk_{C.RESOLUTION}_day_{year}*.nc"))


def _cache_path(year: int) -> Path:
    return C.SITE_DAILY_DIR / f"year={year}.parquet"


def extract_year(year: int, sites_bng: pd.DataFrame) -> tuple[dict, Path | None]:
    """Extract daily tasmax/tasmin/rainfall for all BNG sites for one year.

    Returns (qc_dict, cache_path). Idempotent: skips if the year is already cached.
    """
    import xarray as xr

    cp = _cache_path(year)
    if cp.exists():
        return {"year": year, "status": "cached"}, cp

    # indexers along a 'site' dim -> vectorised pointwise nearest selection
    site_ids = sites_bng.SITE_ID.values
    xi = xr.DataArray(sites_bng.EASTING.values.astype("float64"), dims="site",
                      coords={"site": site_ids})
    yi = xr.DataArray(sites_bng.NORTHING.values.astype("float64"), dims="site",
                      coords={"site": site_ids})

    per_var = {}
    sel_xy = None
    for var in C.VARIABLES:
        files = _year_files(var, year)
        if not files:
            return {"year": year, "status": f"missing {var} files"}, None
        ds = xr.open_mfdataset([str(f) for f in files], combine="by_coords",
                               chunks={STRUCT["time"]: 31}, engine="netcdf4")
        da = ds[var].sel({STRUCT["x"]: xi, STRUCT["y"]: yi}, method="nearest")
        per_var[var] = da.compute()
        if sel_xy is None:
            # record the actually-selected cell centres for distance QC (geometry is constant)
            sel_xy = (da[STRUCT["x"]].values, da[STRUCT["y"]].values)
        ds.close()

    # assemble long dataframe: site x date
    frames = []
    for var, da in per_var.items():
        s = da.to_series().rename(var)  # index (site, time) or (time, site)
        frames.append(s)
    daily = pd.concat(frames, axis=1).reset_index()
    daily = daily.rename(columns={"site": "SITE_ID", STRUCT["time"]: "date"})
    daily["date"] = pd.to_datetime(daily["date"])
    daily["year"] = daily["date"].dt.year
    daily["month"] = daily["date"].dt.month
    daily["doy"] = daily["date"].dt.dayofyear
    daily["tmean"] = (daily["tasmax"] + daily["tasmin"]) / 2.0

    # QC: distance to selected cell + all-NaN sites
    req = sites_bng.set_index("SITE_ID")
    dist = np.hypot(sel_xy[0] - req.loc[site_ids, "EASTING"].values,
                    sel_xy[1] - req.loc[site_ids, "NORTHING"].values)
    nan_sites = (daily.groupby("SITE_ID")[["tasmax", "tasmin", "rainfall"]]
                      .apply(lambda g: g.isna().all().all()))
    nan_ids = nan_sites[nan_sites].index.tolist()

    C.SITE_DAILY_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(cp, index=False)

    qc = {"year": year, "status": "extracted", "n_sites": len(site_ids),
          "max_cell_dist_m": float(np.nanmax(dist)), "n_nan_sites": len(nan_ids),
          "nan_ids": nan_ids}
    return qc, cp


# ==========================================================================
# 6. Features (leakage-safe antecedent window)
# ==========================================================================
def _read_window(year: int) -> pd.DataFrame:
    """Daily rows for the antecedent window of `year`: Dec(year-1) + Jan..May(year)."""
    parts = []
    prev = _cache_path(year - 1)
    if prev.exists():
        d = pd.read_parquet(prev, filters=[("month", "==", 12)])
        parts.append(d)
    cur = _cache_path(year)
    if cur.exists():
        d = pd.read_parquet(cur, filters=[("month", "<=", ANTE_END_MONTH)])
        parts.append(d)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def build_features_for_year(year: int) -> pd.DataFrame:
    w = _read_window(year)
    if w.empty:
        return pd.DataFrame()
    # tag Dec of prev year as belonging to this feature-year's winter
    out = []
    g = w.groupby("SITE_ID")

    # monthly means/totals within the window (prev Dec labelled 'M12P')
    def month_key(m):
        return "M12P" if m == 12 else f"M{m:02d}"

    for sid, grp in g:
        row = {"SITE_ID": sid, "YEAR": year}
        for m, sub in grp.groupby("month"):
            k = month_key(m)
            row[f"TMAX_{k}"] = sub.tasmax.mean()
            row[f"TMIN_{k}"] = sub.tasmin.mean()
            row[f"TMEAN_{k}"] = sub.tmean.mean()
            row[f"RAIN_{k}"] = sub.rainfall.sum()
        # seasonal aggregates
        djf = grp[grp.month.isin(ANTE_WINTER_MONTHS)]
        mam = grp[grp.month.isin(ANTE_SPRING_MONTHS)]
        for name, sub in (("DJF", djf), ("MAM", mam)):
            row[f"TMAX_{name}"] = sub.tasmax.mean()
            row[f"TMIN_{name}"] = sub.tmean.mean() if sub.empty else sub.tasmin.mean()
            row[f"TMEAN_{name}"] = sub.tmean.mean()
            row[f"RAIN_{name}"] = sub.rainfall.sum()
        # GDD: current-year days from GDD_START_DOY through window end
        cur = grp[(grp.year == year) & (grp.doy >= C.GDD_START_DOY)]
        gdd = np.maximum(0.0, cur.tmean - C.T_BASE).sum()
        row["GDD"] = float(gdd)
        out.append(row)
    return pd.DataFrame(out)


def build_all_features(years) -> pd.DataFrame:
    frames = [build_features_for_year(y) for y in years]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ==========================================================================
# 8. Merge
# ==========================================================================
def merge_onto_table(features: pd.DataFrame) -> pd.DataFrame:
    base = pd.read_parquet(C.CLEAN_TABLE)
    n0 = len(base)
    base_year = base["YEAR"].astype("Int64")
    feats = features.copy()
    feats["YEAR"] = feats["YEAR"].astype("Int64")
    merged = base.assign(YEAR=base_year).merge(feats, on=["SITE_ID", "YEAR"], how="left")
    assert len(merged) == n0, f"row count changed on merge: {n0} -> {len(merged)}"
    print(f"merge: {n0} rows in, {len(merged)} rows out (unchanged); "
          f"{merged['GDD'].notna().sum()} rows got climate features")
    return merged


# ==========================================================================
# Main
# ==========================================================================
def main() -> int:
    sites = load_sites()
    bng = sites[sites.is_bng].copy()
    print(f"sites total={len(sites)} | BNG-usable={len(bng)} | "
          f"non-BNG={int((sites.has_coords & ~sites.is_bng).sum())} | "
          f"no-coords={int((~sites.has_coords).sum())}")

    confirm_structure()

    years = list(range(C.YEAR_MIN, C.YEAR_MAX + 1))
    qc_all = []
    for y in years:
        qc, _ = extract_year(y, bng)
        qc_all.append(qc)
        print(f"  {y}: {qc.get('status')}"
              + (f" | maxdist={qc['max_cell_dist_m']:.0f}m nanSites={qc['n_nan_sites']}"
                 if qc.get("status") == "extracted" else ""))

    features = build_all_features(years)
    if features.empty:
        print("No features built (no cached climate yet). Stopping before merge.")
        return 1

    merged = merge_onto_table(features)
    C.FEATURE_TABLE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(C.FEATURE_TABLE, index=False)
    print(f"wrote {C.FEATURE_TABLE}")

    _write_report(sites, bng, qc_all, features, merged)
    return 0


def _write_report(sites, bng, qc_all, features, merged) -> None:
    extracted = [q for q in qc_all if q.get("status") == "extracted"]
    nan_ids = sorted({i for q in extracted for i in q.get("nan_ids", [])})
    max_dist = max((q["max_cell_dist_m"] for q in extracted), default=float("nan"))
    lines = [
        "# Climate merge report\n",
        "## Sites",
        f"- total distinct sites: {len(sites)}",
        f"- usable BNG (England/Scotland/Wales, valid EPSG:27700): {len(bng)}",
        f"- non-BNG coords (NI Irish Grid / Channel Islands / local, negative northings): "
        f"{int((sites.has_coords & ~sites.is_bng).sum())} — EXCLUDED from join (would mis-locate)",
        f"- sites with no coordinates: {int((~sites.has_coords).sum())} — cannot join",
        "\n## Join QC",
        f"- max site-to-cell distance: {max_dist:.0f} m (5km grid => expect <= ~3535 m)",
        f"- all-NaN (sea/edge) sites: {len(nan_ids)}"
        + (f" (snap-to-land {'ON' if SNAP_TO_LAND else 'OFF'})"),
        "\n## Features (site x year)",
        f"- rows: {len(features)}; columns: {len([c for c in features.columns if c not in ('SITE_ID','YEAR')])}",
        "- monthly mean Tmax/Tmin/Tmean + monthly total rainfall (prev Dec = *_M12P, Jan..May = *_M01..M05)",
        "- seasonal DJF & MAM mean Tmax/Tmin/Tmean + total rainfall",
        f"- GDD: sum of max(0, Tmean - {C.T_BASE}) from DOY {C.GDD_START_DOY} through 31 May",
        "\n## Leakage window",
        "- Predictors use ONLY climate from 1 Dec (year-1) through 31 May (year): the winter/spring",
        "  that PRECEDE the flight period. No post-flight climate is used.",
        "- Caveat: very early species (flying in Apr/early May) may have flight dates inside the",
        "  spring window; a per-event dynamic cut-off is a documented future refinement.",
        "\n## Assumptions",
        f"- coordinate names: x={STRUCT['x']} y={STRUCT['y']} time={STRUCT['time']}",
        "- data variable name == folder name (tasmax/tasmin/rainfall); Tmean=(tasmax+tasmin)/2",
        f"- nearest-cell selection via xarray method='nearest'; T_BASE={C.T_BASE}, GDD_START_DOY={C.GDD_START_DOY}",
        f"- merge join keys: SITE_ID + YEAR; base rows preserved ({len(merged)}).",
    ]
    C.REPORT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {C.REPORT_MD}")


if __name__ == "__main__":
    raise SystemExit(main())
