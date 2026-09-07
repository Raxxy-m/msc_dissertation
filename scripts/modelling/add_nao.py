#!/usr/bin/env python
"""
Add the winter (DJFM) station-based NAO index to feature_table.parquet.

Source: Hurrell STATION-BASED DJFM NAO index, NCAR Climate Data Guide. We use the
pre-computed DJFM *seasonal* ascii, not an average of monthly values (the seasonal
normalisation differs).

Year convention (verified against known winters): the DJFM value for year N = mean of
Dec(N-1)..Mar(N), the winter PRECEDING flight year N, so it joins directly onto YEAR = N.
NAO_DJFM_LAG1 = the value for year N-1.

Run:  .venv/bin/python -m scripts.modelling.add_nao
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.modelling import config as C
else:
    from . import config as C


# --------------------------------------------------------------------------
# Download / parse the raw DJFM ascii into a tidy (year, nao_djfm) csv
# --------------------------------------------------------------------------
def _parse_ascii(text: str) -> pd.DataFrame:
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue                      # header / blank lines
        try:
            yr, val = int(parts[0]), float(parts[1])
        except ValueError:
            continue
        rows.append((yr, val))
    df = pd.DataFrame(rows, columns=["year", "nao_djfm"]).sort_values("year")
    df.loc[df["nao_djfm"] == C.NAO_MISSING, "nao_djfm"] = pd.NA
    return df.reset_index(drop=True)


def load_or_download_nao() -> tuple[pd.DataFrame, str]:
    """Return (df, access_date_iso). Uses cached csv if present, else downloads."""
    if C.NAO_CSV.exists():
        df = pd.read_csv(C.NAO_CSV)
        access = date.fromtimestamp(C.NAO_CSV.stat().st_mtime).isoformat()
        return df, access

    import requests
    resp = requests.get(C.NAO_URL, timeout=120)
    resp.raise_for_status()
    df = _parse_ascii(resp.text)
    C.NAO_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.NAO_CSV, index=False)
    return df, date.today().isoformat()


# --------------------------------------------------------------------------
# Sanity check the year convention against known winters
# --------------------------------------------------------------------------
def verify_convention(df: pd.DataFrame) -> list[str]:
    """Assert well-documented winters have the expected sign/magnitude.

    2010 (Dec2009-Mar2010): famously cold, strongly NEGATIVE NAO.
    2020 (Dec2019-Mar2020): mild & stormy, strongly POSITIVE NAO.
    1989/1990: strongly positive.  1996: negative.
    """
    s = df.set_index("year")["nao_djfm"]
    checks = []

    def val(y):
        return float(s.loc[y]) if y in s.index else float("nan")

    assert val(2010) < -2, f"expected 2010 DJFM NAO strongly negative, got {val(2010)}"
    assert val(2020) > 2, f"expected 2020 DJFM NAO strongly positive, got {val(2020)}"
    assert val(1989) > 2 and val(1996) < 0, "1989/1996 sign check failed"
    checks.append(f"2010 = {val(2010):+.2f} (negative, cold winter 2009/10) ✓")
    checks.append(f"2020 = {val(2020):+.2f} (positive, stormy winter 2019/20) ✓")
    checks.append(f"1989 = {val(1989):+.2f}, 1996 = {val(1996):+.2f} ✓")
    return checks


# --------------------------------------------------------------------------
# Join onto the feature table
# --------------------------------------------------------------------------
def add_nao_columns(nao: pd.DataFrame) -> dict:
    base = pd.read_parquet(C.FEATURE_TABLE)
    n0 = len(base)

    # Drop pre-existing NAO cols so the script is idempotent (re-runnable).
    base = base.drop(columns=[c for c in ("NAO_DJFM", "NAO_DJFM_LAG1") if c in base.columns])

    lut = nao.dropna(subset=["nao_djfm"]).set_index("year")["nao_djfm"]
    yr = base["YEAR"].astype("Int64")
    base["NAO_DJFM"] = yr.map(lut).astype("float64")
    base["NAO_DJFM_LAG1"] = (yr - 1).map(lut).astype("float64")

    assert len(base) == n0 == C.EXPECTED_ROWS, (
        f"row count changed: {n0} -> {len(base)} (expected {C.EXPECTED_ROWS})")

    base.to_parquet(C.FEATURE_TABLE, index=False)

    yrs = base["YEAR"].dropna().astype(int)
    covered = sorted(set(yrs[base["NAO_DJFM"].notna()].unique()))
    missing = sorted(set(range(C.YEAR_MIN, C.YEAR_MAX + 1)) - set(covered))
    return {
        "rows": len(base),
        "n_nao": int(base["NAO_DJFM"].notna().sum()),
        "n_nao_lag1": int(base["NAO_DJFM_LAG1"].notna().sum()),
        "nao_year_min": int(nao["year"].min()),
        "nao_year_max": int(nao["year"].max()),
        "covered_years": covered,
        "missing_years_in_range": missing,
    }


def main() -> int:
    nao, access = load_or_download_nao()
    print(f"NAO index: {len(nao)} years, {nao['year'].min()}-{nao['year'].max()} "
          f"(access {access})")
    for line in verify_convention(nao):
        print("  convention:", line)

    qc = add_nao_columns(nao)
    print(f"joined onto feature_table: {qc['rows']} rows (unchanged); "
          f"{qc['n_nao']} rows got NAO_DJFM, {qc['n_nao_lag1']} got LAG1")
    if qc["missing_years_in_range"]:
        print(f"  NOTE: no station NAO for years {qc['missing_years_in_range']} "
              f"(index ends {qc['nao_year_max']})")

    # stash QC for the report writer
    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    qc["access_date"] = access
    pd.Series(qc, dtype="object").to_json(C.OUTPUT_DIR / "_nao_qc.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
