#!/usr/bin/env python
"""
Fix the 2024 winter-NAO gap in feature_table.parquet.

The Hurrell station DJFM index ends 2023, so flight-year 2024 has NAO_DJFM = NaN -- and
the temporal test set (YEAR > 2014) includes 2024, so NAO models would silently drop the
most recent, most climate-relevant test year.

Fix (transparent -- never concatenates the two differently-normalised products): build a
DJFM series from the NOAA CPC monthly NAO ourselves, regress Hurrell on CPC over all
overlapping years, and rescale the CPC 2024 value onto the Hurrell scale. Write it into
NAO_DJFM for 2024 with a NAO_SOURCE column ('hurrell'/'cpc_rescaled'/<NA>) so it is
auditable. Re-asserts the row count, then rebuilds modelling_frame.

Run:  .venv/bin/python -m scripts.modelling.fix_nao_2024
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.modelling import config as C
else:
    from . import config as C


# --------------------------------------------------------------------------
# CPC monthly -> DJFM seasonal series (our own aggregation)
# --------------------------------------------------------------------------
def _parse_cpc_monthly(text: str) -> pd.DataFrame:
    rows = []
    for line in text.splitlines():
        p = line.split()
        if len(p) != 3:
            continue
        try:
            rows.append((int(p[0]), int(p[1]), float(p[2])))
        except ValueError:
            continue
    return pd.DataFrame(rows, columns=["year", "month", "nao"])


def build_cpc_djfm(monthly: pd.DataFrame) -> pd.DataFrame:
    """DJFM(N) = mean of Dec(N-1)+Jan(N)+Feb(N)+Mar(N); requires all 4 months present."""
    idx = monthly.set_index(["year", "month"])["nao"]
    rows = []
    for N in range(int(monthly.year.min()) + 1, int(monthly.year.max()) + 1):
        try:
            vals = [idx[(N - 1, 12)], idx[(N, 1)], idx[(N, 2)], idx[(N, 3)]]
        except KeyError:
            continue
        if all(np.isfinite(vals)):
            rows.append((N, float(np.mean(vals))))
    return pd.DataFrame(rows, columns=["year", "cpc_djfm"])


def load_or_download_cpc() -> tuple[pd.DataFrame, str]:
    if C.CPC_NAO_CSV.exists():
        df = pd.read_csv(C.CPC_NAO_CSV)
        return df, date.fromtimestamp(C.CPC_NAO_CSV.stat().st_mtime).isoformat()
    import requests
    r = requests.get(C.CPC_NAO_URL, timeout=120)
    r.raise_for_status()
    djfm = build_cpc_djfm(_parse_cpc_monthly(r.text))
    C.CPC_NAO_CSV.parent.mkdir(parents=True, exist_ok=True)
    djfm.to_csv(C.CPC_NAO_CSV, index=False)
    return djfm, date.today().isoformat()


# --------------------------------------------------------------------------
# Overlap regression Hurrell ~ CPC and 2024 rescale
# --------------------------------------------------------------------------
def calibrate(hurrell: pd.DataFrame, cpc: pd.DataFrame) -> dict:
    m = hurrell.rename(columns={"nao_djfm": "hurrell"}).merge(cpc, on="year").dropna()
    x, y = m["cpc_djfm"].to_numpy(float), m["hurrell"].to_numpy(float)
    b, a = np.polyfit(x, y, 1)                       # hurrell = a + b*cpc
    r = float(np.corrcoef(x, y)[0, 1])
    yhat = a + b * x
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    return {"a": float(a), "b": float(b), "r": r, "r2": r * r, "rmse": rmse,
            "n_overlap": int(len(m)),
            "overlap_years": [int(m.year.min()), int(m.year.max())]}


# --------------------------------------------------------------------------
# Write into feature_table
# --------------------------------------------------------------------------
def apply_fix() -> dict:
    hurrell = pd.read_csv(C.NAO_CSV)                  # year, nao_djfm (Hurrell, ends 2023)
    cpc, access = load_or_download_cpc()
    cal = calibrate(hurrell, cpc)

    cpc_2024 = cpc.loc[cpc.year == 2024, "cpc_djfm"]
    if cpc_2024.empty:
        raise SystemExit("CPC DJFM has no 2024 value; cannot fill the gap.")
    cpc_2024 = float(cpc_2024.iloc[0])
    rescaled_2024 = cal["a"] + cal["b"] * cpc_2024

    ft = pd.read_parquet(C.FEATURE_TABLE)
    n0 = len(ft)

    # NAO_SOURCE is derived from YEAR COVERAGE (stable across reruns -> idempotent), NOT
    # from the current NaN state: labelling by NaN would relabel the already-filled 2024
    # rows as 'hurrell' on a second run and erase the audit trail.
    hurrell_years = set(hurrell.dropna(subset=["nao_djfm"])["year"].astype(int))
    yr = ft["YEAR"].astype("Int64")
    is_hurrell = yr.isin(hurrell_years) & ft["NAO_DJFM"].notna()
    is_2024 = (yr == 2024)

    # Always (re)write the 2024 value to the CPC-rescaled figure so reruns are stable.
    ft.loc[is_2024, "NAO_DJFM"] = rescaled_2024
    n_fill = int(is_2024.sum())

    src = pd.Series(pd.NA, index=ft.index, dtype="object")
    src[is_hurrell] = "hurrell"
    src[is_2024] = "cpc_rescaled"
    ft["NAO_SOURCE"] = src

    assert len(ft) == n0 == C.EXPECTED_ROWS, (
        f"row count changed: {n0} -> {len(ft)} (expected {C.EXPECTED_ROWS})")
    ft.to_parquet(C.FEATURE_TABLE, index=False)

    qc = {"access_date": access, "cpc_2024_djfm": cpc_2024,
          "rescaled_2024_hurrell": float(rescaled_2024),
          "n_rows_filled_2024": n_fill,
          "n_nao_after": int(ft["NAO_DJFM"].notna().sum()),
          "n_nao_before": int(n0 - ft["NAO_DJFM"].isna().sum() - n_fill),
          "source_counts": ft["NAO_SOURCE"].value_counts(dropna=False).to_dict(),
          **{f"cal_{k}": v for k, v in cal.items()}}
    return qc


def main() -> int:
    qc = apply_fix()
    print(f"CPC DJFM 2024 = {qc['cpc_2024_djfm']:+.3f}  ->  rescaled onto Hurrell scale "
          f"= {qc['rescaled_2024_hurrell']:+.3f}")
    print(f"  overlap regression: hurrell = {qc['cal_a']:+.3f} + {qc['cal_b']:.3f}*cpc  "
          f"(r={qc['cal_r']:.3f}, n={qc['cal_n_overlap']}, "
          f"{qc['cal_overlap_years'][0]}-{qc['cal_overlap_years'][1]})")
    print(f"  filled {qc['n_rows_filled_2024']} rows for flight-year 2024; "
          f"NAO_DJFM non-null now {qc['n_nao_after']:,} (was {qc['n_nao_before']:,})")
    print(f"  NAO_SOURCE counts: {qc['source_counts']}")

    # rebuild the shared frame so downstream models pick up the filled 2024 value
    from scripts.modelling import harness as H
    H.build_modelling_frame(verbose=False)
    frame = pd.read_parquet(C.MODELLING_FRAME)
    n2024 = int(((frame.YEAR == 2024) & frame.NAO_DJFM.notna()).sum())
    print(f"  rebuilt modelling_frame: 2024 rows with NAO now = {n2024:,}")

    (C.OUTPUT_DIR / "models").mkdir(parents=True, exist_ok=True)
    (C.OUTPUT_DIR / "_nao2024_qc.json").write_text(json.dumps(qc, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
