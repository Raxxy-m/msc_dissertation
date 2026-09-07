#!/usr/bin/env python
"""
GDD (growing-degree-day) PROCESS baseline for FIRSTDAY.

Mechanistic rung: an insect emerges once enough thermal energy has accumulated. Daily
Tmean = (tasmax+tasmin)/2, daily GDD = max(0, Tmean - T_BASE), cumulative from
GDD_START_DOY. Predicted emergence = first DOY whose cumulative GDD >= a species-specific
critical sum S*, calibrated per species on TRAIN rows to minimise MAE (optionally fitting
T_BASE too, via FIT_TBASE).

Uses the cached per-site DAILY climate (needs the accumulation trajectory to find the
crossing day, not just the summary GDD column). The cache holds the FULL calendar year, so
we can accumulate to each event's emergence day even for summer flyers -- and since that
only uses days BEFORE the event, it is not leakage (the Dec-May cap is for the fixed ML
aggregate features, not this model).

Run standalone (calibrate + score via the harness, both regimes):
    .venv/bin/python -m scripts.modelling.gdd_baseline
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.modelling import config as C
    from scripts.modelling import harness as H
else:
    from . import config as C
    from . import harness as H


# ==========================================================================
# Daily-cache coverage check (reported)
# ==========================================================================
def check_cache_coverage() -> dict:
    """Confirm whether the daily cache spans the full calendar year or only Dec-May."""
    info = {}
    for y in (C.YEAR_MIN, (C.YEAR_MIN + C.YEAR_MAX) // 2, C.YEAR_MAX):
        d = pd.read_parquet(C.SITE_DAILY_DIR / f"year={y}.parquet", columns=["month", "doy"])
        info[y] = (int(d["month"].min()), int(d["month"].max()),
                   int(d["doy"].min()), int(d["doy"].max()))
    months = {v[1] for v in info.values()}
    info["full_year"] = min(m0 for m0, *_ in info.values()) == 1 and max(months) == 12
    return info


# ==========================================================================
# Cumulative-GDD trajectory store (built once from the daily cache)
# ==========================================================================
def build_stores(site_years: set, t_base: float, need_tmean: bool):
    """Return cum[(site,year)] = cumulative-GDD array (index 0 == DOY 1) at t_base.

    If need_tmean, also return tmean[(site,year)] = daily Tmean array (for per-species
    T_BASE refits). Only site-years present in `site_years` are stored.
    Assumes each year's daily series is contiguous DOY 1..N (verified: full-year cache),
    so predicted DOY = searchsorted(cum, S*) + 1.
    """
    want_sites_by_year = {}
    for s, y in site_years:
        want_sites_by_year.setdefault(y, set()).add(s)

    cum_store, tmean_store = {}, {}
    for y in sorted(want_sites_by_year):
        d = pd.read_parquet(C.SITE_DAILY_DIR / f"year={y}.parquet",
                            columns=["SITE_ID", "doy", "tmean"])
        d = d[d["SITE_ID"].isin(want_sites_by_year[y])].sort_values(["SITE_ID", "doy"])
        for sid, g in d.groupby("SITE_ID", sort=False):
            tm = g["tmean"].to_numpy(float)
            # Sea/edge sites carry NaN series; one NaN poisons cumsum, so skip the whole
            # site-year -- its events then legitimately ABSTAIN.
            if not np.isfinite(tm).all():
                continue
            cum_store[(sid, y)] = np.cumsum(np.clip(tm - t_base, 0.0, None)).astype("float32")
            if need_tmean:
                tmean_store[(sid, y)] = g["tmean"].to_numpy("float32")
    return cum_store, (tmean_store if need_tmean else None)


def _cum_from_tmean(tm: np.ndarray, t_base: float) -> np.ndarray:
    return np.cumsum(np.clip(tm.astype(float) - t_base, 0.0, None)).astype("float32")


# ==========================================================================
# GDD model
# ==========================================================================
class GDDModel(H.Model):
    name = "gdd_process"

    def __init__(self, cum_store, tmean_store=None, fit_tbase=None):
        self.cum_store = cum_store
        self.tmean_store = tmean_store
        self.fit_tbase = C.FIT_TBASE if fit_tbase is None else fit_tbase
        self.params_ = {}          # species -> (S*, T_BASE)
        self.global_param_ = None  # (S*, T_BASE) fallback for thin species

    # ---- helpers ------------------------------------------------------
    def _obs_doy(self, df):
        return (C.april1_doy_vec(df["YEAR"].values) + df["FIRSTDAY"].values).astype(int)

    def _cum_for(self, sid, yr, t_base):
        if abs(t_base - C.T_BASE) < 1e-9:
            return self.cum_store.get((sid, yr))
        tm = self.tmean_store.get((sid, yr)) if self.tmean_store else None
        return None if tm is None else _cum_from_tmean(tm, t_base)

    def _calibrate(self, events, t_base):
        """events: list of (cum_array, obs_doy). Return (best_S*, best_MAE, cross_frac)."""
        obs = np.array([o for _, o in events], dtype=float)
        obs_sum = np.array([c[min(o, len(c)) - 1] for c, o in events], dtype=float)
        lo, hi = np.percentile(obs_sum, [2, 98])
        if hi <= lo:
            hi = lo + 1.0
        cands = np.linspace(lo, hi, C.SSTAR_N_CANDIDATES)

        # predicted DOY per (event, candidate): searchsorted on each event's cum
        n = len(events)
        pred = np.full((n, len(cands)), np.nan, dtype="float32")
        for i, (cum, _) in enumerate(events):
            idx = np.searchsorted(cum, cands, side="left")
            crossed = idx < len(cum)
            pred[i, crossed] = idx[crossed] + 1          # DOY (cum index 0 == DOY 1)

        best = None
        for j in range(len(cands)):
            pj = pred[:, j]
            cross = np.isfinite(pj)
            if cross.mean() < C.SSTAR_MIN_CROSS_FRAC:
                continue
            mae = float(np.abs(pj[cross] - obs[cross]).mean())
            if best is None or mae < best[1]:
                best = (float(cands[j]), mae, float(cross.mean()))
        if best is None:                                  # fall back to central obs_sum
            s = float(np.median(obs_sum))
            return s, np.nan, np.nan
        return best

    def _events_for(self, df, t_base):
        out = []
        obs_doy = self._obs_doy(df)
        for (sid, yr), o in zip(zip(df["SITE_ID"], df["YEAR"]), obs_doy):
            cum = self._cum_for(sid, yr, t_base)
            if cum is not None:
                out.append((cum, int(o)))
        return out

    # ---- interface ----------------------------------------------------
    def fit(self, train_df):
        self.params_.clear()
        tbase_grid = C.TBASE_GRID if self.fit_tbase else [C.T_BASE]

        # global fallback for thin species: best single S* on a capped random sample
        # (a fallback need not be exact; capping keeps the S* grid search bounded).
        GLOBAL_CAP = 20_000
        gtrain = (train_df.sample(GLOBAL_CAP, random_state=C.CV_SEED)
                  if len(train_df) > GLOBAL_CAP else train_df)
        gbest = None
        for tb in tbase_grid:
            ev = self._events_for(gtrain, tb)
            if not ev:
                continue
            s, mae, _ = self._calibrate(ev, tb)
            if gbest is None or (np.isfinite(mae) and mae < gbest[2]):
                gbest = (s, tb, mae if np.isfinite(mae) else np.inf)
        self.global_param_ = (gbest[0], gbest[1]) if gbest else (float("nan"), C.T_BASE)

        for sp, g in train_df.groupby("SPECIES_NAME"):
            if len(g) < C.GDD_MIN_TRAIN_EVENTS:
                continue                                  # thin species -> global fallback
            sbest = None
            for tb in tbase_grid:
                ev = self._events_for(g, tb)
                if not ev:
                    continue
                s, mae, _ = self._calibrate(ev, tb)
                if sbest is None or (np.isfinite(mae) and mae < sbest[2]):
                    sbest = (s, tb, mae if np.isfinite(mae) else np.inf)
            if sbest is not None:
                self.params_[sp] = (sbest[0], sbest[1])
        return self

    def predict(self, df):
        obs_doy_ref = C.april1_doy_vec(df["YEAR"].values)
        out = np.full(len(df), np.nan)
        for i, (sid, yr, sp, a1) in enumerate(
                zip(df["SITE_ID"], df["YEAR"], df["SPECIES_NAME"], obs_doy_ref)):
            sstar, tb = self.params_.get(sp, self.global_param_)
            if not np.isfinite(sstar):
                continue
            cum = self._cum_for(sid, yr, tb)
            if cum is None:
                continue
            idx = int(np.searchsorted(cum, sstar, side="left"))
            if idx >= len(cum):
                continue                                  # non-crossing -> abstain (NaN)
            pred_doy = idx + 1
            out[i] = pred_doy - a1                         # back to FIRSTDAY units
        return out


# ==========================================================================
# Standalone: calibrate, score, save calibration + non-crossing report
# ==========================================================================
def non_crossing_report(model: GDDModel, frame: pd.DataFrame) -> pd.DataFrame:
    """Per-species non-crossing counts using each species' final (temporal-train) S*."""
    pred = model.predict(frame)
    df = frame[["SPECIES_NAME"]].copy()
    df["abstain"] = ~np.isfinite(pred)
    r = (df.groupby("SPECIES_NAME")["abstain"]
           .agg(n="size", n_noncross="sum").reset_index())
    r["frac_noncross"] = r["n_noncross"] / r["n"]
    return r.sort_values("frac_noncross", ascending=False)


def main() -> int:
    cov = check_cache_coverage()
    print("daily-cache coverage:", cov)

    frame = H.load_frame()
    site_years = set(zip(frame["SITE_ID"], frame["YEAR"]))
    print(f"building GDD trajectories for {len(site_years)} site-years "
          f"(T_BASE={C.T_BASE}, fit_tbase={C.FIT_TBASE}) ...")
    cum_store, tmean_store = build_stores(site_years, C.T_BASE, need_tmean=C.FIT_TBASE)

    model = GDDModel(cum_store, tmean_store)
    preds = H.evaluate(model, frame)          # fits per regime/fold internally

    # persist per-species calibration from a full-train fit (temporal train)
    model.fit(frame[frame["temporal_split"] == "train"])
    calib = pd.DataFrame(
        [{"SPECIES_NAME": sp, "S_star": s, "T_BASE": tb}
         for sp, (s, tb) in sorted(model.params_.items())])
    calib.to_csv(C.GDD_CALIB_PATH, index=False)

    nc = non_crossing_report(model, frame)
    print(f"per-species calibrated: {len(model.params_)} species "
          f"(global fallback S*={model.global_param_[0]:.0f}, T_BASE={model.global_param_[1]})")
    print(f"total non-crossing (abstain) rows: {int(nc['n_noncross'].sum())} "
          f"/ {len(frame)} ({nc['n_noncross'].sum()/len(frame):.2%})")

    # merge GDD predictions into the shared predictions/metrics artefacts
    allp = pd.read_parquet(C.PREDICTIONS_PATH) if C.PREDICTIONS_PATH.exists() else pd.DataFrame()
    allp = allp[allp["model"] != model.name] if len(allp) else allp
    allp = pd.concat([allp, preds], ignore_index=True)
    allp.to_parquet(C.PREDICTIONS_PATH, index=False)

    mrows = []
    for (mdl, reg), g in allp.groupby(["model", "regime"]):
        mt = H.metrics_table(g)
        mt.insert(0, "regime", reg); mt.insert(0, "model", mdl)
        mrows.append(mt)
    pd.concat(mrows, ignore_index=True).to_csv(C.METRICS_PATH, index=False)

    ov = H.metrics_table(preds[preds.regime == "temporal"]).iloc[0]
    ovc = H.metrics_table(preds[preds.regime == "grouped_cv"]).iloc[0]
    print(f"\nGDD temporal  : MAE={ov['MAE']:.2f} RMSE={ov['RMSE']:.2f} "
          f"R2={ov['R2']:.3f} bias={ov['bias']:.2f} n={int(ov['n'])}")
    print(f"GDD groupedCV : MAE={ovc['MAE']:.2f} RMSE={ovc['RMSE']:.2f} "
          f"R2={ovc['R2']:.3f} bias={ovc['bias']:.2f} n={int(ovc['n'])}")
    print(f"wrote {C.GDD_CALIB_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
