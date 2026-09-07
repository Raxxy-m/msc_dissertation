#!/usr/bin/env python
"""
Machine-learning rung — RQ2 (predictability & distribution shift) + RQ3 (does NAO add
signal beyond local temperature?). Target = FIRSTDAY.

Models plug into the shared harness, so their metrics/predictions land in the same
output/model_metrics.csv and model_predictions.parquet as the baselines, GDD, and ME models:
  * HGBModel — HistGradientBoosting (PRIMARY): handles NaN natively (no imputing the ~2.8%
    missing climate); species one-hot (see the class note on TreeSHAP).
  * RFModel  — RandomForest (comparator): needs complete X, so climate is median-imputed with
    TRAIN-ONLY medians; species one-hot. Different inductive bias.

Leakage discipline: only pre-flight predictors enter X (monthly/seasonal climate, GDD, the
toggleable NAO block, coordinates, species). Outcomes, abundance, TREND_*/NATIONAL_*, and raw
YEAR are asserted absent; encoders/imputers fit on train only; the target is never imputed.

Run:  .venv/bin/python -m scripts.modelling.ml_baseline
"""
from __future__ import annotations

import sys
import time
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

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

KEYS = ["SITE_ID", "SPECIES_NAME", "YEAR"]


# ==========================================================================
# Frame (Part 0) + feature assembly (Part 1)
# ==========================================================================
def numeric_features(use_nao: bool) -> list[str]:
    fams = ["temp", "rain", "coords"] + (["nao"] if use_nao else [])
    cols = []
    for f in fams:
        cols += C.ML_FEATURES[f]
    return cols


def all_numeric_features() -> list[str]:
    return numeric_features(use_nao=True)


def build_frame(verbose: bool = True) -> pd.DataFrame:
    """Reuse modelling_frame (rows + splits) and join the allowed predictors only."""
    log = (lambda *a: print(*a)) if verbose else (lambda *a: None)
    frame = pd.read_parquet(C.MODELLING_FRAME)
    n0 = len(frame)
    log(f"[frame] modelling_frame rows: {n0}")
    assert n0 == C.FRAME_ROWS, f"frame rows {n0} != expected {C.FRAME_ROWS}"

    need = all_numeric_features()
    # NAO_DJFM / NAO_DJFM_LAG1 already live on the frame; join only the missing predictors
    # (avoids _x/_y suffix collisions).
    to_join = [c for c in need if c not in frame.columns]
    ft = pd.read_parquet(C.FEATURE_TABLE, columns=KEYS + to_join)
    frame = frame.merge(ft, on=KEYS, how="left")
    assert len(frame) == n0, f"row count changed on climate join: {n0} -> {len(frame)}"
    missing = [c for c in need if c not in frame.columns]
    assert not missing, f"expected predictors absent after join: {missing}"
    log(f"[frame] after predictor join ({len(to_join)} new numeric cols; "
        f"{len(need)} total available): {len(frame)} (unchanged)")

    # target completeness (never imputed) — frame guarantees it, assert anyway
    assert frame["FIRSTDAY"].notna().all(), "FIRSTDAY has NaN — target must be complete"
    return frame


def assert_no_leakage(feature_cols: list[str], allow: frozenset = frozenset()) -> None:
    """Hard guard: none of the forbidden outcome/derived/raw-year columns may be features.

    `allow` is an EXPLICIT, documented opt-in (only the raw-YEAR grouped-CV sensitivity uses
    it); it exempts those columns from the forbidden/unknown checks so a deliberate,
    clearly-flagged feature is not mistaken for accidental leakage.
    """
    bad = [c for c in feature_cols if c not in allow
           and (c in set(C.ML_FORBIDDEN) or c.startswith(C.ML_FORBIDDEN_PREFIXES))]
    assert not bad, f"LEAKAGE: forbidden columns in feature set: {bad}"
    # species identity is the only categorical; everything else must be an allowed family
    allowed = set(all_numeric_features()) | {C.ML_SPECIES_COL} | set(allow)
    unknown = [c for c in feature_cols if c not in allowed]
    assert not unknown, f"unexpected feature columns (not in allowed families): {unknown}"


# ==========================================================================
# Models (harness fit/predict interface)
# ==========================================================================
class HGBModel(H.Model):
    """HistGradientBoosting: native NaN handling for climate; species ONE-HOT encoded.

    NOTE on encoding: we use one-hot (not HGB's native categorical) specifically so that
    TreeSHAP is reliable — TreeExplainer mis-attributes HGB native-categorical splits
    (additivity error ~95 days; species credit ~0). One-hot restores exact additivity at a
    negligible accuracy cost (temporal MAE 21.22 vs 21.10 native). Climate NaN is still
    handled natively (no imputation)."""

    def __init__(self, name, use_nao=True, extra_cols=None):
        self.name = name
        self.use_nao = use_nao
        self.num_cols = numeric_features(use_nao)
        self.extra_cols = extra_cols or []      # e.g. ["YEAR"] for the grouped-CV sensitivity
        self.species_levels = None

    def feature_names(self):
        return self.num_cols + self.extra_cols + list(self.species_levels)

    def n_numeric(self):
        return len(self.num_cols) + len(self.extra_cols)

    def _X(self, df):
        # numeric block (NaN preserved for HGB) + one-hot species
        X = df[self.num_cols + self.extra_cols].to_numpy(dtype="float64")
        oh = pd.get_dummies(pd.Categorical(df[C.ML_SPECIES_COL],
                                           categories=self.species_levels))
        oh = oh.reindex(columns=self.species_levels, fill_value=0).to_numpy("float64")
        return np.column_stack([X, oh])

    def fit(self, train_df):
        # extra_cols is the deliberate, grouped-CV-only sensitivity opt-in (e.g. YEAR).
        assert_no_leakage(self.num_cols + self.extra_cols + [C.ML_SPECIES_COL],
                          allow=frozenset(self.extra_cols))
        self.species_levels = sorted(train_df[C.ML_SPECIES_COL].unique())
        self.model_ = HistGradientBoostingRegressor(**C.HGB_PARAMS)
        self.model_.fit(self._X(train_df), train_df["FIRSTDAY"].to_numpy("float64"))
        return self

    def predict(self, df):
        return self.model_.predict(self._X(df))


class RFModel(H.Model):
    """RandomForest with TRAIN-ONLY median imputation + one-hot species."""

    def __init__(self, name, use_nao=True):
        self.name = name
        self.use_nao = use_nao
        self.num_cols = numeric_features(use_nao)
        self.species_levels = None

    def _X(self, df):
        num = df[self.num_cols].to_numpy("float64")
        num = np.where(np.isnan(num), self.medians_, num)      # train medians
        sp = pd.Categorical(df[C.ML_SPECIES_COL], categories=self.species_levels)
        oh = pd.get_dummies(sp).reindex(columns=self.species_levels, fill_value=0)
        return np.column_stack([num, oh.to_numpy("float64")])

    def fit(self, train_df):
        assert_no_leakage(self.num_cols + [C.ML_SPECIES_COL])
        self.species_levels = sorted(train_df[C.ML_SPECIES_COL].unique())
        self.medians_ = np.nanmedian(train_df[self.num_cols].to_numpy("float64"), axis=0)
        self.medians_ = np.where(np.isnan(self.medians_), 0.0, self.medians_)
        self.model_ = RandomForestRegressor(**C.RF_PARAMS)
        self.model_.fit(self._X(train_df), train_df["FIRSTDAY"].to_numpy("float64"))
        return self

    def predict(self, df):
        return self.model_.predict(self._X(df))


# ==========================================================================
# Merge into shared artefacts
# ==========================================================================
def evaluate_and_merge(frame, models) -> pd.DataFrame:
    preds = []
    for m in models:
        t0 = time.time()
        p = H.evaluate(m, frame)
        preds.append(p)
        print(f"  [{m.name}] evaluated both regimes in {time.time()-t0:.0f}s")
    preds = pd.concat(preds, ignore_index=True)

    allp = pd.read_parquet(C.PREDICTIONS_PATH) if C.PREDICTIONS_PATH.exists() else pd.DataFrame()
    names = {m.name for m in models}
    if len(allp):
        allp = allp[~allp["model"].isin(names)]
    allp = pd.concat([allp, preds], ignore_index=True)
    allp.to_parquet(C.PREDICTIONS_PATH, index=False)

    mrows = []
    for (mdl, reg), g in allp.groupby(["model", "regime"]):
        mt = H.metrics_table(g)
        mt.insert(0, "regime", reg); mt.insert(0, "model", mdl)
        mrows.append(mt)
    pd.concat(mrows, ignore_index=True).to_csv(C.METRICS_PATH, index=False)
    return preds


def main() -> int:
    from scripts.modelling.ml_report import build_report

    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    frame = build_frame()

    feats = all_numeric_features() + [C.ML_SPECIES_COL]
    assert_no_leakage(feats)
    print(f"\n[features] {len(feats)} features across families "
          f"{{temp:{len(C.ML_FEATURES['temp'])}, rain:{len(C.ML_FEATURES['rain'])}, "
          f"nao:{len(C.ML_FEATURES['nao'])}, coords:{len(C.ML_FEATURES['coords'])}, "
          f"species:1(cat)}}")
    print(f"[features] leakage guard passed — forbidden columns absent "
          f"(outcomes, SITE_INDEX, TREND_*/NATIONAL_*, raw YEAR).")

    models = [
        HGBModel("ml_hgb", use_nao=True),
        HGBModel("ml_hgb_nonao", use_nao=False),
        RFModel("ml_rf", use_nao=True),
        RFModel("ml_rf_nonao", use_nao=False),
    ]
    print("\n[fit] evaluating ML models through the shared harness (both regimes) ...")
    evaluate_and_merge(frame, models)

    # optional YEAR sensitivity — grouped-CV ONLY (a raw-year feature can't extrapolate to
    # the temporal test), reported as a delta, NOT merged into the shared table.
    year_delta = _year_sensitivity(frame)

    build_report(frame, year_delta=year_delta)
    return 0


def _year_sensitivity(frame) -> dict:
    """Grouped-CV MAE for HGB with a raw YEAR feature added, vs without. CV regime only."""
    base = HGBModel("hgb_year_base", use_nao=True)
    withyr = HGBModel("hgb_year", use_nao=True, extra_cols=["YEAR"])
    out = {}
    for tag, mdl in (("no_year", base), ("with_year", withyr)):
        errs = []
        for fold in range(C.CV_K):
            tr = frame[frame["cv_fold"] != fold]
            te = frame[frame["cv_fold"] == fold]
            mdl.fit(tr)
            errs.append(np.abs(mdl.predict(te) - te["FIRSTDAY"].to_numpy("float64")))
        out[tag] = float(np.concatenate(errs).mean())
    out["delta_mae"] = out["with_year"] - out["no_year"]
    return out


if __name__ == "__main__":
    raise SystemExit(main())
