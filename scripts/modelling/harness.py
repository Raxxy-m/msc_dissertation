#!/usr/bin/env python
"""
Shared evaluation harness for FIRSTDAY models.

- build_modelling_frame(): load feature_table, filter to valid FIRSTDAY on GB/BNG sites
  (logging row counts at every step), and assign the two eval splits ONCE so every model
  is scored on identical data.
- Model interface: any object with .fit(train_df) and .predict(df) -> FIRSTDAY array
  (NaN where it abstains). Naive baselines: species mean, site x species mean, persistence.
- evaluate(model): run both regimes and return tidy predictions.

Regimes: (a) temporal hold-out (train YEAR <= CUTOFF, test after -- distribution shift);
(b) grouped k-fold CV by SITE_ID (a site never spans train & test).

Run standalone to (re)build the frame and score the naive baselines:
    .venv/bin/python -m scripts.modelling.harness
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.modelling import config as C
else:
    from . import config as C

KEYS = ["SITE_ID", "SPECIES_NAME", "YEAR"]


# ==========================================================================
# Modelling frame + fixed split assignment
# ==========================================================================
def _bng_site_ids() -> set:
    """SITE_IDs present in the daily climate cache == the GB/BNG sites (upstream filter)."""
    f = C.SITE_DAILY_DIR / f"year={C.YEAR_MAX}.parquet"
    return set(pd.read_parquet(f, columns=["SITE_ID"])["SITE_ID"].unique())


def _assign_group_kfold(counts: pd.Series, k: int, seed: int) -> dict:
    """Assign sites to k folds, greedily balancing event counts (deterministic, no sklearn):
    shuffle, then drop the largest remaining site into the currently-lightest fold."""
    order = counts.sample(frac=1.0, random_state=seed).sort_values(ascending=False)
    load = np.zeros(k)
    assign = {}
    for site, n in order.items():
        f = int(np.argmin(load))
        assign[site] = f
        load[f] += n
    return assign


def build_modelling_frame(verbose: bool = True) -> pd.DataFrame:
    log = (lambda *a: print(*a)) if verbose else (lambda *a: None)
    funnel = {}

    df = pd.read_parquet(
        C.FEATURE_TABLE,
        columns=KEYS + ["COMMON_NAME", "FIRSTDAY", "FLAG_DOY_IMPLAUSIBLE",
                        "NAO_DJFM", "NAO_DJFM_LAG1"],
    )
    funnel["feature_table_rows"] = len(df)
    log(f"[frame] feature_table rows: {len(df)}")

    df = df.dropna(subset=["FIRSTDAY", "SITE_ID", "SPECIES_NAME", "YEAR"])
    funnel["require_firstday_and_keys"] = len(df)
    log(f"[frame] after require FIRSTDAY + keys present: {len(df)}")

    df["YEAR"] = df["YEAR"].astype(int)
    df = df[(df["YEAR"] >= C.YEAR_MIN) & (df["YEAR"] <= C.YEAR_MAX)]
    funnel["year_in_climate_range"] = len(df)
    log(f"[frame] after YEAR in [{C.YEAR_MIN},{C.YEAR_MAX}] (climate coverage): {len(df)}")

    bng = _bng_site_ids()
    df = df[df["SITE_ID"].isin(bng)]
    funnel["gb_bng_sites"] = len(df)
    log(f"[frame] after keep GB/BNG sites (in daily cache, n={len(bng)}): {len(df)}")

    # FIRSTDAY is days-after-1-April; convert to true DOY and require a real calendar day.
    df["DOY"] = C.april1_doy_vec(df["YEAR"].values) + df["FIRSTDAY"].values
    df = df[(df["DOY"] >= C.DOY_MIN) & (df["DOY"] <= C.DOY_MAX)]
    funnel["implied_doy_valid"] = len(df)
    log(f"[frame] after implied DOY in [{C.DOY_MIN},{C.DOY_MAX}]: {len(df)}")

    df = df[~df["FLAG_DOY_IMPLAUSIBLE"].fillna(False)]
    funnel["not_flag_implausible"] = len(df)
    log(f"[frame] after drop FLAG_DOY_IMPLAUSIBLE: {len(df)}")

    df = df.reset_index(drop=True)

    # ---- fixed splits -----------------------------------------------------
    df["temporal_split"] = np.where(df["YEAR"] <= C.TEMPORAL_CUTOFF, "train", "test")
    site_counts = df.groupby("SITE_ID").size()
    fold_of = _assign_group_kfold(site_counts, C.CV_K, C.CV_SEED)
    df["cv_fold"] = df["SITE_ID"].map(fold_of).astype(int)

    n_tr = int((df["temporal_split"] == "train").sum())
    n_te = int((df["temporal_split"] == "test").sum())
    log(f"[split] temporal: train(<= {C.TEMPORAL_CUTOFF})={n_tr}  test(> {C.TEMPORAL_CUTOFF})={n_te}")
    log(f"[split] grouped {C.CV_K}-fold by SITE_ID; fold sizes: "
        f"{df['cv_fold'].value_counts().sort_index().tolist()}")

    funnel.update(temporal_train=n_tr, temporal_test=n_te,
                  cv_fold_sizes=df["cv_fold"].value_counts().sort_index().tolist(),
                  n_bng_sites=len(bng), temporal_cutoff=C.TEMPORAL_CUTOFF, cv_k=C.CV_K)

    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(C.MODELLING_FRAME, index=False)
    pd.Series(funnel, dtype="object").to_json(C.OUTPUT_DIR / "_frame_funnel.json")
    log(f"[frame] wrote {C.MODELLING_FRAME} ({len(df)} rows)")
    return df


def load_frame() -> pd.DataFrame:
    if not C.MODELLING_FRAME.exists():
        return build_modelling_frame()
    return pd.read_parquet(C.MODELLING_FRAME)


# ==========================================================================
# Metrics
# ==========================================================================
def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[m], y_pred[m]
    n = len(yt)
    if n == 0:
        return {"n": 0, "MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "bias": np.nan}
    err = yp - yt
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"n": n, "MAE": float(np.abs(err).mean()),
            "RMSE": float(np.sqrt((err ** 2).mean())), "R2": r2,
            "bias": float(err.mean())}


def metrics_table(preds: pd.DataFrame, by_species: bool = True,
                  min_species_n: int = 20) -> pd.DataFrame:
    """Overall + per-species metrics for one (model, regime) prediction set."""
    rows = [{"scope": "overall", "species": "ALL",
             **_metrics(preds["y_true"].values, preds["y_pred"].values)}]
    if by_species:
        for sp, g in preds.groupby("SPECIES_NAME"):
            mm = _metrics(g["y_true"].values, g["y_pred"].values)
            if mm["n"] >= min_species_n:
                rows.append({"scope": "species", "species": sp, **mm})
    return pd.DataFrame(rows)


# ==========================================================================
# Model interface + naive baselines
# ==========================================================================
class Model:
    name = "base"

    def fit(self, train_df: pd.DataFrame):
        raise NotImplementedError

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class SpeciesMeanModel(Model):
    name = "species_mean"

    def fit(self, train_df):
        self.global_ = float(train_df["FIRSTDAY"].mean())
        self.by_sp_ = train_df.groupby("SPECIES_NAME")["FIRSTDAY"].mean()
        return self

    def predict(self, df):
        return df["SPECIES_NAME"].map(self.by_sp_).fillna(self.global_).to_numpy(float)


class SiteSpeciesMeanModel(Model):
    """Site x species mean; unseen site-species falls back to species mean, then global."""
    name = "site_species_mean"

    def fit(self, train_df):
        self.global_ = float(train_df["FIRSTDAY"].mean())
        self.by_sp_ = train_df.groupby("SPECIES_NAME")["FIRSTDAY"].mean()
        self.by_ss_ = train_df.groupby(["SITE_ID", "SPECIES_NAME"])["FIRSTDAY"].mean()
        return self

    def predict(self, df):
        ss = df.set_index(["SITE_ID", "SPECIES_NAME"]).index.map(self.by_ss_)
        out = pd.Series(ss, index=df.index, dtype="float64")
        out = out.fillna(df["SPECIES_NAME"].map(self.by_sp_)).fillna(self.global_)
        return out.to_numpy(float)


class PersistenceModel(Model):
    """No-change forecast: FIRSTDAY(site, sp, year) = the same series' value at year-1;
    falls back to the train species mean where no prior year exists. Only past obs are
    used, so it never leaks the target.

    full_history=True looks up year-1 from the whole frame (natural persistence; but in
    site-grouped CV it consults the held-out site's own earlier years, so that CV score
    is not a site-generalisation measure -- reported with a caveat). full_history=False
    looks up from train only.
    """
    name = "persistence"

    def __init__(self, full_history=True, history_df=None):
        self.full_history = full_history
        self._history_df = history_df

    def fit(self, train_df):
        self.global_ = float(train_df["FIRSTDAY"].mean())
        self.by_sp_ = train_df.groupby("SPECIES_NAME")["FIRSTDAY"].mean()
        src = self._history_df if (self.full_history and self._history_df is not None) else train_df
        self.lut_ = src.set_index(["SITE_ID", "SPECIES_NAME", "YEAR"])["FIRSTDAY"]
        self.lut_ = self.lut_[~self.lut_.index.duplicated()]
        return self

    def predict(self, df):
        prev_idx = pd.MultiIndex.from_arrays(
            [df["SITE_ID"], df["SPECIES_NAME"], df["YEAR"] - 1])
        prev = pd.Series(prev_idx.map(self.lut_), index=df.index, dtype="float64")
        self.last_coverage_ = float(prev.notna().mean())    # fraction with a real prior obs
        fallback = df["SPECIES_NAME"].map(self.by_sp_).fillna(self.global_)
        return prev.fillna(fallback).to_numpy(float)


# ==========================================================================
# Evaluation over both regimes
# ==========================================================================
def evaluate(model: Model, frame: pd.DataFrame) -> pd.DataFrame:
    """Return long predictions with columns: model, regime, SITE_ID, SPECIES_NAME,
    YEAR, y_true, y_pred. Temporal = single split; grouped CV = pooled out-of-fold."""
    out = []

    # (a) temporal hold-out
    tr = frame[frame["temporal_split"] == "train"]
    te = frame[frame["temporal_split"] == "test"].copy()
    model.fit(tr)
    te = te.assign(y_true=te["FIRSTDAY"].to_numpy(float), y_pred=model.predict(te),
                   model=model.name, regime="temporal")
    out.append(te[["model", "regime", *KEYS, "y_true", "y_pred"]])

    # (b) grouped k-fold CV (pooled out-of-fold predictions)
    for fold in range(C.CV_K):
        tr = frame[frame["cv_fold"] != fold]
        te = frame[frame["cv_fold"] == fold].copy()
        model.fit(tr)
        te = te.assign(y_true=te["FIRSTDAY"].to_numpy(float), y_pred=model.predict(te),
                       model=model.name, regime="grouped_cv")
        out.append(te[["model", "regime", *KEYS, "y_true", "y_pred"]])

    return pd.concat(out, ignore_index=True)


def evaluate_all(models, frame) -> tuple[pd.DataFrame, pd.DataFrame]:
    preds = pd.concat([evaluate(m, frame) for m in models], ignore_index=True)
    mrows = []
    for (mdl, reg), g in preds.groupby(["model", "regime"]):
        mt = metrics_table(g)
        mt.insert(0, "regime", reg)
        mt.insert(0, "model", mdl)
        mrows.append(mt)
    metrics = pd.concat(mrows, ignore_index=True)
    return preds, metrics


def naive_models(frame) -> list:
    return [
        SpeciesMeanModel(),
        SiteSpeciesMeanModel(),
        PersistenceModel(full_history=True, history_df=frame[KEYS + ["FIRSTDAY"]]),
    ]


def main() -> int:
    frame = build_modelling_frame()
    preds, metrics = evaluate_all(naive_models(frame), frame)
    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(C.PREDICTIONS_PATH, index=False)
    metrics.to_csv(C.METRICS_PATH, index=False)
    print("\nOverall metrics (naive baselines):")
    print(metrics[metrics.scope == "overall"].to_string(index=False))
    print(f"\nwrote {C.PREDICTIONS_PATH}\nwrote {C.METRICS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
