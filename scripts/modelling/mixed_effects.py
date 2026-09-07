#!/usr/bin/env python
"""
Mixed-effects RQ1 model of FIRSTDAY.

RQ1: does the NAO-phenology relationship hold over 1976-2024 and still differ by voltinism
once site-level variation is accounted for? (Updates Westgarth-Smith 2012.) Primary spec,
species fixed, site random intercept:
    FIRSTDAY ~ NAO_DJFM * VOLTINISM + YEAR_c + C(SPECIES_NAME) + (1 | SITE_ID)

Voltinism is nested in species, so its main effect is absorbed by the species fixed effects;
the NAO x voltinism interaction is identified from within-species year-to-year NAO variation.
We build explicit interaction columns (NAO_c_multi, NAO_c_var) rather than let patsy make a
rank-deficient design. NAO_DJFM and TMEAN_MAM are mean-centred (NAO_c, TMAM_c) -- leaves the
slopes unchanged but cuts main-vs-interaction collinearity.

Three estimation routes, all reported (never silently switched):
  1. MixedLM (primary spec) -> site random-intercept variance & ICC.
  2. FE-OLS with cluster-robust SEs (SITE_ID, two-way SITE_ID x YEAR where possible) for
     every spec -> coefficients + 95% CIs (the coefficient workhorse).
  3. Two-stage: per-species OLS FIRSTDAY ~ NAO_c + YEAR_c, then precision-weighted meta of
     the NAO slopes by voltinism (a transparent Westgarth-Smith-style cross-check).

Run:  .venv/bin/python -m scripts.modelling.mixed_effects
"""
from __future__ import annotations

import sys
import time
import warnings
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

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

KEYS = ["SITE_ID", "SPECIES_NAME", "YEAR"]


# ==========================================================================
# Analysis frame: modelling_frame + voltinism + spring temperature
# ==========================================================================
def build_analysis_frame(exclude_2024: bool | None = None) -> tuple[pd.DataFrame, dict]:
    exclude_2024 = C.NAO_EXCLUDE_2024 if exclude_2024 is None else exclude_2024
    prov = {}

    frame = pd.read_parquet(C.MODELLING_FRAME)
    prov["frame_rows"] = len(frame)

    volt = pd.read_csv(C.VOLTINISM_CSV)[["SPECIES_NAME", "voltinism"]]
    n0 = len(frame)
    frame = frame.merge(volt, on="SPECIES_NAME", how="left")
    assert len(frame) == n0, f"voltinism merge changed rows: {n0} -> {len(frame)}"
    prov["after_voltinism_merge"] = len(frame)

    # spring temperature (and GDD) live in feature_table, not the frame
    clim = pd.read_parquet(C.FEATURE_TABLE, columns=KEYS + ["TMEAN_MAM", "GDD"])
    n0 = len(frame)
    frame = frame.merge(clim, on=KEYS, how="left")
    assert len(frame) == n0, f"climate merge changed rows: {n0} -> {len(frame)}"
    prov["after_climate_merge"] = len(frame)

    # Exclude species without a usable voltinism class, distinguishing WHY:
    #   (a) MANUAL_REVIEW aggregate (Thymelicus lineola/sylvestris) — cannot take one class;
    #   (b) unclassified in Cook (voltinism NaN) — the migrants Vanessa atalanta / cardui,
    #       which have no fixed resident-UK generation structure, so Cook leaves it blank.
    is_manual = frame["voltinism"].isin(C.VOLT_DROP)
    is_unclassified = frame["voltinism"].isna()
    drop = is_manual | is_unclassified
    prov["dropped_rows_total"] = int(drop.sum())
    prov["dropped_manual_review_species"] = sorted(
        frame.loc[is_manual, "SPECIES_NAME"].unique().tolist())
    prov["dropped_unclassified_species"] = sorted(
        frame.loc[is_unclassified, "SPECIES_NAME"].unique().tolist())
    prov["dropped_species"] = sorted(frame.loc[drop, "SPECIES_NAME"].unique().tolist())
    frame = frame[~drop].copy()
    prov["after_drop"] = len(frame)

    if exclude_2024:
        n0 = len(frame)
        frame = frame[frame["YEAR"] != 2024].copy()
        prov["excluded_2024_rows"] = n0 - len(frame)
    prov["exclude_2024"] = exclude_2024

    # centred predictors (slopes unchanged; reduces collinearity) + interaction columns
    frame["YEAR_c"] = frame["YEAR"] - frame["YEAR"].mean()
    frame["NAO_c"] = frame["NAO_DJFM"] - frame["NAO_DJFM"].mean()
    frame["TMAM_c"] = frame["TMEAN_MAM"] - frame["TMEAN_MAM"].mean()
    is_multi = (frame["voltinism"] == "multivoltine").astype(float)
    is_var = (frame["voltinism"] == "variable").astype(float)
    frame["is_multi"], frame["is_var"] = is_multi, is_var
    frame["NAO_c_multi"] = frame["NAO_c"] * is_multi
    frame["NAO_c_var"] = frame["NAO_c"] * is_var
    frame["TMAM_c_multi"] = frame["TMAM_c"] * is_multi
    frame["TMAM_c_var"] = frame["TMAM_c"] * is_var

    prov["voltinism_counts"] = frame.drop_duplicates("SPECIES_NAME")["voltinism"]\
        .value_counts().to_dict()
    prov["n_species"] = frame["SPECIES_NAME"].nunique()
    prov["n_sites"] = frame["SITE_ID"].nunique()
    prov["year_mean"] = float(frame["YEAR"].mean())
    prov["nao_mean"] = float(frame["NAO_DJFM"].mean())
    prov["tmam_mean"] = float(frame["TMEAN_MAM"].mean())

    # require the model predictors to be present (never impute the target FIRSTDAY)
    need = ["FIRSTDAY", "NAO_DJFM", "TMEAN_MAM", "YEAR_c"]
    n0 = len(frame)
    frame = frame.dropna(subset=need)
    prov["dropped_missing_predictors"] = n0 - len(frame)
    prov["analysis_rows"] = len(frame)
    return frame, prov


# ==========================================================================
# Design matrix (manual, to avoid the patsy C()/config-`C` name clash and to give
# clean named coefficients). Columns: const + continuous terms + species dummies.
# ==========================================================================
def _design_matrix(frame: pd.DataFrame, cont_terms: list[str],
                   species_levels: list[str] | None = None) -> pd.DataFrame:
    if species_levels is None:
        species_levels = sorted(frame["SPECIES_NAME"].unique())
    sp = pd.Categorical(frame["SPECIES_NAME"], categories=species_levels)
    D = pd.get_dummies(sp, prefix="SP", drop_first=True).astype(float)
    D.index = frame.index
    X = pd.DataFrame({"const": 1.0}, index=frame.index)
    for t in cont_terms:
        X[t] = frame[t].astype(float)
    return pd.concat([X, D], axis=1)


# ==========================================================================
# FE-OLS with cluster-robust SEs (coefficient workhorse)
# ==========================================================================
def fit_ols_cluster(frame: pd.DataFrame, cont_terms: list[str], label: str,
                    twoway: bool = True) -> dict:
    """OLS with species fixed effects (dummies) and cluster-robust covariance.

    Tries two-way clustering (SITE_ID, YEAR); falls back to one-way SITE_ID.
    Returns coefficients (non-species terms), CIs, and fit stats.
    """
    X = _design_matrix(frame, cont_terms)
    y = frame["FIRSTDAY"].astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.OLS(y, X)
        cov_used = "cluster_twoway_site_year"
        try:
            if not twoway:
                raise ValueError("twoway disabled")
            groups2 = frame[["SITE_ID", "YEAR"]].to_numpy()
            res = model.fit(cov_type="cluster",
                            cov_kwds={"groups": groups2, "use_correction": True})
        except Exception:
            res = model.fit(cov_type="cluster",
                            cov_kwds={"groups": frame["SITE_ID"].to_numpy()})
            cov_used = "cluster_oneway_site"

    ci = res.conf_int()
    keep = [p for p in res.params.index if not p.startswith("SP_")]
    coefs = pd.DataFrame({
        "term": keep,
        "estimate": [res.params[p] for p in keep],
        "se": [res.bse[p] for p in keep],
        "ci_low": [ci.loc[p, 0] for p in keep],
        "ci_high": [ci.loc[p, 1] for p in keep],
        "p": [res.pvalues[p] for p in keep],
    })
    return {"label": label, "formula": f"FIRSTDAY ~ {' + '.join(cont_terms)} + species_FE",
            "cov_used": cov_used, "n": int(res.nobs), "r2": float(res.rsquared),
            "r2_adj": float(res.rsquared_adj), "coefs": coefs, "res": res}


def slope_by_class(fit: dict, base: str, multi: str | None, var: str | None) -> pd.DataFrame:
    """Linear-combination slopes per voltinism class with cluster-robust SEs.

    univoltine = base ; multivoltine = base + multi ; variable = base + var.
    """
    res = fit["res"]
    cov = res.cov_params()
    p = res.params
    out = []

    def comb(terms, name):
        vec = pd.Series(0.0, index=p.index)
        for t in terms:
            vec[t] = 1.0
        est = float(vec @ p)
        se = float(np.sqrt(vec.values @ cov.values @ vec.values))
        out.append({"class": name, "slope": est, "se": se,
                    "ci_low": est - 1.96 * se, "ci_high": est + 1.96 * se})

    comb([base], "univoltine")
    if multi:
        comb([base, multi], "multivoltine")
    if var:
        comb([base, var], "variable")
    return pd.DataFrame(out)


# ==========================================================================
# MixedLM (site random intercept) -> ICC
# ==========================================================================
def fit_mixedlm(frame: pd.DataFrame, cont_terms: list[str]) -> dict:
    X = _design_matrix(frame, cont_terms)
    y = frame["FIRSTDAY"].astype(float)
    t0 = time.time()
    status, res = "ok", None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            md = sm.MixedLM(y, X, groups=frame["SITE_ID"].to_numpy(),
                            exog_re=np.ones((len(frame), 1)))
            res = md.fit(method="lbfgs", maxiter=C.MIXEDLM_MAXITER, reml=True)
        if not res.converged:
            status = "not_converged"
    except Exception as e:  # noqa: BLE001
        status = f"failed: {type(e).__name__}: {e}"
    elapsed = time.time() - t0

    out = {"cont_terms": cont_terms, "status": status, "seconds": round(elapsed, 1)}
    if res is not None:
        try:
            sig2_site = float(np.atleast_2d(res.cov_re)[0, 0])
            sig2_res = float(res.scale)
            fe = res.fe_params
            out.update({
                "site_var": sig2_site, "resid_var": sig2_res,
                "site_sd": sig2_site ** 0.5, "resid_sd": sig2_res ** 0.5,
                "icc": sig2_site / (sig2_site + sig2_res),
                "nao_c": float(fe.get("NAO_c", np.nan)),
                "nao_c_multi": float(fe.get("NAO_c_multi", np.nan)),
                "nao_c_var": float(fe.get("NAO_c_var", np.nan)),
                "year_c": float(fe.get("YEAR_c", np.nan)),
            })
        except Exception as e:  # noqa: BLE001
            out["status"] = f"postprocess_failed: {e}"
    return out


# ==========================================================================
# Two-stage: per-species NAO slopes, meta-analysed by voltinism
# ==========================================================================
def two_stage(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stage 1: per-species OLS FIRSTDAY ~ NAO_c + YEAR_c (SEs). Stage 2: precision-
    weighted mean slope per voltinism class (+ between-species heterogeneity)."""
    rows = []
    for sp, g in frame.groupby("SPECIES_NAME"):
        if len(g) < 30 or g["NAO_c"].std() == 0:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = smf.ols("FIRSTDAY ~ NAO_c + YEAR_c", data=g).fit()
        rows.append({"SPECIES_NAME": sp, "voltinism": g["voltinism"].iloc[0],
                     "n": len(g), "nao_slope": r.params["NAO_c"],
                     "nao_se": r.bse["NAO_c"], "year_slope": r.params["YEAR_c"]})
    per_sp = pd.DataFrame(rows)

    meta = []
    for cls, g in per_sp.groupby("voltinism"):
        w = 1.0 / (g["nao_se"] ** 2)
        wmean = float((w * g["nao_slope"]).sum() / w.sum())
        wse = float(np.sqrt(1.0 / w.sum()))
        meta.append({"class": cls, "n_species": len(g),
                     "nao_slope_wmean": wmean, "se": wse,
                     "ci_low": wmean - 1.96 * wse, "ci_high": wmean + 1.96 * wse,
                     "slope_sd_across_species": float(g["nao_slope"].std())})
    return per_sp, pd.DataFrame(meta)


# ==========================================================================
# VIFs (continuous terms of the 'both' model)
# ==========================================================================
def compute_vifs(frame: pd.DataFrame, terms: list[str]) -> pd.DataFrame:
    X = sm.add_constant(frame[terms].astype(float))
    rows = []
    for i, t in enumerate(X.columns):
        if t == "const":
            continue
        rows.append({"term": t, "VIF": float(variance_inflation_factor(X.values, i))})
    return pd.DataFrame(rows)


# ==========================================================================
# Harness adapter: FE-OLS marginal predictor (fixed species template)
# ==========================================================================
class FEOLSModel(H.Model):
    """FE-OLS model plugged into the shared harness. Uses a FIXED species-dummy template
    (from the full analysis frame) so train/test designs align; predicts the marginal
    (fixed-effect) mean, i.e. random site intercept = 0 for held-out sites."""

    def __init__(self, name, cont_terms, species_levels):
        self.name = name
        self.cont_terms = cont_terms
        self.species_levels = species_levels

    def _design(self, df):
        X = df[self.cont_terms].astype(float).to_numpy()
        sp = pd.Categorical(df["SPECIES_NAME"], categories=self.species_levels)
        D = pd.get_dummies(sp, drop_first=True).astype(float).to_numpy()
        return np.column_stack([np.ones(len(df)), X, D])

    def fit(self, train_df):
        X = self._design(train_df)
        y = train_df["FIRSTDAY"].to_numpy(float)
        self.beta_, *_ = np.linalg.lstsq(X, y, rcond=None)
        return self

    def predict(self, df):
        return self._design(df) @ self.beta_


def harness_eval_and_merge(frame_full: pd.DataFrame, models: list) -> pd.DataFrame:
    """Evaluate ME models via the shared harness (both regimes) and merge into the
    shared model_predictions/model_metrics artefacts alongside baselines + GDD."""
    preds = pd.concat([H.evaluate(m, frame_full) for m in models], ignore_index=True)

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

    C.ME_PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(C.ME_PRED_PATH, index=False)
    return preds


# ==========================================================================
# Diagnostics
# ==========================================================================
def diagnostics(fit: dict, frame: pd.DataFrame, path: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scipy.stats as ss

    # --- Shared figure style (keep IDENTICAL across all plotting modules) ---------
    plt.rcParams.update({
        "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    })

    res = fit["res"]
    fitted = res.fittedvalues.to_numpy()
    resid = res.resid.to_numpy()
    d = frame.loc[res.resid.index, ["YEAR", "SPECIES_NAME"]].copy()
    d["resid"] = resid

    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    # residuals vs fitted (hexbin — 600k points)
    ax[0, 0].hexbin(fitted, resid, gridsize=60, cmap="Blues", mincnt=1)
    ax[0, 0].axhline(0, color="r", lw=1)
    ax[0, 0].set(xlabel="Fitted FIRSTDAY", ylabel="Residual",
                 title="(a) Residuals vs fitted")
    # QQ
    samp = pd.Series(resid).sample(min(20000, len(resid)), random_state=C.CV_SEED)
    ss.probplot(samp, dist="norm", plot=ax[0, 1])
    ax[0, 1].set_title("(b) Residual QQ (20k sample)")
    # residuals by year
    by_yr = d.groupby("YEAR")["resid"].mean()
    ax[1, 0].axhline(0, color="grey", lw=0.8)
    ax[1, 0].plot(by_yr.index, by_yr.values, "o-", ms=3)
    ax[1, 0].set(xlabel="Year", ylabel="Mean residual", title="(c) Mean residual by year")
    # residuals by species (sorted)
    by_sp = d.groupby("SPECIES_NAME")["resid"].mean().sort_values()
    ax[1, 1].axhline(0, color="grey", lw=0.8)
    ax[1, 1].plot(range(len(by_sp)), by_sp.values, "o", ms=3)
    ax[1, 1].set(xlabel="Species (sorted)", ylabel="Mean residual",
                 title="(d) Mean residual by species")
    fig.suptitle(f"Mixed-effects diagnostics — {fit['label']} (FE-OLS fit)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return {"resid_mean": float(resid.mean()), "resid_sd": float(resid.std()),
            "worst_year": [int(by_yr.abs().idxmax()), float(by_yr.abs().max())],
            "fig": path.name}


# ==========================================================================
# Driver
# ==========================================================================
CONT_PRIMARY = ["NAO_c", "NAO_c_multi", "NAO_c_var", "YEAR_c"]
CONT_BOTH = ["NAO_c", "NAO_c_multi", "NAO_c_var",
             "TMAM_c", "TMAM_c_multi", "TMAM_c_var", "YEAR_c"]


def run(exclude_2024: bool = False) -> dict:
    frame, prov = build_analysis_frame(exclude_2024=exclude_2024)
    R = {"prov": prov, "exclude_2024": exclude_2024}

    # ---- specifications (FE-OLS + cluster-robust) -----------------------
    specs = {
        "primary_nao": CONT_PRIMARY,
        "primary_nao_noyear": ["NAO_c", "NAO_c_multi", "NAO_c_var"],
        "climate_only": ["TMAM_c", "TMAM_c_multi", "TMAM_c_var", "YEAR_c"],
        "both": CONT_BOTH,
    }
    fits = {k: fit_ols_cluster(frame, terms, k) for k, terms in specs.items()}
    R["fits"] = fits

    # per-class slopes (days per NAO unit / per degC)
    R["nao_slopes"] = {
        "primary": slope_by_class(fits["primary_nao"], "NAO_c", "NAO_c_multi", "NAO_c_var"),
        "primary_noyear": slope_by_class(fits["primary_nao_noyear"], "NAO_c",
                                         "NAO_c_multi", "NAO_c_var"),
        "both": slope_by_class(fits["both"], "NAO_c", "NAO_c_multi", "NAO_c_var"),
    }
    R["temp_slopes"] = {
        "climate_only": slope_by_class(fits["climate_only"], "TMAM_c",
                                       "TMAM_c_multi", "TMAM_c_var"),
        "both": slope_by_class(fits["both"], "TMAM_c", "TMAM_c_multi", "TMAM_c_var"),
    }

    # binary sensitivity (univoltine vs multivoltine; 'variable' excluded)
    bin_frame = frame[frame["voltinism"].isin(["univoltine", "multivoltine"])].copy()
    R["binary_fit"] = fit_ols_cluster(bin_frame, ["NAO_c", "NAO_c_multi", "YEAR_c"],
                                      "binary_uni_vs_multi")
    R["binary_slopes"] = slope_by_class(R["binary_fit"], "NAO_c", "NAO_c_multi", None)
    R["binary_n"] = len(bin_frame)

    # ---- MixedLM primary (ICC) ------------------------------------------
    R["mixedlm"] = fit_mixedlm(frame, specs["primary_nao"])

    # ---- two-stage cross-check ------------------------------------------
    per_sp, meta = two_stage(frame)
    R["two_stage_per_species"] = per_sp
    R["two_stage_meta"] = meta

    # ---- VIFs (both model continuous terms) -----------------------------
    R["vifs"] = compute_vifs(frame, CONT_BOTH)

    # ---- diagnostics on primary -----------------------------------------
    R["diag"] = diagnostics(fits["primary_nao"], frame,
                            C.MODELS_DIR / "me_diagnostics.png")

    # ---- harness eval of primary + both ---------------------------------
    species_levels = sorted(frame["SPECIES_NAME"].unique())
    frame_full = frame  # already has all needed columns + splits from modelling_frame
    me_models = [
        FEOLSModel("me_nao", CONT_PRIMARY, species_levels),
        FEOLSModel("me_nao_temp", CONT_BOTH, species_levels),
    ]
    hp = harness_eval_and_merge(frame_full, me_models)
    R["harness_metrics"] = pd.concat(
        [H.metrics_table(g).assign(model=mdl, regime=reg)
         for (mdl, reg), g in hp.groupby(["model", "regime"])],
        ignore_index=True)

    # like-for-like anchor: species-mean climatology on the SAME ME analysis subset
    # (the shared species_mean baseline scores the full frame incl. the erratic migrant
    # Vanessa species the ME models drop, so its MAE is not directly comparable).
    anchor = H.evaluate(H.SpeciesMeanModel(), frame_full)
    anchor["model"] = "species_mean_mesubset"
    R["anchor_metrics"] = pd.concat(
        [H.metrics_table(g).assign(model="species_mean_mesubset", regime=reg)
         for reg, g in anchor.groupby("regime")], ignore_index=True)
    R["frame"] = frame
    return R


def _print_summary(R):
    print(f"\nanalysis rows: {R['prov']['analysis_rows']:,} | "
          f"species {R['prov']['n_species']} | sites {R['prov']['n_sites']:,} | "
          f"voltinism {R['prov']['voltinism_counts']}")
    print(f"excluded (no voltinism class): aggregate="
          f"{R['prov']['dropped_manual_review_species']} "
          f"migrants={R['prov']['dropped_unclassified_species']}")
    print("\nprimary NAO slopes (days per NAO unit, with YEAR_c):")
    print(R["nao_slopes"]["primary"].to_string(index=False))
    ml = R["mixedlm"]
    print(f"\nMixedLM: status={ml['status']} ({ml.get('seconds')}s) "
          + (f"ICC={ml.get('icc'):.3f} site_sd={ml.get('site_sd'):.2f} "
             f"resid_sd={ml.get('resid_sd'):.2f}" if "icc" in ml else ""))
    print("\ntwo-stage meta (NAO slope by voltinism):")
    print(R["two_stage_meta"].to_string(index=False))
    print("\nVIFs (both model):")
    print(R["vifs"].to_string(index=False))


def main() -> int:
    from scripts.modelling.me_report import write_report

    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("===== primary run (2024 via CPC-rescaled NAO) =====")
    R_main = run(exclude_2024=False)
    _print_summary(R_main)

    print("\n===== sensitivity run (2024 excluded) =====")
    R_excl = run(exclude_2024=True)
    print("primary NAO slopes (2024 excluded):")
    print(R_excl["nao_slopes"]["primary"].to_string(index=False))

    # coefficient CSV (every FE-OLS specification, main run)
    coef_frames = []
    for k, f in R_main["fits"].items():
        c = f["coefs"].copy(); c.insert(0, "spec", k); coef_frames.append(c)
    cb = R_main["binary_fit"]["coefs"].copy(); cb.insert(0, "spec", "binary_uni_vs_multi")
    coef_frames.append(cb)
    pd.concat(coef_frames, ignore_index=True).to_csv(C.ME_COEF_CSV, index=False)
    print(f"\nwrote {C.ME_COEF_CSV}")

    write_report(R_main, R_excl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
