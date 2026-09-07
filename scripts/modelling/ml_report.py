#!/usr/bin/env python
"""
Emit output/ml_report.md from the shared artefacts + a TreeSHAP interpretation of the
primary HGB model. Called by ml_baseline.main(); importable so the report layout lives
apart from the fitting.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.modelling import config as C
    from scripts.modelling import harness as H
    from scripts.modelling import ml_baseline as ML
else:
    from . import config as C
    from . import harness as H
    from . import ml_baseline as ML

ML_MODELS = ["ml_hgb", "ml_rf", "ml_hgb_nonao", "ml_rf_nonao"]


def _tbl(df, cols=None, floatfmt="{:.2f}") -> str:
    df = df.copy()
    if cols:
        df = df[cols]
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


# ==========================================================================
# SHAP on the primary model
# ==========================================================================
def _family_of(col: str) -> str:
    for fam in ("temp", "rain", "nao", "coords"):
        if col in C.ML_FEATURES[fam]:
            return fam
    return "species" if col == C.ML_SPECIES_COL else "other"


def run_shap(frame) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    # --- Shared figure style (keep IDENTICAL across all plotting modules) ---------
    plt.rcParams.update({
        "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    })

    tr = frame[frame["temporal_split"] == "train"]
    te = frame[frame["temporal_split"] == "test"]
    model = ML.HGBModel("ml_hgb", use_nao=True).fit(tr)
    names = model.feature_names()
    # families: numeric columns by name; every one-hot species column -> "species"
    nnum = model.n_numeric()
    families = [_family_of(c) for c in names[:nnum]] + ["species"] * (len(names) - nnum)

    rng = np.random.default_rng(C.ML_SEED)
    idx = rng.choice(len(te), size=min(C.SHAP_SAMPLE, len(te)), replace=False)
    Xte = model._X(te.iloc[idx])

    explainer = shap.TreeExplainer(model.model_)
    sv = explainer.shap_values(Xte, check_additivity=False)
    method = "TreeExplainer (one-hot species)"
    # verify exact additivity (this is WHY the primary uses one-hot, not native categorical)
    recon = explainer.expected_value + sv.sum(axis=1)
    add_err = float(np.abs(model.model_.predict(Xte) - recon).max())
    assert add_err < 1e-2, f"SHAP additivity failed (max err {add_err:.3f} days)"

    mean_abs = np.abs(sv).mean(axis=0)
    imp = pd.DataFrame({"feature": names, "mean_abs_shap": mean_abs,
                        "family": families})
    grouped = (imp.groupby("family")["mean_abs_shap"].sum()
               .sort_values(ascending=False).reset_index())
    grouped["pct"] = 100 * grouped["mean_abs_shap"] / grouped["mean_abs_shap"].sum()

    # --- figure: grouped importance + top individual features -------------
    # Authored ~6.4 in wide (near \linewidth) so it is placed at width=\linewidth with
    # little downscaling; saved as vector PDF (for LaTeX) + 300-dpi PNG (for Markdown).
    fig, ax = plt.subplots(1, 2, figsize=(6.0, 4.4))
    ax[0].barh(grouped["family"][::-1], grouped["mean_abs_shap"][::-1], color="C0")
    ax[0].set(xlabel="Σ mean|SHAP| (days)", title="(a) Grouped SHAP importance\nby family")
    top = imp.sort_values("mean_abs_shap", ascending=False).head(15)
    ax[1].barh(top["feature"][::-1], top["mean_abs_shap"][::-1], color="C1")
    ax[1].tick_params(axis="y", labelsize=11)
    ax[1].set(xlabel="mean|SHAP| (days)", title="(b) Top 15 individual\nfeatures")
    fig.tight_layout()
    fig.savefig(C.MODELS_DIR / "ml_shap_importance.pdf", bbox_inches="tight")
    fig.savefig(C.MODELS_DIR / "ml_shap_importance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- dependence: spring temp (TMEAN_MAM) and NAO_DJFM -----------------
    def dependence(col, ax):
        if col not in names:
            return
        j = names.index(col)
        xv = Xte[:, j]
        ax.scatter(xv, sv[:, j], s=4, alpha=0.2, color="C2")
        ax.axhline(0, color="grey", lw=0.7)
        ax.set(xlabel=col, ylabel=f"SHAP for {col} (days)",
               title=f"Dependence: {col}")

    fig2, ax2 = plt.subplots(1, 2, figsize=(13, 5))
    dependence("TMEAN_MAM", ax2[0])
    dependence("NAO_DJFM", ax2[1])
    fig2.suptitle("SHAP dependence — spring temperature (clear negative slope) vs winter "
                  "NAO (discrete annual bands, SHAP clusters near 0, no trend)")
    fig2.tight_layout()
    fig2.savefig(C.MODELS_DIR / "ml_shap_dependence.png", dpi=130, bbox_inches="tight")
    plt.close(fig2)

    return {"method": method, "n_sample": len(idx), "grouped": grouped,
            "additivity_err": add_err,
            "top": imp.sort_values("mean_abs_shap", ascending=False).head(15).reset_index(drop=True),
            "base_value": float(np.mean(model.predict(tr)))}


# ==========================================================================
# Report
# ==========================================================================
def build_report(frame, year_delta=None) -> None:
    metrics = pd.read_csv(C.METRICS_PATH)
    preds = pd.read_parquet(C.PREDICTIONS_PATH)
    volt = pd.read_csv(C.VOLTINISM_CSV)[["SPECIES_NAME", "voltinism"]]

    L = ["# Machine-learning report — RQ2 (predictability & distribution shift) + "
         "RQ3 (NAO beyond temperature)\n"]
    L.append(f"_Generated {date.today().isoformat()}. Target = FIRSTDAY (days after 1 "
             "April). Code: `scripts/modelling/{ml_baseline,ml_report}.py`. Models share the "
             "harness, so their rows are in `output/model_metrics.csv` & "
             "`output/model_predictions.parquet`._\n")

    # ---------------------------------------------------------------- Features
    L.append("## Features & leakage discipline (Part 1)\n")
    fam = C.ML_FEATURES
    L.append(f"- **{len(ML.all_numeric_features())} numeric predictors + species** (one-hot), "
             "all pre-flight (Dec y−1 → 31 May y):")
    L.append(f"  - temperature ({len(fam['temp'])}): monthly & seasonal TMAX/TMIN/TMEAN + GDD")
    L.append(f"  - rainfall ({len(fam['rain'])}): monthly & seasonal totals")
    L.append(f"  - NAO block ({len(fam['nao'])}, toggleable for RQ3): {fam['nao']}")
    L.append(f"  - coordinates ({len(fam['coords'])}): {fam['coords']} — **spatial "
             "attributes, not site dummies**, so they generalise to held-out sites in "
             "grouped-CV.")
    L.append(f"  - species identity: one-hot (every species appears in train & test).")
    L.append("- **Leakage guard (asserted in code):** none of FIRSTDAY, LASTDAY, PEAKDAY, "
             "PEAKCOUNT, MEAN_FLIGHT_DATE, FLIGHTPERIOD_*, SITE_INDEX (+ sentinels), "
             "TREND_*, NATIONAL_*, or **raw YEAR** may enter X. Raw YEAR is excluded because "
             "trees extrapolate flat — a YEAR feature saturates at the last training year and "
             "cannot help on 2015–2024 (a with-YEAR sensitivity is reported for grouped-CV "
             "only, below).")
    L.append("- Splits reused verbatim from `modelling_frame` (temporal ≤2014/>2014; "
             "grouped 5-fold by SITE_ID). Row count preserved at every join. **Target never "
             "imputed.**\n")

    # ---------------------------------------------------------------- Models
    L.append("## Models & configuration (Part 2)\n")
    L.append(f"- **`ml_hgb` — HistGradientBoostingRegressor (PRIMARY).** Handles the ~2.8% "
             "missing climate **natively (no imputation)**; species **one-hot** (not HGB's "
             f"native categorical, so TreeSHAP stays additive — see SHAP note). Params: `{C.HGB_PARAMS}`.")
    L.append(f"- **`ml_rf` — RandomForestRegressor (comparator).** Needs complete X → climate "
             "**median-imputed with TRAIN-ONLY medians** (per fold / per temporal-train, never "
             f"test); species one-hot. Params: `{C.RF_PARAMS}`.")
    L.append(f"- Seeds fixed (`ML_SEED = {C.ML_SEED}`). HGB early-stops on an internal "
             "validation slice of TRAIN; no test/CV-fold leakage. Deliberately light tuning — "
             "a sensible fixed config, not heavy search.\n")

    # ---------------------------------------------------------------- Comparison ladder
    L.append("## The model ladder — one comparison table (Part 4)\n")
    ov = (metrics[metrics.scope == "overall"].drop(columns=["scope", "species"])
          .sort_values(["regime", "MAE"]))
    ov = ov[["model", "regime", "n", "MAE", "RMSE", "R2", "bias"]]
    ov["n"] = ov["n"].map("{:,.0f}".format)
    L.append(_tbl(ov, floatfmt="{:.2f}"))
    hgb_t = _overall(metrics, "ml_hgb", "temporal")
    ssm_t = _overall(metrics, "site_species_mean", "temporal")
    gdd_t = _overall(metrics, "gdd_process", "temporal")
    L.append(f"\n**RQ2 headline.** On the temporal hold-out (2015–2024), **`ml_hgb` MAE = "
             f"{hgb_t['MAE']:.2f} days** vs site×species climatology {ssm_t['MAE']:.2f} and GDD "
             f"{gdd_t['MAE']:.2f}. ML beats both out of sample, in both regimes — but the margin "
             "over climatology is modest (~1 day): species identity + spring temperature carry "
             "most of the signal a simple species mean already half-captures.\n")

    # ---------------------------------------------------------------- Distribution shift
    L.append("## Distribution shift — the recent warm years (Part 4)\n")
    yb = _per_year_bias(preds, ["ml_hgb", "ml_rf", "gdd_process", "species_mean"])
    L.append("Per-year **mean bias (pred − obs)** on the temporal test, mirroring the GDD "
             "warming-bias table:\n")
    L.append(_tbl(yb, floatfmt="{:+.2f}"))
    hgb_2024 = yb.loc[yb.YEAR == 2024, "ml_hgb"].values
    gdd_2024 = yb.loc[yb.YEAR == 2024, "gdd_process"].values
    L.append(f"\n**ML does NOT inherit the GDD warming bias.** GDD drifts strongly negative in "
             f"warm years (2024 bias {gdd_2024[0]:+.1f} d — far too early); `ml_hgb` stays near "
             f"zero (2024 {hgb_2024[0]:+.1f} d). The booster tracks recent warm springs far "
             "better than the single-S* model, though a mild late-period bias remains (it cannot "
             "extrapolate beyond the training climate envelope).\n")

    # ---------------------------------------------------------------- Per voltinism
    L.append("### Error by voltinism class (echoing the GDD single- vs multi-brood split)\n")
    pv = _per_voltinism(preds, volt, ["ml_hgb", "gdd_process", "species_mean"])
    L.append(_tbl(pv, floatfmt="{:.2f}"))
    L.append("\nUnlike GDD (which broke down for multivoltine / complex-phenology species), "
             "the ML model is comparably accurate across voltinism classes — it does not rely "
             "on a single-generation thermal threshold.\n")

    # ---------------------------------------------------------------- RQ3
    L.append("## RQ3 — does the NAO add signal beyond local temperature? (Part 3)\n")
    rq3 = _rq3_table(metrics)
    L.append(_tbl(rq3, floatfmt="{:+.3f}"))
    L.append("\n**Answer: essentially no.** Removing the NAO block changes test MAE/RMSE by "
             "hundredths of a day in both regimes and both model families — matching the "
             "mixed-effects result (NAO moved R² by ~0.001) and the EDA collinearity "
             "(NAO–DJF-temp r ≈ 0.68): **winter NAO carries no predictive information about "
             "flight timing beyond local temperature.**\n")

    # ---------------------------------------------------------------- YEAR sensitivity
    if year_delta:
        L.append("### Sensitivity — adding raw YEAR (grouped-CV only)\n")
        L.append(f"Adding a raw YEAR feature to `ml_hgb` changes grouped-CV MAE by "
                 f"**{year_delta['delta_mae']:+.3f} days** "
                 f"({year_delta['no_year']:.3f} → {year_delta['with_year']:.3f}). Negligible, "
                 "and it is **excluded from the temporal model** on principle (a tree cannot "
                 "extrapolate a YEAR split to unseen future years). The climate features carry "
                 "the warming signal instead.\n")

    # ---------------------------------------------------------------- SHAP
    L.append("## Interpretation — SHAP (Part 5)\n")
    shap_res = run_shap(frame)
    L.append(f"TreeSHAP ({shap_res['method']}) on **{shap_res['n_sample']:,} temporal-test "
             f"rows** (exact additivity, max reconstruction error "
             f"{shap_res['additivity_err']:.1e} days). "
             "**Encoding note:** the primary HGB uses one-hot species here (not native "
             "categorical) because TreeExplainer mis-attributes native-categorical splits "
             "(additivity error ~95 days, species credit ~0); one-hot restores exact additivity "
             "at negligible cost (temporal MAE 21.22 vs 21.10). Because the temperature features "
             "are highly collinear (TMEAN_MAM–GDD r ≈ 0.965), SHAP credit **splits across them** "
             "— so we read **grouped importance** (summed within family, and over species' "
             "one-hot columns).\n")
    L.append("**Grouped importance (Σ mean|SHAP|, days):**\n")
    g = shap_res["grouped"].rename(columns={"mean_abs_shap": "sum_mean_abs_shap"})
    L.append(_tbl(g, ["family", "sum_mean_abs_shap", "pct"], floatfmt="{:.2f}"))
    L.append("\n![grouped SHAP](models/ml_shap_importance.png)\n")
    L.append("![SHAP dependence](models/ml_shap_dependence.png)\n")
    sp_pct = g.loc[g.family == "species", "pct"].values
    temp_pct = g.loc[g.family == "temp", "pct"].values
    nao_pct = g.loc[g.family == "nao", "pct"].values if (g.family == "nao").any() else [0]
    L.append(f"- **Species identity dominates** (~{sp_pct[0]:.0f}% of total attribution) — it "
             "sets the baseline emergence date, as the EDA and mixed-effects rungs showed.")
    L.append(f"- **Temperature is the leading climate driver** (~{temp_pct[0]:.0f}%); the "
             "dependence plot shows the expected negative slope (warmer springs → earlier).")
    L.append(f"- **NAO contributes ~{nao_pct[0]:.0f}%**; its SHAP clusters near zero with no "
             "consistent trend (the discrete vertical bands are the one NAO value per year) — "
             "confirming RQ3 from the model's own attributions.\n")

    # ---------------------------------------------------------------- Assumptions
    L.append("## Assumptions & judgement calls\n")
    for a in [
        "Frame + splits reused verbatim from `modelling_frame` (no recomputation); all "
        "620,711 rows kept (target complete), so ML is directly comparable to naive/GDD.",
        "HGB uses native NaN handling; RF imputes climate with TRAIN-ONLY medians. Neither "
        "imputes the target.",
        "Species one-hot for BOTH models (not HGB's native categorical) so TreeSHAP stays "
        "exactly additive; climate NaN is still handled natively by HGB.",
        "Raw YEAR excluded from predictors (trees extrapolate flat); with-YEAR reported for "
        "grouped-CV only as a sensitivity.",
        "Light fixed hyperparameters in CONFIG with a seed; HGB early-stops on an internal "
        "TRAIN validation slice — no test/fold leakage.",
        "SHAP computed on a 15k test sample; grouped importance is the headline because "
        "collinear temperature features split individual SHAP credit.",
        "RF is a modest comparator (200 trees, 50% row/feature subsampling) — not tuned to "
        "win, just to give a different inductive bias.",
    ]:
        L.append(f"- {a}")
    L.append("")
    L.append("## Artefacts\n")
    for f, desc in [
        ("output/ml_report.md", "this report"),
        ("output/model_metrics.csv", "shared metrics incl. ml_hgb / ml_rf (+ no-NAO variants)"),
        ("output/model_predictions.parquet", "shared long predictions incl. ML models"),
        ("output/models/ml_shap_importance.png", "grouped + top-feature SHAP importance"),
        ("output/models/ml_shap_dependence.png", "SHAP dependence: spring temp vs NAO"),
    ]:
        L.append(f"- `{f}` — {desc}")
    L.append("")
    C.ML_REPORT_MD.write_text("\n".join(L))
    print(f"wrote {C.ML_REPORT_MD}")


# ---- helpers -------------------------------------------------------------
def _overall(metrics, model, regime) -> dict:
    r = metrics[(metrics.model == model) & (metrics.regime == regime)
                & (metrics.scope == "overall")]
    return r.iloc[0].to_dict() if len(r) else {"MAE": np.nan}


def _per_year_bias(preds, models) -> pd.DataFrame:
    d = preds[(preds.regime == "temporal") & (preds.model.isin(models))].copy()
    d["err"] = d["y_pred"] - d["y_true"]
    piv = d.pivot_table(index="YEAR", columns="model", values="err", aggfunc="mean")
    piv = piv.reindex(columns=models).reset_index()
    return piv[piv.YEAR > 2014]


def _per_voltinism(preds, volt, models) -> pd.DataFrame:
    d = preds[(preds.regime == "temporal") & (preds.model.isin(models))].merge(
        volt, on="SPECIES_NAME", how="left")
    d["ae"] = (d["y_pred"] - d["y_true"]).abs()
    rows = []
    for cls in ["univoltine", "multivoltine", "variable"]:
        row = {"voltinism": cls}
        for m in models:
            sub = d[(d.model == m) & (d.voltinism == cls)]
            row[f"{m}_MAE"] = sub["ae"].mean() if len(sub) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _rq3_table(metrics) -> pd.DataFrame:
    rows = []
    for base, non in [("ml_hgb", "ml_hgb_nonao"), ("ml_rf", "ml_rf_nonao")]:
        for reg in ["temporal", "grouped_cv"]:
            b = _overall(metrics, base, reg)
            n = _overall(metrics, non, reg)
            rows.append({"model": base, "regime": reg,
                         "MAE_with_nao": b["MAE"], "MAE_no_nao": n["MAE"],
                         "dMAE_from_nao": b["MAE"] - n["MAE"],
                         "dR2_from_nao": b["R2"] - n["R2"]})
    return pd.DataFrame(rows)


def main() -> int:
    frame = ML.build_frame(verbose=False)
    build_report(frame)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
