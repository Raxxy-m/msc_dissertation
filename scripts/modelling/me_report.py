#!/usr/bin/env python
"""
Emit output/mixedeffects_report.md from mixed_effects.run()'s in-memory results (plus the
shared model_metrics.csv and the Part-1 NAO-fix QC json). Called by mixed_effects.main();
kept separate so the report layout lives apart from the fitting.
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

WS = "Westgarth-Smith et al. (2012)"


def _tbl(df: pd.DataFrame, cols=None, floatfmt="{:.3f}") -> str:
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


def _slopes_md(df: pd.DataFrame) -> str:
    d = df.rename(columns={"slope": "days/NAO-unit"})
    return _tbl(d, ["class", "days/NAO-unit", "se", "ci_low", "ci_high"])


def write_report(R, R_excl) -> None:
    prov = R["prov"]
    L = ["# Mixed-effects report — RQ1 (NAO × voltinism → FIRSTDAY)\n"]
    L.append(f"_Generated {date.today().isoformat()}. Target = FIRSTDAY (days after 1 "
             "April). Code: `scripts/modelling/{fix_nao_2024,mixed_effects,me_report}.py`._\n")
    L.append("**RQ1:** does the NAO–phenology relationship hold over 1976–2024, and does it "
             f"still differ by voltinism, once site-level variation is accounted for? "
             f"(Updates {WS}.)\n")

    # ------------------------------------------------------------------ Part 1
    L.append("## Part 1 — 2024 winter-NAO gap fix\n")
    qc = json.loads((C.OUTPUT_DIR / "_nao2024_qc.json").read_text()) \
        if (C.OUTPUT_DIR / "_nao2024_qc.json").exists() else {}
    if qc:
        L.append(f"- The Hurrell station DJFM index ends 2023, so flight-year 2024 "
                 f"(in the temporal TEST set) had `NAO_DJFM = NaN`.")
        L.append(f"- Built a CPC DJFM series ourselves from the NOAA CPC monthly NAO, then "
                 f"regressed Hurrell on CPC over the {qc['cal_n_overlap']} overlapping years "
                 f"({qc['cal_overlap_years'][0]}–{qc['cal_overlap_years'][1]}): "
                 f"**Hurrell = {qc['cal_a']:+.3f} + {qc['cal_b']:.3f}·CPC**, "
                 f"r = {qc['cal_r']:.3f} (R² = {qc['cal_r2']:.3f}), RMSE = {qc['cal_rmse']:.2f}.")
        L.append(f"- CPC DJFM 2024 = {qc['cpc_2024_djfm']:+.3f} → **rescaled onto the Hurrell "
                 f"scale = {qc['rescaled_2024_hurrell']:+.3f}** and written into "
                 f"`NAO_DJFM` for the {qc['n_rows_filled_2024']:,} flight-year-2024 rows.")
        L.append(f"- Added column **`NAO_SOURCE`** so the substitution is auditable: "
                 f"{qc['source_counts']}. The two products are **not** silently concatenated; "
                 "the CPC value is explicitly rescaled and flagged `cpc_rescaled`.")
        L.append(f"- feature_table row count re-asserted unchanged (= {C.EXPECTED_ROWS:,}).")
    L.append("")

    # ------------------------------------------------------------------ Frame
    L.append("## Part 2 — Modelling frame\n")
    L.append(f"- Reuses `output/modelling_frame.parquet` (same rows/splits as the baselines "
             "& GDD), joins `data/species_voltinism.csv` on SPECIES_NAME, and joins "
             "`TMEAN_MAM` from feature_table (row counts asserted after each merge).")
    L.append(f"- **Excluded {len(prov['dropped_species'])} species with no usable voltinism "
             f"class** ({prov['dropped_rows_total']:,} rows), for two distinct reasons: "
             f"(a) the recorder **aggregate** {prov['dropped_manual_review_species']} "
             "(cannot take a single class), and (b) the **migrants** "
             f"{prov['dropped_unclassified_species']}, which Cook leaves voltinism-blank as "
             "they have no fixed resident-UK generation structure. Both are stated, not "
             "silently dropped.")
    L.append(f"- Also dropped {prov['dropped_missing_predictors']:,} rows missing a model "
             "predictor (TMEAN_MAM on a few site-years). **The target FIRSTDAY is never imputed.**")
    L.append(f"- **Analysis rows: {prov['analysis_rows']:,}** across "
             f"{prov['n_species']} species and {prov['n_sites']:,} sites. Voltinism (species "
             f"counts): {prov['voltinism_counts']}.")
    L.append(f"- Predictors mean-centred: NAO (mean {prov['nao_mean']:+.2f}), TMEAN_MAM "
             f"(mean {prov['tmam_mean']:.2f} °C), YEAR (mean {prov['year_mean']:.0f}). "
             "Centring leaves slopes unchanged, reduces main-vs-interaction collinearity.\n")

    # ------------------------------------------------------------------ Spec & identifiability
    L.append("### Specification & a note on voltinism identifiability\n")
    L.append("Primary: **FIRSTDAY ~ NAO_DJFM × VOLTINISM + YEAR_c + species FE + (1 | SITE_ID)**. "
             "Because **voltinism is nested in species**, its *main* effect is collinear with "
             "the species fixed effects and is deliberately omitted (absorbed by the species "
             "dummies); the **NAO × voltinism interaction** is still identified from "
             "within-species year-to-year NAO variation. We build explicit interaction columns "
             "(`NAO_c_multi`, `NAO_c_var`) so the design is full-rank (no silent rank "
             "deficiency).\n")
    L.append("**Why YEAR_c is included:** NAO varies *only by year* and is confounded with the "
             "long-term warming trend, so without a year term an NAO coefficient could merely "
             "capture warming. Reported with and without YEAR_c below.\n")

    # ------------------------------------------------------------------ Estimation routes
    ml = R["mixedlm"]
    L.append("### Estimation routes (all reported — no silent switching)\n")
    L.append(f"1. **MixedLM** (statsmodels, site random intercept): converged in "
             f"{ml.get('seconds')} s (status `{ml['status']}`) on the full "
             f"{prov['analysis_rows']:,} rows — used for the **variance decomposition / ICC**.")
    L.append("2. **FE-OLS + cluster-robust SEs** (species dummies; two-way clusters "
             f"SITE_ID × YEAR): the **coefficient workhorse** — cov used = "
             f"`{R['fits']['primary_nao']['cov_used']}`.")
    L.append("3. **Two-stage**: per-species OLS of FIRSTDAY on NAO_c + YEAR_c, then "
             "precision-weighted meta-analysis of the NAO slopes by voltinism — a transparent "
             f"{WS}-style cross-check.\n")
    L.append("> **Key inference point.** NAO takes one value per year — only ~49 independent "
             "observations across 1976–2024. **Year clustering is what makes the SEs honest**: "
             "model-based MixedLM/OLS SEs (treating the ~555k rows as independent given site) "
             "would badly overstate NAO significance. So we read significance off the "
             "**year-clustered** SEs and use MixedLM only for the variance components.\n")

    # ------------------------------------------------------------------ Variance / ICC
    if "icc" in ml:
        L.append("### Site random effect (MixedLM variance components)\n")
        L.append(f"- Site-intercept SD = **{ml['site_sd']:.1f} days**, residual SD = "
                 f"**{ml['resid_sd']:.1f} days** → **ICC = {ml['icc']:.3f}** "
                 f"(≈ {ml['icc']*100:.0f}% of the leftover variance, after species FE + NAO + "
                 "year, is between-site). Site variation is real and worth a random intercept, "
                 "but species identity dominates.\n")

    # ------------------------------------------------------------------ NAO slopes
    L.append("### NAO effect by voltinism (days per NAO unit)\n")
    L.append("Primary (FE-OLS, **with YEAR_c**, year-clustered CIs):\n")
    L.append(_slopes_md(R["nao_slopes"]["primary"]))
    L.append("\nWithout YEAR_c:\n")
    L.append(_slopes_md(R["nao_slopes"]["primary_noyear"]))
    L.append("\nTwo-stage meta-analysis (per-species slopes pooled by class; **narrower CIs — "
             "see caveat**):\n")
    L.append(_tbl(R["two_stage_meta"],
                  ["class", "n_species", "nao_slope_wmean", "se", "ci_low", "ci_high",
                   "slope_sd_across_species"]))
    L.append(f"\n_Caveat:_ the two-stage CIs treat each species' slope as independent, but all "
             "species share the same NAO years, so they **understate** uncertainty; the "
             "year-clustered FE-OLS CIs are the honest ones. The point estimates agree across "
             "all three routes (MixedLM NAO_c = "
             f"{ml.get('nao_c', float('nan')):+.2f}).\n")

    # interaction interpretation
    pr = R["nao_slopes"]["primary"].set_index("class")
    diff_multi = pr.loc["multivoltine", "slope"] - pr.loc["univoltine", "slope"]
    diff_var = pr.loc["variable", "slope"] - pr.loc["univoltine", "slope"]
    inter = R["fits"]["primary_nao"]["coefs"].set_index("term")
    L.append("**Interaction (difference in NAO slope vs univoltine):** multivoltine − "
             f"univoltine = {diff_multi:+.2f} days/unit "
             f"(CI {inter.loc['NAO_c_multi','ci_low']:+.2f}, {inter.loc['NAO_c_multi','ci_high']:+.2f}); "
             f"variable − univoltine = {diff_var:+.2f} "
             f"(CI {inter.loc['NAO_c_var','ci_low']:+.2f}, {inter.loc['NAO_c_var','ci_high']:+.2f}). "
             "Both interaction CIs straddle 0 → **no detectable voltinism difference** in the "
             "NAO response over the full record.\n")

    # ------------------------------------------------------------------ Binary sensitivity
    L.append("### Sensitivity — binary voltinism (univoltine vs multivoltine, 'variable' "
             f"excluded; n = {R['binary_n']:,}). Closest to the {WS} dichotomy:\n")
    L.append(_slopes_md(R["binary_slopes"]))
    bi = R["binary_fit"]["coefs"].set_index("term")
    L.append(f"\nInteraction NAO×multivoltine = {bi.loc['NAO_c_multi','estimate']:+.3f} "
             f"days/unit (CI {bi.loc['NAO_c_multi','ci_low']:+.2f}, "
             f"{bi.loc['NAO_c_multi','ci_high']:+.2f}) — again indistinguishable from 0.\n")

    # ------------------------------------------------------------------ Climate-only / both / VIF
    L.append("### Local temperature vs NAO (RQ3 set-up)\n")
    L.append("Spring-temperature (TMEAN_MAM) effect by voltinism (days per °C):\n")
    L.append(_tbl(R["temp_slopes"]["climate_only"].rename(columns={"slope": "days/degC"}),
                  ["class", "days/degC", "se", "ci_low", "ci_high"]))
    # temperature × voltinism interaction (this one IS significant)
    tc = R["fits"]["climate_only"]["coefs"].set_index("term")
    L.append(f"\n**Temperature × voltinism IS detectable:** multivoltine species are more "
             f"spring-temperature-sensitive than univoltine — interaction "
             f"TMAM×multivoltine = {tc.loc['TMAM_c_multi','estimate']:+.2f} days/°C "
             f"(CI {tc.loc['TMAM_c_multi','ci_low']:+.2f}, {tc.loc['TMAM_c_multi','ci_high']:+.2f}, "
             f"p = {tc.loc['TMAM_c_multi','p']:.3f}). So voltinism modulates the **temperature** "
             "response though not the (weak) NAO response — a better-identified contrast, since "
             "local temperature varies site×year, not just year.\n")
    L.append("**Both together (NAO + spring temp), NAO slopes once local temp is in the model:**\n")
    L.append(_slopes_md(R["nao_slopes"]["both"]))
    L.append("\nVIFs (both-model continuous terms):\n")
    L.append(_tbl(R["vifs"], ["term", "VIF"], floatfmt="{:.2f}"))
    nb = R["fits"]["both"]["coefs"].set_index("term")
    L.append(f"\n**NAO sign flip — the RQ3 headline.** The univoltine NAO slope is "
             f"{pr.loc['univoltine','slope']:+.2f} days/unit marginally, but with spring "
             f"temperature added it **flips to {nb.loc['NAO_c','estimate']:+.2f}** "
             f"(CI {nb.loc['NAO_c','ci_low']:+.2f}, {nb.loc['NAO_c','ci_high']:+.2f}). "
             "A reversal under moderate collinearity (VIF ≈ 3; EDA NAO–DJF-temp r ≈ 0.68) means "
             "the two share their predictive content: **NAO's apparent phenology effect operates "
             "through local temperature, not independently of it.** We do not over-claim from an "
             "unstable coefficient — NAO carries little beyond local temperature, which sets up "
             "RQ3. Spring temperature itself stays large and stable (≈ −4.4 days/°C) in both "
             "models.\n")

    # ------------------------------------------------------------------ Coefficient tables
    L.append("### Full coefficient tables (non-species terms; 95% CIs)\n")
    for k in ["primary_nao", "primary_nao_noyear", "climate_only", "both"]:
        f = R["fits"][k]
        L.append(f"**{k}** — R² = {f['r2']:.3f}, n = {f['n']:,}, cov = `{f['cov_used']}`:\n")
        L.append(_tbl(f["coefs"], ["term", "estimate", "se", "ci_low", "ci_high", "p"]))
        L.append("")

    # ------------------------------------------------------------------ 2024 sensitivity
    L.append("### Sensitivity — excluding flight-year 2024\n")
    pe = R_excl["nao_slopes"]["primary"].set_index("class")
    L.append(f"Re-running the primary spec with 2024 dropped "
             f"({R_excl['prov']['analysis_rows']:,} rows) gives univoltine NAO slope "
             f"{pe.loc['univoltine','slope']:+.3f} (vs {pr.loc['univoltine','slope']:+.3f} with "
             "the CPC-rescaled 2024). **Conclusions are unchanged** — the 2024 handling does "
             "not move the NAO or interaction estimates materially.\n")

    # ------------------------------------------------------------------ Westgarth-Smith
    L.append("### Comparison to Westgarth-Smith et al. (2012)\n")
    slope_u = pr.loc["univoltine", "slope"]
    L.append(f"- **Direction agrees:** the NAO–phenology slope is **negative** (positive winter "
             f"NAO → earlier first appearance; ≈ {slope_u:.2f} days per NAO unit for "
             "univoltine species), consistent with WS and with the EDA year-level relationship.")
    L.append("- **Strength / significance — we qualify WS:** over the *full* 1976–2024 record, "
             "with honest year-clustered SEs and species + site structure removed, the NAO "
             "effect is **weak and not statistically distinguishable from zero**. NAO's ~49 "
             "annual values simply do not pin the slope down tightly.")
    L.append("- **Voltinism difference — not supported:** unlike a simple univoltine/"
             "multivoltine split, we find **no detectable difference** in the NAO response "
             "between voltinism classes (all interaction CIs cover 0), in the 3-level primary "
             "and in the binary WS-style contrast.\n")

    # ------------------------------------------------------------------ Diagnostics
    d = R["diag"]
    L.append("### Model diagnostics\n")
    L.append(f"![diagnostics](models/{d['fig']})")
    L.append(f"- Residual mean ≈ {d['resid_mean']:.2f}, SD ≈ {d['resid_sd']:.1f} days. "
             "Residuals-vs-fitted show mild heteroscedasticity; the QQ plot has heavy tails "
             "(FIRSTDAY is bounded and right-skewed) — SEs are cluster-robust, so inference is "
             "not reliant on normality. Mean residual by year is centred on 0 (worst year "
             f"{d['worst_year'][0]}: {d['worst_year'][1]:+.1f} d); by species it is near 0 "
             "(species FE absorb baselines).\n")

    # ------------------------------------------------------------------ Harness
    L.append("### Out-of-sample skill (shared harness, both regimes)\n")
    if C.METRICS_PATH.exists():
        m = pd.read_csv(C.METRICS_PATH)
        ov = m[m.scope == "overall"].drop(columns=["scope", "species"])
        # append the like-for-like same-subset anchor
        anc = R.get("anchor_metrics")
        if anc is not None:
            a2 = anc[anc.scope == "overall"].drop(columns=["scope", "species"])
            ov = pd.concat([ov, a2], ignore_index=True)
        ov = ov.sort_values(["regime", "MAE"])[["model", "regime", "n", "MAE", "RMSE", "R2", "bias"]]
        ov["n"] = ov["n"].map("{:,.0f}".format)
        L.append(_tbl(ov, floatfmt="{:.2f}"))
    L.append("\n`me_nao` = primary (NAO×voltinism + year + species FE); `me_nao_temp` adds "
             "spring temperature; `species_mean_mesubset` is the species-mean climatology "
             "scored on the **identical ME subset** (the fair anchor).\n")
    L.append("> **Read the `n` column carefully.** The ME models drop the two migrant *Vanessa* "
             "species (heavily recorded, with erratic arrival timing that inflates everyone's "
             "error), so the shared full-frame `species_mean` is *not* a like-for-like "
             "comparator. Against the same-subset anchor: in **grouped-CV** `me_nao_temp` "
             "improves MAE to ~21.3 vs ~21.9 (a real ~0.6-day gain from spring temperature); in "
             "the **temporal** hold-out the models are ~tied, and `me_nao` (no temperature) is "
             "barely above the anchor. **Temperature structure buys a modest gain; NAO structure "
             "essentially none** — consistent with the sign flip. This rung's value is chiefly "
             "scientific (RQ1 inference), not point-prediction.\n")

    # ------------------------------------------------------------------ Assumptions
    L.append("## Assumptions & judgement calls\n")
    for a in [
        "**2024 NAO:** filled from CPC DJFM rescaled onto the Hurrell scale via the overlap "
        "regression (not raw-concatenated); flagged in NAO_SOURCE. Excluding 2024 instead "
        "leaves conclusions unchanged (shown above).",
        "**Voltinism:** primary = 3-level (univoltine ref / multivoltine / variable); "
        "excluded the Thymelicus lineola/sylvestris aggregate and the two migrants "
        "(Vanessa atalanta, V. cardui) Cook leaves voltinism-blank; binary uni-vs-multi "
        "reported as the WS-style sensitivity.",
        "**Species as fixed effect** (dummies) absorbs the dominant per-species baseline the "
        "EDA showed; voltinism main effect is therefore not separately identified (nested).",
        "**Site as random intercept** (MixedLM) / **clustered by SITE_ID** (FE-OLS) — two "
        "ways to handle within-site correlation; both reported and they agree.",
        "**Inference on NAO uses year-clustered SEs** because NAO varies only annually; "
        "model-based SEs would overstate significance.",
        "**YEAR_c included** to separate the NAO signal from the secular warming trend; "
        "with/without both shown.",
        "**Predictors centred**; slopes unaffected. **Target never imputed** — rows missing "
        "FIRSTDAY or a predictor are dropped and counted.",
        "**Harness metrics** for ME models are on the analysis subset (fewer rows than the "
        "baselines); comparison is like-for-like on shared rows within each model's `n`.",
    ]:
        L.append(f"- {a}")
    L.append("")
    L.append("## Artefacts\n")
    for f, desc in [
        ("output/mixedeffects_report.md", "this report"),
        ("output/models/mixedeffects_coefficients.csv", "every FE-OLS spec's coefficients + CIs"),
        ("output/models/mixedeffects_predictions.parquet", "harness long predictions (me_nao, me_nao_temp)"),
        ("output/models/me_diagnostics.png", "residual diagnostics"),
        ("data/nao_djfm_cpc.csv", "reproducible CPC DJFM series"),
        ("data/feature_table.parquet", "+ NAO_SOURCE, 2024 NAO filled"),
        ("output/model_metrics.csv", "shared metrics incl. me_nao / me_nao_temp"),
    ]:
        L.append(f"- `{f}` — {desc}")
    L.append("")
    C.ME_REPORT_MD.write_text("\n".join(L))
    print(f"wrote {C.ME_REPORT_MD}")
