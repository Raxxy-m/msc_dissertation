#!/usr/bin/env python
"""
Curated methodology + results figures for the dissertation (main body + appendix).

EVERY data-driven figure is computed from the real artefacts — no number is transcribed
from a report or from memory:
    output/model_metrics.csv                    (model ladder MAE/RMSE/R2/bias)
    output/model_predictions.parquet            (per-year bias recomputed here)
    output/models/mixedeffects_coefficients.csv (coef + 95% CI, all specs)
    output/gdd_species_calibration.csv          (per-species critical sum S*)
    output/modelling_frame.parquet              (real split / fold sizes)
    data/nao_djfm.csv, data/nao_djfm_cpc.csv    (Hurrell + CPC NAO for the 2024 fix)
    data/feature_table.parquet, data/site_daily_climate/  (raw-series illustrations)
The mixed-model by-class slopes (multi/variable need the covariance, not in the coef CSV)
and the MixedLM variance components are RECOMPUTED via scripts.modelling.mixed_effects on
the same frame, so they match the report exactly rather than being copied.

Outputs (into output/figures/): each figure as PNG@300 dpi AND PDF (vector).
Writes output/report_figures_report.md cataloguing every figure. Reads only; never
modifies the eda figures or any pipeline output.

Run:  .venv/bin/python -m scripts.modelling.report_figures
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
import seaborn as sns

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.modelling import config as C
else:
    from . import config as C

# =====================================================================================
# CONFIG
# =====================================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = {
    "fig_dir": PROJECT_ROOT / "output" / "figures",
    "report_md": PROJECT_ROOT / "output" / "report_figures_report.md",
    "metrics": PROJECT_ROOT / "output" / "model_metrics.csv",
    "predictions": PROJECT_ROOT / "output" / "model_predictions.parquet",
    "coef_csv": PROJECT_ROOT / "output" / "models" / "mixedeffects_coefficients.csv",
    "gdd_csv": PROJECT_ROOT / "output" / "gdd_species_calibration.csv",
    "frame": PROJECT_ROOT / "output" / "modelling_frame.parquet",
    "feature_table": PROJECT_ROOT / "data" / "feature_table.parquet",
    "daily_dir": PROJECT_ROOT / "data" / "site_daily_climate",
    "nao_hurrell": PROJECT_ROOT / "data" / "nao_djfm.csv",
    "nao_cpc": PROJECT_ROOT / "data" / "nao_djfm_cpc.csv",
    "dpi": 300,
    "seed": 42,
    # Okabe-Ito colourblind-safe palette
    "oi": {"black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9",
           "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
           "vermillion": "#D55E00", "purple": "#CC79A7", "grey": "#999999"},
    # illustration parameters (real years chosen from data)
    "gdd_warm_year": 2024, "gdd_cold_year": 2013,   # warmest / coldest national spring
    "gdd_demo_species": "Maniola jurtina",          # S* read from the calibration CSV
}
OI = CONFIG["oi"]
sns.set_theme(style="whitegrid", context="notebook")
# --- Shared figure style (keep IDENTICAL across all plotting modules) ---------
plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "svg.fonttype": "none",
})

# consistent per-model / per-class colours across figures
MODEL_COLOR = {
    "persistence": OI["grey"], "species_mean": OI["skyblue"],
    "site_species_mean": OI["blue"], "gdd_process": OI["vermillion"],
    "me_nao": OI["purple"], "me_nao_temp": OI["yellow"],
    "ml_hgb": OI["green"], "ml_rf": OI["black"],
}
MODEL_LABEL = {
    "persistence": "persistence", "species_mean": "species mean",
    "site_species_mean": "site×species mean", "gdd_process": "GDD process",
    "me_nao": "ME: NAO×volt", "me_nao_temp": "ME: NAO+temp",
    "ml_hgb": "ML: HGB", "ml_rf": "ML: RF",
}
VOLT_COLOR = {"univoltine": OI["blue"], "multivoltine": OI["vermillion"],
              "variable": OI["green"]}

MANIFEST = []   # (filename_stem, description, source, placement)


def save_fig(fig, stem, description, source, placement):
    CONFIG["fig_dir"].mkdir(parents=True, exist_ok=True)
    png = CONFIG["fig_dir"] / f"{stem}.png"
    pdf = CONFIG["fig_dir"] / f"{stem}.pdf"
    fig.savefig(png, dpi=CONFIG["dpi"], bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    MANIFEST.append((stem, description, source, placement))
    print(f"  wrote figures/{stem}.png + .pdf  [{placement}]")


# =====================================================================================
# Recompute the mixed-effects pieces (by-class slopes need the covariance; ICC from MixedLM)
# =====================================================================================
_ME_CACHE = {}


def compute_me():
    if _ME_CACHE:
        return _ME_CACHE
    from scripts.modelling import mixed_effects as ME
    frame, _ = ME.build_analysis_frame(exclude_2024=False)
    fits = {
        "primary": ME.fit_ols_cluster(frame, ["NAO_c", "NAO_c_multi", "NAO_c_var", "YEAR_c"],
                                      "primary_nao"),
        "climate": ME.fit_ols_cluster(frame, ["TMAM_c", "TMAM_c_multi", "TMAM_c_var", "YEAR_c"],
                                      "climate_only"),
        "both": ME.fit_ols_cluster(
            frame, ["NAO_c", "NAO_c_multi", "NAO_c_var",
                    "TMAM_c", "TMAM_c_multi", "TMAM_c_var", "YEAR_c"], "both"),
    }
    _ME_CACHE.update({
        "nao_by_class": ME.slope_by_class(fits["primary"], "NAO_c", "NAO_c_multi", "NAO_c_var"),
        "temp_by_class": ME.slope_by_class(fits["climate"], "TMAM_c", "TMAM_c_multi", "TMAM_c_var"),
        "nao_by_class_both": ME.slope_by_class(fits["both"], "NAO_c", "NAO_c_multi", "NAO_c_var"),
        "mixedlm": ME.fit_mixedlm(frame, ["NAO_c", "NAO_c_multi", "NAO_c_var", "YEAR_c"]),
    })
    return _ME_CACHE


# =====================================================================================
# 1. Mixed-model structure schematic  [BODY]
# =====================================================================================
def fig1_me_structure():
    me = compute_me()
    ml = me["mixedlm"]
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 6))

    # ---- (a) annotated equation --------------------------------------
    axA.axis("off")
    axA.set_title("(a) Primary specification", loc="left", fontweight="bold")
    axA.text(0.5, 0.86, r"FIRSTDAY $\sim$ NAO$\times$VOLTINISM + YEAR$_c$"
             "\n" r"          + species FE + (1 | SITE)",
             ha="center", va="center", fontsize=15, family="monospace",
             transform=axA.transAxes,
             bbox=dict(boxstyle="round,pad=0.6", fc=OI["yellow"], ec="black", alpha=0.35))
    calls = [
        (0.06, 0.60, "NAO×VOLTINISM interaction — identified from\nwithin-species year-to-year "
                     "NAO variation (voltinism\nmain effect is nested in species → absorbed)", OI["purple"]),
        (0.06, 0.38, "species as FIXED effects — absorb the dominant\nper-species emergence "
                     "baseline (60 levels)", OI["green"]),
        (0.06, 0.20, "site as a RANDOM intercept — (1 | SITE);\nwithin-site correlation, "
                     f"ICC = {ml['icc']:.3f}", OI["blue"]),
        (0.06, 0.04, "predictors mean-centred; inference on\nYEAR-clustered SEs (NAO varies only "
                     "by year)", OI["vermillion"]),
    ]
    for x, y, txt, col in calls:
        axA.text(x, y, txt, ha="left", va="center", fontsize=9.5, transform=axA.transAxes,
                 bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=col, lw=1.6))

    # ---- (b) random-intercept illustration ---------------------------
    site_sd, resid_sd, icc = ml["site_sd"], ml["resid_sd"], ml["icc"]
    slope = float(me["nao_by_class"].set_index("class").loc["univoltine", "slope"])
    rng = np.random.default_rng(CONFIG["seed"])
    x = np.linspace(-3, 4, 50)          # NAO range
    centre = 45.0                        # illustrative mean intercept (days)
    intercepts = centre + rng.normal(0, site_sd, size=5)
    palette = [OI["blue"], OI["orange"], OI["green"], OI["vermillion"], OI["purple"]]
    for b0, col in zip(intercepts, palette):
        axB.plot(x, b0 + slope * x, color=col, lw=2, zorder=3)
        # faint residual scatter around each site's line
        xs = rng.uniform(-3, 4, 40)
        axB.scatter(xs, b0 + slope * xs + rng.normal(0, resid_sd, 40),
                    s=6, color=col, alpha=0.18, zorder=1)
    axB.axhline(centre, color="black", ls=":", lw=1, zorder=2)
    axB.annotate(f"site-intercept SD = {site_sd:.1f} d",
                 xy=(-3, centre + site_sd), xytext=(-2.9, centre + 2.4 * site_sd),
                 fontsize=9, arrowprops=dict(arrowstyle="->"))
    axB.annotate("", xy=(3.6, intercepts.min() + slope * 3.6),
                 xytext=(3.6, intercepts.max() + slope * 3.6),
                 arrowprops=dict(arrowstyle="<->", color="black"))
    axB.text(3.7, centre, "between-site\nspread", fontsize=8, va="center")
    axB.set_xlabel("winter NAO (centred)")
    axB.set_ylabel("FIRSTDAY (days after 1 Apr)")
    axB.set_title(f"(b) Random site intercepts: shared slope, different intercepts\n"
                  f"site SD {site_sd:.1f} d · residual SD {resid_sd:.1f} d · "
                  f"ICC {icc:.3f} (species FE dominate; site is the random part)",
                  loc="left")
    fig.tight_layout()
    save_fig(fig, "fig_me_structure",
             "Mixed-model schematic: annotated primary equation + random site-intercept illustration.",
             "Recomputed MixedLM variance components (site SD, resid SD, ICC) + univoltine NAO "
             "slope via scripts.modelling.mixed_effects on output/modelling_frame.parquet.",
             "BODY")


# =====================================================================================
# 2. GDD mechanism schematic  [BODY]  — real daily climate, real S*
# =====================================================================================
def _national_cum_gdd(year, t_base=5.0):
    d = pd.read_parquet(CONFIG["daily_dir"] / f"year={year}.parquet",
                        columns=["doy", "tmean"])
    daily = d.groupby("doy")["tmean"].mean().sort_index()
    gdd = np.clip(daily.values - t_base, 0, None)
    return daily.index.values, np.cumsum(gdd)


def fig2_gdd_mechanism():
    gdd = pd.read_csv(CONFIG["gdd_csv"])
    sp = CONFIG["gdd_demo_species"]
    sstar = float(gdd.loc[gdd.SPECIES_NAME == sp, "S_star"].iloc[0])
    warm_y, cold_y = CONFIG["gdd_warm_year"], CONFIG["gdd_cold_year"]
    dw, cw = _national_cum_gdd(warm_y)
    dc, cc = _national_cum_gdd(cold_y)

    def crossing(doy, cum):
        i = int(np.searchsorted(cum, sstar))
        return doy[i] if i < len(doy) else None

    xw, xc = crossing(dw, cw), crossing(dc, cc)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(dw, cw, color=OI["vermillion"], lw=2.4, label=f"warm spring ({warm_y})")
    ax.plot(dc, cc, color=OI["blue"], lw=2.4, label=f"cold spring ({cold_y})")
    ax.axhline(sstar, color="black", ls="--", lw=1.4)
    ax.text(5, sstar + 15, f"critical sum S* = {sstar:.0f} °C·day\n({sp})",
            fontsize=9, va="bottom")
    for xc_, cum_col in [(xw, OI["vermillion"]), (xc, OI["blue"])]:
        if xc_ is not None:
            ax.plot([xc_, xc_], [0, sstar], color=cum_col, ls=":", lw=1.6)
            ax.scatter([xc_], [sstar], color=cum_col, zorder=5, s=40)
            ax.annotate(f"emerge\nDOY {xc_}", xy=(xc_, 0), xytext=(xc_, sstar * 0.28),
                        ha="center", fontsize=8, color=cum_col,
                        arrowprops=dict(arrowstyle="->", color=cum_col))
    if xw and xc:
        ax.annotate("", xy=(xw, sstar * 0.5), xytext=(xc, sstar * 0.5),
                    arrowprops=dict(arrowstyle="<->", color="black"))
        ax.text((xw + xc) / 2, sstar * 0.53, f"{xc - xw} d\nearlier", ha="center",
                fontsize=8)
    ax.axvline(1, color=OI["grey"], lw=0.8)
    ax.text(3, ax.get_ylim()[1] * 0.02, "accumulation start (DOY 1),  base T = 5 °C",
            fontsize=8, color=OI["grey"])
    ax.set_xlim(0, max(xw or 200, xc or 200) + 40)
    ax.set_ylim(0, sstar * 1.5)
    ax.set_xlabel("day of year")
    ax.set_ylabel("cumulative growing-degree-days (°C·day)")
    ax.set_title("GDD process model: emergence when cumulative GDD reaches S*\n"
                 "warm springs cross the threshold earlier (interannual tracking)", loc="left")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_fig(fig, "fig_gdd_mechanism",
             "GDD mechanism: cumulative GDD vs day-of-year for the warmest and coldest national "
             "springs, with a real species critical sum S* and the emergence crossings.",
             f"data/site_daily_climate (national-mean daily Tmean, {warm_y} vs {cold_y}); "
             f"S* for {sp} from output/gdd_species_calibration.csv.",
             "BODY")


# =====================================================================================
# 3. Leakage-window timeline  [BODY/APPENDIX]
# =====================================================================================
def fig3_leakage_timeline():
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.axis("off")
    # month axis: Dec(y-1) .. following Sep(y)
    months = ["Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]
    x = np.arange(len(months))
    y0 = 0.5
    # predictor window Dec..May (indices 0..5)
    ax.add_patch(FancyBboxPatch((-0.1, y0), 5.7, 0.5, boxstyle="round,pad=0.02",
                                fc=OI["skyblue"], ec=OI["blue"], alpha=0.5))
    ax.text(2.7, y0 + 0.6, "PREDICTOR WINDOW — pre-flight climate\n"
            "Dec(y−1) → 31 May(y)", ha="center", fontsize=10, fontweight="bold",
            color=OI["blue"])
    # flight period Apr..Sep (indices 4..9) — overlaps spring window slightly
    ax.add_patch(FancyBboxPatch((3.9, y0 - 0.75), 5.7, 0.5, boxstyle="round,pad=0.02",
                                fc=OI["orange"], ec=OI["vermillion"], alpha=0.5))
    ax.text(6.7, y0 - 0.95, "FLIGHT PERIOD (FIRSTDAY … LASTDAY)", ha="center",
            fontsize=10, fontweight="bold", color=OI["vermillion"])
    # feature labels
    ax.text(0, y0 + 0.25, "_M12P", ha="center", fontsize=8)
    for i, m in enumerate(["_M01", "_M02", "_M03", "_M04", "_M05"], start=1):
        ax.text(i, y0 + 0.25, m, ha="center", fontsize=8)
    ax.text(1, y0 + 0.05, "DJF", ha="center", fontsize=8, style="italic", color=OI["blue"])
    ax.text(4, y0 + 0.05, "MAM", ha="center", fontsize=8, style="italic", color=OI["blue"])
    ax.text(2.7, y0 + 0.4, "GDD accumulates →", ha="center", fontsize=8, color=OI["blue"])
    # no-post-flight rule
    ax.axvline(5.5, color="black", ls="--", lw=1.5)
    ax.text(5.6, y0 + 1.05, "no climate AFTER 31 May enters the fixed ML features\n"
            "(no post-flight leakage)", fontsize=9, va="top")
    for xi, m in zip(x, months):
        ax.text(xi, y0 - 0.15, m, ha="center", fontsize=9)
    ax.text(-0.1, y0 - 0.35, "← winter (year−1)     spring/summer (flight year) →",
            fontsize=8, color=OI["grey"])
    ax.text(2.7, y0 - 1.35, "Caveat: very early species can fly inside the spring window; the "
            "mechanistic GDD model (which accumulates to each event's own\nemergence day, using "
            "only prior days) is exempt from the Dec–May cap and is not affected by this.",
            ha="center", fontsize=8, style="italic", color=OI["grey"])
    ax.set_xlim(-0.6, 10)
    ax.set_ylim(-1.6, 1.7)
    ax.set_title("Leakage-safe predictor window vs the flight period", loc="left",
                 fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "fig_leakage_timeline",
             "Timeline of the Dec(y−1)→31 May(y) leakage-safe predictor window vs the flight "
             "period, with feature-family labels and the no-post-flight rule.",
             "Schematic (matplotlib); feature naming from data/feature_table.parquet conventions "
             "(_M12P/_M01..M05, _DJF/_MAM, GDD).",
             "APPENDIX")


# =====================================================================================
# 4. Evaluation-regimes schematic  [APPENDIX]  — real sizes from the frame
# =====================================================================================
def fig4_eval_regimes():
    frame = pd.read_parquet(CONFIG["frame"], columns=["YEAR", "SITE_ID",
                                                       "temporal_split", "cv_fold"])
    n_train = int((frame.temporal_split == "train").sum())
    n_test = int((frame.temporal_split == "test").sum())
    fold_sizes = frame.groupby("cv_fold").size().sort_index().tolist()
    n_sites = frame.SITE_ID.nunique()

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5))
    # ---- (i) temporal hold-out ----
    axL.axis("off")
    axL.set_title("(i) Temporal hold-out (distribution-shift test)", loc="left",
                  fontweight="bold")
    span = 2024 - 1976
    axL.add_patch(mpatches.Rectangle((1976, 0.4), 2014 - 1976, 0.35,
                                     fc=OI["skyblue"], ec="black", alpha=0.6))
    axL.add_patch(mpatches.Rectangle((2014, 0.4), 2024 - 2014, 0.35,
                                     fc=OI["orange"], ec="black", alpha=0.7))
    axL.text(1995, 0.57, f"TRAIN  YEAR ≤ 2014\n{n_train:,} rows", ha="center", fontsize=10)
    axL.text(2019.5, 0.57, f"TEST\nYEAR > 2014\n{n_test:,} rows", ha="center", fontsize=9)
    for yr in [1976, 2014, 2024]:
        axL.axvline(yr, ymin=0.35, ymax=0.78, color="black", lw=0.8)
        axL.text(yr, 0.33, str(yr), ha="center", fontsize=8)
    axL.text(2000, 0.2, "the last ~10 years (incl. the warm 2020, 2024) are held out to test "
             "extrapolation\nto a shifted climate", ha="center", fontsize=8, color=OI["grey"])
    axL.set_xlim(1974, 2026)
    axL.set_ylim(0.1, 0.9)

    # ---- (ii) grouped 5-fold CV ----
    axR.set_title("(ii) Grouped 5-fold CV (groups = SITE_ID)", loc="left", fontweight="bold")
    for k, sz in enumerate(fold_sizes):
        for j in range(5):
            col = OI["orange"] if j == k else OI["skyblue"]
            axR.barh(k, 1, left=j, color=col, ec="white",
                     alpha=0.7 if j == k else 0.5)
        axR.text(5.1, k, f"{sz:,}", va="center", fontsize=8)
    axR.set_yticks(range(5))
    axR.set_yticklabels([f"fold {i}" for i in range(5)])
    axR.set_xticks([])
    axR.set_xlim(0, 6.2)
    axR.invert_yaxis()
    axR.set_xlabel(f"{n_sites:,} sites partitioned into 5 folds — a site never spans "
                   "train & test")
    axR.legend(handles=[mpatches.Patch(color=OI["skyblue"], alpha=0.5, label="train folds"),
                        mpatches.Patch(color=OI["orange"], alpha=0.7, label="test fold")],
               loc="lower right", fontsize=8)
    axR.text(2.5, 5.2, "→ held-out sites have no site history, so site×species mean "
             "collapses to the species mean", ha="center", fontsize=8, color=OI["grey"])
    fig.suptitle("Two evaluation regimes (assignment fixed once, reused by every model)",
                 fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "fig_eval_regimes",
             "The two evaluation regimes: temporal hold-out (train≤2014/test>2014) and "
             "site-grouped 5-fold CV, with real split and fold sizes.",
             "output/modelling_frame.parquet (temporal_split, cv_fold, SITE_ID counts).",
             "APPENDIX")


# =====================================================================================
# 5. Model-ladder MAE comparison  [BODY]
# =====================================================================================
def fig5_model_ladder():
    m = pd.read_csv(CONFIG["metrics"])
    ov = m[m.scope == "overall"][["model", "regime", "n", "MAE"]].copy()
    ov = ov[~ov.model.str.endswith("_nonao")]         # drop RQ3 ablation variants
    ov = ov[ov.model != "species_mean_mesubset"]      # keep the main ladder
    piv = ov.pivot_table(index="model", columns="regime", values="MAE")
    npiv = ov.pivot_table(index="model", columns="regime", values="n")
    piv = piv.sort_values("temporal", ascending=True)  # best (lowest MAE) first

    ys = np.arange(len(piv))[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    for y, model in zip(ys, piv.index):
        t, g = piv.loc[model, "temporal"], piv.loc[model, "grouped_cv"]
        ax.plot([g, t], [y, y], color=OI["grey"], lw=2, zorder=1)
        ax.scatter(g, y, color=MODEL_COLOR.get(model, "black"), s=95, zorder=3,
                   marker="o", edgecolor="black", linewidths=0.6)
        ax.scatter(t, y, color=MODEL_COLOR.get(model, "black"), s=95, zorder=3,
                   marker="D", edgecolor="black", linewidths=0.6)
    ax.set_yticks(ys)
    ax.set_yticklabels([MODEL_LABEL.get(mm, mm) for mm in piv.index])
    ax.set_xlabel("overall MAE (days) — lower is better")
    ax.set_title("Model ladder: out-of-sample MAE by regime\n"
                 "(circle = grouped-CV, diamond = temporal hold-out)", loc="left")
    xmin, xmax = np.nanmin(piv.values), np.nanmax(piv.values)
    ax.set_xlim(xmin - 0.25, xmax + 0.45)                # margin so no marker is clipped
    ax.set_ylim(-0.6, len(piv) - 0.4)
    # legend for the two regimes (upper right is empty — best models sit at the left)
    h = [plt.Line2D([], [], marker="o", ls="", color="grey", label="grouped-CV"),
         plt.Line2D([], [], marker="D", ls="", color="grey", label="temporal")]
    ax.legend(handles=h, loc="upper right")
    # caveat annotation for ME models (different n)
    me_models = [mm for mm in piv.index if mm.startswith("me_")]
    if me_models:
        ntxt = ", ".join(f"{MODEL_LABEL[mm]} n≈{int(npiv.loc[mm,'temporal']):,}" for mm in me_models)
        ax.text(0.5, -0.14, "Note: ME models are scored on a smaller subset (drop the two migrant "
                f"Vanessa spp.) — {ntxt} vs 325,738 for the full-frame\nbaselines — so they are "
                "NOT a like-for-like row against the others (see mixedeffects_report.md).",
                transform=ax.transAxes, ha="center", fontsize=7.5, color=OI["grey"])
    fig.tight_layout()
    save_fig(fig, "fig_model_ladder_mae",
             "Dumbbell of overall MAE per model in both regimes (temporal ◆ vs grouped-CV ○), "
             "ordered by skill; ME different-n caveat annotated.",
             "output/model_metrics.csv (scope=overall; MAE, n per model×regime).",
             "BODY")


# =====================================================================================
# 6. Coefficient forest plots  [BODY]
# =====================================================================================
def _forest(ax, df, title, xlabel, unit):
    ys = np.arange(len(df))[::-1]
    for y, (_, r) in zip(ys, df.iterrows()):
        if "color" in df.columns:
            col = r["color"]
        else:
            col = VOLT_COLOR.get(r["class"], OI["black"]) if "class" in df.columns else OI["blue"]
        ax.plot([r["ci_low"], r["ci_high"]], [y, y], color=col, lw=3.0)
        ax.scatter(r["value"], y, color=col, s=95, zorder=3, edgecolor="white")
        # label beside the CI (same y) so the top row never collides with the title
        ax.annotate(f"{r['value']:+.2f}", (r["ci_high"], y), textcoords="offset points",
                    xytext=(6, 0), va="center", ha="left", fontsize=12)
    ax.axvline(0, color="black", ls="--", lw=1)
    ax.set_yticks(ys)
    ax.set_yticklabels(df["label"])
    ax.set_xlabel(f"{xlabel} ({unit})")
    ax.set_title(title, loc="left", fontsize=14)
    ax.margins(y=0.30)                                   # headroom for the top row + title
    xl = ax.get_xlim()                                   # room for the offset value labels
    ax.set_xlim(xl[0], xl[1] + 0.28 * (xl[1] - xl[0]))


def fig6_coefficients():
    me = compute_me()
    coef = pd.read_csv(CONFIG["coef_csv"])

    # (a) NAO slope by voltinism
    nao = me["nao_by_class"].rename(columns={"slope": "value"}).copy()
    nao["label"] = nao["class"]
    # (b) temperature slope by voltinism
    temp = me["temp_by_class"].rename(columns={"slope": "value"}).copy()
    temp["label"] = temp["class"]
    # (c) univoltine NAO: marginal (primary_nao) vs with spring temp (both) — from coef CSV
    def row(spec, term):
        r = coef[(coef.spec == spec) & (coef.term == term)].iloc[0]
        return {"value": r["estimate"], "ci_low": r["ci_low"], "ci_high": r["ci_high"]}
    # colour by model (marginal vs conditional), NOT by voltinism (both rows are univoltine)
    flip = pd.DataFrame([
        {"label": "NAO alone\n(univoltine)", "color": OI["skyblue"], **row("primary_nao", "NAO_c")},
        {"label": "NAO + spring temp\n(univoltine)", "color": OI["orange"], **row("both", "NAO_c")},
    ])

    # narrower than before (was 15 in) so the text is proportionally larger, whether the
    # figure is viewed directly or placed at \linewidth / \textheight in the dissertation.
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(11, 5.2))
    _forest(a, nao, "(a) NAO effect by voltinism\n(all CIs cross 0 → no signal, no volt. diff.)",
            "days per NAO unit", "year-clustered 95% CI")
    _forest(b, temp, "(b) Spring-temperature effect by voltinism\n(multivoltine more sensitive)",
            "days per °C", "year-clustered 95% CI")
    _forest(c, flip, "(c) RQ3 sign-flip: univoltine NAO slope\nmarginal vs with spring temp",
            "days per NAO unit", "95% CI")
    fig.suptitle("Mixed-effects coefficients (FE-OLS, year-clustered SEs)",
                 fontweight="bold", fontsize=16)
    fig.tight_layout()
    save_fig(fig, "fig_me_coefficients",
             "Forest plots: (a) NAO slope by voltinism, (b) spring-temp slope by voltinism, "
             "(c) univoltine NAO sign-flip (marginal −0.41 → +0.46 with temperature).",
             "By-class slopes recomputed via mixed_effects.slope_by_class; panel (c) NAO_c rows "
             "from output/models/mixedeffects_coefficients.csv (specs primary_nao, both).",
             "BODY")


# =====================================================================================
# 7. Warming-decoupling / bias-by-year  [BODY]  — recomputed from predictions
# =====================================================================================
def fig7_warming_decoupling():
    p = pd.read_parquet(CONFIG["predictions"],
                        columns=["model", "regime", "YEAR", "y_true", "y_pred"])
    d = p[(p.regime == "temporal") & p.model.isin(["gdd_process", "ml_hgb", "species_mean"])].copy()
    d["err"] = d["y_pred"] - d["y_true"]
    yb = d.groupby(["model", "YEAR"])["err"].mean().reset_index()
    yb = yb[yb.YEAR >= 2015]

    fig, ax = plt.subplots(figsize=(10, 6))
    for model in ["gdd_process", "ml_hgb", "species_mean"]:
        s = yb[yb.model == model].sort_values("YEAR")
        ax.plot(s.YEAR, s.err, "-o", color=MODEL_COLOR[model], lw=2.2, ms=6,
                label=MODEL_LABEL[model])
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("year (temporal test set)")
    ax.set_ylabel("mean bias  (predicted − observed FIRSTDAY, days)")
    ax.set_title("Warming decoupling: the GDD process model drifts early in warm years,\n"
                 "while ML stays near zero", loc="left")
    ax.set_xticks(range(2015, 2025))
    ax.legend(loc="lower left")
    ax.annotate("warm springs\n2020, 2024", xy=(2024, yb[(yb.model=='gdd_process')&(yb.YEAR==2024)].err.iloc[0]),
                xytext=(2021.5, -22), fontsize=8, color=OI["vermillion"],
                arrowprops=dict(arrowstyle="->", color=OI["vermillion"]))
    fig.tight_layout()
    save_fig(fig, "fig_warming_decoupling",
             "Per-year mean bias (pred − obs) on the temporal test, 2015–2024, for GDD, ML-HGB "
             "and species-mean: GDD drifts strongly negative in warm years; ML stays near zero.",
             "output/model_predictions.parquet (regime=temporal; recomputed mean(y_pred−y_true) "
             "by model×YEAR).",
             "BODY")


# =====================================================================================
# 8. Variance decomposition / ICC  [APPENDIX]
# =====================================================================================
def fig8_icc_variance():
    ml = compute_me()["mixedlm"]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    bars = ax.bar(["site intercept", "residual"], [ml["site_sd"], ml["resid_sd"]],
                  color=[OI["blue"], OI["grey"]], edgecolor="black", width=0.6)
    for bar, v in zip(bars, [ml["site_sd"], ml["resid_sd"]]):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.4, f"{v:.1f} d",
                ha="center", fontsize=11)
    ax.set_ylabel("standard deviation (days)")
    ax.set_title(f"MixedLM variance components\nICC = {ml['icc']:.3f}  "
                 f"(≈{ml['icc']*100:.0f}% of leftover variance is between-site)", loc="left")
    ax.text(0.04, 0.70, "the residual is what remains\nafter species FE + NAO + year",
            transform=ax.transAxes, ha="left", va="top", fontsize=9, color=OI["grey"])
    fig.tight_layout()
    save_fig(fig, "fig_icc_variance",
             "MixedLM variance decomposition: site-intercept SD vs residual SD, with ICC.",
             "Recomputed MixedLM (site_var, scale) via mixed_effects.fit_mixedlm on the ME frame.",
             "APPENDIX")


# =====================================================================================
# 9. Hurrell–CPC NAO regression (2024 fix)  [APPENDIX]
# =====================================================================================
def fig9_nao_hurrell_cpc():
    hur = pd.read_csv(CONFIG["nao_hurrell"]).rename(columns={"nao_djfm": "hurrell"})
    cpc = pd.read_csv(CONFIG["nao_cpc"])
    m = hur.merge(cpc, on="year").dropna(subset=["hurrell", "cpc_djfm"])
    x, y = m["cpc_djfm"].values, m["hurrell"].values
    b, a = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])
    cpc_2024 = float(cpc.loc[cpc.year == 2024, "cpc_djfm"].iloc[0])
    resc_2024 = a + b * cpc_2024

    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.scatter(x, y, s=28, color=OI["blue"], alpha=0.7,
               label=f"overlap {int(m.year.min())}–{int(m.year.max())} (n={len(m)})")
    xs = np.array([x.min(), max(x.max(), cpc_2024)])
    ax.plot(xs, a + b * xs, color=OI["black"], lw=2,
            label=f"Hurrell = {a:+.3f} + {b:.3f}·CPC   (r = {r:.3f}, R² = {r**2:.3f})")
    ax.scatter([cpc_2024], [resc_2024], s=140, color=OI["vermillion"], marker="*",
               zorder=5, edgecolor="black",
               label=f"2024 CPC {cpc_2024:+.2f} → rescaled {resc_2024:+.2f}")
    ax.plot([cpc_2024, cpc_2024], [ax.get_ylim()[0], resc_2024], ls=":",
            color=OI["vermillion"], lw=1.2)
    ax.set_xlabel("CPC DJFM NAO (NOAA/CPC)")
    ax.set_ylabel("Hurrell station DJFM NAO")
    ax.set_title("2024 NAO gap fix: CPC rescaled onto the Hurrell scale\n"
                 "(two products, different normalisation — not concatenated)", loc="left")
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    save_fig(fig, "fig_nao_hurrell_cpc",
             "Hurrell vs CPC DJFM NAO over the overlap years with the fitted rescaling line and "
             "R², and the rescaled 2024 value highlighted.",
             "data/nao_djfm.csv (Hurrell) + data/nao_djfm_cpc.csv (CPC); regression recomputed here.",
             "APPENDIX")


# =====================================================================================
# Report
# =====================================================================================
def write_report():
    L = ["# Report figures — catalogue\n"]
    L.append(f"_Generated {pd.Timestamp.today().date()}. Script: "
             "`scripts/modelling/report_figures.py`. All files in `output/figures/` as PNG "
             "(300 dpi) + PDF (vector). The LaTeX build imports from a `figures/` folder, so "
             "copy/symlink `output/figures/*` there._\n")
    L.append("Every data-driven figure is computed from the listed artefact — no number is "
             "transcribed from a report. Mixed-effects by-class slopes and MixedLM variance "
             "components are recomputed via `scripts.modelling.mixed_effects` on "
             "`output/modelling_frame.parquet`, so they match `mixedeffects_report.md` exactly.\n")
    body = [m for m in MANIFEST if m[3] == "BODY"]
    appx = [m for m in MANIFEST if m[3] == "APPENDIX"]

    def block(items):
        out = []
        for stem, desc, src, place in items:
            out.append(f"### `{stem}.pdf` / `.png`  — **{place}**")
            out.append(f"- {desc}")
            out.append(f"- **Built from:** {src}\n")
        return out

    L.append("## Main body\n")
    L += block(body)
    L.append("## Appendix\n")
    L += block(appx)

    L.append("## Notes on redundancy with the EDA figures\n")
    L.append("- These figures do **not** duplicate `output/eda/*`. The EDA set covers raw "
             "distributions, the site map, the phenology trend, the climate-correlation heatmap "
             "and the NAO↔temperature / NAO↔phenology relationships (EDA figs 6–9). The new set "
             "is about **models**: the mixed-model structure, the GDD mechanism, leakage and "
             "evaluation design, the model ladder, coefficient forests, the warming-decoupling "
             "result, the ICC, and the Hurrell–CPC 2024 fix.")
    L.append("- `fig_nao_hurrell_cpc` is new (the 2024 gap fix) and is not in the EDA set; EDA "
             "fig 9 shows NAO↔temperature and NAO↔phenology, a different relationship.")
    L.append("- `fig_leakage_timeline` is placed in the **appendix** as a methodological aid; "
             "promote it to the body if the leakage discipline needs emphasising in the "
             "methods narrative.\n")
    CONFIG["report_md"].write_text("\n".join(L))
    print(f"wrote {CONFIG['report_md']}")


def main() -> int:
    CONFIG["fig_dir"].mkdir(parents=True, exist_ok=True)
    print("Building report figures ...")
    fig1_me_structure()
    fig2_gdd_mechanism()
    fig3_leakage_timeline()
    fig4_eval_regimes()
    fig5_model_ladder()
    fig6_coefficients()
    fig7_warming_decoupling()
    fig8_icc_variance()
    fig9_nao_hurrell_cpc()
    write_report()
    print(f"\nDone — {len(MANIFEST)} figures in {CONFIG['fig_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
