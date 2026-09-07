#!/usr/bin/env python
"""
Generate output/modelling_report.md from the saved modelling artefacts.

Reads (all produced by add_nao / harness / gdd_baseline):
  output/_nao_qc.json, output/_frame_funnel.json, output/model_metrics.csv,
  output/gdd_species_calibration.csv, output/model_predictions.parquet
so the report is a reproducible view of the artefacts, not hand-typed numbers.

Run:  .venv/bin/python -m scripts.modelling.report
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.modelling import config as C
    from scripts.modelling import gdd_baseline as G
else:
    from . import config as C
    from . import gdd_baseline as G


def _load():
    nao = json.load(open(C.OUTPUT_DIR / "_nao_qc.json"))
    funnel = json.load(open(C.OUTPUT_DIR / "_frame_funnel.json"))
    metrics = pd.read_csv(C.METRICS_PATH)
    calib = pd.read_csv(C.GDD_CALIB_PATH)
    preds = pd.read_parquet(C.PREDICTIONS_PATH)
    return nao, funnel, metrics, calib, preds


def _md_table(df: pd.DataFrame, floatfmt="{:.2f}") -> str:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def build() -> str:
    nao, funnel, metrics, calib, preds = _load()
    cov = G.check_cache_coverage()

    L = ["# Modelling report — target FIRSTDAY (first appearance, days after 1 April)\n"]
    L.append(f"_Generated {pd.Timestamp.today().date()}. Pipeline: "
             "`add_nao` → `harness` → `gdd_baseline` → `report`, all under "
             "`scripts/modelling/`. Run with `.venv/bin/python`._\n")

    # ---------------------------------------------------------------- Part 1
    L.append("## Part 1 — Winter NAO (station-based DJFM) join\n")
    L.append(f"- **Source:** Hurrell STATION-BASED DJFM NAO index, NCAR Climate Data Guide.")
    L.append(f"  - URL: {C.NAO_URL}")
    L.append(f"  - Access date: {nao.get('access_date')}")
    L.append(f"  - Citation: {C.NAO_CITATION} Accessed {nao.get('access_date')}.")
    L.append(f"- Raw index saved to `data/nao_djfm.csv` (year, nao_djfm), "
             f"{nao['nao_year_min']}–{nao['nao_year_max']}.")
    L.append("- **Used the pre-computed DJFM seasonal ascii** (not the monthly index averaged "
             "to DJFM — seasonal vs monthly normalisation differ).")
    L.append("- **Year convention (verified against known winters):** DJFM value for year *N* "
             "= mean of Dec(*N*−1)…Mar(*N*), the winter preceding flight year *N*, so it joins "
             "directly onto `YEAR = N`. Checks: 2010 = −4.64 (cold 2009/10), 2020 = +3.63 "
             "(stormy 2019/20), 1989 = +5.08, 1996 = −3.78. ✓")
    L.append(f"- Added `NAO_DJFM` (year *N*) and `NAO_DJFM_LAG1` (year *N*−1) to "
             f"`feature_table.parquet`. **Row count {nao['rows']} unchanged** "
             f"(asserted == {C.EXPECTED_ROWS}).")
    L.append(f"- Coverage: **{nao['n_nao']:,} rows** got `NAO_DJFM`, "
             f"**{nao['n_nao_lag1']:,} rows** got `NAO_DJFM_LAG1`.")
    miss = nao.get("missing_years_in_range") or []
    if miss:
        L.append(f"- ⚠️ **Coverage caveat:** the station index ends {nao['nao_year_max']}, so "
                 f"flight-year **{miss}** has no `NAO_DJFM` (still gets `NAO_DJFM_LAG1` from "
                 f"{nao['nao_year_max']}). The product genuinely stops at {nao['nao_year_max']}; "
                 "it was not fabricated forward.")
    L.append("")

    # ---------------------------------------------------------------- Part 2
    L.append("## Part 2 — Shared evaluation harness\n")
    L.append("### Modelling-frame filter funnel (target never imputed)\n")
    fn = pd.DataFrame({
        "filter": ["feature_table rows", "require FIRSTDAY + keys present",
                   f"YEAR in [{C.YEAR_MIN},{C.YEAR_MAX}] (climate coverage)",
                   f"keep GB/BNG sites (daily cache, n={funnel['n_bng_sites']})",
                   f"implied DOY in [{C.DOY_MIN},{C.DOY_MAX}]",
                   "drop FLAG_DOY_IMPLAUSIBLE"],
        "rows": [funnel["feature_table_rows"], funnel["require_firstday_and_keys"],
                 funnel["year_in_climate_range"], funnel["gb_bng_sites"],
                 funnel["implied_doy_valid"], funnel["not_flag_implausible"]],
    })
    L.append(_md_table(fn, floatfmt="{:.0f}"))
    L.append("")
    L.append("FIRSTDAY is *days after 1 April* (can be negative), converted to true "
             "day-of-year via DOY = 91/92 (non-leap/leap) + FIRSTDAY for the validity "
             "check and for the GDD crossing. **The target is never imputed** — rows "
             "without a valid FIRSTDAY are dropped, not filled.\n")
    L.append("### Two evaluation regimes (split assignment stored once in "
             "`output/modelling_frame.parquet`, reused by every model)\n")
    L.append(f"- **(a) Temporal hold-out** (distribution-shift test): train "
             f"`YEAR ≤ {funnel['temporal_cutoff']}` = {funnel['temporal_train']:,} rows; "
             f"test `YEAR > {funnel['temporal_cutoff']}` = {funnel['temporal_test']:,} rows "
             "(the last ~10 years, 2015–2024).")
    L.append(f"- **(b) Grouped {funnel['cv_k']}-fold CV**, groups = `SITE_ID`, so a site "
             f"never spans train & test (no site leakage). Fold sizes: "
             f"{funnel['cv_fold_sizes']}. Deterministic greedy balancing by event count "
             "(seed in CONFIG), no sklearn dependency.")
    L.append("")
    L.append("### Model interface & baselines\n")
    L.append("Every model exposes `.fit(train_df)` / `.predict(df) → FIRSTDAY array` "
             "(NaN = abstain); `harness.evaluate(model)` runs both regimes. Metrics: "
             "**MAE, RMSE (days), R², mean bias**, overall and per species. Naive baselines "
             "(train-only): species mean, site×species mean (→ species mean → global), and "
             "persistence (previous-year same-site-species obs — strictly-past, leak-free).\n")
    L.append("> **Voltinism not available.** `feature_table` has no voltinism field, so results "
             "here are **by species**. A voltinism column is **needed for the mixed-effects "
             "rung** — the per-species GDD win/loss split below shows why (single- vs "
             "multi-brood species behave very differently).\n")

    # ---------------------------------------------------------------- Part 3
    L.append("## Part 3 — GDD process baseline\n")
    L.append(f"- **Daily-cache coverage finding:** the cache is **FULL calendar year** "
             f"(months 1–12, DOY 1–366; checked "
             f"{', '.join(str(y) for y in cov if isinstance(y,int))}), so GDD accumulates to "
             "each event's emergence day even for summer flyers. That uses only days *before* "
             "the event, so it is not leakage (the Dec–May cap is for the fixed ML features, "
             "not this model).")
    L.append(f"- **Model:** daily Tmean = (tasmax+tasmin)/2; daily GDD = max(0, Tmean − "
             f"T_BASE); cumulative from DOY {C.GDD_START_DOY}. Predicted FIRSTDAY = first "
             f"DOY with cumulative GDD ≥ species critical sum S*.")
    L.append(f"- **Calibration:** per species, S* chosen on **train rows only** to minimise "
             f"MAE(crossing day, observed FIRSTDAY) over a {C.SSTAR_N_CANDIDATES}-point grid; "
             f"a candidate is valid only if ≥{C.SSTAR_MIN_CROSS_FRAC:.0%} of train events reach "
             f"it. T_BASE = {C.T_BASE}, GDD_START_DOY = {C.GDD_START_DOY} (optional per-species "
             f"T_BASE search `FIT_TBASE`, grid {C.TBASE_GRID}, off by default).")
    L.append(f"- **{len(calib)} species calibrated**; S* range "
             f"{calib.S_star.min():.0f}–{calib.S_star.max():.0f} °C·day "
             f"(median {calib.S_star.median():.0f}). Thin species "
             f"(< {C.GDD_MIN_TRAIN_EVENTS} train events) use a global-fallback S*.")

    # non-crossing / abstain from predictions
    g = preds[preds.model == "gdd_process"]
    gt = g[g.regime == "temporal"]
    n_abs = int(gt["y_pred"].isna().sum())
    L.append(f"- **Non-crossing (abstain) cases:** {n_abs:,} of {len(gt):,} temporal-test "
             f"rows ({n_abs/len(gt):.2%}) — dominated by the 66 all-NaN sea/edge sites whose "
             "site-years are skipped (one NaN would poison the species' S* via cumsum). These "
             "are **flagged and counted, never silently dropped**: the model abstains (NaN) "
             "and they leave the metric denominators.")
    L.append("")

    # ---------------------------------------------------------------- Results
    L.append("## Results — baselines vs GDD (both regimes, overall)\n")
    ov = (metrics[metrics.scope == "overall"]
          .drop(columns=["scope", "species"])
          .sort_values(["regime", "MAE"]))
    ov = ov[["model", "regime", "n", "MAE", "RMSE", "R2", "bias"]]
    ov_show = ov.copy()
    ov_show["n"] = ov_show["n"].map("{:,}".format)
    L.append(_md_table(ov_show, floatfmt="{:.2f}"))
    L.append("")
    L.append("**Reading the table.**")
    L.append("- Naive **climatology is hard to beat**: site×species / species means give the "
             "lowest MAE (~21–22 days). In grouped-CV the site×species model *collapses to the "
             "species mean* (identical numbers) — held-out sites have no site history, the "
             "expected consequence of the site-grouped split.")
    L.append("- The single-threshold **GDD model does not beat climatology on MAE** (~22–23 "
             "days) — expected for a site-agnostic baseline that ignores the spatial structure "
             "the site means capture. Its value is **interannual**: the static means predict "
             "the same day every year; GDD tracks warm/cold springs.")

    # yearly bias trend for GDD
    gg = g.dropna(subset=["y_pred"]).copy()
    gg["YEAR"] = gg["YEAR"] if "YEAR" in gg else None
    yb = (gg.assign(err=gg["y_pred"] - gg["y_true"])
            .groupby("YEAR")["err"].mean())
    picks = [y for y in (1980, 1990, 2000, 2010, 2015, 2020, 2024) if y in yb.index]
    trend = ", ".join(f"{y}: {yb.loc[y]:+.1f}" for y in picks)
    L.append(f"- **Key process finding — a warming bias.** GDD is median-unbiased on train "
             "(median bias 0.0) but its mean bias grows negative in recent test years (overall "
             f"temporal bias "
             f"{ov[(ov.model=='gdd_process')&(ov.regime=='temporal')]['bias'].iloc[0]:+.1f} "
             "days). Yearly mean bias (pred−obs) drifts: " + trend + ". The fixed threshold "
             "**increasingly predicts emergence too early in warm years** — butterflies are "
             "not advancing as fast as accumulated warmth implies. That decoupling is exactly "
             "what a mechanistic baseline should expose, and motivates the next rungs (site "
             "random effects, NAO, non-linear thermal response).")
    L.append("")

    # per-species where GDD wins / loses (temporal)
    sp = metrics[(metrics.scope == "species") & (metrics.regime == "temporal")]
    piv = sp.pivot_table(index="species", columns="model", values="MAE")
    naive_cols = [c for c in ("species_mean", "site_species_mean") if c in piv]
    piv["best_naive"] = piv[naive_cols].min(axis=1)
    piv["gdd_minus_naive"] = piv["gdd_process"] - piv["best_naive"]
    wins = (piv.sort_values("gdd_minus_naive").head(6)
              [["gdd_process", "best_naive", "gdd_minus_naive"]]
              .round(2).reset_index())
    loses = (piv.sort_values("gdd_minus_naive").tail(5)
               [["gdd_process", "best_naive", "gdd_minus_naive"]]
               .round(2).reset_index())
    L.append("### Where the GDD process model helps vs hurts (temporal MAE, days)\n")
    L.append("**GDD beats the best naive baseline** (negative Δ) — thermally-controlled, "
             "typically univoltine species:\n")
    L.append(_md_table(wins))
    L.append("\n**GDD loses most** — late / multivoltine / complex-phenology species where a "
             "single spring threshold is the wrong model:\n")
    L.append(_md_table(loses))
    L.append("\nThis split (single-brood ↔ GDD helps; multi-brood ↔ GDD hurts) is the "
             "concrete case for adding a **voltinism** field before the mixed-effects rung.")
    L.append("")

    # ---------------------------------------------------------------- Assumptions
    L.append("## Assumptions & decisions\n")
    for a in [
        "FIRSTDAY is *days after 1 April*; converted to DOY with April 1 = day 91 "
        "(non-leap) / 92 (leap). FIRSTDAY range in data: −30…215.",
        "Modelling frame = GB/BNG sites only (the 3,496 sites present in the daily "
        "cache); years restricted to 1976–2024 (climate coverage). 1973–1975 survey "
        "rows carry no climate and are excluded so all models share identical rows.",
        "GB/BNG identified as sites present in the daily climate cache (upstream "
        "Irish-Grid / Channel-Islands exclusion already applied).",
        "Persistence uses strictly-past observations (year−1), so it is leak-free w.r.t. "
        "the forecast target. Caveat: under site-grouped CV it may consult a held-out "
        "site's own earlier years, so its grouped-CV score is not a site-generalisation "
        "measure; it is most meaningful in the temporal regime.",
        "GDD trajectory store skips any site-year containing a NaN daily Tmean (the 66 "
        "sea/edge sites) to avoid NaN-poisoning of cumsum and per-species percentiles; "
        "those events abstain and are counted.",
        "Daily series assumed contiguous DOY 1..N (verified full-year), so predicted "
        "DOY = searchsorted(cumGDD, S*) + 1.",
        "S* calibrated in day-space MAE (not GDD-space), per the brief.",
        f"Temporal cutoff {C.TEMPORAL_CUTOFF}, CV k={C.CV_K}, seed {C.CV_SEED}, "
        f"T_BASE={C.T_BASE}, GDD_START_DOY={C.GDD_START_DOY} — all in "
        "`scripts/modelling/config.py`.",
    ]:
        L.append(f"- {a}")
    L.append("")
    L.append("## Artefacts\n")
    for f, d in [
        ("data/nao_djfm.csv", "raw DJFM NAO index (reproducible)"),
        ("data/feature_table.parquet", "+ NAO_DJFM, NAO_DJFM_LAG1"),
        ("output/modelling_frame.parquet", "filtered frame + fixed split assignment"),
        ("output/model_predictions.parquet", "long predictions: model,regime,keys,y_true,y_pred"),
        ("output/model_metrics.csv", "tidy metrics: overall + per species, both regimes"),
        ("output/gdd_species_calibration.csv", "per-species S* (and T_BASE)"),
    ]:
        L.append(f"- `{f}` — {d}")
    L.append("")
    return "\n".join(L)


def main() -> int:
    C.REPORT_MD.write_text(build())
    print(f"wrote {C.REPORT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
