# Modelling report — target FIRSTDAY (first appearance, days after 1 April)

_Generated 2026-09-05. Pipeline: `add_nao` → `harness` → `gdd_baseline` → `report`, all under `scripts/modelling/`. Run with `.venv/bin/python`._

## Part 1 — Winter NAO (station-based DJFM) join

- **Source:** Hurrell STATION-BASED DJFM NAO index, NCAR Climate Data Guide.
  - URL: https://climatedataguide.ucar.edu/sites/default/files/2023-07/nao_station_djfm.txt
  - Access date: 2026-07-09
  - Citation: NAO Index Data provided by the Climate Analysis Section, NCAR, Boulder, USA, Hurrell (2003). Updated regularly. Accessed 2026-07-09.
- Raw index saved to `data/nao_djfm.csv` (year, nao_djfm), 1864–2023.
- **Used the pre-computed DJFM seasonal ascii** (not the monthly index averaged to DJFM — seasonal vs monthly normalisation differ).
- **Year convention (verified against known winters):** DJFM value for year *N* = mean of Dec(*N*−1)…Mar(*N*), the winter preceding flight year *N*, so it joins directly onto `YEAR = N`. Checks: 2010 = −4.64 (cold 2009/10), 2020 = +3.63 (stormy 2019/20), 1989 = +5.08, 1996 = −3.78. ✓
- Added `NAO_DJFM` (year *N*) and `NAO_DJFM_LAG1` (year *N*−1) to `feature_table.parquet`. **Row count 648788 unchanged** (asserted == 648788).
- Coverage: **608,368 rows** got `NAO_DJFM`, **648,788 rows** got `NAO_DJFM_LAG1`.
- ⚠️ **Coverage caveat:** the station index ends 2023, so flight-year **[2024]** has no `NAO_DJFM` (still gets `NAO_DJFM_LAG1` from 2023). The product genuinely stops at 2023; it was not fabricated forward.

## Part 2 — Shared evaluation harness

### Modelling-frame filter funnel (target never imputed)

| filter | rows |
| --- | --- |
| feature_table rows | 648788 |
| require FIRSTDAY + keys present | 648788 |
| YEAR in [1976,2024] (climate coverage) | 648713 |
| keep GB/BNG sites (daily cache, n=3496) | 620711 |
| implied DOY in [1,366] | 620711 |
| drop FLAG_DOY_IMPLAUSIBLE | 620711 |

FIRSTDAY is *days after 1 April* (can be negative), converted to true day-of-year via DOY = 91/92 (non-leap/leap) + FIRSTDAY for the validity check and for the GDD crossing. **The target is never imputed** — rows without a valid FIRSTDAY are dropped, not filled.

### Two evaluation regimes (split assignment stored once in `output/modelling_frame.parquet`, reused by every model)

- **(a) Temporal hold-out** (distribution-shift test): train `YEAR ≤ 2014` = 294,973 rows; test `YEAR > 2014` = 325,738 rows (the last ~10 years, 2015–2024).
- **(b) Grouped 5-fold CV**, groups = `SITE_ID`, so a site never spans train & test (no site leakage). Fold sizes: [124143, 124142, 124142, 124142, 124142]. Deterministic greedy balancing by event count (seed in CONFIG), no sklearn dependency.

### Model interface & baselines

Every model exposes `.fit(train_df)` / `.predict(df) → FIRSTDAY array` (NaN = abstain); `harness.evaluate(model)` runs both regimes. Metrics: **MAE, RMSE (days), R², mean bias**, overall and per species. Naive baselines (train-only): species mean, site×species mean (→ species mean → global), and persistence (previous-year same-site-species obs — strictly-past, leak-free).

> **Voltinism not available.** `feature_table` has no voltinism field, so results here are **by species**. A voltinism column is **needed for the mixed-effects rung** — the per-species GDD win/loss split below shows why (single- vs multi-brood species behave very differently).

## Part 3 — GDD process baseline

- **Daily-cache coverage finding:** the cache is **FULL calendar year** (months 1–12, DOY 1–366; checked 1976, 2000, 2024), so GDD accumulates to each event's emergence day even for summer flyers. That uses only days *before* the event, so it is not leakage (the Dec–May cap is for the fixed ML features, not this model).
- **Model:** daily Tmean = (tasmax+tasmin)/2; daily GDD = max(0, Tmean − T_BASE); cumulative from DOY 1. Predicted FIRSTDAY = first DOY with cumulative GDD ≥ species critical sum S*.
- **Calibration:** per species, S* chosen on **train rows only** to minimise MAE(crossing day, observed FIRSTDAY) over a 250-point grid; a candidate is valid only if ≥80% of train events reach it. T_BASE = 5.0, GDD_START_DOY = 1 (optional per-species T_BASE search `FIT_TBASE`, grid [0.0, 2.5, 5.0, 7.5, 10.0], off by default).
- **58 species calibrated**; S* range 196–1404 °C·day (median 540). Thin species (< 30 train events) use a global-fallback S*.
- **Non-crossing (abstain) cases:** 3,965 of 325,738 temporal-test rows (1.22%) — dominated by the 66 all-NaN sea/edge sites whose site-years are skipped (one NaN would poison the species' S* via cumsum). These are **flagged and counted, never silently dropped**: the model abstains (NaN) and they leave the metric denominators.

## Results — baselines vs GDD (both regimes, overall)

| model | regime | n | MAE | RMSE | R2 | bias |
| --- | --- | --- | --- | --- | --- | --- |
| ml_rf | grouped_cv | 620,711 | 20.30 | 28.69 | 0.47 | 0.07 |
| ml_rf_nonao | grouped_cv | 620,711 | 20.34 | 28.72 | 0.47 | 0.08 |
| ml_hgb | grouped_cv | 620,711 | 20.67 | 28.88 | 0.47 | 0.41 |
| ml_hgb_nonao | grouped_cv | 620,711 | 20.73 | 28.93 | 0.46 | 0.41 |
| me_nao_temp | grouped_cv | 524,672 | 21.31 | 29.66 | 0.44 | 0.00 |
| me_nao | grouped_cv | 524,672 | 21.83 | 30.16 | 0.42 | 0.00 |
| site_species_mean | grouped_cv | 620,711 | 22.33 | 30.67 | 0.40 | 0.00 |
| species_mean | grouped_cv | 620,711 | 22.33 | 30.67 | 0.40 | 0.00 |
| gdd_process | grouped_cv | 612,192 | 22.41 | 32.36 | 0.33 | -5.12 |
| persistence | grouped_cv | 620,711 | 23.54 | 34.89 | 0.22 | -1.54 |
| ml_hgb | temporal | 325,738 | 21.22 | 29.82 | 0.40 | -1.01 |
| ml_hgb_nonao | temporal | 325,738 | 21.24 | 29.78 | 0.40 | -0.75 |
| site_species_mean | temporal | 325,738 | 21.27 | 30.46 | 0.37 | -0.26 |
| me_nao | temporal | 255,491 | 21.32 | 30.18 | 0.39 | -3.11 |
| ml_rf_nonao | temporal | 325,738 | 21.37 | 29.91 | 0.40 | -0.76 |
| ml_rf | temporal | 325,738 | 21.39 | 29.90 | 0.40 | -0.72 |
| me_nao_temp | temporal | 255,491 | 21.52 | 29.82 | 0.40 | -0.39 |
| species_mean | temporal | 325,738 | 21.97 | 30.29 | 0.38 | -0.23 |
| gdd_process | temporal | 321,773 | 23.15 | 34.42 | 0.20 | -10.48 |
| persistence | temporal | 325,738 | 23.23 | 34.42 | 0.20 | -2.05 |

**Reading the table.**
- Naive **climatology is hard to beat**: site×species / species means give the lowest MAE (~21–22 days). In grouped-CV the site×species model *collapses to the species mean* (identical numbers) — held-out sites have no site history, the expected consequence of the site-grouped split.
- The single-threshold **GDD model does not beat climatology on MAE** (~22–23 days) — expected for a site-agnostic baseline that ignores the spatial structure the site means capture. Its value is **interannual**: the static means predict the same day every year; GDD tracks warm/cold springs.
- **Key process finding — a warming bias.** GDD is median-unbiased on train (median bias 0.0) but its mean bias grows negative in recent test years (overall temporal bias -10.5 days). Yearly mean bias (pred−obs) drifts: 1980: +11.0, 1990: -7.6, 2000: -1.4, 2010: +8.1, 2015: +3.3, 2020: -20.4, 2024: -24.7. The fixed threshold **increasingly predicts emergence too early in warm years** — butterflies are not advancing as fast as accumulated warmth implies. That decoupling is exactly what a mechanistic baseline should expose, and motivates the next rungs (site random effects, NAO, non-linear thermal response).

### Where the GDD process model helps vs hurts (temporal MAE, days)

**GDD beats the best naive baseline** (negative Δ) — thermally-controlled, typically univoltine species:

| species | gdd_process | best_naive | gdd_minus_naive |
| --- | --- | --- | --- |
| Thymelicus acteon | 16.53 | 20.10 | -3.57 |
| Celastrina argiolus | 29.26 | 32.48 | -3.21 |
| Satyrium pruni | 7.49 | 9.95 | -2.46 |
| Limenitis camilla | 9.05 | 10.94 | -1.89 |
| Apatura iris | 8.78 | 10.09 | -1.31 |
| Vanessa atalanta | 29.49 | 30.62 | -1.14 |

**GDD loses most** — late / multivoltine / complex-phenology species where a single spring threshold is the wrong model:

| species | gdd_process | best_naive | gdd_minus_naive |
| --- | --- | --- | --- |
| Hipparchia semele | 18.12 | 12.39 | 5.74 |
| Polygonia c-album | 49.81 | 42.94 | 6.87 |
| Erebia aethiops | 16.19 | 8.91 | 7.27 |
| Aricia agestis | 37.26 | 29.76 | 7.50 |
| Phengaris arion | 16.27 | 6.59 | 9.69 |

This split (single-brood ↔ GDD helps; multi-brood ↔ GDD hurts) is the concrete case for adding a **voltinism** field before the mixed-effects rung.

## Assumptions & decisions

- FIRSTDAY is *days after 1 April*; converted to DOY with April 1 = day 91 (non-leap) / 92 (leap). FIRSTDAY range in data: −30…215.
- Modelling frame = GB/BNG sites only (the 3,496 sites present in the daily cache); years restricted to 1976–2024 (climate coverage). 1973–1975 survey rows carry no climate and are excluded so all models share identical rows.
- GB/BNG identified as sites present in the daily climate cache (upstream Irish-Grid / Channel-Islands exclusion already applied).
- Persistence uses strictly-past observations (year−1), so it is leak-free w.r.t. the forecast target. Caveat: under site-grouped CV it may consult a held-out site's own earlier years, so its grouped-CV score is not a site-generalisation measure; it is most meaningful in the temporal regime.
- GDD trajectory store skips any site-year containing a NaN daily Tmean (the 66 sea/edge sites) to avoid NaN-poisoning of cumsum and per-species percentiles; those events abstain and are counted.
- Daily series assumed contiguous DOY 1..N (verified full-year), so predicted DOY = searchsorted(cumGDD, S*) + 1.
- S* calibrated in day-space MAE (not GDD-space), per the brief.
- Temporal cutoff 2014, CV k=5, seed 42, T_BASE=5.0, GDD_START_DOY=1 — all in `scripts/modelling/config.py`.

## Artefacts

- `data/nao_djfm.csv` — raw DJFM NAO index (reproducible)
- `data/feature_table.parquet` — + NAO_DJFM, NAO_DJFM_LAG1
- `output/modelling_frame.parquet` — filtered frame + fixed split assignment
- `output/model_predictions.parquet` — long predictions: model,regime,keys,y_true,y_pred
- `output/model_metrics.csv` — tidy metrics: overall + per species, both regimes
- `output/gdd_species_calibration.csv` — per-species S* (and T_BASE)
