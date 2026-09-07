# Report figures — catalogue

_Generated 2026-08-28. Script: `scripts/modelling/report_figures.py`. All files in `output/figures/` as PNG (300 dpi) + PDF (vector). The LaTeX build imports from a `figures/` folder, so copy/symlink `output/figures/*` there._

Every data-driven figure is computed from the listed artefact — no number is transcribed from a report. Mixed-effects by-class slopes and MixedLM variance components are recomputed via `scripts.modelling.mixed_effects` on `output/modelling_frame.parquet`, so they match `mixedeffects_report.md` exactly.

## Main body

### `fig_me_structure.pdf` / `.png`  — **BODY**
- Mixed-model schematic: annotated primary equation + random site-intercept illustration.
- **Built from:** Recomputed MixedLM variance components (site SD, resid SD, ICC) + univoltine NAO slope via scripts.modelling.mixed_effects on output/modelling_frame.parquet.

### `fig_gdd_mechanism.pdf` / `.png`  — **BODY**
- GDD mechanism: cumulative GDD vs day-of-year for the warmest and coldest national springs, with a real species critical sum S* and the emergence crossings.
- **Built from:** data/site_daily_climate (national-mean daily Tmean, 2024 vs 2013); S* for Maniola jurtina from output/gdd_species_calibration.csv.

### `fig_model_ladder_mae.pdf` / `.png`  — **BODY**
- Dumbbell of overall MAE per model in both regimes (temporal ◆ vs grouped-CV ○), ordered by skill; ME different-n caveat annotated.
- **Built from:** output/model_metrics.csv (scope=overall; MAE, n per model×regime).

### `fig_me_coefficients.pdf` / `.png`  — **BODY**
- Forest plots: (a) NAO slope by voltinism, (b) spring-temp slope by voltinism, (c) univoltine NAO sign-flip (marginal −0.41 → +0.46 with temperature).
- **Built from:** By-class slopes recomputed via mixed_effects.slope_by_class; panel (c) NAO_c rows from output/models/mixedeffects_coefficients.csv (specs primary_nao, both).

### `fig_warming_decoupling.pdf` / `.png`  — **BODY**
- Per-year mean bias (pred − obs) on the temporal test, 2015–2024, for GDD, ML-HGB and species-mean: GDD drifts strongly negative in warm years; ML stays near zero.
- **Built from:** output/model_predictions.parquet (regime=temporal; recomputed mean(y_pred−y_true) by model×YEAR).

## Appendix

### `fig_leakage_timeline.pdf` / `.png`  — **APPENDIX**
- Timeline of the Dec(y−1)→31 May(y) leakage-safe predictor window vs the flight period, with feature-family labels and the no-post-flight rule.
- **Built from:** Schematic (matplotlib); feature naming from data/feature_table.parquet conventions (_M12P/_M01..M05, _DJF/_MAM, GDD).

### `fig_eval_regimes.pdf` / `.png`  — **APPENDIX**
- The two evaluation regimes: temporal hold-out (train≤2014/test>2014) and site-grouped 5-fold CV, with real split and fold sizes.
- **Built from:** output/modelling_frame.parquet (temporal_split, cv_fold, SITE_ID counts).

### `fig_icc_variance.pdf` / `.png`  — **APPENDIX**
- MixedLM variance decomposition: site-intercept SD vs residual SD, with ICC.
- **Built from:** Recomputed MixedLM (site_var, scale) via mixed_effects.fit_mixedlm on the ME frame.

### `fig_nao_hurrell_cpc.pdf` / `.png`  — **APPENDIX**
- Hurrell vs CPC DJFM NAO over the overlap years with the fitted rescaling line and R², and the rescaled 2024 value highlighted.
- **Built from:** data/nao_djfm.csv (Hurrell) + data/nao_djfm_cpc.csv (CPC); regression recomputed here.

## Notes on redundancy with the EDA figures

- These figures do **not** duplicate `output/eda/*`. The EDA set covers raw distributions, the site map, the phenology trend, the climate-correlation heatmap and the NAO↔temperature / NAO↔phenology relationships (EDA figs 6–9). The new set is about **models**: the mixed-model structure, the GDD mechanism, leakage and evaluation design, the model ladder, coefficient forests, the warming-decoupling result, the ICC, and the Hurrell–CPC 2024 fix.
- `fig_nao_hurrell_cpc` is new (the 2024 gap fix) and is not in the EDA set; EDA fig 9 shows NAO↔temperature and NAO↔phenology, a different relationship.
- `fig_leakage_timeline` is placed in the **appendix** as a methodological aid; promote it to the body if the leakage discipline needs emphasising in the methods narrative.
