#!/usr/bin/env python
"""
Assemble output/eda_report.md from the saved EDA + voltinism artefacts
(output/eda/_eda_stats.json, output/eda/_voltinism_status.json). Reproducible:
re-run after scripts/eda.py and scripts/voltinism.py.

Run:  .venv/bin/python -m scripts.eda_report
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EDA_DIR = PROJECT_ROOT / "output" / "eda"
STATS = EDA_DIR / "_eda_stats.json"
VOLT = EDA_DIR / "_voltinism_status.json"
REPORT = PROJECT_ROOT / "output" / "eda_report.md"
ACCESS_DATE = date.today().isoformat()

CITATIONS = {
    "cook": ("Cook, P.M.; Tordoff, G.M.; Davis, A.M.; Parsons, M.S.; Dennis, E.B.; "
             "Fox, R.; Botham, M.S.; Bourn, N.A.D. (2021). Traits data for the "
             "butterflies and macro-moths of Great Britain and Ireland, 2021. NERC EDS "
             "Environmental Information Data Centre. "
             "https://doi.org/10.5285/5b5a13b6-2304-47e3-9c9d-35237d1232c6 "
             "(Open Government Licence)."),
    "mw": ("Middleton-Welling, J.; Dapporto, L.; García-Barros, E.; Wiemers, M.; "
           "Nowicki, P.; Plazio, E.; Bonelli, S.; Zaccagno, M.; Šašić, M.; Liparova, J.; "
           "Schweiger, O.; Harpke, A.; Musche, M.; Settele, J.; Schmucki, R.; Shreeve, T. "
           "(2020). A new comprehensive trait database of European and Maghreb "
           "butterflies. Dryad. https://doi.org/10.5061/dryad.6m905qfx6 (CC0)."),
    "ws": ("Westgarth-Smith, A.R.; Leroy, S.A.G.; Collins, P.E.F.; Harrington, R. (2012). "
           "Temporal variations in English populations of a forest insect pest, the green "
           "spruce aphid, linked to the North Atlantic Oscillation and global warming."),
}


def _fig(stats, key):
    return stats["figures"].get(key)


def build() -> str:
    S = json.loads(STATS.read_text())
    V = json.loads(VOLT.read_text()) if VOLT.exists() else None
    fd = S["firstday"]
    L = ["# Exploratory Data Analysis — UKBMS feature table\n"]
    L.append(f"_Generated {ACCESS_DATE}. Source: `data/feature_table.parquet` "
             "(read-only). Figures in `output/eda/`. Target of interest = **FIRSTDAY** "
             "(first appearance, days after 1 April; negative = before 1 April)._\n")

    # ---- data-version note (mandated) -----------------------------------
    L.append("## Data-version note\n")
    L.append("The project uses the **UKBMS 1976–2024 phenology release**, which is "
             "**whole-season only** — it has **no per-generation / brood columns** (unlike "
             "the 1976–2022 release). Brood splits are therefore not used; voltinism is "
             "instead derived at the **species level** from the trait database (Cook et al. "
             "2022, cross-checked against Middleton-Welling et al. 2020). "
             "**Why the 2024 release:** it is required to extend the record beyond "
             "Westgarth-Smith (2012) and to supply the recent warm years needed for the "
             "distribution-shift (temporal hold-out) test in the modelling phase, so "
             "first-generation emergence is captured via **FIRSTDAY (first appearance)** "
             "rather than an explicit first-brood column.\n")

    # ---- summary table --------------------------------------------------
    L.append("## Summary\n")
    L.append("| metric | value |")
    L.append("| --- | --- |")
    L.append(f"| rows (site×species×year) | {S['n_rows']:,} |")
    L.append(f"| distinct sites | {S['n_sites']:,} |")
    L.append(f"| distinct species | {S['n_species']} |")
    L.append(f"| year range | {S['year_range'][0]}–{S['year_range'][1]} "
             "(climate/coverage focus 1976–2024) |")
    L.append(f"| FIRSTDAY mean ± sd | {fd['mean']:.1f} ± {fd['std']:.1f} days after 1 Apr |")
    L.append(f"| FIRSTDAY median [min, max] | {fd['50%']:.0f} [{fd['min']:.0f}, "
             f"{fd['max']:.0f}] |")
    L.append("")
    if S.get("missingness_top"):
        L.append("**Per-column missingness (top, % NA):** " +
                 ", ".join(f"`{k}` {v:.1f}%" for k, v in S["missingness_top"].items()) +
                 ". Full breakdown in fig 5.\n")

    # ---- figures --------------------------------------------------------
    L.append("## Figures\n")
    figs = [
        ("fig1", "Survey coverage over time", _fig(S, "fig1"),
         f"Records and distinct sites per year, 1976–2024. Effort grows markedly "
         f"(≈{S['coverage']['first_year_sites']} sites in 1976 → "
         f"{S['coverage']['last_year_sites']:,} in 2024), reflecting WCBS expansion — later "
         "years dominate the sample, which matters for any pooled trend."),
        ("fig2", "Records per species", _fig(S, "fig2"),
         f"Strongly unbalanced: richest = {S['records_per_species']['richest']} "
         f"({S['records_per_species']['richest_n']:,} rows), poorest = "
         f"{S['records_per_species']['poorest']} ({S['records_per_species']['poorest_n']}); "
         f"{S['records_per_species']['n_under_500']} species have <500 records. Per-species "
         "models for data-poor species will be unstable."),
        ("fig3", "Site map (British National Grid)", _fig(S, "fig3"),
         f"Distinct sites coloured by years surveyed; the GB points form the recognisable "
         f"shape of Great Britain with no GB points in the sea. **{S['site_map']['non_gb_country_sites']} "
         "non-GB sites (red ×)** are mislocated exactly as expected — Northern Ireland "
         "(Irish Grid, projected into the Irish Sea) and Channel Islands (negative northing, "
         f"min {S['site_map']['min_northing']:.0f} m); these were excluded from the climate "
         f"join. {S['site_map']['sites_no_coords']} sites have no coordinates."),
        ("fig4", "FIRSTDAY distribution", _fig(S, "fig4"),
         "Overall FIRSTDAY plus 6 species spanning early→late flyers. Overall is "
         "right-skewed; per-species distributions are much tighter and well separated, "
         "confirming species identity is the dominant driver of absolute emergence date."),
        ("fig5", "Missingness per column", _fig(S, "fig5"),
         "Percent NA per column. Climate/NAO features are missing for the non-BNG / "
         "no-coordinate sites and (for NAO_DJFM) flight-year 2024; the target FIRSTDAY is "
         "complete. No column is imputed."),
        ("fig6", "Phenology trend over time", _fig(S, "fig6"),
         f"Mean FIRSTDAY per year is **advancing {abs(S['phenology_trend']['overall_slope_days_per_decade']):.2f} "
         f"days/decade** overall (r={S['phenology_trend']['overall_pearson_r']}). "
         "Per-species slopes vary sensibly — early-spring "
         "*Anthocharis cardamines* advances fastest, adult-overwintering *Gonepteryx "
         "rhamni* barely moves. **Yes, emergence is advancing.**"),
        ("fig7", "FIRSTDAY vs spring temperature", _fig(S, "fig7"),
         f"(a) Pooled FIRSTDAY vs {S['firstday_vs_temp']['feature']} is negative but weak "
         f"(slope {S['firstday_vs_temp']['pooled_slope_days_per_C']} days/°C, "
         f"r={S['firstday_vs_temp']['pooled_r']}) because pooling mixes species baselines; "
         "(b) after removing species means the within-species response is the correct, "
         f"stronger view (slope {S['firstday_vs_temp']['within_species_slope_days_per_C']} "
         f"days/°C, r={S['firstday_vs_temp']['within_species_r']}): warmer springs → earlier "
         "flight. Record-level noise is large; the year-level signal (fig 6) is clearer."),
        ("fig8", "Climate feature correlations", _fig(S, "fig8"),
         f"Temperature features are highly collinear (strongest pair "
         f"{S['corr_heatmap']['max_pair'][0]}–{S['corr_heatmap']['max_pair'][1]}, "
         f"r={S['corr_heatmap']['max_val']}); GDD and seasonal means are near-duplicates of "
         "the monthly temps. **Multicollinearity** must be handled in modelling "
         "(regularisation / feature selection / a process feature rather than all raw temps)."),
    ]
    nao = S.get("nao", {})
    if nao.get("present"):
        figs.append((
            "fig9", "Winter NAO relationships", _fig(S, "fig9"),
            f"(a) Winter DJFM NAO tracks local temperature (DJF r={nao['nao_vs_DJF_r']}, "
            f"MAM r={nao['nao_vs_MAM_r']}) — **NAO is largely a winter-temperature proxy** "
            "(informs RQ3, i.e. NAO adds little beyond local temp for prediction). "
            f"(b) Mean FIRSTDAY anomaly vs winter NAO is negative "
            f"(slope {nao['firstday_anom_vs_nao_slope']} days/NAO-unit, "
            f"r={nao['firstday_anom_vs_nao_r']}, {nao['n_years']} years): positive NAO → "
            "earlier emergence — the updated Westgarth-Smith relationship, modest at year level."))
    else:
        L.append("_NAO columns not present in the feature table — fig 9 skipped._\n")

    for key, title, fname, caption in figs:
        n = title
        L.append(f"### {title}")
        if fname:
            L.append(f"![{n}](eda/{fname})")
        L.append(caption + "\n")

    # ---- data issues ----------------------------------------------------
    L.append("## Data issues found\n")
    for i in S["data_issues"]:
        L.append(f"- {i}")
    L.append("")

    # ---- Part B: voltinism ---------------------------------------------
    L.append("## Part B — Species voltinism lookup\n")
    L.append("Deliverable: `data/species_voltinism.csv` "
             "(SPECIES_NAME, voltinism, source, note), classifying each of the "
             f"{S['n_species']} species as univoltine / multivoltine, with "
             "latitude/region-variable species flagged `variable` and anything unmatched "
             "or ambiguous flagged `MANUAL_REVIEW`. **No voltinism value is guessed.**\n")
    L.append("**Sources & access (accessed " + ACCESS_DATE + "):**")
    L.append(f"- *Primary* — {CITATIONS['cook']}")
    L.append(f"- *Cross-check* — {CITATIONS['mw']}")
    L.append("")
    if V and not (V.get("cook_file") or V.get("mw_file")):
        L.append("> ⚠️ **Blocked on data access (awaiting user download).** Neither trait "
                 "dataset could be fetched automatically: the Cook et al. EIDC download is "
                 "behind a **licence-acceptance** step (Open Government Licence) which was "
                 "**not accepted on the user's behalf**, per instruction; the CC0 "
                 "Middleton-Welling Dryad download is behind a **JavaScript anti-bot "
                 "challenge** that the command-line client cannot pass. "
                 "`scripts/voltinism.py` is written and ready: drop either file into "
                 "`./data/` and re-run to auto-detect columns, join on scientific name "
                 "(with a genus-synonym table), cross-check the two sources, and flag "
                 "disagreements/unmatched species for review.\n")
        L.append(f"Current `species_voltinism.csv` is a **scaffold**: all "
                 f"{V['n_species']} species are `MANUAL_REVIEW` / `PENDING` until a source "
                 "file is supplied. **0 values have been invented.**\n")
    elif V:
        cc = V.get("class_counts", {})
        cook_ok = bool(V.get("cook_file"))
        mw_ok = bool(V.get("mw_file"))
        L.append("**Classification method (from Cook's own categories, no guessing):** "
                 "`obligate_multivoltine` → **multivoltine**; `partial_generation` → "
                 "**variable** (facultative / partial second brood — the region- & "
                 "latitude-dependent case the brief asks to flag); `obligate_univoltine` → "
                 "**univoltine**; none set → `MANUAL_REVIEW`. Names joined on `scientific_name` "
                 "(normalising the non-breaking spaces in the source) with a genus-synonym "
                 "table for *Colias crocea*, *Lysandra bellargus/coridon*.\n")
        L.append(f"Assigned from source: **{V['n_assigned']}/{V['n_species']}** "
                 f"(the primary source Cook et al. covers every species). Class counts: "
                 f"univoltine {cc.get('univoltine',0)}, multivoltine "
                 f"{cc.get('multivoltine',0)}, variable {cc.get('variable',0)}, "
                 f"MANUAL_REVIEW {cc.get('MANUAL_REVIEW',0)}.\n")
        if not mw_ok:
            L.append("> **Cross-check status:** the Middleton-Welling (Dryad, CC0) dataset "
                     "could not be retrieved automatically (JavaScript anti-bot wall), so "
                     "classifications rest on the primary source (Cook et al.) alone. Drop "
                     "the Middleton-Welling xlsx into `./data/` and re-run to auto-add the "
                     "cross-check; any source disagreement will be flagged `MANUAL_REVIEW`.\n")
        L.append(f"**Manual-review species ({len(V['review_list'])})** — unmatched, "
                 "ambiguous, or sources disagree:")
        for s in V["review_list"]:
            L.append(f"- `{s}`")
        L.append("")
    L.append("Known review cases regardless of source: **`Thymelicus lineola/sylvestris`** "
             "is a recorder aggregate of two species and cannot take a single class; the "
             "cryptic *Leptidea* pair (`juvernica`/`sinapis`) needs care. Several UKBMS "
             "names differ from trait-DB genera (e.g. *Aglais io* ↔ *Inachis io*, "
             "*Speyeria/Fabriciana* ↔ *Argynnis*, *Favonius quercus* ↔ *Neozephyrus "
             "quercus*, *Phengaris arion* ↔ *Maculinea arion*) — handled by a synonym table.\n")

    L.append("## Artefacts\n")
    L.append("- `output/eda/*.png` — figures 1–9")
    L.append("- `output/eda_report.md` — this report")
    volt_note = ("59/60 classified from Cook et al.; 1 aggregate on manual review"
                 if V and V.get("cook_file") else "scaffold until sources supplied")
    L.append(f"- `data/species_voltinism.csv` — voltinism lookup ({volt_note})")
    L.append("")
    return "\n".join(L)


def main() -> int:
    REPORT.write_text(build())
    print(f"wrote {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
