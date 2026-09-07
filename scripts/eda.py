#!/usr/bin/env python
"""
PART A — exploratory data analysis of data/feature_table.parquet (READ ONLY).

Target of interest = FIRSTDAY (first appearance, "days after 1 April"; can be negative).
Saves figures to output/eda/*.png and (with scripts/voltinism.py) writes
output/eda_report.md. NAO figures are produced only if the NAO columns exist.

Run:  .venv/bin/python -m scripts.eda        (or: .venv/bin/python scripts/eda.py)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import seaborn as sns

# =====================================================================================
# CONFIG
# =====================================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = {
    "feature_table": PROJECT_ROOT / "data" / "feature_table.parquet",
    "eda_dir": PROJECT_ROOT / "output" / "eda",
    "stats_json": PROJECT_ROOT / "output" / "eda" / "_eda_stats.json",
    "dpi": 130,
    "year_min": 1976, "year_max": 2024,          # coverage window (data starts 1973)
    # British National Grid plausibility bounds (EPSG:27700), metres.
    "bng_e": (0, 700_000), "bng_n": (0, 1_300_000),
    "non_gb_countries": {"Northern Ireland", "Channel Islands", "Isle of Man"},
    # 6 common species spanning early -> late flyers (validated by median FIRSTDAY below).
    "focus_species": ["Anthocharis cardamines", "Gonepteryx rhamni", "Pieris rapae",
                      "Maniola jurtina", "Aphantopus hyperantus", "Pyronia tithonus"],
    # climate features for the correlation heatmap (temperature multicollinearity focus).
    "heatmap_feats": (["TMEAN_M12P", "TMEAN_M01", "TMEAN_M02", "TMEAN_M03", "TMEAN_M04",
                       "TMEAN_M05", "TMEAN_DJF", "TMEAN_MAM", "GDD",
                       "RAIN_DJF", "RAIN_MAM", "NAO_DJFM", "NAO_DJFM_LAG1"]),
    "spring_temp": "TMEAN_MAM",
    "scatter_sample": 25_000,
    "seed": 42,
}
sns.set_theme(style="whitegrid", context="notebook")

# --- Shared figure style (keep IDENTICAL across all plotting modules). ---------
# Figures are authored close to their display size (~5.8 in wide for a full-\linewidth
# float) so LaTeX does little/no downscaling and these font sizes survive to the page.
# Applied AFTER sns.set_theme so it wins over the seaborn "notebook" context.
SHARED_RCPARAMS = {
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
}
plt.rcParams.update(SHARED_RCPARAMS)
FD_LABEL = "FIRSTDAY (days after 1 April)"


def _save(fig, name):
    """Save both a vector PDF (for LaTeX) and a 300-dpi PNG (for the Markdown report).

    `name` ends in .png (kept as the return value so the stats JSON / eda_report.md
    keep referencing the raster). The .pdf twin is what the dissertation imports.
    """
    CONFIG["eda_dir"].mkdir(parents=True, exist_ok=True)
    stem = name[:-4] if name.lower().endswith(".png") else name
    fig.savefig(CONFIG["eda_dir"] / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(CONFIG["eda_dir"] / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return f"{stem}.png"


def _lin(x, y):
    """Return (slope, intercept, pearson_r, n) over finite pairs."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return (np.nan, np.nan, np.nan, len(x))
    b, a = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])
    return (float(b), float(a), r, int(len(x)))


# =====================================================================================
# Figures
# =====================================================================================
def fig1_coverage(df, S):
    d = df[(df.YEAR >= CONFIG["year_min"]) & (df.YEAR <= CONFIG["year_max"])]
    per = d.groupby("YEAR").agg(records=("FIRSTDAY", "size"),
                                sites=("SITE_ID", "nunique"))
    fig, ax1 = plt.subplots(figsize=(6.0, 3.6))
    ax1.plot(per.index, per["records"], color="C0", marker="o", ms=3, label="records")
    ax1.set_xlabel("Year"); ax1.set_ylabel("Records (site×species rows)", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax2 = ax1.twinx()
    ax2.plot(per.index, per["sites"], color="C3", marker="s", ms=3, label="distinct sites")
    ax2.set_ylabel("Distinct sites monitored", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3"); ax2.grid(False)
    ax1.set_title("Survey coverage over time (1976–2024)\nrecords and distinct sites per year")
    S["coverage"] = {"first_year_records": int(per["records"].iloc[0]),
                     "last_year_records": int(per["records"].iloc[-1]),
                     "first_year_sites": int(per["sites"].iloc[0]),
                     "last_year_sites": int(per["sites"].iloc[-1])}
    return _save(fig, "fig01_coverage_by_year.png")


def fig2_records_per_species(df, S):
    c = df.groupby("SPECIES_NAME").size().sort_values()
    fig, ax = plt.subplots(figsize=(8, 12))
    ax.barh(c.index, c.values, color="C0")
    ax.set_xlabel("Records (site×species×year rows)")
    ax.set_title("Records per species (sorted)\ndata-rich vs data-poor species")
    ax.margins(y=0.005)
    S["records_per_species"] = {"richest": c.index[-1], "richest_n": int(c.iloc[-1]),
                                "poorest": c.index[0], "poorest_n": int(c.iloc[0]),
                                "n_under_500": int((c < 500).sum())}
    return _save(fig, "fig02_records_per_species.png")


def fig3_site_map(df, S):
    s = df.drop_duplicates("SITE_ID")[
        ["SITE_ID", "EASTING", "NORTHING", "N_YRS_SURVEYED", "COUNTRY"]].copy()
    coord = s.dropna(subset=["EASTING", "NORTHING"])
    non_gb = coord["COUNTRY"].isin(CONFIG["non_gb_countries"])
    (e0, e1), (n0, n1) = CONFIG["bng_e"], CONFIG["bng_n"]
    out_bounds = ~(coord["EASTING"].between(e0, e1) & coord["NORTHING"].between(n0, n1))
    gb = coord[~non_gb]
    fig, ax = plt.subplots(figsize=(5.0, 6.3))
    sc = ax.scatter(gb["EASTING"], gb["NORTHING"], c=gb["N_YRS_SURVEYED"],
                    cmap="viridis", s=8, alpha=0.7)
    bad = coord[non_gb]
    ax.scatter(bad["EASTING"], bad["NORTHING"], c="red", marker="x", s=25,
               label=f"non-GB country (n={len(bad)})")
    fig.colorbar(sc, ax=ax, label="N_YRS_SURVEYED", shrink=0.6)
    ax.set_xlabel("Easting (m, BNG)"); ax.set_ylabel("Northing (m, BNG)")
    ax.xaxis.set_major_locator(MaxNLocator(4))       # avoid crowded easting ticks
    ax.set_aspect("equal"); ax.legend(loc="upper left")
    ax.set_title("UKBMS site map (British National Grid)\ncoloured by years surveyed; "
                 "red = non-GB coords (mislocated)")
    S["site_map"] = {"sites_with_coords": int(len(coord)),
                     "sites_no_coords": int(s["EASTING"].isna().sum()),
                     "non_gb_country_sites": int(non_gb.sum()),
                     "out_of_bng_bounds": int(out_bounds.sum()),
                     "min_northing": float(coord["NORTHING"].min())}
    return _save(fig, "fig03_site_map.png")


def fig4_firstday_dist(df, S):
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(3, 4)
    ax0 = fig.add_subplot(gs[:, 0:2])
    ax0.hist(df["FIRSTDAY"].dropna(), bins=60, color="C0")
    ax0.set_xlabel(FD_LABEL); ax0.set_ylabel("Records")
    ax0.set_title("FIRSTDAY distribution (all species)")
    order = (df[df.SPECIES_NAME.isin(CONFIG["focus_species"])]
             .groupby("SPECIES_NAME")["FIRSTDAY"].median().sort_values())
    for i, sp in enumerate(order.index):
        ax = fig.add_subplot(gs[i // 2, 2 + i % 2])
        ax.hist(df.loc[df.SPECIES_NAME == sp, "FIRSTDAY"].dropna(), bins=35, color="C2")
        ax.set_title(f"{sp}\n(median {order[sp]:.0f})", fontsize=11)
        ax.tick_params(labelsize=10)
    fig.suptitle("FIRSTDAY overall and for 6 species spanning early→late flyers")
    fig.tight_layout()
    return _save(fig, "fig04_firstday_distribution.png")


def fig5_missingness(df, S):
    miss = (df.isna().mean() * 100).sort_values(ascending=False)
    miss = miss[miss > 0]
    # 71 columns have some NA — a single-column bar chart would be a ~14 in tall strip
    # (illegible when fit to a page). Split the sorted list across two side-by-side
    # panels so every label reads at 9 pt on one page. Same data, layout only.
    half = (len(miss) + 1) // 2
    parts = [miss.iloc[:half], miss.iloc[half:]]
    xmax = float(miss.max()) * 1.35   # headroom so the value label clears the bar
    # 71 long column names in two columns: keep a moderate aspect so the bars stay
    # visible (a very tall figure squashes the x-axis) and labels read at ~9-10 pt.
    fig, axes = plt.subplots(1, 2, figsize=(6.6, max(5.6, 0.165 * half)),
                             layout="constrained")
    for ax, part in zip(axes, parts):
        ax.barh(part.index, part.values, color="C1")
        ax.invert_yaxis()
        ax.set_xlim(0, xmax)
        ax.set_xticks([0, 5])
        ax.set_xlabel("% missing")
        ax.tick_params(axis="y", labelsize=9)
        for y, v in enumerate(part.values):
            ax.text(v + xmax * 0.03, y, f"{v:.1f}", va="center", fontsize=8)
    fig.suptitle("Missingness per column (columns with any NA)")
    S["missingness_top"] = {k: round(float(v), 2) for k, v in miss.head(15).items()}
    return _save(fig, "fig05_missingness.png")


def fig6_phenology_trend(df, S):
    d = df[(df.YEAR >= CONFIG["year_min"]) & (df.YEAR <= CONFIG["year_max"])]
    yr = d.groupby("YEAR")["FIRSTDAY"].mean()
    b, a, r, n = _lin(yr.index.values.astype(float), yr.values)
    # Full-width SIDEWAYS figure in the dissertation: authored ~9 in wide so that, placed
    # with width=\textheight (9.03 in) in a rotated float, it renders at ~1:1 with no
    # downscaling — every per-panel title and tick stays legible (>=9 pt).
    fig = plt.figure(figsize=(9.0, 5.6), layout="constrained")
    gs = fig.add_gridspec(3, 4)
    ax0 = fig.add_subplot(gs[:, 0:2])
    ax0.plot(yr.index, yr.values, "o-", ms=3, color="C0")
    xs = np.array([yr.index.min(), yr.index.max()], float)
    ax0.plot(xs, a + b * xs, "r--", label=f"slope {b*10:+.2f} days/decade")
    ax0.set_xlabel("Year"); ax0.set_ylabel("Mean " + FD_LABEL)
    ax0.legend(); ax0.set_title("Phenology trend (all species)\nmean FIRSTDAY per year")
    ax0.xaxis.set_major_locator(MaxNLocator(6, integer=True))
    sp_slopes = {}
    for i, sp in enumerate(CONFIG["focus_species"]):
        ax = fig.add_subplot(gs[i // 2, 2 + i % 2])
        g = d[d.SPECIES_NAME == sp].groupby("YEAR")["FIRSTDAY"].mean()
        if len(g) >= 3:
            sb, sa, sr, sn = _lin(g.index.values.astype(float), g.values)
            ax.plot(g.index, g.values, ".", ms=3, color="C2")
            ax.plot(xs, sa + sb * xs, "r--", lw=1)
            ax.set_title(f"{sp}\n{sb*10:+.1f} d/decade", fontsize=12)
            sp_slopes[sp] = round(sb * 10, 2)
        ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))  # fewer year ticks
        ax.tick_params(labelsize=11)
    fig.suptitle("Mean FIRSTDAY over time with linear trend "
                 "(negative slope = advancing/earlier emergence)")
    S["phenology_trend"] = {"overall_slope_days_per_decade": round(b * 10, 3),
                            "overall_pearson_r": round(r, 3), "n_years": n,
                            "species_slopes_days_per_decade": sp_slopes}
    return _save(fig, "fig06_phenology_trend.png")


def fig7_firstday_vs_temp(df, S):
    col = CONFIG["spring_temp"]
    d = df[["FIRSTDAY", col, "SPECIES_NAME"]].dropna().copy()
    # species-mean-removed anomalies isolate the WITHIN-species thermal response
    d["fd_anom"] = d["FIRSTDAY"] - d.groupby("SPECIES_NAME")["FIRSTDAY"].transform("mean")
    d["t_anom"] = d[col] - d.groupby("SPECIES_NAME")[col].transform("mean")
    b, a, r, n = _lin(d[col].values, d["FIRSTDAY"].values)          # pooled (raw)
    bw, aw, rw, nw = _lin(d["t_anom"].values, d["fd_anom"].values)  # within-species
    samp = d.sample(min(CONFIG["scatter_sample"], len(d)), random_state=CONFIG["seed"])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5))
    axL.scatter(samp[col], samp["FIRSTDAY"], s=5, alpha=0.15, color="C0")
    xs = np.array([d[col].min(), d[col].max()])
    axL.plot(xs, a + b * xs, "r-", lw=2, label=f"slope {b:+.2f} days/°C   r={r:+.2f}")
    axL.set_xlabel(f"Spring mean temperature {col} (°C)"); axL.set_ylabel(FD_LABEL)
    axL.legend(); axL.set_title("(a) Pooled: FIRSTDAY vs spring temperature\n"
                                "(weak — mixes species with different baselines)")
    axR.scatter(samp["t_anom"], samp["fd_anom"], s=5, alpha=0.15, color="C2")
    xs2 = np.array([d["t_anom"].min(), d["t_anom"].max()])
    axR.plot(xs2, aw + bw * xs2, "r-", lw=2, label=f"slope {bw:+.2f} days/°C   r={rw:+.2f}")
    axR.axhline(0, color="grey", lw=0.6); axR.axvline(0, color="grey", lw=0.6)
    axR.set_xlabel(f"{col} anomaly (°C, species-mean removed)")
    axR.set_ylabel("FIRSTDAY anomaly (days)")
    axR.legend(); axR.set_title("(b) Within-species thermal sensitivity\n"
                                "warmer springs → earlier first appearance")
    fig.tight_layout()
    S["firstday_vs_temp"] = {"feature": col, "pooled_slope_days_per_C": round(b, 3),
                             "pooled_r": round(r, 3),
                             "within_species_slope_days_per_C": round(bw, 3),
                             "within_species_r": round(rw, 3), "n": n}
    return _save(fig, "fig07_firstday_vs_spring_temp.png")


def fig8_corr_heatmap(df, S):
    feats = [c for c in CONFIG["heatmap_feats"] if c in df.columns]
    corr = df[feats].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                vmin=-1, vmax=1, square=True, cbar_kws={"shrink": 0.7},
                annot_kws={"size": 7}, ax=ax)
    ax.set_title("Correlation of climate features (multicollinearity check)")
    # strongest off-diagonal collinearity
    cv = corr.values.copy(); np.fill_diagonal(cv, np.nan)
    c2 = pd.DataFrame(cv, index=corr.index, columns=corr.columns)
    mx = c2.abs().stack().idxmax()
    S["corr_heatmap"] = {"features": feats,
                         "max_pair": list(mx), "max_val": round(float(c2.loc[mx]), 3)}
    return _save(fig, "fig08_climate_correlation.png")


def fig9_nao(df, S):
    if "NAO_DJFM" not in df.columns:
        S["nao"] = {"present": False}
        return None
    d = df[(df.YEAR >= CONFIG["year_min"]) & (df.YEAR <= CONFIG["year_max"])]
    # (a) year-level winter NAO vs winter/spring local temperature
    yr = d.groupby("YEAR").agg(NAO=("NAO_DJFM", "mean"),
                               DJF=("TMEAN_DJF", "mean"), MAM=("TMEAN_MAM", "mean")).dropna()
    # (b) per-year FIRSTDAY anomaly (species-mean removed) vs winter NAO
    sp_mean = d.groupby("SPECIES_NAME")["FIRSTDAY"].transform("mean")
    anom = (d.assign(fd_anom=d["FIRSTDAY"] - sp_mean)
              .groupby("YEAR").agg(fd_anom=("fd_anom", "mean"),
                                   NAO=("NAO_DJFM", "mean")).dropna())
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(13, 5.5))
    b1, a1, r1, _ = _lin(yr["NAO"].values, yr["DJF"].values)
    axa.scatter(yr["NAO"], yr["DJF"], color="C0", label=f"DJF temp  r={r1:+.2f}")
    xs = np.array([yr["NAO"].min(), yr["NAO"].max()])
    axa.plot(xs, a1 + b1 * xs, "C0--")
    b2, a2, r2, _ = _lin(yr["NAO"].values, yr["MAM"].values)
    axa.scatter(yr["NAO"], yr["MAM"], color="C1", marker="^", label=f"MAM temp  r={r2:+.2f}")
    axa.plot(xs, a2 + b2 * xs, "C1--")
    axa.set_xlabel("Winter NAO (station DJFM)"); axa.set_ylabel("Mean local temperature (°C)")
    axa.legend(); axa.set_title("(a) Winter NAO vs local winter/spring temperature\n"
                                "(RQ3: is NAO a proxy for local temp?)")
    b3, a3, r3, _ = _lin(anom["NAO"].values, anom["fd_anom"].values)
    axb.scatter(anom["NAO"], anom["fd_anom"], color="C3")
    xs3 = np.array([anom["NAO"].min(), anom["NAO"].max()])
    axb.plot(xs3, a3 + b3 * xs3, "r--",
             label=f"slope {b3:+.2f} days/NAO-unit  r={r3:+.2f}")
    axb.axhline(0, color="grey", lw=0.8)
    axb.set_xlabel("Winter NAO (station DJFM)")
    axb.set_ylabel("Mean FIRSTDAY anomaly (days, species-mean removed)")
    axb.legend(); axb.set_title("(b) Phenology anomaly vs winter NAO\n"
                                "(updated Westgarth-Smith relationship)")
    fig.tight_layout()
    S["nao"] = {"present": True,
                "nao_vs_DJF_r": round(r1, 3), "nao_vs_MAM_r": round(r2, 3),
                "firstday_anom_vs_nao_slope": round(b3, 3),
                "firstday_anom_vs_nao_r": round(r3, 3),
                "n_years": int(len(anom))}
    return _save(fig, "fig09_nao_relationships.png")


# =====================================================================================
# Driver
# =====================================================================================
def run_part_a() -> dict:
    df = pd.read_parquet(CONFIG["feature_table"])
    S = {"n_rows": int(len(df)), "n_sites": int(df.SITE_ID.nunique()),
         "n_species": int(df.SPECIES_NAME.nunique()),
         "year_range": [int(df.YEAR.min()), int(df.YEAR.max())],
         "firstday": {k: round(float(v), 2) for k, v in
                      df.FIRSTDAY.describe().to_dict().items()},
         "figures": {}}
    S["figures"]["fig1"] = fig1_coverage(df, S)
    S["figures"]["fig2"] = fig2_records_per_species(df, S)
    S["figures"]["fig3"] = fig3_site_map(df, S)
    S["figures"]["fig4"] = fig4_firstday_dist(df, S)
    S["figures"]["fig5"] = fig5_missingness(df, S)
    S["figures"]["fig6"] = fig6_phenology_trend(df, S)
    S["figures"]["fig7"] = fig7_firstday_vs_temp(df, S)
    S["figures"]["fig8"] = fig8_corr_heatmap(df, S)
    S["figures"]["fig9"] = fig9_nao(df, S)

    # ---- data issues -----------------------------------------------------
    issues = []
    sm = S["site_map"]
    if sm["min_northing"] < 0:
        issues.append(f"{sm['non_gb_country_sites']} sites have non-GB coordinates "
                      f"(Northern Ireland Irish-Grid / Channel Islands / Isle of Man); "
                      f"min northing {sm['min_northing']:.0f} m (Channel Islands are "
                      "negative). These plot mislocated on a BNG map and were excluded "
                      "from the climate join upstream.")
    issues.append(f"{sm['sites_no_coords']} distinct sites have no coordinates "
                  "(cannot be mapped or climate-joined).")
    issues.append(f"FIRSTDAY ranges {S['firstday']['min']:.0f}…{S['firstday']['max']:.0f} "
                  "days after 1 April; negative values = species first seen before 1 April "
                  "(early spring flyers), plausible not erroneous.")
    if S.get("nao", {}).get("present"):
        issues.append("NAO_DJFM missing for flight-year 2024 (station index ends 2023); "
                      "NAO figures use 1976–2023 fully, 2024 only via LAG1.")
    rps = S["records_per_species"]
    issues.append(f"Data-poor species: {rps['n_under_500']} of {S['n_species']} species "
                  f"have <500 records (poorest: {rps['poorest']}, {rps['poorest_n']}); "
                  "per-species models for these will be unstable.")
    S["data_issues"] = issues

    CONFIG["stats_json"].parent.mkdir(parents=True, exist_ok=True)
    CONFIG["stats_json"].write_text(json.dumps(S, indent=2))
    return S


if __name__ == "__main__":
    st = run_part_a()
    print("=== EDA summary ===")
    print(f"rows={st['n_rows']:,}  sites={st['n_sites']}  species={st['n_species']}  "
          f"years={st['year_range'][0]}–{st['year_range'][1]}")
    print("FIRSTDAY:", st["firstday"])
    print(f"phenology slope: {st['phenology_trend']['overall_slope_days_per_decade']} "
          f"days/decade (r={st['phenology_trend']['overall_pearson_r']})")
    fv = st["firstday_vs_temp"]
    print(f"FIRSTDAY vs {fv['feature']}: pooled {fv['pooled_slope_days_per_C']} days/°C "
          f"(r={fv['pooled_r']}); within-species {fv['within_species_slope_days_per_C']} "
          f"days/°C (r={fv['within_species_r']})")
    if st.get("nao", {}).get("present"):
        print("NAO vs DJF r:", st["nao"]["nao_vs_DJF_r"],
              "| FIRSTDAY-anom vs NAO r:", st["nao"]["firstday_anom_vs_nao_r"])
    print("\ndata issues:")
    for i in st["data_issues"]:
        print(" -", i)
    print("\nfigures:", [v for v in st["figures"].values() if v])
