#!/usr/bin/env python
"""
PART B — build a per-species voltinism lookup (univoltine vs bivoltine/multivoltine)
for the ~60 UKBMS species, needed for RQ1 and the mixed-effects model.

Sources (NOT auto-downloadable — see eda_report.md "voltinism source access"):
  * PRIMARY   Cook et al. (2021, 2024 update) traits data, EIDC id
              5b5a13b6-2304-47e3-9c9d-35237d1232c6 -> licence-acceptance gated; user
              downloaded the RO-Crate into ./traits/ (read via load_cook()).
  * CROSSCHECK Middleton-Welling et al. (2020) European & Maghreb trait DB,
              Dryad doi:10.5061/dryad.6m905qfx6 (CC0) -> anti-bot gated; optional xlsx in data/.

Cook voltinism is coded as one-hot indicator columns (obligate_univoltine /
obligate_multivoltine / partial_generation) under a two-row header; load_cook() handles
that layout and the non-breaking spaces in scientific names. The generic load_source()
handles the Middleton-Welling cross-check (auto-detecting name/voltinism columns).
It never guesses: anything not clearly resolved from a source is left as 'MANUAL_REVIEW'.

If no source file is present it writes a SCAFFOLD (all 60 species -> MANUAL_REVIEW /
PENDING) so the schema exists and the review list is explicit, then prints instructions.

Run:  .venv/bin/python -m scripts.voltinism
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

# =====================================================================================
# CONFIG
# =====================================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = {
    "feature_table": PROJECT_ROOT / "data" / "feature_table.parquet",
    "out_csv": PROJECT_ROOT / "data" / "species_voltinism.csv",
    # Cook et al. traits RO-Crate (user-downloaded into ./traits/). The ecological traits
    # CSV has a TWO-ROW header (group row + field row) and one-hot voltinism indicators.
    "cook_globs": ["traits/data/ecological_traits_2024.csv",
                   "traits/data/ecological_traits*.csv", "data/*ecological_traits*.csv"],
    "cook_encoding": "cp1252",   # contains Windows-1252 bytes + non-breaking spaces
    "mw_globs": ["data/*European*Maghreb*Trait*data*.xlsx", "data/*middleton*welling*.xlsx",
                 "data/traits_middleton*/**/*.xlsx", "traits/**/*European*Maghreb*.xlsx"],
    # Column auto-detection hints (checked case-insensitively, substring match):
    "name_col_hints": ["scientific_name", "scientificname", "taxon", "species_name",
                       "species", "binomial", "latin"],
    "volt_col_hints": ["voltinism", "volt", "generations", "n_gen", "broods", "brood"],
    # Synonyms: UKBMS SPECIES_NAME -> alternative binomials used in trait DBs (genus revisions).
    # Only those actually needed to match Cook 2024 are required; extras are harmless.
    "synonyms": {
        "Colias croceus": ["Colias crocea"],
        "Polyommatus bellargus": ["Lysandra bellargus"],
        "Polyommatus coridon": ["Lysandra coridon"],
        # kept for the Middleton-Welling cross-check (different genera there):
        "Aglais io": ["Inachis io"],
        "Speyeria aglaja": ["Argynnis aglaja"],
        "Fabriciana adippe": ["Argynnis adippe"],
        "Favonius quercus": ["Neozephyrus quercus", "Quercusia quercus"],
        "Ochlodes sylvanus": ["Ochlodes venata", "Ochlodes venatus"],
        "Phengaris arion": ["Maculinea arion"],
    },
    # Aggregates / cryptic pairs that CANNOT be resolved to a single species -> review.
    "force_review": {"Thymelicus lineola/sylvestris"},
}


# =====================================================================================
# Value -> class mapping (used once real source columns are known)
# =====================================================================================
def classify_text(v) -> str | None:
    """Map a free-text / coded voltinism value to a class, or None if unclear."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip().lower()
    if s in ("", "na", "nan", "unknown", "?"):
        return None
    # explicit variability first
    if any(k in s for k in ("variable", "partial", "varies", "1-2", "1–2", "1 to 2",
                            "uni/bi", "uni-bi")):
        return "variable"
    if "univolt" in s or s in ("1", "1.0", "one", "monovoltine"):
        return "univoltine"
    if any(k in s for k in ("bivolt", "multivolt", "polyvolt", "plurivolt")):
        return "multivoltine"
    # pure integer generations
    m = re.fullmatch(r"(\d+(?:\.\d+)?)", s)
    if m:
        n = float(m.group(1))
        return "univoltine" if n <= 1 else "multivoltine"
    return None


def classify_minmax(vmin, vmax) -> str | None:
    """From min/max generations-per-year columns."""
    try:
        lo, hi = float(vmin), float(vmax)
    except (TypeError, ValueError):
        return None
    if pd.isna(lo) or pd.isna(hi):
        return None
    if hi <= 1:
        return "univoltine"
    if lo <= 1 < hi:
        return "variable"
    return "multivoltine"


# =====================================================================================
# Source loading (auto-detect columns)
# =====================================================================================
def _find(globs):
    for g in globs:
        hits = sorted(glob.glob(str(PROJECT_ROOT / g), recursive=True))
        if hits:
            return Path(hits[0])
    return None


def _detect_col(cols, hints):
    low = {c: c.lower() for c in cols}
    for h in hints:
        for c, cl in low.items():
            if h in cl:
                return c
    return None


def load_cook(path: Path) -> tuple[pd.DataFrame, dict]:
    """Cook et al. 2024 ecological_traits CSV -> df[name, cook].

    Structure: row 0 = trait GROUP, row 1 = field names, rows 2+ = species. Voltinism is
    three one-hot indicator columns under the 'voltinism' group. Scientific names contain
    non-breaking spaces (\\xa0) which must be normalised before joining.
    Mapping (Cook's own categories):
        obligate_multivoltine == 1 -> multivoltine
        partial_generation    == 1 -> variable   (facultative/partial 2nd brood, region-dep.)
        obligate_univoltine   == 1 -> univoltine
        none set                   -> None (unclassified -> MANUAL_REVIEW downstream)
    """
    raw = pd.read_csv(path, encoding=CONFIG["cook_encoding"], header=None, dtype=str)
    fields = [str(f).strip() for f in raw.iloc[1]]
    data = raw.iloc[2:].reset_index(drop=True)
    data.columns = fields
    norm = lambda s: str(s).replace("\xa0", " ").strip()

    def onehot(col):
        return data[col].fillna("0").astype(str).str.strip().eq("1") if col in data else \
            pd.Series(False, index=data.index)

    uni, multi, partial = (onehot("obligate_univoltine"),
                           onehot("obligate_multivoltine"),
                           onehot("partial_generation"))
    cls = np.where(multi, "multivoltine",
                   np.where(partial, "variable",
                            np.where(uni, "univoltine", None)))
    out = pd.DataFrame({"name": data["scientific_name"].map(norm), "cook": cls})
    out = out[out["name"].ne("") & out["name"].ne("nan")].drop_duplicates("name")
    meta = {"file": path.name, "rows": len(out),
            "n_voltinism_coded": int(out["cook"].notna().sum()),
            "name_col": "scientific_name", "volt_cols":
            ["obligate_univoltine", "obligate_multivoltine", "partial_generation"]}
    return out, meta


def load_source(path: Path, label: str) -> tuple[pd.DataFrame, dict]:
    """Return (df[name, voltinism_class], meta). Auto-detects columns; defensive."""
    if path.suffix.lower() in (".xlsx", ".xls"):
        raw = pd.read_excel(path)
    else:
        raw = pd.read_csv(path)
    name_col = _detect_col(raw.columns, CONFIG["name_col_hints"])
    volt_col = _detect_col(raw.columns, CONFIG["volt_col_hints"])
    meta = {"file": path.name, "rows": len(raw), "name_col": name_col,
            "volt_col": volt_col, "all_cols": list(raw.columns)}
    if name_col is None or volt_col is None:
        meta["error"] = ("could not auto-detect name/voltinism columns; set them "
                         "explicitly in CONFIG after inspecting all_cols")
        return pd.DataFrame(columns=["name", label]), meta
    # try min/max generation columns too
    vmin = _detect_col(raw.columns, ["vol_min", "gen_min", "min_gen", "voltinism_min"])
    vmax = _detect_col(raw.columns, ["vol_max", "gen_max", "max_gen", "voltinism_max"])
    out = pd.DataFrame({"name": raw[name_col].astype(str).str.strip()})
    if vmin and vmax:
        out[label] = [classify_minmax(a, b) for a, b in zip(raw[vmin], raw[vmax])]
    else:
        out[label] = raw[volt_col].map(classify_text)
    out = out.dropna(subset=[label]).drop_duplicates("name")
    return out, meta


def _name_variants(sp: str) -> list[str]:
    return [sp, *CONFIG["synonyms"].get(sp, [])]


def lookup(species: list[str], src: pd.DataFrame, col: str) -> dict:
    """Match each UKBMS species (and its synonyms, and species-epithet fallback) to src."""
    by_name = dict(zip(src["name"], src[col]))
    by_epithet = {}
    for nm, cl in zip(src["name"], src[col]):
        toks = nm.split()
        if len(toks) >= 2:
            by_epithet.setdefault(toks[1].lower(), cl)
    res = {}
    for sp in species:
        val = None
        for v in _name_variants(sp):
            if v in by_name:
                val = by_name[v]
                break
        if val is None:                      # last resort: species-epithet match
            ep = sp.split()[1].lower() if len(sp.split()) > 1 else ""
            val = by_epithet.get(ep)
        res[sp] = val
    return res


# =====================================================================================
# Build
# =====================================================================================
def build() -> tuple[pd.DataFrame, dict]:
    species = sorted(pd.read_parquet(CONFIG["feature_table"],
                                     columns=["SPECIES_NAME"])["SPECIES_NAME"]
                     .dropna().unique())
    cook_path, mw_path = _find(CONFIG["cook_globs"]), _find(CONFIG["mw_globs"])
    status = {"n_species": len(species), "cook_file": str(cook_path) if cook_path else None,
              "mw_file": str(mw_path) if mw_path else None, "source_meta": {}}

    cook_map, mw_map = {}, {}
    if cook_path:
        cdf, cmeta = load_cook(cook_path)
        status["source_meta"]["cook"] = cmeta
        cook_map = lookup(species, cdf, "cook") if not cdf.empty else {}
    if mw_path:
        mdf, mmeta = load_source(mw_path, "mw")
        status["source_meta"]["mw"] = mmeta
        mw_map = lookup(species, mdf, "mw") if not mdf.empty else {}

    rows = []
    for sp in species:
        c = cook_map.get(sp)
        m = mw_map.get(sp)
        if sp in CONFIG["force_review"]:
            rows.append((sp, "MANUAL_REVIEW", "—",
                         "species aggregate / cryptic pair — cannot assign single class"))
            continue
        if c and m:
            if c == m:
                rows.append((sp, c, "Cook 2021 (2024 update) + Middleton-Welling 2020 (agree)", ""))
            else:
                rows.append((sp, "MANUAL_REVIEW", "Cook 2021 (2024 update) vs Middleton-Welling 2020",
                             f"sources DISAGREE: Cook={c}, MW={m}"))
        elif c:
            rows.append((sp, c, "Cook et al. 2021 (2024 update)", "no cross-check match in MW" if mw_path else
                         "MW cross-check not available"))
        elif m:
            rows.append((sp, m, "Middleton-Welling 2020",
                         "not matched in Cook (primary) — verify"))
        else:
            why = "no trait source file present yet" if not (cook_path or mw_path) \
                  else "species not matched to any source (synonym?) — review"
            rows.append((sp, "MANUAL_REVIEW", "PENDING", why))

    df = pd.DataFrame(rows, columns=["SPECIES_NAME", "voltinism", "source", "note"])
    status["n_assigned"] = int((~df["voltinism"].eq("MANUAL_REVIEW")).sum())
    status["review_list"] = df.loc[df["voltinism"] == "MANUAL_REVIEW", "SPECIES_NAME"].tolist()
    status["class_counts"] = df["voltinism"].value_counts().to_dict()
    return df, status


def main() -> int:
    df, status = build()
    CONFIG["out_csv"].parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CONFIG["out_csv"], index=False)
    print(f"wrote {CONFIG['out_csv']}  ({len(df)} species)")
    print("class counts:", status["class_counts"])
    print(f"assigned from source: {status['n_assigned']}/{status['n_species']}")
    if not (status["cook_file"] or status["mw_file"]):
        print("\n*** NO TRAIT SOURCE FILE FOUND — wrote SCAFFOLD (all species MANUAL_REVIEW).")
        print("    Place the downloaded files in ./data/ and re-run:")
        print("      Cook 2022 CSV  -> matches data/cook*trait*.csv (primary)")
        print("      Middleton-Welling xlsx -> matches data/*European*Maghreb*Trait*data*.xlsx")
    else:
        print(f"\nmanual-review species ({len(status['review_list'])}):")
        for s in status["review_list"]:
            print("   -", s)
    import json
    (PROJECT_ROOT / "output" / "eda").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "output" / "eda" / "_voltinism_status.json").write_text(
        json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
