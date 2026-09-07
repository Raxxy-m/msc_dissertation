#!/usr/bin/env python
"""
Orchestrate the full modelling phase end-to-end (idempotent):

    Stage 3 (baselines + GDD)
      add_nao       -> NAO columns on feature_table + data/nao_djfm.csv
      harness       -> modelling frame + fixed splits + naive baselines
      gdd_baseline  -> GDD process baseline (both regimes)
      report        -> output/modelling_report.md
    Stage 3b (RQ1 mixed-effects rung)
      fix_nao_2024  -> fill 2024 NAO (CPC-rescaled) + NAO_SOURCE; rebuild frame
      mixed_effects -> NAO x voltinism models + harness eval + output/mixedeffects_report.md
    Stage 3c (RQ2/RQ3 machine-learning rung)
      ml_baseline   -> HGB + RF (with/without NAO) via harness + SHAP + output/ml_report.md

Run:  .venv/bin/python -m scripts.modelling.run_all
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.modelling import (add_nao, harness, gdd_baseline, report,
                                    fix_nao_2024, mixed_effects, ml_baseline)
else:
    from . import (add_nao, harness, gdd_baseline, report,
                   fix_nao_2024, mixed_effects, ml_baseline)


def main() -> int:
    steps = (add_nao.main, harness.main, gdd_baseline.main, report.main,
             fix_nao_2024.main, mixed_effects.main, ml_baseline.main)
    for step in steps:
        print(f"\n===== {step.__module__} =====")
        rc = step()
        if rc:
            return rc
    print("\nmodelling phase complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
