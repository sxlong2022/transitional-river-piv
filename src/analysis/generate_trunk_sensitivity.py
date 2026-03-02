"""
generate_trunk_sensitivity.py
=============================
Trunk aggregation sensitivity analysis:
Sweep (min_trunk_length_m, endpoint_tol_m) and report effective trunk count.

Usage:
    conda activate riverpiv
    python -m src.analysis.generate_trunk_sensitivity
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.quantitative_relationships import analyze_trunk_level_relationships

sites = ["HuangHe-A", "HuangHe-B"]
masks = {"HuangHe-A": 4, "HuangHe-B": 4}
min_lengths = [2000.0, 3000.0, 5000.0, 8000.0]
endpoint_tols = [40.0, 80.0, 120.0, 160.0]

print("| Site | `min_trunk_length_m` | `endpoint_tol_m` | Effective Trunks ($N_{eff}$) |")
print("|:---|---:|---:|---:|")

for site in sites:
    mask = masks[site]
    npz_path = PROJECT_ROOT / "results" / "PostprocessedPIV" / site / f"{site}_mask{mask}_link_sBCMn_flat_step20_metric_v2.npz"
    if not npz_path.exists():
        npz_path = PROJECT_ROOT / "results" / "PostprocessedPIV" / site / f"{site}_mask{mask}_link_sBCMn_flat_step100_metric.npz"
    if not npz_path.exists():
        print(f"| {site} | — | — | FILE NOT FOUND |")
        continue

    for ml in min_lengths:
        for et in endpoint_tols:
            try:
                res = analyze_trunk_level_relationships(
                    npz_path,
                    k_trunks=100,
                    endpoint_tol_m=et,
                    weight_by="length_B",
                    min_trunk_length_m=ml,
                )
                k_eff = int(res.diagnostics.get("k_trunks", 0))
                print(f"| {site} | {ml:.0f} | {et:.0f} | {k_eff} |")
            except Exception as e:
                print(f"| {site} | {ml:.0f} | {et:.0f} | ERROR: {e} |")
