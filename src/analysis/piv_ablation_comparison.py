"""
piv_ablation_comparison.py
==========================
Run PIV ablation study for Juruá-A Mask 1:
Compare single-angle baselines (0°, 15°, 30°, 45° each alone)
against multi-angle (0+15+30+45°) fusion.

For each configuration, reports:
 - Raw coverage (grid points with valid displacement)
 - Filtered coverage (points passing uncertainty guards)
 - Keep ratio (%)
 - Median raw displacement magnitude (before filtering, in m/yr)
 - Per-angle information when applicable (CV, directional std)

Usage:
    conda activate riverpiv
    python -m src.analysis.piv_ablation_comparison
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.piv_analysis.jurua_multitilt import run_multitilt_jurua
from src.config import PROJECT_ROOT as CFG_ROOT


def calculate_metrics(stats: dict, label: str) -> dict:
    """Calculate summary metrics from compute_vector_stats output."""
    u_mean = stats["u_mean"]
    v_mean = stats["v_mean"]
    n_mean = stats["n_mean"]
    n_std  = stats["n_std"]
    t_std  = stats["theta_std"]
    bad    = stats["bad_mask"]
    N_arr  = stats["N"]

    # Raw coverage (finite displacement)
    raw_mask   = np.isfinite(u_mean)
    raw_count  = int(np.sum(raw_mask))

    # Filtered coverage
    filt_mask  = raw_mask & (~bad)
    filt_count = int(np.sum(filt_mask))

    keep_ratio = (filt_count / raw_count * 100.0) if raw_count > 0 else 0.0

    # Raw median displacement magnitude (before quality filtering)
    n_raw = np.hypot(u_mean[raw_mask], v_mean[raw_mask])
    median_disp = float(np.nanmedian(n_raw)) if n_raw.size > 0 else np.nan

    # Mean number of realizations per valid point
    mean_N = float(np.nanmean(N_arr[raw_mask])) if raw_count > 0 else 0.0

    # Precision & directional stability (only meaningful after filtering)
    if filt_count > 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            cv = n_std[filt_mask] / n_mean[filt_mask]
        avg_cv      = float(np.nanmean(cv))
        avg_dir_std = float(np.rad2deg(np.nanmean(t_std[filt_mask])))
    else:
        avg_cv      = np.nan
        avg_dir_std = np.nan

    return {
        "Configuration": label,
        "Angles": "",   # filled by caller
        "Raw (n)": raw_count,
        "Filtered (n)": filt_count,
        "Keep (%)": round(keep_ratio, 1),
        "Median |M| raw (m/yr)": round(median_disp, 2) if np.isfinite(median_disp) else "—",
        "Mean N": round(mean_N, 1),
        "Precision CV": round(avg_cv, 3) if np.isfinite(avg_cv) else "—",
        "Dir Std (°)": round(avg_dir_std, 1) if np.isfinite(avg_dir_std) else "—",
    }


def main():
    print("PIV Ablation Study: Juruá-A, Mask 1")
    print("=" * 60)

    rows = []

    # ── Single-angle baselines ──────────────────────────────────────
    for tilt in (0, 15, 30, 45):
        label = f"Single-Angle ({tilt}°)"
        print(f"Running {label}...")
        _, _, stats, _ = run_multitilt_jurua(
            site="Jurua-A", mask_level=1, tilt_degs=(tilt,)
        )
        m = calculate_metrics(stats, label)
        m["Angles"] = f"{tilt}°"
        rows.append(m)

    # ── Multi-angle fusion ──────────────────────────────────────────
    print("Running Multi-Angle (0+15+30+45°) fusion...")
    _, _, stats_multi, _ = run_multitilt_jurua(
        site="Jurua-A", mask_level=1, tilt_degs=(0, 15, 30, 45)
    )
    m = calculate_metrics(stats_multi, "Multi-Angle Fusion")
    m["Angles"] = "0°+15°+30°+45°"
    rows.append(m)

    # ── Output ──────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    # Reorder columns
    cols = [
        "Configuration", "Angles", "Raw (n)", "Filtered (n)", "Keep (%)",
        "Median |M| raw (m/yr)", "Mean N", "Precision CV", "Dir Std (°)",
    ]
    df = df[cols]

    print("\n" + df.to_markdown(index=False))

    # Save CSV
    out_csv = CFG_ROOT / "results" / "PostprocessedPIV" / "piv_ablation_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved to {out_csv}")


if __name__ == "__main__":
    main()
