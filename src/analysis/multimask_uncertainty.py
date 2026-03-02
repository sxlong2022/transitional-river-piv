"""Uncertainty quantification for multi-mask Mn profiles.

Reads the output from `src.pipeline.jurua_pipeline`:

    results/PostprocessedPIV/{site}/jurua_multimask_multitilt_centerline_profiles_step4b.npz

Computes and outputs:
- Mn_mean(s) and Mn_std(s) (available directly from npz if already computed)
- Coefficient of variation CV(s) = Mn_std / |Mn_mean|
- Directional consistency: statistics on whether Mn signs are consistent across masks
- Exceedance probability: given threshold T, compute P(|Mn| > T)

And plots:
- CV and sigma vs s curves
- Directional consistency and exceedance probability along the channel (simple strip or line plot)
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import argparse
import numpy as np
import matplotlib.pyplot as plt

from src.config import PROJECT_ROOT
from src.postprocessing.postprocess import get_postprocessed_dir
from src.visualization.quicklook import describe_output_root


def load_multimask_profiles(site: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load aggregated multi-mask Mn profile results.

    Returns: mask_levels, s, Mn_stack, Mn_mean, Mn_std
    """

    out_dir = get_postprocessed_dir(PROJECT_ROOT, site)
    npz_path = out_dir / "jurua_multimask_multitilt_centerline_profiles_step4b.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Multi-mask profile aggregation result not found: {npz_path}\n"
            "Please run first: python -m src.pipeline.jurua_pipeline --mask-levels 1 2 3 4 ..."
        )

    data = np.load(npz_path)
    mask_levels = data["mask_levels"]
    s = data["s"]
    Mn_stack = data["Mn_stack"]
    Mn_mean = data["Mn_mean"]
    Mn_std = data["Mn_std"]

    return mask_levels, s, Mn_stack, Mn_mean, Mn_std


def compute_metrics(
    Mn_stack: np.ndarray,
    Mn_mean: np.ndarray,
    Mn_std: np.ndarray,
    thresholds: Tuple[float, ...] = (5.0,),
) -> dict:
    """Compute a series of uncertainty metrics based on multi-mask Mn profiles."""

    # Coefficient of variation CV = sigma / |mean|
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = Mn_std / np.abs(Mn_mean)
        cv[~np.isfinite(cv)] = np.nan

    # Directional consistency: whether all masks have the same sign
    sign = np.sign(Mn_stack)  # [-1, 0, 1]
    # Only consider non-NaN masks
    valid = np.isfinite(Mn_stack)
    # Count positives and negatives along the mask dimension
    n_pos = np.sum((sign > 0) & valid, axis=0)
    n_neg = np.sum((sign < 0) & valid, axis=0)
    n_tot = np.sum(valid, axis=0)

    dir_consistency = np.zeros_like(Mn_mean, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        # Defined as max(n_pos, n_neg) / n_tot, range [0,1]
        dir_consistency = np.maximum(n_pos, n_neg) / n_tot
        dir_consistency[~np.isfinite(dir_consistency)] = np.nan

    # Threshold exceedance probability P(|Mn| > T)
    prob_exceed = {}
    abs_Mn = np.abs(Mn_stack)
    for T in thresholds:
        with np.errstate(divide="ignore", invalid="ignore"):
            n_exceed = np.sum((abs_Mn > T) & valid, axis=0)
            p = n_exceed / n_tot
            p[~np.isfinite(p)] = np.nan
        prob_exceed[T] = p

    return {
        "cv": cv,
        "dir_consistency": dir_consistency,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_tot": n_tot,
        "prob_exceed": prob_exceed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse multi-mask centerline Mn profiles and quantify uncertainty metrics.",
    )
    parser.add_argument("--site", default="Jurua-A", help="Site name, e.g., 'Jurua-A'")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[5.0],
        help="Migration rate threshold list (m/yr), used to compute exceedance probability, e.g., 5 10",
    )

    args = parser.parse_args()

    mask_levels, s, Mn_stack, Mn_mean, Mn_std = load_multimask_profiles(args.site)
    metrics = compute_metrics(Mn_stack, Mn_mean, Mn_std, thresholds=tuple(args.thresholds))

    cv = metrics["cv"]
    dir_consistency = metrics["dir_consistency"]
    prob_exceed = metrics["prob_exceed"]

    out_dir = get_postprocessed_dir(PROJECT_ROOT, args.site)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save numerical results for later statistics or comparison with Yellow River sites
    out_npz = out_dir / "jurua_multimask_multitilt_centerline_uncertainty_metrics.npz"
    np.savez(
        out_npz,
        mask_levels=mask_levels,
        s=s,
        Mn_stack=Mn_stack,
        Mn_mean=Mn_mean,
        Mn_std=Mn_std,
        cv=cv,
        dir_consistency=dir_consistency,
        **{f"prob_exceed_T{T:g}": prob_exceed[T] for T in prob_exceed},
    )
    print("Saved multi-mask uncertainty metrics to:", out_npz)

    out_root = describe_output_root(PROJECT_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)

    s_km = s / 1000.0

    # To avoid physically meaningless extreme CV spikes when |Mn_mean| is very small,
    # we mask CV for sections where |Mn_mean| < 0.5 m/yr during visualization (for plotting only).
    mean_abs = np.abs(Mn_mean)
    cv_plot = cv.copy()
    cv_plot[(~np.isfinite(mean_abs)) | (mean_abs < 0.5)] = np.nan

    # Figure 1: sigma and CV profiles
    out_png_sigma_cv = out_root / "jurua_multimask_multitilt_centerline_sigma_cv.png"
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(s_km, Mn_std, "C0-", label="σ(Mn) across masks")
    ax1.set_xlabel("Centerline distance s (km)")
    ax1.set_ylabel("Std of Mn (m/yr)", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")

    ax2 = ax1.twinx()
    ax2.plot(s_km, cv_plot, "C1-", label="CV(Mn), |mean|≥0.5 m/yr")
    ax2.set_ylabel("CV = σ/|mean| (masked where |mean|<0.5)", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")

    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"{args.site} multi-mask Mn uncertainty: σ and CV")

    fig.tight_layout()
    fig.savefig(out_png_sigma_cv, dpi=200)
    plt.close(fig)
    print("Saved sigma / CV profile to:", out_png_sigma_cv)

    # Figure 2: Directional consistency and threshold exceedance probability
    out_png_dir_prob = out_root / "jurua_multimask_multitilt_centerline_dir_prob.png"
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(s_km, dir_consistency, "k-", label="Directional consistency")

    for T, p in prob_exceed.items():
        ax.plot(s_km, p, label=f"P(|Mn| > {T:g} m/yr)")

    ax.set_xlabel("Centerline distance s (km)")
    ax.set_ylabel("Probability / consistency")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{args.site} multi-mask Mn: direction & exceedance")
    ax.legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png_dir_prob, dpi=200)
    plt.close(fig)
    print("Saved directional consistency and exceedance probability plot to:", out_png_dir_prob)


if __name__ == "__main__":
    main()
