"""Jurua-A Mask1 multi-tilt time series PIV example:

- Performs time series PIV + time statistics for each of Mask1_Tilt00 / 15 / 30 / 45;
- Retilts each tilt's time-averaged vector field back to a unified coordinate system in vector space;
- Treats the 4 tilts as 4 realizations and performs compute_vector_stats fusion on their **common intersection grid**;
- Outputs all / filtered preferred vector field plots.

Usage (from project root):
    python -m src.piv_analysis.jurua_multitilt
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import argparse
import numpy as np

from src.config import PROJECT_ROOT
from src.postprocessing.postprocess import compute_vector_stats, retilt_vectors
from src.piv_analysis.jurua_timeseries import run_timeseries_jurua
from src.visualization.quicklook import describe_output_root


TILTS_DEG = (0, 15, 30, 45)


def _center_crop(arr: np.ndarray, ny_target: int, nx_target: int) -> np.ndarray:
    """Center-crops a 2D array to a given size, used for aligning grids of different tilts.

    Assumes:
    - PIV grids of different tilts are symmetrically expanded around the image center (from preprocessing padding);
    - Therefore, taking the central sub-block of each grid as the common intersection region for the 4 tilts is a reasonable approximation.
    """

    ny, nx = arr.shape
    if ny < ny_target or nx < nx_target:
        raise RuntimeError("Target crop size larger than original array, cannot center-crop")

    sy = (ny - ny_target) // 2
    sx = (nx - nx_target) // 2
    return arr[sy : sy + ny_target, sx : sx + nx_target]


def run_multitilt_jurua(
    site: str = "Jurua-A",
    mask_level: int = 1,
    tilt_degs: Tuple[int, ...] = TILTS_DEG,
):
    """Fuses time-averaged vector fields from multiple tilts under a single mask level.

    Process:
    - Calls run_timeseries_jurua for each tilt_deg to obtain time-averaged field for that tilt;
    - Performs retilt in vector space with phi = -tilt_deg;
    - Computes grid sizes for all tilts, finds the common minimum grid, and center-crops all fields and reference coordinates;
    - Calls compute_vector_stats on this common intersection grid for multi-realization fusion.

    Returns:
    x, y        : common grid coordinates
    stats_tilt  : fused statistics dictionary (same as compute_vector_stats)
    per_tilt    : information for each tilt: {"u_mean", "v_mean", "pairs", "phi_data"}
    """

    # First loop: perform time series PIV + retilt for each tilt, but don't stack yet; record shapes
    raw: List[Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Tuple[str, str]], float]] = []
    ny_list: List[int] = []
    nx_list: List[int] = []

    for tilt_deg in tilt_degs:
        x, y, stats_time, in_dir, pairs = run_timeseries_jurua(
            site=site,
            mask_level=mask_level,
            tilt_deg=tilt_deg,
        )

        u_mean = stats_time["u_mean"]
        v_mean = stats_time["v_mean"]

        # During image preprocessing, phi = -tilt_deg was used for rotation; here we do retilt in vector space with the same angle
        phi_data = -float(tilt_deg)
        u_untilt, v_untilt = retilt_vectors(u_mean, v_mean, phi_deg=phi_data)

        ny, nx = u_untilt.shape
        ny_list.append(ny)
        nx_list.append(nx)

        raw.append((int(tilt_deg), x, y, u_untilt, v_untilt, pairs, phi_data))

    # Common intersection grid size = minimum ny/nx among all tilts
    ny_min = min(ny_list)
    nx_min = min(nx_list)

    # Reference coordinates: use the first tilt's x, y, and do the same center crop
    first_tilt, x0, y0, _, _, _, _ = raw[0]
    x_ref = _center_crop(x0, ny_min, nx_min)
    y_ref = _center_crop(y0, ny_min, nx_min)

    # Second pass: center-crop and stack retilted vectors from each tilt
    u_fields: List[np.ndarray] = []
    v_fields: List[np.ndarray] = []
    per_tilt: Dict[int, Dict[str, object]] = {}

    for tilt_deg, x, y, u_untilt, v_untilt, pairs, phi_data in raw:
        u_c = _center_crop(u_untilt, ny_min, nx_min)
        v_c = _center_crop(v_untilt, ny_min, nx_min)

        u_fields.append(u_c)
        v_fields.append(v_c)

        per_tilt[tilt_deg] = {
            "u_mean": u_c,
            "v_mean": v_c,
            "pairs": pairs,
            "phi_data": phi_data,
            "shape": u_c.shape,
        }

    u_stack = np.stack(u_fields, axis=0)
    v_stack = np.stack(v_fields, axis=0)

    stats_tilt = compute_vector_stats(u_stack, v_stack)

    return x_ref, y_ref, stats_tilt, per_tilt


def main() -> None:
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(
        description="Fuse multi-tilt time-averaged PIV fields for a given site/mask.",
    )
    parser.add_argument("--site", default="Jurua-A", help="Site name, e.g., 'Jurua-A'")
    parser.add_argument("--mask-level", type=int, default=1, help="Mask level, e.g., 1 for Mask1")
    parser.add_argument(
        "--tilts",
        type=int,
        nargs="+",
        default=list(TILTS_DEG),
        help="List of tilts to fuse, e.g., 0 15 30 45",
    )

    args = parser.parse_args()

    tilt_tuple = tuple(int(t) for t in args.tilts)

    x, y, stats_tilt, per_tilt = run_multitilt_jurua(
        site=args.site,
        mask_level=args.mask_level,
        tilt_degs=tilt_tuple,
    )

    u_mean = stats_tilt["u_mean"]
    v_mean = stats_tilt["v_mean"]
    bad = stats_tilt["bad_mask"]

    print("Tilts involved:", sorted(per_tilt.keys()))
    for tilt in sorted(per_tilt.keys()):
        info = per_tilt[tilt]
        print(
            f"  Tilt{tilt:02d}: Intersection grid shape = {info['shape']}, "
            f"Number of valid time pairs = {len(info['pairs'])}, Sample pairs = {info['pairs'][:2]}"
        )

    print("Fused u_mean / v_mean shape:", u_mean.shape)
    print("Fused grid points with at least 1 valid sample:", int(np.isfinite(u_mean).sum()))
    print("Fused bad_mask grid points:", int(bad.sum()))

    out_root = describe_output_root(PROJECT_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)

    # Plot 1: Multi-tilt fused time-averaged vector field (unfiltered)
    out_png_all = out_root / f"jurua_multitilt_mask{args.mask_level}_all.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    mag = np.hypot(u_mean, v_mean)
    q = ax.quiver(x, y, u_mean, v_mean, mag, cmap="viridis", scale=50)
    plt.colorbar(q, ax=ax, label="|v_mean| (multi-tilt, pixel/yr)")
    ax.set_aspect("equal")
    ax.set_title(f"{args.site} Mask{args.mask_level} multi-tilt preferred PIV (all)")
    fig.savefig(out_png_all, dpi=200)
    plt.close(fig)

    # Plot 2: Multi-tilt fused vector field after applying uncertainty mask
    out_png_filt = out_root / f"jurua_multitilt_mask{args.mask_level}_filtered.png"
    u_f = u_mean.copy()
    v_f = v_mean.copy()
    u_f[bad] = np.nan
    v_f[bad] = np.nan

    fig, ax = plt.subplots(figsize=(8, 6))
    mag_f = np.hypot(u_f, v_f)
    q = ax.quiver(x, y, u_f, v_f, mag_f, cmap="viridis", scale=50)
    plt.colorbar(q, ax=ax, label="|v_mean| (multi-tilt, filtered, pixel/yr)")
    ax.set_aspect("equal")
    ax.set_title(f"{args.site} Mask{args.mask_level} multi-tilt preferred PIV (filtered)")
    fig.savefig(out_png_filt, dpi=200)
    plt.close(fig)

    print("Saved multi-tilt fused vector field (all) to:", out_png_all)
    print("Saved multi-tilt fused vector field (filtered) to:", out_png_filt)


if __name__ == "__main__":
    main()
