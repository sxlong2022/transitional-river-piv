"""Jurua-A time series PIV example: Iterates over multiple years under Mask1_Tilt00 and outputs time statistics.

Assumptions:
- Data_and_code/Data/PreparedImagery/Jurua-A/Mask1_Tilt00 is extracted;
- Filenames follow the pattern `Jurua-A_YYYY_01-01_12-31_mask_thresh.tif`;
- Dependencies such as openpiv, rasterio, matplotlib are installed in the conda environment.

Usage (from project root):
    python -m src.piv_analysis.jurua_timeseries
"""

from pathlib import Path
from typing import List, Tuple

import argparse
import numpy as np
import rasterio
from openpiv import pyprocess, validation, filters

from src.config import PROJECT_ROOT
from src.postprocessing.postprocess import compute_vector_stats
from src.preprocessing.prepared_imagery import get_prepared_imagery_dir
from src.visualization.quicklook import describe_output_root


def run_timeseries_jurua(
    site: str = "Jurua-A",
    mask_level: int = 1,
    tilt_deg: int = 0,
    window_size: int = 64,
    overlap: int = 32,
):
    """Performs PIV on all years for a given site/mask level/tilt and computes time statistics.

    Returns:
    x, y : grid coordinates
    stats : statistics result dictionary from compute_vector_stats
    in_dir : input directory Path
    pairs : list of (file_a_name, file_b_name) that actually participated in computation
    """

    base = get_prepared_imagery_dir(site)
    tilt_str = f"Mask{mask_level}_Tilt{abs(tilt_deg):02d}"
    in_dir = base / tilt_str

    # Only use *_mask_thresh.tif files to avoid interference from Color / Removed* etc.
    tifs = sorted(in_dir.glob("*mask_thresh.tif"))
    if len(tifs) < 2:
        raise RuntimeError(f"Not enough tif files in {in_dir}")

    u_fields: List[np.ndarray] = []
    v_fields: List[np.ndarray] = []
    x = y = None
    pairs: List[Tuple[str, str]] = []

    for i in range(len(tifs) - 1):
        f_a = tifs[i]
        f_b = tifs[i + 1]
        pairs.append((f_a.name, f_b.name))

        with rasterio.open(f_a) as src:
            frame_a = src.read(1)
        with rasterio.open(f_b) as src:
            frame_b = src.read(1)

        u, v, sig2noise = pyprocess.extended_search_area_piv(
            frame_a.astype(np.int32),
            frame_b.astype(np.int32),
            window_size=window_size,
            overlap=overlap,
            dt=1.0,
            search_area_size=window_size,
            sig2noise_method="peak2peak",
        )

        _x, _y = pyprocess.get_coordinates(
            image_size=frame_a.shape,
            search_area_size=window_size,
            overlap=overlap,
        )

        # Simple quality control
        u, v, mask = validation.sig2noise_val(u, v, sig2noise, threshold=1.3)
        u, v = filters.replace_outliers(u, v, method="localmean")

        if x is None:
            x, y = _x, _y

        u_fields.append(u.astype(float))
        v_fields.append(v.astype(float))

    u_stack = np.stack(u_fields, axis=0)
    v_stack = np.stack(v_fields, axis=0)

    # Use relaxed default thresholds for statistics: sigma_n_factor=2, theta_std_deg=120, min_samples=2
    stats = compute_vector_stats(u_stack, v_stack)

    return x, y, stats, in_dir, pairs


def main() -> None:
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(
        description="Run time-series PIV for a single site/mask/tilt (Jurua example).",
    )
    parser.add_argument("--site", default="Jurua-A", help="Site name, e.g., 'Jurua-A'")
    parser.add_argument("--mask-level", type=int, default=1, help="Mask level, e.g., 1 for Mask1")
    parser.add_argument("--tilt-deg", type=int, default=0, help="Tilt angle (degrees), e.g., 0/15/30/45")
    parser.add_argument("--window-size", type=int, default=64, help="PIV window size (pixels)")
    parser.add_argument("--overlap", type=int, default=32, help="PIV window overlap (pixels)")

    args = parser.parse_args()

    x, y, stats, in_dir, pairs = run_timeseries_jurua(
        site=args.site,
        mask_level=args.mask_level,
        tilt_deg=args.tilt_deg,
        window_size=args.window_size,
        overlap=args.overlap,
    )

    u_mean = stats["u_mean"]
    v_mean = stats["v_mean"]
    bad = stats["bad_mask"]

    print("PIV input directory:", in_dir)
    print("Site / Mask / Tilt:", args.site, f"Mask{args.mask_level}", f"Tilt{args.tilt_deg:02d}")
    print("Number of image pairs for computation:", len(pairs))
    if pairs:
        print("First 5 image pairs:")
        for a, b in pairs[:5]:
            print("  ", a, "→", b)
    print("u_mean / v_mean shape:", u_mean.shape)
    print("Grid points with at least 1 valid sample:", int(np.isfinite(u_mean).sum()))
    print("Grid points marked as bad in uncertainty mask:", int(bad.sum()))

    out_root = describe_output_root(PROJECT_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)

    # Plot 1: Time-averaged vector field (unfiltered)
    out_png_all = out_root / "jurua_timeseries_piv_all.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    mag = np.hypot(u_mean, v_mean)
    q = ax.quiver(x, y, u_mean, v_mean, mag, cmap="viridis", scale=50)
    plt.colorbar(q, ax=ax, label="|v_mean| (pixel/yr)")
    ax.set_aspect("equal")
    ax.set_title("Jurua-A time-averaged PIV (Mask1_Tilt00, all)")
    fig.savefig(out_png_all, dpi=200)
    plt.close(fig)

    # Plot 2: Time-averaged vector field after applying uncertainty mask
    out_png_filt = out_root / "jurua_timeseries_piv_filtered.png"
    u_f = u_mean.copy()
    v_f = v_mean.copy()
    u_f[bad] = np.nan
    v_f[bad] = np.nan

    fig, ax = plt.subplots(figsize=(8, 6))
    mag_f = np.hypot(u_f, v_f)
    q = ax.quiver(x, y, u_f, v_f, mag_f, cmap="viridis", scale=50)
    plt.colorbar(q, ax=ax, label="|v_mean| (filtered, pixel/yr)")
    ax.set_aspect("equal")
    ax.set_title("Jurua-A time-averaged PIV (Mask1_Tilt00, filtered)")
    fig.savefig(out_png_filt, dpi=200)
    plt.close(fig)

    print("Saved time-averaged vector field (all) to:", out_png_all)
    print("Saved time-averaged vector field (filtered) to:", out_png_filt)


if __name__ == "__main__":
    main()
