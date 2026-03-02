"""Step 4A: Georeferencing and physical unit conversion for Jurua-A Mask1 multi-tilt preferred vector field.

Functions:
- Calls `run_multitilt_jurua` to obtain multi-tilt preferred PIV field (pixel grid / pixel per PIV-grid-year);
- Reads a georeferenced mask GeoTIFF from `Data/GEOTIFFS/Jurua-A/mask{mask_level}`;
- Linearly stretches PIV grid (x_pix, y_pix) to the GeoTIFF's row/column extent to obtain (col, row);
- Uses affine transformation to strictly map small displacements (Δcol, Δrow) to geographic displacements (ΔX, ΔY);
- Converts velocity from pixel/yr to m/yr (based on average year interval inferred from filenames);
- Saves results as .npz and outputs a physical unit vector plot.

Usage (from project root):
    python -m src.morphodynamics.jurua_georef_multitilt

Notes:
- Assumes the multi-tilt PIV grid covers the main river segment of the reference mask,
  so linear scaling can be used to map PIV grid to GeoTIFF row/column extent;
- Compared to the previous "approx m/yr, using √(a^2+e^2)" version, this script
  uses the full affine matrix to convert (u_pix, v_pix) → (vx_m, vy_m), which is more rigorous.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import argparse
import numpy as np
import rasterio

from src.config import DATA_DIR, PROJECT_ROOT
from src.morphodynamics.coupling import project_velocity_on_normal
from src.piv_analysis.jurua_multitilt import run_multitilt_jurua
from src.postprocessing.postprocess import get_postprocessed_dir
from src.preprocessing.prepared_imagery import get_geotiffs_dir
from src.visualization.quicklook import describe_output_root


def _choose_reference_mask(site: str, mask_level: int, year: int | None = None) -> Path:
    """Selects a georeferenced mask GeoTIFF as the reference raster.

    Defaults to preferring the given year (e.g., 1987), falling back to the first file in the directory if not found.
    Supports both Jurua-A and HuangHe-A/B site paths.
    """

    root = get_geotiffs_dir(site) / f"mask{mask_level}"
    if not root.exists():
        raise FileNotFoundError(f"Reference mask directory does not exist: {root}")

    if year is not None:
        # Try multiple possible filename formats
        candidates = [
            root / f"{site}_{year}_01-01_12-31_mask.tif",
            root / f"{site}_{year}_01-01_12-31_mask{mask_level}.tif",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

    tifs = sorted(root.glob("*.tif"))
    if not tifs:
        raise FileNotFoundError(f"No .tif mask files found under {root}")

    return tifs[0]


def _infer_mean_dt_years(pairs: list[Tuple[str, str]]) -> float:
    """Infers average year interval from filenames, e.g., 'Jurua-A_1987_01-01_12-31_mask_thresh.tif'."""

    dts = []
    for a, b in pairs:
        try:
            ya = int(a.split("_")[1])
            yb = int(b.split("_")[1])
            dts.append(yb - ya)
        except Exception:
            continue
    if not dts:
        return 1.0
    return float(np.mean(dts))


def _meters_per_degree(lat_deg: float) -> Tuple[float, float]:
    lat_rad = np.deg2rad(float(lat_deg))
    m_per_deg_lat = (
        111132.92
        - 559.82 * np.cos(2 * lat_rad)
        + 1.175 * np.cos(4 * lat_rad)
        - 0.0023 * np.cos(6 * lat_rad)
    )
    m_per_deg_lon = (
        111412.84 * np.cos(lat_rad)
        - 93.5 * np.cos(3 * lat_rad)
        + 0.118 * np.cos(5 * lat_rad)
    )
    return float(m_per_deg_lon), float(m_per_deg_lat)


def georef_multitilt_jurua(
    site: str = "Jurua-A",
    mask_level: int = 1,
    ref_year: int | None = 1987,
) -> dict:
    """Performs georeferencing and m/yr unit conversion for Jurua-A Mask1 multi-tilt preferred vector field.

    Returns a dict containing several arrays with main keys:
    - X_geo, Y_geo : geographic coordinates (same shape as PIV grid)
    - u_m_per_year, v_m_per_year : m/yr vector components strictly converted via affine matrix
    - u_pix, v_pix : vector components in original pixel grid coordinates (pixel per PIV-grid-year)
    - dt_mean_years: average year interval estimated from filenames
    - transform, crs, ref_path: reference GeoTIFF information
    """

    # 1. Multi-tilt preferred field (pixel grid coordinates + "pixel per PIV-grid-year")
    x_pix, y_pix, stats_tilt, per_tilt = run_multitilt_jurua(site=site, mask_level=mask_level)

    u_pix = stats_tilt["u_mean"]
    v_pix = stats_tilt["v_mean"]

    # Use Tilt00 pair info to estimate average year interval
    pairs_tilt0: list[Tuple[str, str]] = per_tilt[0]["pairs"]  # type: ignore[index]
    dt_mean_years = _infer_mean_dt_years(pairs_tilt0)

    # 2. Select reference mask and read affine transform
    ref_path = _choose_reference_mask(site=site, mask_level=mask_level, year=ref_year)
    with rasterio.open(ref_path) as src:
        transform = src.transform
        crs = src.crs
        height, width = src.height, src.width

    ny, nx = u_pix.shape

    # 3. Linearly stretch PIV grid (x_pix, y_pix) to GeoTIFF row/column extent
    x_min, x_max = float(x_pix.min()), float(x_pix.max())
    y_min, y_max = float(y_pix.min()), float(y_pix.max())

    if x_max == x_min or y_max == y_min:
        raise RuntimeError("PIV grid coordinate range is zero, cannot perform linear mapping")

    # Normalize to [0, width-1] / [0, height-1]
    col = (x_pix - x_min) / (x_max - x_min) * (width - 1)
    row = (y_pix - y_min) / (y_max - y_min) * (height - 1)

    # 4. Affine transform to geographic coordinates (X, Y)
    #   rasterio.transform convention: x = a*col + b*row + c; y = d*col + e*row + f
    a, b, c, d, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
    X_geo = a * col + b * row + c
    Y_geo = d * col + e * row + f

    # 5. Use full affine matrix to convert (u_pix, v_pix) → (ΔX, ΔY)
    #   Steps:
    #   - Linear relationship from (x_pix, y_pix) → (col, row):
    #       col = kx * x_pix + const
    #       row = ky * y_pix + const
    #   - Small displacements (u_pix, v_pix) in PIV grid correspond to GeoTIFF rows/cols:
    #       Δcol = kx * u_pix
    #       Δrow = ky * v_pix
    #   - Then use affine matrix:
    #       ΔX = a*Δcol + b*Δrow
    #       ΔY = d*Δcol + e*Δrow

    kx = (width - 1) / (x_max - x_min)
    ky = (height - 1) / (y_max - y_min)

    dcol = kx * u_pix
    drow = ky * v_pix

    dX = a * dcol + b * drow
    dY = d * dcol + e * drow

    if dt_mean_years <= 0:
        dt_mean_years = 1.0

    # Strictly normalize output velocity to m/yr; if reference raster is geographic (degrees), do local approximation
    is_geographic = bool(crs is not None and getattr(crs, "is_geographic", False))
    if is_geographic:
        lon0 = float(np.nanmean(X_geo))
        lat0 = float(np.nanmean(Y_geo))
        m_per_deg_lon0, m_per_deg_lat0 = _meters_per_degree(lat0)
        X_m = (X_geo - lon0) * m_per_deg_lon0
        Y_m = (Y_geo - lat0) * m_per_deg_lat0
        dX_m = dX * m_per_deg_lon0
        dY_m = dY * m_per_deg_lat0
        u_m_per_year = dX_m / dt_mean_years
        v_m_per_year = dY_m / dt_mean_years
    else:
        lon0 = float("nan")
        lat0 = float("nan")
        m_per_deg_lon0 = float("nan")
        m_per_deg_lat0 = float("nan")
        X_m = X_geo
        Y_m = Y_geo
        u_m_per_year = dX / dt_mean_years
        v_m_per_year = dY / dt_mean_years

    return {
        "X_geo": X_geo,
        "Y_geo": Y_geo,
        "X_m": X_m,
        "Y_m": Y_m,
        "u_m_per_year": u_m_per_year,
        "v_m_per_year": v_m_per_year,
        "u_pix": u_pix,
        "v_pix": v_pix,
        "dt_mean_years": dt_mean_years,
        "is_geographic": bool(is_geographic),
        "lon0": lon0,
        "lat0": lat0,
        "m_per_deg_lon0": m_per_deg_lon0,
        "m_per_deg_lat0": m_per_deg_lat0,
        "crs": crs,
        "transform": transform,
        "ref_path": ref_path,
    }


def main() -> None:
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(
        description="Step 4A: georeference multi-tilt preferred PIV field and convert to m/yr (strict affine).",
    )
    parser.add_argument("--site", default="Jurua-A", help="Site name, e.g., 'Jurua-A'")
    parser.add_argument("--mask-level", type=int, default=1, help="Mask level, e.g., 1 for Mask1")
    parser.add_argument("--ref-year", type=int, default=None, help="Preferred year for reference mask (optional); if omitted, auto-selects first available file")

    args = parser.parse_args()

    result = georef_multitilt_jurua(site=args.site, mask_level=args.mask_level, ref_year=args.ref_year)

    X_geo = result["X_geo"]
    Y_geo = result["Y_geo"]
    u_m_per_year = result["u_m_per_year"]
    v_m_per_year = result["v_m_per_year"]
    dt_mean_years = result["dt_mean_years"]
    ref_path = result["ref_path"]

    print("Reference mask:", ref_path)
    print("Estimated average year interval dt_mean_years:", dt_mean_years)

    # Save npz results for subsequent analysis or RivGraph coupling
    out_dir = get_postprocessed_dir(PROJECT_ROOT, args.site)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / f"jurua_mask{args.mask_level}_multitilt_georef_step4a_strict.npz"

    np.savez(
        out_npz,
        X_geo=X_geo,
        Y_geo=Y_geo,
        X_m=result["X_m"],
        Y_m=result["Y_m"],
        u_m_per_year=u_m_per_year,
        v_m_per_year=v_m_per_year,
        dt_mean_years=dt_mean_years,
        is_geographic=bool(result.get("is_geographic", False)),
        lon0=float(result.get("lon0", float("nan"))),
        lat0=float(result.get("lat0", float("nan"))),
        m_per_deg_lon0=float(result.get("m_per_deg_lon0", float("nan"))),
        m_per_deg_lat0=float(result.get("m_per_deg_lat0", float("nan"))),
    )

    print("Saved strict m/yr vector field to:", out_npz)

    # Plot physical unit vector field (m/yr)
    out_root = describe_output_root(PROJECT_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)
    out_png = out_root / f"jurua_multitilt_mask{args.mask_level}_georef_m_per_year_strict.png"

    fig, ax = plt.subplots(figsize=(8, 6))
    mag = np.hypot(u_m_per_year, v_m_per_year)
    q = ax.quiver(X_geo, Y_geo, u_m_per_year, v_m_per_year, mag, cmap="viridis", scale=50_000)
    plt.colorbar(q, ax=ax, label="|v| (m/yr, strict affine)")
    ax.set_aspect("equal")
    ax.set_title(f"{args.site} Mask{args.mask_level} multi-tilt preferred PIV (m/yr, Step 4A strict)")
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    print("Saved physical unit vector plot to:", out_png)


if __name__ == "__main__":
    main()
