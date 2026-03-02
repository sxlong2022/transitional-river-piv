"""
Preprocessing script to generate PreparedImagery from GEOTIFFS.

Functions:
1. Read raw DSWE masks from data/GEOTIFFS/{site}/mask{level}/
2. Select sparse year sequence (aligned with Jurua-A)
3. Threshold masks (ensure binary)
4. Create subdirectories for multiple tilts (0°, 15°, 30°, 45°)
5. Apply image rotation for non-zero tilts

Output structure:
    data/PreparedImagery/{site}/
        ├── Mask{level}_Tilt00/
        │   ├── {site}_{year}_01-01_12-31_mask_thresh.tif
        │   └── ...
        ├── Mask{level}_Tilt15/
        ├── Mask{level}_Tilt30/
        └── Mask{level}_Tilt45/

Usage:
    python -m src.preprocessing.generate_prepared_imagery --site HuangHe-A --mask-level 2

Dependencies:
    - rasterio
    - scipy (for image rotation)
    - numpy
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import rasterio
from rasterio.transform import Affine
from scipy.ndimage import rotate as ndimage_rotate

# =====================================================================================
# === Configuration ============================================================================
# =====================================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

# Tilt angles (consistent with Jurua-A)
TILT_ANGLES = (0, 15, 30, 45)

# Sparse years used for Jurua-A (one every 3-4 years, total 10 years)
JURUA_YEAR_PATTERN = [1987, 1990, 1993, 1997, 2001, 2005, 2009, 2013, 2017, 2021]

# =====================================================================================
# Recommended years for Yellow River sites (based on historical evolution context)
# =====================================================================================

# HuangHe-A (Zhengzhou segment): Focus on changes around Xiaolangdi
# - 1986-1997: Dry period
# - 1999: Xiaolangdi operation (key turning point)
# - 2002: Water and sediment regulation begins
HUANGHE_A_YEARS = [
    1986,  # Landsat 5 start
    1990,  # Pre-mid dry period
    1994,  # Dry period
    1997,  # Most severe dry year
    1999,  # Xiaolangdi operation (key!)
    2002,  # Water and sediment regulation begins
    2005,
    2010,
    2015,
    2020,
    2024,  # Latest
]

# HuangHe-B (Ningmeng segment): Relatively uniform sampling after Longyangxia
HUANGHE_B_YEARS = [
    1986,  # Longyangxia just started
    1990,
    1995,
    2000,
    2005,
    2010,
    2015,
    2020,
    2024,
]

# Site to recommended years mapping
SITE_YEAR_PATTERNS = {
    "Jurua-A": JURUA_YEAR_PATTERN,
    "HuangHe-A": HUANGHE_A_YEARS,
    "HuangHe-B": HUANGHE_B_YEARS,
}


def select_sparse_years(
    available_years: List[int],
    target_count: int = 10,
    min_interval: int = 3,
) -> List[int]:
    """Selects sparse time series from available years.

    Strategy:
    - Try to select at equal intervals, approximately (end - start) / (target_count - 1)
    - Ensure first and last years are included
    - Interval not less than min_interval
    """
    if len(available_years) <= target_count:
        return available_years

    available_years = sorted(available_years)
    start, end = available_years[0], available_years[-1]
    span = end - start

    # Calculate ideal interval
    ideal_interval = span / (target_count - 1) if target_count > 1 else span
    interval = max(ideal_interval, min_interval)

    selected = [start]
    current = start

    while len(selected) < target_count - 1:
        target_year = current + interval
        # Find the available year closest to target_year
        closest = min(
            [y for y in available_years if y > current],
            key=lambda y: abs(y - target_year),
            default=None,
        )
        if closest is None:
            break
        selected.append(closest)
        current = closest

    # Ensure last year is included
    if end not in selected:
        selected.append(end)

    return sorted(set(selected))


def threshold_mask(arr: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Thresholds mask to ensure binary.

    Original DSWE masks can be 0/1 integers or 0.0-1.0 floats.
    Output: uint8 type, 0 or 255 (consistent with Jurua-A's mask_thresh).
    """
    binary = (arr > threshold).astype(np.uint8) * 255
    return binary


def rotate_image_with_padding(
    arr: np.ndarray,
    angle_deg: float,
    fill_value: int = 0,
) -> np.ndarray:
    """Rotates image and pads edges.

    Consistent with Tilt preprocessing in the paper:
    - Positive angle means counter-clockwise rotation
    - Rotated image size may change
    - Edges are padded with fill_value
    """
    if angle_deg == 0:
        return arr.copy()

    # scipy.ndimage.rotate: reshape=True expands image to accommodate rotated content
    rotated = ndimage_rotate(
        arr,
        angle=-angle_deg,  # Negative sign because we want positive angle to be counter-clockwise
        reshape=True,
        order=0,  # Nearest neighbor interpolation, preserves binary nature
        mode="constant",
        cval=fill_value,
    )
    return rotated.astype(arr.dtype)


def process_single_tif(
    src_path: Path,
    dst_path: Path,
    tilt_deg: int,
) -> Tuple[bool, str]:
    """Processes a single GeoTIFF file.

    Returns: (success, message)
    """
    try:
        with rasterio.open(src_path) as src:
            arr = src.read(1)
            profile = src.profile.copy()
            original_transform = src.transform

        # 1. Threshold
        arr_thresh = threshold_mask(arr)

        # 2. Rotate if needed
        if tilt_deg != 0:
            arr_out = rotate_image_with_padding(arr_thresh, tilt_deg)
            # Update dimensions in profile
            new_h, new_w = arr_out.shape
            profile.update(
                height=new_h,
                width=new_w,
                # After rotation, transform becomes complex; simplify: keep original resolution
                # Actual georef will be done in Step 4A via affine alignment
            )
        else:
            arr_out = arr_thresh

        # Update profile
        profile.update(dtype=rasterio.uint8, count=1)

        # Write output
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(arr_out, 1)

        return True, f"OK -> {dst_path.name}"

    except Exception as e:
        return False, f"ERROR: {e}"


def generate_prepared_imagery(
    site: str,
    mask_level: int = 2,
    tilt_angles: Tuple[int, ...] = TILT_ANGLES,
    year_selection: str = "auto",
    target_years: int = 10,
):
    """Generates PreparedImagery for the specified site.

    Parameters
    ----
    site : str
        Site name, e.g., "HuangHe-A"
    mask_level : int
        DSWE mask level (1-4)
    tilt_angles : tuple
        List of tilts to generate
    year_selection : str
        "auto" = automatic sparse selection
        "all" = use all available years
        "jurua" = use same years as Jurua-A (if available)
    target_years : int
        Target number of years (only effective when year_selection="auto")
    """
    # Input directory
    geotiff_dir = DATA_ROOT / "GEOTIFFS" / site / f"mask{mask_level}"
    if not geotiff_dir.exists():
        raise FileNotFoundError(f"GEOTIFFS directory does not exist: {geotiff_dir}")

    # Scan available tif files
    # Filename format: {site}_{year}_01-01_12-31_mask{level}.tif or {site}_{year}_01-01_12-31_mask.tif
    tifs = sorted(geotiff_dir.glob("*.tif"))
    if not tifs:
        raise FileNotFoundError(f"No GEOTIFF files found: {geotiff_dir}")

    # Parse years
    year_to_tif = {}
    for tif in tifs:
        # Try to extract year from filename
        name = tif.stem
        parts = name.split("_")
        for part in parts:
            if part.isdigit() and len(part) == 4:
                year = int(part)
                year_to_tif[year] = tif
                break

    available_years = sorted(year_to_tif.keys())
    print(f"Found {len(available_years)} years: {available_years[0]} - {available_years[-1]}")

    # Select years
    if year_selection == "all":
        selected_years = available_years
    elif year_selection == "site":
        # Use site-specific recommended years (based on historical context)
        if site in SITE_YEAR_PATTERNS:
            pattern = SITE_YEAR_PATTERNS[site]
            selected_years = [y for y in pattern if y in available_years]
            missing = [y for y in pattern if y not in available_years]
            if missing:
                print(f"Note: The following recommended years are not available: {missing}")
        else:
            print(f"Warning: Site '{site}' has no preset year pattern, using automatic selection")
            selected_years = select_sparse_years(available_years, target_years)
    elif year_selection == "jurua":
        # Backward compatibility
        selected_years = [y for y in JURUA_YEAR_PATTERN if y in available_years]
        if not selected_years:
            print("Warning: No years matching Jurua-A, will use automatic selection")
            selected_years = select_sparse_years(available_years, target_years)
    else:  # auto
        selected_years = select_sparse_years(available_years, target_years)

    print(f"Selected {len(selected_years)} years: {selected_years}")

    # Output root directory
    prepared_root = DATA_ROOT / "PreparedImagery" / site

    # Process each tilt
    for tilt_deg in tilt_angles:
        tilt_dir = prepared_root / f"Mask{mask_level}_Tilt{abs(tilt_deg):02d}"
        print(f"\n{'=' * 60}")
        print(f"Processing Tilt = {tilt_deg}° -> {tilt_dir}")
        print(f"{'=' * 60}")

        for year in selected_years:
            src_tif = year_to_tif[year]
            # Output filename: {site}_{year}_01-01_12-31_mask_thresh.tif
            dst_name = f"{site}_{year}_01-01_12-31_mask_thresh.tif"
            dst_tif = tilt_dir / dst_name

            success, msg = process_single_tif(src_tif, dst_tif, tilt_deg)
            status = "✓" if success else "✗"
            print(f"  {status} {year}: {msg}")

    print(f"\n{'=' * 60}")
    print(f"Done! PreparedImagery saved to: {prepared_root}")
    print(f"{'=' * 60}")

    return prepared_root


def main():
    parser = argparse.ArgumentParser(
        description="Generate PreparedImagery from GEOTIFFS (for PIV analysis)",
        epilog="""
Examples:
    # Generate mask2 PreparedImagery for HuangHe-A (using site-specific recommended years, default)
    python -m src.preprocessing.generate_prepared_imagery --site HuangHe-A --mask-level 2

    # HuangHe-A recommended years: 1986,1990,1994,1997,1999,2002,2005,2010,2015,2020,2024
    #   (focus on key nodes: Xiaolangdi 1999, Water-sediment regulation 2002)
    
    # HuangHe-B recommended years: 1986,1990,1995,2000,2005,2010,2015,2020,2024
    #   (relatively uniform sampling after Longyangxia)

    # Use all available years
    python -m src.preprocessing.generate_prepared_imagery --site HuangHe-A --mask-level 2 --year-selection all

    # Use automatic sparse selection (equal intervals)
    python -m src.preprocessing.generate_prepared_imagery --site HuangHe-A --mask-level 2 --year-selection auto

    # Only generate Tilt00 (no rotation, quick test)
    python -m src.preprocessing.generate_prepared_imagery --site HuangHe-A --mask-level 2 --tilts 0
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--site", required=True, help="Site name, e.g., HuangHe-A")
    parser.add_argument("--mask-level", type=int, default=2, choices=[1, 2, 3, 4],
                        help="DSWE mask level (default: 2)")
    parser.add_argument("--tilts", nargs="+", type=int, default=[0, 15, 30, 45],
                        help="Tilt list (default: 0 15 30 45)")
    parser.add_argument("--year-selection", choices=["site", "auto", "all", "jurua"], default="site",
                        help="Year selection strategy: site=site-specific recommendations (default), auto=automatic sparse, all=all years, jurua=aligned with Jurua-A")
    parser.add_argument("--target-years", type=int, default=10,
                        help="Target number of years (only for auto mode)")

    args = parser.parse_args()

    generate_prepared_imagery(
        site=args.site,
        mask_level=args.mask_level,
        tilt_angles=tuple(args.tilts),
        year_selection=args.year_selection,
        target_years=args.target_years,
    )


if __name__ == "__main__":
    main()
