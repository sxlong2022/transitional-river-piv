"""
generate_imagery_metadata.py
============================
Scan Landsat-derived DSWE GeoTIFF directories for each study site and produce
a summary table of available annual composites (year range, count, dimensions,
CRS, resolution).

The Yellow River data resides under  data/GEOTIFFS/{site}/mask{level}/
The Juruá-A data comes from the Chadwick et al. (2023) Dryad release,
stored under  文献/.../Data_and_code/Data/GEOTIFFS/Jurua-A/mask{level}/

Output is a Markdown table for the Supplement (Table S1).

Usage:
    conda activate riverpiv
    python -m src.analysis.generate_imagery_metadata
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ----- Data directories -----
DATA_DIR_YR = PROJECT_ROOT / "data" / "GEOTIFFS"
DATA_DIR_JURUA = (
    PROJECT_ROOT
    / "文献"
    / "Remote Sensing of Riverbank Migration Using Particle Image Velocimetry"
    / "Data_and_code"
    / "Data"
    / "GEOTIFFS"
)

# Sites and their primary mask levels used in the paper
SITES = {
    "Jurua-A":   {"masks": [1], "data_root": DATA_DIR_JURUA},
    "HuangHe-A": {"masks": [4], "data_root": DATA_DIR_YR},
    "HuangHe-B": {"masks": [4], "data_root": DATA_DIR_YR},
}


def extract_year(filename: str) -> int | None:
    """Extract 4-digit year from a GeoTIFF filename."""
    m = re.search(r"_(\d{4})_", filename)
    return int(m.group(1)) if m else None


def get_raster_info(tif_path: Path) -> dict:
    """Read basic raster metadata via rasterio."""
    try:
        import rasterio
        with rasterio.open(tif_path) as src:
            return {
                "rows": src.height,
                "cols": src.width,
                "crs": str(src.crs) if src.crs else "N/A",
                "res_x": abs(src.transform.a) if src.transform else None,
                "res_y": abs(src.transform.e) if src.transform else None,
            }
    except Exception:
        return {"rows": "?", "cols": "?", "crs": "?", "res_x": "?", "res_y": "?"}


def main():
    print("**Table S1.** Summary of Landsat-derived DSWE annual composites.\n")
    print("| Site | Mask Level | Year Range | Composites (n) "
          "| Dimensions (rows × cols) | CRS | Nominal Resolution |")
    print("|:---|:---:|:---:|---:|:---:|:---|:---:|")

    for site, cfg in SITES.items():
        data_root = cfg["data_root"]
        for mask in cfg["masks"]:
            mask_dir = data_root / site / f"mask{mask}"
            if not mask_dir.exists():
                mask_dir = data_root / site
            if not mask_dir.exists():
                print(f"| {site} | {mask} | — | 0 | — | — | — |")
                continue

            tifs = sorted(mask_dir.glob("*.tif"))
            years = sorted({y for t in tifs if (y := extract_year(t.name)) is not None})

            if not years:
                print(f"| {site} | {mask} | — | 0 | — | — | — |")
                continue

            year_range = f"{years[0]}–{years[-1]}"
            n = len(years)

            info = get_raster_info(tifs[0])
            dims = f"{info['rows']} × {info['cols']}"
            crs = info["crs"]
            # All datasets are Landsat-based at 30 m
            nom_res = "30 m"

            print(f"| {site} | {mask} | {year_range} | {n} | {dims} | {crs} | {nom_res} |")

    # Also list individual years per site
    print("\n\n**Detail:** Individual years available per site.\n")
    print("| Site | Mask Level | Available Years |")
    print("|:---|:---:|:---|")
    for site, cfg in SITES.items():
        data_root = cfg["data_root"]
        for mask in cfg["masks"]:
            mask_dir = data_root / site / f"mask{mask}"
            if not mask_dir.exists():
                mask_dir = data_root / site
            if not mask_dir.exists():
                print(f"| {site} | {mask} | — |")
                continue
            tifs = sorted(mask_dir.glob("*.tif"))
            years = sorted({y for t in tifs if (y := extract_year(t.name)) is not None})
            years_str = ", ".join(str(y) for y in years) if years else "—"
            print(f"| {site} | {mask} | {years_str} |")


if __name__ == "__main__":
    main()
