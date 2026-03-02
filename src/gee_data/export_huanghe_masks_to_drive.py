"""
Export annual water masks for Yellow River target segments to Google Drive

Computes water masks using Jones DSWE method from Landsat Collection 2 and exports to Google Drive.
Users manually download from Google Drive and place in local data/GEOTIFFS/{site}/ directory.

Usage:
    python -m src.gee_data.export_huanghe_masks_to_drive --site HuangHe-A --start-year 1985 --end-year 2023

Dependencies:
    - earthengine-api

Authentication required before first use:
    earthengine authenticate
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import ee


# =====================================================================================
# === GEE Initialization ======================================================================
# =====================================================================================

def initialize_gee():
    """Initialize Google Earth Engine."""
    try:
        ee.Initialize(project="fast-banner-452901-c8")
        print("Google Earth Engine initialized successfully.")
        return True
    except Exception as e:
        print(f"Error initializing Google Earth Engine: {e}")
        print("Please run 'earthengine authenticate' to complete authorization first.")
        return False


# =====================================================================================
# === Boundary Polygon Definition: Read from local Shapefile ============================================
# =====================================================================================

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data"


def get_roi(site: str) -> ee.Geometry:
    """Read site boundary from local Shapefile and convert to EE Geometry (without relying on shapely).

    Expected locations:
        data/GIS/{site}.shp
        or data/GIS/{site}/{site}.shp
    """
    try:
        import fiona
    except ImportError:
        raise ImportError(
            "fiona is required to read local Shapefile. Please install in riverpiv environment:\n"
            "    pip install fiona"
        )

    # Try root directory Shapefile first, then subdirectory
    shp_root = DATA_ROOT / "GIS" / f"{site}.shp"
    shp_dir = DATA_ROOT / "GIS" / site / f"{site}.shp"

    if shp_root.exists():
        shp_path = shp_root
    elif shp_dir.exists():
        shp_path = shp_dir
    else:
        raise FileNotFoundError(
            f"Shapefile not found: {shp_root} or {shp_dir}.\n"
            f"Please confirm that convert_kml_to_shp.py has generated {site}.shp under data/GIS."
        )

    print(f"Using Shapefile to define ROI: {shp_path}")

    with fiona.open(shp_path) as src:
        if len(src) == 0:
            raise ValueError(f"No features in Shapefile: {shp_path}")

        # Assume CRS is WGS84 (convert_kml_to_shp.py writes EPSG:4326 from KML directly)
        feat = next(iter(src))
        geom = feat["geometry"]
        if geom is None:
            raise ValueError(f"Feature geometry is empty: {shp_path}")

        gtype = geom.get("type")
        coords = geom.get("coordinates")

        if gtype == "Polygon":
            outer = coords[0]
        elif gtype == "MultiPolygon":
            # Take first polygon's outer ring
            outer = coords[0][0]
        else:
            raise ValueError(f"Geometry type is not Polygon/MultiPolygon, but {gtype}")

        # Discard possible Z values, keep only (lon, lat)
        coords_2d = [[x, y] for (x, y, *_) in outer]

    return ee.Geometry.Polygon([coords_2d])


# =====================================================================================
# === Jones DSWE Water Mask Computation =========================================================
# =====================================================================================

def compute_dswe_mask(image, water_level: int = 2):
    """
    Compute Jones et al. (2019) DSWE water mask.

    Implements the exact lookup table logic from GEE_watermasks (evan-greenbrg).
    Reference: https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/media/files/LSDS-2084_LandsatC2_L3_DSWE_ADD-v1.pdf

    water_level:
        1 = High confidence water
        2 = Moderate confidence water  
        3 = Potential wetland
        4 = Low confidence water / partial surface water
    """
    image = ee.Image(image)

    # Extract bands (unified naming: SR_B2=Blue, SR_B3=Green, SR_B4=Red, SR_B5=NIR, SR_B6=SWIR1, SR_B7=SWIR2)
    blue = image.select("SR_B2")
    green = image.select("SR_B3")
    red = image.select("SR_B4")
    nir = image.select("SR_B5")
    swir1 = image.select("SR_B6")
    swir2 = image.select("SR_B7")

    # === Compute water indices ===
    # MNDWI = (Green - SWIR1) / (Green + SWIR1)
    mndwi = green.subtract(swir1).divide(green.add(swir1))

    # MBSRV = Green + Red
    mbsrv = green.add(red)

    # MBSRN = NIR + SWIR1
    mbsrn = nir.add(swir1)

    # NDVI = (NIR - Red) / (NIR + Red)
    ndvi = nir.subtract(red).divide(nir.add(red))

    # AWEsh = Blue + 2.5*Green - 1.5*(NIR+SWIR1) - 0.25*SWIR2
    awesh = blue.add(green.multiply(2.5)).subtract(mbsrn.multiply(1.5)).subtract(swir2.multiply(0.25))

    # === DSWE 5 tests (Jones et al. 2019) ===
    # Test 1: MNDWI > 0.124
    t1 = mndwi.gt(0.124).toInt()

    # Test 2: MBSRV > MBSRN
    t2 = mbsrv.gt(mbsrn).toInt()

    # Test 3: AWEsh > 0
    t3 = awesh.gt(0).toInt()

    # Test 4: MNDWI > -0.44 AND SWIR1 < 900 AND NIR < 1500 AND NDVI < 0.7
    # Threshold note: GEE_watermasks uses raw DN (0-10000 = 0-1 reflectance), we scaled to 0-1
    # 900/10000 = 0.09, 1500/10000 = 0.15
    t4 = mndwi.gt(-0.44).And(swir1.lt(0.09)).And(nir.lt(0.15)).And(ndvi.lt(0.7)).toInt()

    # Test 5: MNDWI > -0.5 AND Blue < 1000 AND SWIR1 < 3000 AND SWIR2 < 1000 AND NIR < 2500
    # 1000/10000 = 0.1, 3000/10000 = 0.3, 2500/10000 = 0.25
    t5 = mndwi.gt(-0.5).And(blue.lt(0.1)).And(swir1.lt(0.3)).And(swir2.lt(0.1)).And(nir.lt(0.25)).toInt()

    # === Combine t1-t5 using 5-digit decimal encoding ===
    # t_code = t1 + t2*10 + t3*100 + t4*1000 + t5*10000
    t_code = t1.add(t2.multiply(10)).add(t3.multiply(100)).add(t4.multiply(1000)).add(t5.multiply(10000))

    # === Exact lookup table classification (exactly consistent with GEE_watermasks) ===
    dswe = ee.Image(0).toInt()

    # No Water (class 0): t_code in [0, 1, 10, 100, 1000]
    # Already 0 by default, no extra processing needed

    # High confidence water (class 1): t_code in [1111, 10111, 11101, 11110, 11111]
    high_conf = t_code.eq(1111).Or(t_code.eq(10111)).Or(t_code.eq(11101)).Or(t_code.eq(11110)).Or(t_code.eq(11111))
    dswe = dswe.where(high_conf, 1)

    # Moderate confidence water (class 2): t_code in [111, 1011, 1101, 1110, 10011, 10101, 10110, 11001, 11010, 11100]
    mod_conf = (
        t_code.eq(111).Or(t_code.eq(1011)).Or(t_code.eq(1101)).Or(t_code.eq(1110))
        .Or(t_code.eq(10011)).Or(t_code.eq(10101)).Or(t_code.eq(10110))
        .Or(t_code.eq(11001)).Or(t_code.eq(11010)).Or(t_code.eq(11100))
    )
    dswe = dswe.where(mod_conf, 2)

    # Potential wetland (class 3): t_code == 11000
    pot_wet = t_code.eq(11000)
    dswe = dswe.where(pot_wet, 3)

    # Low confidence water (class 4): t_code in [11, 101, 110, 1001, 1010, 1100, 10000, 10001, 10010, 10100]
    low_conf = (
        t_code.eq(11).Or(t_code.eq(101)).Or(t_code.eq(110))
        .Or(t_code.eq(1001)).Or(t_code.eq(1010)).Or(t_code.eq(1100))
        .Or(t_code.eq(10000)).Or(t_code.eq(10001)).Or(t_code.eq(10010)).Or(t_code.eq(10100))
    )
    dswe = dswe.where(low_conf, 4)

    # === Generate binary mask based on water_level ===
    # water_level=1: only high confidence (class 1)
    # water_level=2: high + moderate (class 1,2)
    # water_level=3: high + moderate + potential wetland (class 1,2,3)
    # water_level=4: all water (class 1,2,3,4)
    water_mask = dswe.gte(1).And(dswe.lte(water_level))

    return water_mask.rename("water_mask")


def get_landsat_collection(year: int, roi: ee.Geometry, start_date: str, end_date: str):
    """Get Landsat Collection 2 Level-2 data for specified year."""
    
    # Select sensor based on year
    if year <= 1984:
        collection_id = "LANDSAT/LT04/C02/T1_L2"
        sensor = "L4"
    elif year <= 2012:
        if year <= 1999:
            collection_id = "LANDSAT/LT05/C02/T1_L2"
            sensor = "L5"
        else:
            collection_id = "LANDSAT/LE07/C02/T1_L2"
            sensor = "L7"
    elif year <= 2021:
        collection_id = "LANDSAT/LC08/C02/T1_L2"
        sensor = "L8"
    else:
        collection_id = "LANDSAT/LC09/C02/T1_L2"
        sensor = "L9"
    
    # For L4/L5/L7, band names need mapping
    if sensor in ["L4", "L5", "L7"]:
        # Map TM/ETM+ bands to OLI-style naming, preserve QA_PIXEL for cloud mask if possible
        def rename_bands(img):
            img = ee.Image(img)
            # Rename spectral main bands to unified SR_B2..SR_B7
            spec = img.select(
                ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"],
                ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]
            )
            # If QA_PIXEL exists, add it back
            band_names = img.bandNames()
            qa_exists = band_names.contains("QA_PIXEL")

            def _with_qa():
                return spec.addBands(img.select("QA_PIXEL"))

            def _no_qa():
                return spec

            out = ee.Image(ee.Algorithms.If(qa_exists, _with_qa(), _no_qa()))
            return out.copyProperties(img, ["system:time_start"])
        
        collection = ee.ImageCollection(collection_id) \
            .filterBounds(roi) \
            .filterDate(start_date, end_date) \
            .map(rename_bands)
    else:
        collection = ee.ImageCollection(collection_id) \
            .filterBounds(roi) \
            .filterDate(start_date, end_date)
    
    return collection, sensor


def scale_landsat(image):
    """Apply Landsat C2 L2 scaling factors."""
    optical = image.select("SR_B.").multiply(0.0000275).add(-0.2)
    return optical.copyProperties(image, ["system:time_start"])


def mask_clouds(image):
    """Use QA_PIXEL for cloud masking; if image lacks QA_PIXEL, return as-is.

    Note:
    - Landsat C2 L2 should theoretically all include QA_PIXEL, but during band renaming or
      in some older scenes, this band may be missing. To prevent Image.select from throwing
      errors, we do a guarded selection here.
    - For images without QA_PIXEL, equivalent to no cloud masking, relying on DSWE's own robustness.
    """
    image = ee.Image(image)
    band_names = image.bandNames()
    qa_exists = band_names.contains("QA_PIXEL")

    def _apply_mask(img):
        qa = img.select("QA_PIXEL")
        # Bit 3: Cloud, Bit 4: Cloud Shadow
        cloud_mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
        return img.updateMask(cloud_mask)

    def _no_mask(img):
        return img

    return ee.Image(ee.Algorithms.If(qa_exists, _apply_mask(image), _no_mask(image)))


# =====================================================================================
# === Main Export Function ======================================================================
# =====================================================================================

def export_annual_masks(
    site: str,
    start_year: int = 1986,
    end_year: int = 2024,
    water_level: int = 2,
    drive_folder: str = "HuangHe_WaterMasks",
):
    """
    Export annual water masks to Google Drive.
    
    Parameters
    ------
    site : str
        Site name, e.g., "HuangHe-A".
    start_year : int
        Start year.
    end_year : int
        End year.
    water_level : int
        DSWE water confidence level (1-4).
    drive_folder : str
        Target Google Drive folder name.
    """
    
    roi = get_roi(site)
    
    print(f"\n{'=' * 60}")
    print(f"Exporting {site} annual water masks to Google Drive")
    print(f"Time range: {start_year} - {end_year}")
    print(f"Water Level: {water_level}")
    print(f"Drive folder: {drive_folder}")
    print(f"{'=' * 60}\n")
    
    tasks = []
    
    for year in range(start_year, end_year + 1):
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        
        print(f"Processing year {year}...")
        
        try:
            # Get Landsat data
            collection, sensor = get_landsat_collection(year, roi, start_date, end_date)
            
            # Check if data exists
            count = collection.size().getInfo()
            if count == 0:
                print(f"  ⚠️ No available images for year {year}, skipping.")
                continue
            
            print(f"  📦 Found {count} {sensor} images")
            
            # Apply cloud mask and scaling
            collection = collection.map(mask_clouds).map(scale_landsat)
            
            # Compute annual median composite
            composite = collection.median()
            
            # Compute DSWE water mask
            water_mask = compute_dswe_mask(composite, water_level=water_level)
            
            # Clip to ROI
            water_mask = water_mask.clip(roi).toUint8()
            
            # Export to Drive
            task_name = f"{site}_{year}_01-01_12-31_mask{water_level}"
            
            task = ee.batch.Export.image.toDrive(
                image=water_mask,
                description=task_name,
                folder=drive_folder,
                fileNamePrefix=task_name,
                scale=30,
                region=roi,
                maxPixels=1e13,
                crs="EPSG:4326"
            )
            task.start()
            tasks.append((year, task_name, task))
            
            print(f"  🚀 Export task submitted: {task_name}")
            
            # Avoid API rate limiting
            time.sleep(1)
            
        except Exception as e:
            print(f"  ❌ Error processing year {year}: {e}")
            continue
    
    print(f"\n{'=' * 60}")
    print(f"Submitted {len(tasks)} export tasks to Google Drive")
    print(f"Check task status at https://code.earthengine.google.com/tasks")
    print(f"After completion, download from Google Drive folder '{drive_folder}'")
    print(f"{'=' * 60}")
    
    return tasks


# =====================================================================================
# === CLI Entry Point ========================================================================
# =====================================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Export annual water masks for Yellow River target segments to Google Drive",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Export water_level=2 masks for HuangHe-A site from 1985-2023
    python -m src.gee_data.export_huanghe_masks_to_drive --site HuangHe-A

    # Export only 2000-2020
    python -m src.gee_data.export_huanghe_masks_to_drive --site HuangHe-A --start-year 2000 --end-year 2020

    # Export water_level=1 (high confidence)
    python -m src.gee_data.export_huanghe_masks_to_drive --site HuangHe-A --water-level 1
        """,
    )
    parser.add_argument("--site", required=True, help="Site name: HuangHe-A or HuangHe-B")
    parser.add_argument("--start-year", type=int, default=1986, help="Start year")
    parser.add_argument("--end-year", type=int, default=2024, help="End year")
    parser.add_argument("--water-level", type=int, default=2, choices=[1, 2, 3, 4],
                        help="DSWE water confidence level (1=high, 2=moderate, 3=wetland, 4=low)")
    parser.add_argument("--drive-folder", default="HuangHe_WaterMasks",
                        help="Target Google Drive folder name")

    args = parser.parse_args()

    if not initialize_gee():
        return

    export_annual_masks(
        site=args.site,
        start_year=args.start_year,
        end_year=args.end_year,
        water_level=args.water_level,
        drive_folder=args.drive_folder,
    )


if __name__ == "__main__":
    main()
