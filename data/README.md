# Data Directory

Due to file size constraints, large GeoTIFF imagery and intermediate vector datasets are not included in this repository.

To successfully run the workflow, your `data/` directory must be structured as follows:

```
data/
├── GEOTIFFS/
│   ├── Jurua-A/
│   │   ├── mask1/
│   │   │   ├── 1987.tif
│   │   │   ├── 1988.tif
│   │   │   └── ...
│   │   └── mask2/ ...
│   ├── HuangHe-A/
│   └── HuangHe-B/
├── GIS/
│   ├── Jurua-A.shp
│   ├── HuangHe-A.shp
│   └── HuangHe-B.shp
└── PreparedImagery/
    ├── Jurua-A/
    └── ...
```

## 1. Obtaining Juruá River Data

The Juruá River benchmark dataset (Chadwick et al., 2023) is publicly available via the Dryad Digital Repository:
**DOI:** 10.25349/D9HG82

Download the archive, extract the water masks, and place the consecutive yearly `.tif` composites into the respective `data/GEOTIFFS/Jurua-A/mask{1,2,3,4}` directories based on their classification scheme.

## 2. Obtaining Yellow River Data

Yellow River Dynamic Surface Water Extent (DSWE) annual composites can be reproduced via Google Earth Engine using the included script:

```bash
# Export DSWE Mask 2 for HuangHe-A (1985-2023) to your Google Drive
python -m src.gee_data.export_huanghe_masks_to_drive \
    --site HuangHe-A \
    --start-year 1985 --end-year 2023 \
    --water-level 2
```

Requires Google Earth Engine authentication (`earthengine authenticate`) and the `earthengine-api` Python package. Once exported to your Google Drive, place the files in `data/GEOTIFFS/HuangHe-A/mask2`.

## 3. Data Preparation

Once the raw GeoTIFFs are in place, generate the padded, masked, and tilted binary imagery (used directly by OpenPIV) by running the preprocessing script:

```bash
python -m src.preprocessing.generate_prepared_imagery --site Jurua-A
```
This will populate the `data/PreparedImagery/` directory, after which you safely run the main PIV scripts.
