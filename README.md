# Transitional Meander PIV Workflow

This repository contains the source code for the paper: **"An enhanced satellite PIV and graph-based skeletonization workflow for diagnosing migration regimes in regulated transitional rivers"** by Xiaolong Song et al., submitted to *Computers & Geosciences*.

The codebase implements a comprehensive workflow for extracting subpixel riverbank migration from dense optical satellite time series. It utilizes multi-angle Particle Image Velocimetry (PIV) fusion, continuous normal-intersection sampling, Union-Find/DAG-based trunk aggregation, and multi-mask uncertainty propagation.

## 1. Installation

The workflow requires Python 3.9 or higher. We recommend using `conda` for environment management.

```bash
# Clone the repository
git clone https://github.com/[username]/transitional-meander-piv.git
cd transitional-meander-piv

# Create and activate a conda environment
conda create -n riverpiv python=3.10
conda activate riverpiv

# Install dependencies
pip install -r requirements.txt
```

## 2. Quick Test

To verify that your Python environment and core dependencies (OpenPIV, NumPy, etc.) are correctly configured, run the following synthetic Quick Test. **No data download is required for this test.**

```bash
python -m tests.quick_test
```

This script will generate a synthetic image pair, run the OpenPIV extended search area routine, validate the displacements, and aggregate the vector statistics. You should see `=== Quick Test Completed Successfully ==!` printed at the end.

## 3. Data Acquisition

### Juruá River Benchmark
The original raw and water-classified imagery for the Juruá River benchmark (Chadwick et al., 2023) can be downloaded from the Dryad Digital Repository:
[https://doi.org/10.5061/dryad.8pk0p2ntg](https://doi.org/10.5061/dryad.8pk0p2ntg)

### Yellow River Data
The workflow includes a Google Earth Engine script to export Dynamic Surface Water Extent (DSWE) water masks for the Yellow River sites. For detailed instructions, please see `data/README.md`.

*Note: You can specify the root path for data storage by setting the `RIVERPIV_DATA_ROOT` environment variable. By default, the code looks for a `data/` directory at the project root.*

## 4. Usage: Running the Pipeline

Once the data is preprocessed and placed in `data/PreparedImagery/`, `data/GEOTIFFS/`, and `data/GIS/`, you can run the complete pipeline using the provided runner.

For example, to run the multi-angle PIV fusion, skeleton georeferencing, RivGraph node-link generation, and continuous trunk profile extraction for the Juruá-A benchmark over 4 mask threshold levels:

```bash
python -m src.pipeline.jurua_pipeline \
    --site Jurua-A \
    --mask-levels 1 2 3 4 \
    --tilts 0 15 30 45 \
    --step-m 100 \
    --ref-year 1987
```

To calculate the multi-mask uncertainty profiles after running the pipeline:

```bash
python -m src.analysis.multimask_uncertainty --site Jurua-A --thresholds 5 10
```

## 5. Repository Structure

- `src/piv_analysis/`: Multi-tilt PIV analysis and fusion via OpenPIV.
- `src/postprocessing/`: Retilting and temporal vector statistics.
- `src/morphodynamics/`: Georeferencing, skeletonization and continuous profile extraction.
- `src/analysis/`: Diagnostic plots, ablation studies, and uncertainty metrics.
- `src/pipeline/`: Consolidated, single-command run wrappers.
- `src/gee_data/`: Earth Engine DSWE export routines.
- `tests/`: Installation verification tests.

## 6. Dependencies

Major packages used in this workflow:
- `openpiv` [Zenodo DOI: 10.5281/zenodo.167606](https://doi.org/10.5281/zenodo.167606)
- `rivgraph` (Schwenk & Hariharan, 2021)
- `numpy`, `scipy`, `matplotlib`, `rasterio`

## 7. References

If you use this codebase or the multi-angle PIV workflow, please cite:

**The Paper:**
> Song, X., et al. (2026). An enhanced satellite PIV and graph-based skeletonization workflow for diagnosing migration regimes in regulated transitional rivers. *Computers & Geosciences* (Submitted).

**OpenPIV-Python:**
> Liberzon, A., Lasagna, D., Aubert, M., Bachant, P., Kirkham, J., Leu, R., & Borg, J. (2021). OpenPIV/openpiv-python: OpenPIV-Python (Version 0.20.5). Zenodo. https://doi.org/10.5281/zenodo.167606

## 8. License

This code is distributed under the MIT License. See the `LICENSE` file for details.
