# Transitional River PIV Workflow

This repository contains the source code for the paper: **"An enhanced satellite PIV and graph-based skeletonization workflow for diagnosing migration regimes in regulated transitional rivers"** by Xiaolong Song et al., submitted to *Computers & Geosciences*.

The codebase implements a comprehensive workflow for extracting subpixel riverbank migration from dense optical satellite time series. It utilizes multi-angle Particle Image Velocimetry (PIV) fusion, continuous normal-intersection sampling, Union-Find/DAG-based trunk aggregation, and multi-mask uncertainty propagation.

## 1. Installation

The workflow requires Python 3.9 (specifically required by RivGraph). We recommend using `conda` for environment management.

```bash
# Clone the repository
git clone https://github.com/sxlong2022/transitional-river-piv.git
cd transitional-river-piv

# Create and activate a conda environment
conda create -n riverpiv python=3.9
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
[https://doi.org/10.25349/D9HG82](https://doi.org/10.25349/D9HG82)

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

## 6. Citation

If you use this codebase or the multi-angle PIV workflow, please cite both the paper and the archived software:

**The Paper:**
> Song, X., et al. (2026). An enhanced satellite PIV and graph-based skeletonization workflow for diagnosing migration regimes in regulated transitional rivers. *Computers & Geosciences* (Submitted).

**The Software:**
> Song, X. (2026). sxlong2022/transitional-river-piv: multi-angle Satellite PIV and Trunk Aggregation Workflow (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.18831632

## 7. Dependencies

This code is distributed under the MIT License. See the `LICENSE` file for details.
