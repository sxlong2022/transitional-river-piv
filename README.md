# Transitional River PIV Workflow

This repository contains the source code for the paper: **"An enhanced satellite PIV and graph-based skeletonization workflow for diagnosing migration regimes in regulated transitional rivers"** by Xiaolong Song et al., published in *Computers & Geosciences* (2026).

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

## 4. Sample Data & Reproducing Figures

To allow immediate testing and reproduction of the paper's figures without requiring hours of data downloading and processing, this repository includes the **final diagnostic metrics and continuous profiles** as sample data. 

These lightweight data files are located in `results/RivGraph/` and `results/PostprocessedPIV/`. You can reproduce Figs. 4–9, A1, S2, and S3 directly using the analysis scripts. Figs. 1 and 3 require the full satellite imagery and water masks (see Section 3 for data acquisition); Fig. 2 is provided as a static vector graphic at `results/figures/paper/Fig2.svg`.

> **Site name mapping**: In the codebase and script arguments, `HuangHe-A` and `HuangHe-B` correspond to the **YR-A** and **YR-B** study reaches referred to in the paper.

| Figure | Description | Script / Command |
| --- | --- | --- |
| Fig. 4 | Trunk organization contrast (YR-A vs YR-B) | Run for each site × mask combination: `python -m src.analysis.plot_fig4_trunk_overlay --site HuangHe-A --mask-level 2 --preset paper`, `--mask-level 4`, then repeat with `--site HuangHe-B`. (Fig. 4 shows Mask 2 and Mask 4 columns side-by-side.) |
| Fig. 5 | Along-channel B, C, Mn profiles | `python -m src.analysis.plot_fig5_trunk_profiles --site HuangHe-A --mask-level 4 --preset paper` (repeat with `--site HuangHe-B`) |
| Fig. 6a | Trunk-scale structure statistics — ACF/CCF (YR-A) | `python -m src.analysis.plot_fig6_spectral_structure --site HuangHe-A --mask-level 4` |
| Fig. 6b | Trunk-scale structure statistics — ACF/CCF (YR-B) | `python -m src.analysis.plot_fig6_spectral_structure --site HuangHe-B --mask-level 4` |
| Fig. 7 | Link-scale scatter: \|Mn\| vs \|C\| and B | `python -m src.analysis.plot_paper_panels --mode scatter_2x2 --sites HuangHe-A HuangHe-B --masks 4` |
| Fig. 8 | Conceptual migration-regime schematic | `python -m src.analysis.plot_fig8_conceptual` |
| Fig. 9 | Synthesis of regime indicators | `python -m src.analysis.plot_fig9_synthesis` |
| Fig. A1 | Trunk aggregation algorithm illustration | `python -m src.analysis.plot_figA1_concept --preset paper` |
| Fig. S2 | Fourier spectra of width & curvature | `python -m src.analysis.plot_paper_panels --mode fft_spectra --sites HuangHe-A HuangHe-B Jurua-A --masks 4 4 1` |
| Fig. S3 | Dimensionless \|C\|B vs \|Mn\| | `python -m src.analysis.plot_paper_panels --mode dimless_cb --sites HuangHe-A HuangHe-B Jurua-A --masks 4 4 1` |

For example, to reproduce Fig. 9 (Synthesis):
```bash
python -m src.analysis.plot_fig9_synthesis
```
The resulting figure will be saved to `results/figures/paper/Fig9_Synthesis.png`.

## 5. Usage: Running the Pipeline

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

## 6. Repository Structure

- `src/piv_analysis/`: Multi-tilt PIV analysis and fusion via OpenPIV.
- `src/postprocessing/`: Retilting and temporal vector statistics.
- `src/morphodynamics/`: Georeferencing, skeletonization and continuous profile extraction.
- `src/analysis/`: Diagnostic plots, ablation studies, and uncertainty metrics.
- `src/pipeline/`: Consolidated, single-command run wrappers.
- `src/gee_data/`: Earth Engine DSWE export routines.
- `tests/`: Installation verification tests.

## 7. Citation

If you use this codebase or the multi-angle PIV workflow, please cite both the paper and the archived software:

**The Paper:**
> Song, X., Feng, H., Xu, H., & Bai, Y. (2026). An enhanced satellite PIV and graph-based skeletonization workflow for diagnosing migration regimes in regulated transitional rivers. *Computers & Geosciences*, 214, 106183. https://doi.org/10.1016/j.cageo.2026.106183

**The Software:**
> Song, X. (2026). sxlong2022/transitional-river-piv: multi-angle Satellite PIV and Trunk Aggregation Workflow (v1.1.0). Zenodo. https://doi.org/10.5281/zenodo.19493042

## 8. Core Dependencies & Acknowledgments

This workflow builds heavily upon two excellent open-source libraries:
- **[OpenPIV](https://github.com/OpenPIV/openpiv-python)**: Used as the core engine for computing subpixel riverbank displacements.
- **[RivGraph](https://github.com/VeinsOfTheEarth/RivGraph)**: Used for deriving and aggregating graph-based channel skeletons. Note that RivGraph's specific dependency chain (e.g., specific versions of GDAL and Fiona) is the reason this project strictly requires **Python 3.9**.

## 9. License

This code is distributed under the MIT License. See the `LICENSE` file for details.
