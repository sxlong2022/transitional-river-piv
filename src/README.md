# src Directory Structure and Recommended Execution Order

This directory contains all source code for the PIV riverbank migration Python project. For clarity, each submodule is briefly described in the order: "Data → PIV → Postprocessing → Morphodynamics → Pipeline & Analysis".

---

## 1. Top-Level Configuration

- `config.py`
  - Defines path constants such as `PROJECT_ROOT` and `DATA_DIR`
  - All scripts use this to locate `Data_and_code/` and `results/`

---

## 2. Data Preparation and Validation

- `preprocessing/`
  - `prepared_imagery.py`:
    - Provides helper functions like `get_prepared_imagery_dir(site)`
    - Points to `PreparedImagery/{SiteName}/MaskX_TiltYY` directories

- `validation/`
  - `check_data.py`:
    - Helper code for checking whether input data exists and dimensions are reasonable (can be extended as needed)

- `data_acquisition/` (currently a placeholder)
  - `gee_interface.py`:
    - Reserved interface for future Google Earth Engine image downloads

---

## 3. PIV Computation (Time Series and Multi-Tilt)

- `piv_analysis/`
  - `minimal_example_jurua.py`
    - Minimal example to verify OpenPIV usage on a single Jurua-A image pair
  - `jurua_timeseries.py`
    - Step 1: Multi-year time series PIV for a single site / single mask / single tilt
    - Performs PIV on consecutive yearly images using OpenPIV, then computes time-averaged vectors and uncertainty masks via `compute_vector_stats`
    - Supports command-line arguments: `--site`, `--mask-level`, `--tilt-deg`, etc.
  - `jurua_multitilt.py`
    - Step 2: Fuses multiple tilts (0/15/30/45°) under a single mask
    - Calls `run_timeseries_jurua` for each tilt, performs retilt in vector space, and fuses multiple realizations on the common intersection grid
    - Outputs multi-tilt fused pixel velocity fields (all / filtered)

---

## 4. PIV Postprocessing and Utility Functions

- `postprocessing/`
  - `postprocess.py`
    - `compute_vector_stats(u_stack, v_stack)`:
      - Computes time-averaged vectors, sample count, variance, and uncertainty mask `bad_mask`
    - `retilt_vectors(u, v, phi_deg)`:
      - Rotates vector fields from different tilts back to a unified coordinate system
    - `get_postprocessed_dir(PROJECT_ROOT, site)`:
      - Locates `results/PostprocessedPIV/{site}/` output directory

---

## 5. Morphodynamics Coupling (Step 4A / 4B)

- `morphodynamics/`
  - `coupling.py`
    - `project_velocity_on_normal(vx, vy, nx, ny)`:
      - Projects velocity vectors onto a given normal (used for centerline normal migration rate calculation)
  - `jurua_georef_multitilt.py`
    - Step 3 / 4A:
      - Reads reference masks with affine matrices from `GEOTIFFS/{site}/maskX`
      - Linearly maps the multi-tilt fused PIV grid to GeoTIFF row/column coordinates
      - Uses the full affine matrix (a, b, c, d, e, f) to strictly convert pixel displacements to geographic displacements → m/yr
      - Saves `jurua_maskX_multitilt_georef_step4a_strict.npz` and vector plots
    - Supports command-line arguments: `--site`, `--mask-level`, `--ref-year`
  - `jurua_centerline_profile.py`
    - Step 4 / 4B:
      - Automatically identifies centerlines (LineString or Polygon exterior) from `GIS/{site}.gpkg` and reprojects to the same CRS as Step 4A
      - Densifies sampling along the centerline at fixed intervals, performs nearest-neighbor PIV velocity sampling, and projects onto normals to obtain `Mn` profiles
      - Saves `jurua_maskX_multitilt_centerline_profile_step4b.npz` and map/profile plots
    - Supports command-line arguments: `--site`, `--mask-level`, `--step-m`, `--ref-year`

---

## 6. Visualization and Quicklook

- `visualization/`
  - `quicklook.py`
    - `describe_output_root(PROJECT_ROOT)`:
      - Returns the `results/figures/` root directory
    - Additional quicklook plotting tools can be added as needed

---

## 7. Pipeline and Batch Execution

- `pipeline/`
  - `jurua_pipeline.py`
    - **One-command pipeline script**:
      - Sequentially calls for a given `site` and multiple `mask_levels` / `tilts`:
        1. `run_multitilt_jurua` (Step 1–2)
        2. `georef_multitilt_jurua` (Step 4A)
        3. `build_centerline_profile` (Step 4B)
      - When multiple `mask_levels` are provided, multi-mask Mn profiles are stacked to compute mean and standard deviation
    - Recommended command examples (from project root):

      ```bash
      # Single mask (Mask1) full workflow
      python -m src.pipeline.jurua_pipeline \
        --site Jurua-A --mask-levels 1 --tilts 0 15 30 45 --step-m 100 --ref-year 1987

      # Multi-mask (Mask1–4) full workflow + multi-mask uncertainty
      python -m src.pipeline.jurua_pipeline \
        --site Jurua-A --mask-levels 1 2 3 4 --tilts 0 15 30 45 --step-m 100 --ref-year 1987
      ```

---

## 8. Analysis and Uncertainty Diagnostics

- `analysis/`
  - `multimask_uncertainty.py`
    - Reads multi-mask Mn profile aggregation results `jurua_multimask_multitilt_centerline_profiles_step4b.npz`
    - Computes:
      - `σ_Mn(s)`, coefficient of variation `CV(s)`
      - Direction consistency `dir_consistency(s)` (whether erosion/deposition distributions have the same sign across masks)
      - Exceedance probability `P(|Mn| > T)` for a given threshold T
    - Outputs:
      - `jurua_multimask_multitilt_centerline_uncertainty_metrics.npz`
      - `jurua_multimask_multitilt_centerline_sigma_cv.png`
      - `jurua_multimask_multitilt_centerline_dir_prob.png`
    - Example usage:

      ```bash
      python -m src.analysis.multimask_uncertainty --site Jurua-A --thresholds 5 10
      ```

---

## 9. Recommended Execution Order (Using Jurua-A as Example)

1. **Check/Prepare Data**: Ensure files are complete under `PreparedImagery/`, `GEOTIFFS/`, and `GIS/`.
2. **Run Complete Workflow with One Command** (Recommended):
   - Use `pipeline/jurua_pipeline.py` to run multi-tilt, multi-mask, Step 4A and Step 4B all at once.
3. **Debug or Modify by Step**:
   - PIV algorithm related → Modify `piv_analysis/` and `postprocessing/`.
   - Affine/registration related → Modify `morphodynamics/jurua_georef_multitilt.py`.
   - Centerline and normal profile related → Modify `morphodynamics/jurua_centerline_profile.py`.
   - Uncertainty and statistical analysis → Use or extend scripts in `analysis/`.

When adapting this workflow to other river segments such as the Yellow River:
- Keep the directory structure unchanged and prepare `PreparedImagery`, `GEOTIFFS`, and `GIS` for the new site;
- Replace `--site Jurua-A` with the new site name in command-line arguments;
- If needed, add new site-specific pipeline scripts in `pipeline/`.
