"""Step 4B: Normal projection and migration rate profile of Jurua-A multi-tilt preferred field with centerline.

Overview:
- Reads m/yr vector field output from Step 4A (strict affine version):
    results/PostprocessedPIV/Jurua-A/jurua_mask1_multitilt_georef_step4a_strict.npz
- Automatically identifies and reads centerline layer from Data/GIS/Jurua-A.gpkg;
- Densifies sampling along the centerline at fixed intervals to obtain a series of points (Xc, Yc) and their along-stream distance s;
- At each sampling point, retrieves nearest-neighbor (u, v) (m/yr) from the PIV grid;
- Computes tangent and normal unit vectors (tx, ty) / (nx, ny) along the centerline;
- Uses project_velocity_on_normal to project (u, v) onto the normal to obtain normal migration rate Mn;
- Saves profile data as npz and outputs two plots:
    1) Map view: centerline colored by Mn (m/yr);
    2) Profile view: s vs Mn curve.

Usage (from project root):
    python -m src.morphodynamics.jurua_centerline_profile
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import argparse
import fiona
import numpy as np
import rasterio
from pyproj import CRS, Transformer
from shapely.geometry import LineString, MultiLineString, Polygon, MultiPolygon, shape
from shapely.ops import transform as shp_transform

from src.config import PROJECT_ROOT
from src.morphodynamics.coupling import project_velocity_on_normal
from src.morphodynamics.jurua_georef_multitilt import _choose_reference_mask
from src.postprocessing.postprocess import get_postprocessed_dir
from src.preprocessing.prepared_imagery import get_gis_dir
from src.visualization.quicklook import describe_output_root


def _load_centerline(site: str = "Jurua-A", mask_level: int = 1, ref_year: int | None = 1987) -> LineString:
    """Automatically identifies and reads centerline LineString from Data/GIS/{site}.gpkg, reprojecting to the same CRS as the PIV grid.

    Strategy:
    - Uses fiona.listlayers to list all layers;
    - Prefers layers with "center" in the name; if none, falls back to the first layer;
    - Supports LineString / MultiLineString / Polygon / MultiPolygon:
      * Line features: directly used as candidate centerline;
      * Polygon features: uses exterior boundary as candidate line;
    - Finally selects the longest LineString from all candidate lines as the centerline.
    """

    gis_dir = get_gis_dir(site)

    path_candidates = [
        gis_dir / f"{site}.gpkg",
        gis_dir / site / f"{site}.gpkg",
        gis_dir / f"{site}.shp",
        gis_dir / site / f"{site}.shp",
    ]

    vector_path = next((p for p in path_candidates if p.exists()), None)
    if vector_path is None:
        tried = "\n".join(str(p) for p in path_candidates)
        raise FileNotFoundError(f"Cannot find centerline vector file, tried:\n{tried}")

    layer = None
    if vector_path.suffix.lower() == ".gpkg":
        layers = fiona.listlayers(str(vector_path))
        if not layers:
            raise RuntimeError(f"No layers found in GeoPackage {vector_path}")

        candidates = [lyr for lyr in layers if "center" in lyr.lower()]
        layer = candidates[0] if candidates else layers[0]

    line_geoms: List[LineString | MultiLineString] = []
    src_crs_raw = None

    if layer is None:
        src_ctx = fiona.open(vector_path)
    else:
        src_ctx = fiona.open(vector_path, layer=layer)

    with src_ctx as src:
        src_crs_raw = src.crs
        for feat in src:
            geom = shape(feat["geometry"]) if feat["geometry"] is not None else None
            if geom is None:
                continue

            if isinstance(geom, (LineString, MultiLineString)):
                line_geoms.append(geom)
            elif isinstance(geom, (Polygon, MultiPolygon)):
                if isinstance(geom, Polygon):
                    line_geoms.append(LineString(geom.exterior.coords))
                else:
                    for poly in geom.geoms:
                        line_geoms.append(LineString(poly.exterior.coords))

    if not line_geoms:
        raise RuntimeError(f"No usable line/polygon geometries found in layer {layer} for building centerline")

    # Expand all MultiLineStrings to LineStrings and select the longest one (still in GPKG original CRS)
    line_candidates: List[LineString] = []
    for g in line_geoms:
        if isinstance(g, LineString):
            line_candidates.append(g)
        elif isinstance(g, MultiLineString):
            line_candidates.extend(list(g.geoms))

    if not line_candidates:
        raise RuntimeError(f"Could not construct any LineString centerline candidates from layer {layer}")

    longest = max(line_candidates, key=lambda g: g.length)

    # Reproject centerline to the same CRS as Step 4A reference mask (typically projected coordinates, unit m)
    ref_path = _choose_reference_mask(site=site, mask_level=mask_level, year=ref_year)
    with rasterio.open(ref_path) as ref_src:
        ref_crs_raw = ref_src.crs

    src_crs = CRS.from_user_input(src_crs_raw) if src_crs_raw else None
    ref_crs = CRS.from_user_input(ref_crs_raw) if ref_crs_raw else None

    if src_crs and ref_crs and src_crs != ref_crs:
        transformer = Transformer.from_crs(src_crs, ref_crs, always_xy=True)

        def _transform_xy(x: np.ndarray, y: np.ndarray, z: np.ndarray | None = None):
            x2, y2 = transformer.transform(x, y)
            return x2, y2

        longest = shp_transform(_transform_xy, longest)

    return longest


def _densify_line(line: LineString, step: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Densifies LineString sampling at given arc length interval step (m).

    Returns:
    - s : along-stream distance array, shape (n,)
    - xs, ys : sampled point coordinates, shape (n,)
    """

    length = line.length
    if length <= 0:
        raise RuntimeError("Centerline length is 0, cannot densify sampling")

    n_step = int(np.ceil(length / step))
    s = np.linspace(0.0, length, n_step + 1)
    xs = np.empty_like(s)
    ys = np.empty_like(s)
    for i, si in enumerate(s):
        pt = line.interpolate(si)
        xs[i] = pt.x
        ys[i] = pt.y
    return s, xs, ys


def _compute_tangent_normal(xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Computes tangent (tx, ty) and normal (nx, ny) unit vectors from sampled point coordinates."""

    # Use along-stream distance as approximate step size to compute derivative for tangent
    # Assumes sampling points are equally spaced
    dx = np.gradient(xs)
    dy = np.gradient(ys)
    t_norm = np.hypot(dx, dy)
    t_norm[t_norm == 0] = 1.0
    tx = dx / t_norm
    ty = dy / t_norm

    # Define normal: rotate tangent 90 degrees clockwise (convention for nx, ny)
    nx = ty
    ny = -tx

    return tx, ty, nx, ny


def _sample_velocity_nearest(
    X_grid: np.ndarray,
    Y_grid: np.ndarray,
    U_grid: np.ndarray,
    V_grid: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """At each (xs, ys) location, retrieves nearest-neighbor vector values from a regular grid.

    Grid size is typically ~1e4, with hundreds of sampling points along the centerline, so simple point-by-point nearest neighbor is sufficient.
    """

    u_s = np.empty_like(xs)
    v_s = np.empty_like(ys)

    for i, (x0, y0) in enumerate(zip(xs, ys)):
        dist2 = (X_grid - x0) ** 2 + (Y_grid - y0) ** 2
        j, k = np.unravel_index(np.argmin(dist2), X_grid.shape)
        u_s[i] = U_grid[j, k]
        v_s[i] = V_grid[j, k]

    return u_s, v_s


def build_centerline_profile(
    site: str = "Jurua-A",
    mask_level: int = 1,
    step_m: float = 100.0,
    ref_year: int | None = 1987,
) -> dict:
    """Builds normal migration rate profile of multi-tilt preferred field along the centerline.

    Returns dictionary containing:
    - s : along-stream distance (m)
    - xs, ys : centerline sampled point coordinates
    - tx, ty  : tangent unit vectors
    - nx, ny  : normal unit vectors
    - u_s, v_s: (u, v) sampled onto centerline (m/yr)
    - Mn      : normal migration rate (m/yr)
    - X_grid, Y_grid, U_grid, V_grid : PIV grid and velocity field (for debugging)
    """

    # 1. Load Step 4A strict PIV results
    out_dir = get_postprocessed_dir(PROJECT_ROOT, site)
    npz_path = out_dir / f"jurua_mask{mask_level}_multitilt_georef_step4a_strict.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Step 4A strict output not found: {npz_path}\nPlease run first: python -m src.morphodynamics.jurua_georef_multitilt"
        )

    data = np.load(npz_path)
    X_grid = data["X_geo"]
    Y_grid = data["Y_geo"]
    U_grid = data["u_m_per_year"]
    V_grid = data["v_m_per_year"]

    # 2. Read centerline (and reproject to match PIV grid CRS) and densify sampling
    centerline = _load_centerline(site, mask_level=mask_level, ref_year=ref_year)
    s, xs, ys = _densify_line(centerline, step=step_m)

    # 3. Compute tangent and normal
    tx, ty, nx, ny = _compute_tangent_normal(xs, ys)

    # 4. Sample PIV velocities along centerline (nearest neighbor)
    u_s, v_s = _sample_velocity_nearest(X_grid, Y_grid, U_grid, V_grid, xs, ys)

    # 5. Project onto normal to obtain normal migration rate profile
    Mn = project_velocity_on_normal(u_s, v_s, nx, ny)

    return {
        "s": s,
        "xs": xs,
        "ys": ys,
        "tx": tx,
        "ty": ty,
        "nx": nx,
        "ny": ny,
        "u_s": u_s,
        "v_s": v_s,
        "Mn": Mn,
        "X_grid": X_grid,
        "Y_grid": Y_grid,
        "U_grid": U_grid,
        "V_grid": V_grid,
    }


def main() -> None:
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(
        description="Step 4B: project georeferenced PIV onto river centerline normals to build a migration profile.",
    )
    parser.add_argument("--site", default="Jurua-A", help="Site name, e.g., 'Jurua-A'")
    parser.add_argument("--mask-level", type=int, default=1, help="Mask level, e.g., 1 for Mask1")
    parser.add_argument("--step-m", type=float, default=100.0, help="Sampling interval along centerline (meters)")
    parser.add_argument("--ref-year", type=int, default=1987, help="Preferred year for selecting reference mask")

    args = parser.parse_args()

    result = build_centerline_profile(
        site=args.site,
        mask_level=args.mask_level,
        step_m=args.step_m,
        ref_year=args.ref_year,
    )

    s = result["s"]
    xs = result["xs"]
    ys = result["ys"]
    Mn = result["Mn"]

    # Negative values erode to one side, positive to the other; actual direction determined by normal convention

    out_dir = get_postprocessed_dir(PROJECT_ROOT, args.site)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / f"jurua_mask{args.mask_level}_multitilt_centerline_profile_step4b.npz"

    np.savez(out_npz, s=s, xs=xs, ys=ys, Mn=Mn)
    print("Saved centerline normal migration rate profile to:", out_npz)

    out_root = describe_output_root(PROJECT_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)

    # Plot 1: Map view, centerline colored by Mn
    out_png_map = out_root / f"jurua_mask{args.mask_level}_multitilt_centerline_Mn_map.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(xs, ys, c=Mn, cmap="RdBu_r", s=10)
    plt.colorbar(sc, ax=ax, label="Normal migration rate Mn (m/yr)")
    ax.set_aspect("equal")
    ax.set_title(f"{args.site} Mask{args.mask_level} multi-tilt: centerline normal migration (Step 4B)")
    fig.savefig(out_png_map, dpi=200)
    plt.close(fig)

    # Plot 2: Profile view, s vs Mn
    out_png_prof = out_root / f"jurua_mask{args.mask_level}_multitilt_centerline_Mn_profile.png"
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(s / 1000.0, Mn, "k-")
    ax.set_xlabel("Centerline distance s (km)")
    ax.set_ylabel("Normal migration rate Mn (m/yr)")
    ax.set_title(f"{args.site} Mask{args.mask_level} multi-tilt: centerline normal migration profile")
    ax.grid(True, alpha=0.3)
    fig.savefig(out_png_prof, dpi=200)
    plt.close(fig)

    print("Saved centerline normal migration rate map to:", out_png_map)
    print("Saved centerline normal migration rate profile to:", out_png_prof)


if __name__ == "__main__":
    main()
