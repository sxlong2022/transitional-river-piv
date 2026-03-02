"""RivGraph Link Profiles: calculates s, x, y, B(s), C(s) along each link.

This module implements the geometric extraction part:
- Reads binary water mask raster and RivGraph link vector;
- Densifies sampling along each link's centerline using arc length to obtain (s, x, y);
- Computes local width B(s) using normal intersection method;
- Computes curvature C(s) based on centerline geometry.

Normal migration Mn(s) from PIV can be further overlaid on these profiles.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import fiona
import numpy as np
import rasterio
from rasterio.transform import rowcol
from shapely.geometry import LineString, MultiLineString, shape


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


def _iter_lines_from_vector(path: Path) -> Iterable[Tuple[str, LineString]]:
    """Iterates through a vector file to extract LineString geometries and their IDs.

    Prefers "id" or "link_id" fields; otherwise uses sequential index.
    """

    with fiona.open(path) as src:
        for idx, feat in enumerate(src):
            geom = feat["geometry"]
            if geom is None:
                continue
            shp = shape(geom)
            lines: list[LineString] = []
            if isinstance(shp, LineString):
                lines = [shp]
            elif isinstance(shp, MultiLineString):
                lines = list(shp.geoms)
            if not lines:
                continue

            props = feat.get("properties", {}) or feat
            link_id = (
                str(props.get("id"))
                if props.get("id") is not None
                else (
                    str(props.get("link_id"))
                    if props.get("link_id") is not None
                    else str(idx)
                )
            )

            for li, line in enumerate(lines):
                # Append sub-index if a single feature contains multiple LineStrings
                yield (f"{link_id}_{li}" if li > 0 else link_id, line)


def _densify_line(line: LineString, step: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Densifies sampling of a LineString at a given arc length interval step (m).

    Returns:
    - s : along-stream distance array, shape (n,)
    - xs, ys : sampled point coordinates, shape (n,)
    """

    if step <= 0:
        raise ValueError("step must be positive")

    length = line.length
    if length <= 0:
        raise RuntimeError("LineString length is 0, cannot densify sampling")

    # If original geometry already contains enough vertices, prefer keeping them
    # (prevents curvature from becoming zero on short links due to large step)
    coords = np.asarray(line.coords, dtype=float)
    if coords.ndim == 2 and coords.shape[0] >= 3:
        xs0 = coords[:, 0]
        ys0 = coords[:, 1]
        ds0 = np.hypot(np.diff(xs0), np.diff(ys0))
        keep = np.concatenate([[True], ds0 > 0])
        xs0 = xs0[keep]
        ys0 = ys0[keep]
        if xs0.size >= 3:
            ds = np.hypot(np.diff(xs0), np.diff(ys0))
            s = np.concatenate([[0.0], np.cumsum(ds)])
            return s, xs0, ys0

    n_step = int(np.ceil(length / step))
    # Minimum of 3 points (n_step>=2) for curvature calculation
    n_step = max(n_step, 2)
    s = np.linspace(0.0, length, n_step + 1)
    xs = np.empty_like(s)
    ys = np.empty_like(s)
    for i, si in enumerate(s):
        pt = line.interpolate(si)
        xs[i] = pt.x
        ys[i] = pt.y
    return s, xs, ys


def _densify_line_always(line: LineString, step: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if step <= 0:
        raise ValueError("step must be positive")

    length = line.length
    if length <= 0:
        raise RuntimeError("LineString length is 0, cannot densify sampling")

    n_step = int(np.ceil(length / step))
    n_step = max(n_step, 2)
    s = np.linspace(0.0, length, n_step + 1)
    xs = np.empty_like(s)
    ys = np.empty_like(s)
    for i, si in enumerate(s):
        pt = line.interpolate(si)
        xs[i] = pt.x
        ys[i] = pt.y
    return s, xs, ys


def _compute_tangent_normal(xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Computes tangent (tx, ty) and normal (nx, ny) unit vectors from sampled coordinates."""

    dx = np.gradient(xs)
    dy = np.gradient(ys)
    t_norm = np.hypot(dx, dy)
    t_norm[t_norm == 0] = 1.0
    tx = dx / t_norm
    ty = dy / t_norm

    # Normal: rotate tangent 90 degrees clockwise
    nx = ty
    ny = -tx

    return tx, ty, nx, ny


def _compute_curvature(xs: np.ndarray, ys: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Computes discrete curvature C(s) from (xs, ys) and arc length s."""

    dx = np.gradient(xs)
    dy = np.gradient(ys)
    t_norm = np.hypot(dx, dy)
    t_norm[t_norm == 0] = 1.0
    tx = dx / t_norm
    ty = dy / t_norm

    theta = np.unwrap(np.arctan2(ty, tx))
    # Differentiate orientation with respect to arc length
    dtheta_ds = np.gradient(theta, s)
    return dtheta_ds


def _sample_width_along_normal(
    mask: np.ndarray,
    transform,
    xs: np.ndarray,
    ys: np.ndarray,
    nx: np.ndarray,
    ny: np.ndarray,
    search_halfwidth: float,
    sample_spacing: float,
    min_valid_fraction: float = 0.1,
) -> np.ndarray:
    """Computes local width B(s) by finding water-land boundaries along the normal direction.

    Enhanced version:
    1. Does not strictly require the center point to be in water (accounts for PIV grid offsets);
    2. Finds the longest continuous water segment along the normal as B;
    3. Only returns valid width if the water segment is sufficiently long.

    Parameters:
    - mask: binary water mask array (1=water, 0=land)
    - transform: raster affine transform (rasterio.Affine)
    - xs, ys: sampled coordinates
    - nx, ny: normal unit vectors
    - search_halfwidth: normal search halfwidth (m)
    - sample_spacing: search point spacing along normal (m)
    - min_valid_fraction: minimum required fraction of water pixels along normal
    """

    h, w = mask.shape
    widths = np.full_like(xs, np.nan, dtype=float)

    # Precompute relative coordinates
    u_vals = np.arange(-search_halfwidth, search_halfwidth + sample_spacing, sample_spacing, dtype=float)
    if u_vals.size < 3:
        return widths

    for i, (x0, y0, nx0, ny0) in enumerate(zip(xs, ys, nx, ny)):
        # Construct sample points along normal
        xu = x0 + u_vals * nx0
        yu = y0 + u_vals * ny0

        # Convert to row/col indices
        rr, cc = rowcol(transform, xu, yu)
        rr = np.asarray(rr)
        cc = np.asarray(cc)
        # Handle out-of-bounds as land
        in_bounds = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        water = np.zeros_like(u_vals, dtype=np.uint8)
        water[in_bounds] = mask[rr[in_bounds], cc[in_bounds]]

        # Quality control: skip if water percentage is too low
        water_frac = water.sum() / float(water.size)
        if water_frac < min_valid_fraction:
            continue

        # Find longest continuous water segment (run-length method)
        padded = np.concatenate([[0], water, [0]])
        diff = np.diff(padded.astype(np.int8))
        # Rising edge (0→1)
        starts = np.where(diff == 1)[0]
        # Falling edge (1→0)
        ends = np.where(diff == -1)[0]

        if starts.size == 0 or ends.size == 0:
            continue

        # Segment lengths
        lengths = ends - starts
        max_idx = np.argmax(lengths)
        run_start = starts[max_idx]
        run_end = ends[max_idx]

        # Calculate u range for this segment
        left_u = u_vals[run_start]
        right_u = u_vals[run_end - 1]

        width_candidate = right_u - left_u
        # Basic validity check
        if width_candidate > 0:
            widths[i] = width_candidate

    return widths


def compute_link_profiles(
    mask_raster_path: str,
    links_vector_path: str,
    step_m: float = 100.0,
    normal_search_halfwidth_m: float | None = None,
    sample_spacing_factor: float = 0.5,
    min_valid_fraction: float = 0.05,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Computes geometric profiles (s, x, y, B, C) for RivGraph links.

    Returns dict: {link_id: {"s": ..., "x": ..., "y": ..., "B": ..., "C": ...}}

    Parameters:
    - normal_search_halfwidth_m: normal search halfwidth (m). Default is 30 pixels
      (~900m @ 30m resolution), sufficient for Jurua-A (~500-600m width).
    - min_valid_fraction: water pixel fraction threshold. Default 0.05 (5%).
    """

    mask_path = Path(mask_raster_path)
    vec_path = Path(links_vector_path)

    if not mask_path.exists():
        raise FileNotFoundError(f"Mask raster not found: {mask_path}")
    if not vec_path.exists():
        raise FileNotFoundError(f"RivGraph links vector not found: {vec_path}")

    with rasterio.open(mask_path) as src:
        mask_raw = src.read(1)
        # Normalize non-zero to 1 for binary mask (1=water, 0=land)
        mask = (mask_raw > 0).astype(np.uint8)
        transform = src.transform
        crs = src.crs
        is_geographic = bool(crs is not None and getattr(crs, "is_geographic", False))
        # Cell size for default search params
        cell_size_x = abs(transform.a)
        cell_size_y = abs(transform.e)
        if is_geographic:
            mid_lat = float((src.bounds.top + src.bounds.bottom) / 2.0)
            m_per_deg_lon, m_per_deg_lat = _meters_per_degree(mid_lat)
            base_cell = max(cell_size_x * m_per_deg_lon, cell_size_y * m_per_deg_lat)
        else:
            base_cell = max(cell_size_x, cell_size_y)

    if normal_search_halfwidth_m is None:
        # Default search halfwidth 30 pixels
        normal_search_halfwidth_m = 30.0 * base_cell

    sample_spacing = max(base_cell * sample_spacing_factor, base_cell * 0.25)

    profiles: Dict[str, Dict[str, np.ndarray]] = {}

    for link_id, line in _iter_lines_from_vector(vec_path):
        if is_geographic:
            coords = np.asarray(line.coords, dtype=float)
            xs0 = coords[:, 0]
            ys0 = coords[:, 1]
            lat0 = float(np.nanmean(ys0)) if np.isfinite(ys0).any() else 0.0
            m_per_deg_lon0, m_per_deg_lat0 = _meters_per_degree(lat0)

            line_m = LineString(list(zip(xs0 * m_per_deg_lon0, ys0 * m_per_deg_lat0)))
            s, xm, ym = _densify_line_always(line_m, step=step_m)

            xs = xm / m_per_deg_lon0
            ys = ym / m_per_deg_lat0

            _, _, nx_m, ny_m = _compute_tangent_normal(xm, ym)
            nx = nx_m / m_per_deg_lon0
            ny = ny_m / m_per_deg_lat0
            C = _compute_curvature(xm, ym, s)
        else:
            s, xs, ys = _densify_line_always(line, step=step_m)
            _, _, nx, ny = _compute_tangent_normal(xs, ys)
            C = _compute_curvature(xs, ys, s)

        B = _sample_width_along_normal(
            mask=mask,
            transform=transform,
            xs=xs,
            ys=ys,
            nx=nx,
            ny=ny,
            search_halfwidth=normal_search_halfwidth_m,
            sample_spacing=sample_spacing,
            min_valid_fraction=min_valid_fraction,
        )

        profiles[link_id] = {
            "s": s,
            "x": xs,
            "y": ys,
            "B": B,
            "C": C,
        }

    return profiles

