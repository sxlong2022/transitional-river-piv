"""Morphodynamics coupling placeholder module: RivGraph + PIV normal projection will be implemented here."""

from typing import Dict, Tuple

import numpy as np


def _meters_per_degree(lat_deg: float) -> tuple[float, float]:
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


def project_velocity_on_normal(vx: np.ndarray, vy: np.ndarray, nx: np.ndarray, ny: np.ndarray) -> np.ndarray:
    """Projects velocity vector (vx, vy) onto normal (nx, ny), returning normal migration rate M.

    Assumes (nx, ny) are already unit vectors; ensure consistent normal direction convention in actual usage.
    """
    return vx * nx + vy * ny


def _compute_tangent_normal_from_xy(
    xs: np.ndarray,
    ys: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dx = np.gradient(xs)
    dy = np.gradient(ys)
    t_norm = np.hypot(dx, dy)
    t_norm[t_norm == 0] = 1.0
    tx = dx / t_norm
    ty = dy / t_norm

    # Normal: tangent rotated 90 degrees clockwise
    nx = ty
    ny = -tx

    return tx, ty, nx, ny


def _looks_like_lonlat(xs: np.ndarray, ys: np.ndarray) -> bool:
    if xs.size == 0 or ys.size == 0:
        return False
    if not (np.isfinite(xs).any() and np.isfinite(ys).any()):
        return False
    return (np.nanmax(np.abs(xs)) <= 180.0) and (np.nanmax(np.abs(ys)) <= 90.0)


def _sample_velocity_nearest_grid(
    X_grid: np.ndarray,
    Y_grid: np.ndarray,
    U_grid: np.ndarray,
    V_grid: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Performs nearest-neighbor sampling of velocity components (u, v) for given coordinates (xs, ys) on a regular PIV grid."""

    u_s = np.empty_like(xs, dtype=float)
    v_s = np.empty_like(ys, dtype=float)

    for i, (x0, y0) in enumerate(zip(xs, ys)):
        dist2 = (X_grid - x0) ** 2 + (Y_grid - y0) ** 2
        j, k = np.unravel_index(np.argmin(dist2), X_grid.shape)
        u_s[i] = U_grid[j, k]
        v_s[i] = V_grid[j, k]

    return u_s, v_s


def add_Mn_to_link_profiles(
    link_profiles: Dict[str, Dict[str, np.ndarray]],
    X_grid: np.ndarray,
    Y_grid: np.ndarray,
    U_grid: np.ndarray,
    V_grid: np.ndarray,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Computes normal migration rate Mn(s) on each RivGraph link profile based on the PIV grid."""

    out: Dict[str, Dict[str, np.ndarray]] = {}

    for link_id, prof in link_profiles.items():
        x = prof.get("x")
        y = prof.get("y")

        if x is None or y is None:
            out[link_id] = dict(prof)
            continue

        xs = np.asarray(x, dtype=float)
        ys = np.asarray(y, dtype=float)

        # 1. Compute tangent and normal
        if _looks_like_lonlat(xs, ys):
            lat0 = float(np.nanmean(ys))
            m_per_deg_lon0, m_per_deg_lat0 = _meters_per_degree(lat0)
            xm = xs * m_per_deg_lon0
            ym = ys * m_per_deg_lat0
            _, _, nx, ny = _compute_tangent_normal_from_xy(xm, ym)
        else:
            _, _, nx, ny = _compute_tangent_normal_from_xy(xs, ys)

        # 2. Nearest-neighbor velocity sampling on PIV grid
        u_s, v_s = _sample_velocity_nearest_grid(
            X_grid=X_grid,
            Y_grid=Y_grid,
            U_grid=U_grid,
            V_grid=V_grid,
            xs=xs,
            ys=ys,
        )

        # 3. Project onto normal to obtain normal migration rate profile
        Mn = project_velocity_on_normal(u_s, v_s, nx, ny)

        prof_out = dict(prof)
        prof_out["Mn"] = Mn
        out[link_id] = prof_out

    return out
