"""PIV postprocessing module: Result paths and vector statistics utilities.

Current implementations:
- get_postprocessed_dir: Convention for future PostprocessedPIV output location;
- retilt_vectors: Performs counter-tilt rotation on vector components by a given angle;
- compute_vector_stats: Performs time/ensemble statistics on a set of (u, v) vector fields and provides uncertainty masks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def get_postprocessed_dir(project_root: Path, site: str) -> Path:
    """Returns the directory for PostprocessedPIV output according to current project convention.

    Currently placed at `results/PostprocessedPIV/{site}` under project root,
    coexisting with original author's Data/PostprocessedPIV without conflict.
    """

    return project_root / "results" / "PostprocessedPIV" / site


def retilt_vectors(u: np.ndarray, v: np.ndarray, phi_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """Performs counter-tilt rotation on vector (u, v) by a given angle.

    Logic corresponds to MATLAB's RetiltPIV_fun:

        [theta, n] = cart2pol(u, v)
        thetaretilt = theta + phi*(pi/180)
        [uretilt, vretilt] = pol2cart(thetaretilt, n)

    Assumes:
    - u, v are vector components measured in some "rotated" image coordinate system;
    - phi_deg is the rotation angle Tilt.phi used during preprocessing (e.g., -15, -30, etc.);
    - Returns (u_r, v_r) as vector components after "counter-rotating back to original coordinate system".
    """

    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    theta = np.arctan2(v, u)
    n = np.hypot(u, v)

    theta_retilt = theta + np.deg2rad(phi_deg)

    u_r = n * np.cos(theta_retilt)
    v_r = n * np.sin(theta_retilt)

    return u_r, v_r


def compute_vector_stats(
    u_stack: np.ndarray,
    v_stack: np.ndarray,
    sigma_n_factor: float = 2.0,
    theta_std_deg: float = 120.0,
    min_samples: int = 2,
    amp_eps: float = 1e-3,
) -> Dict[str, np.ndarray]:
    """Performs statistics on a set of (u, v) vector fields, returning mean, standard deviation, sample count, and uncertainty mask.

    Parameters
    ------
    u_stack, v_stack : ndarray
        Shape (n_realizations, ny, nx), NaN indicates invalid values.
    sigma_n_factor : float
        Condition cond1: Only check sigma_n >= factor * n_mean when n_mean > amp_eps.
    theta_std_deg : float
        Condition cond2: Direction standard deviation >= this angle threshold (degrees).
    min_samples : int
        Condition cond3: Valid sample count < min_samples.
    amp_eps : float
        Values too small are considered "essentially stationary" and do not trigger cond1.

    Returns
    ------
    dict, containing the following keys:
        - u_mean, v_mean
        - n_mean, n_std
        - theta_mean, theta_std
        - N        (valid sample count)
        - bad_mask (boolean mask for grid points with high uncertainty)
    """

    u_stack = np.asarray(u_stack, dtype=float)
    v_stack = np.asarray(v_stack, dtype=float)

    # Magnitude and direction
    n_stack = np.hypot(u_stack, v_stack)
    theta_stack = np.arctan2(v_stack, u_stack)

    # Valid sample count
    valid = np.isfinite(n_stack)
    N = valid.sum(axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        u_mean = np.nanmean(u_stack, axis=0)
        v_mean = np.nanmean(v_stack, axis=0)
        n_mean = np.nanmean(n_stack, axis=0)
        n_std = np.nanstd(n_stack, axis=0)

    # Circular mean and standard deviation (unit: radians)
    sin_sum = np.nansum(np.sin(theta_stack), axis=0)
    cos_sum = np.nansum(np.cos(theta_stack), axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        theta_mean = np.arctan2(sin_sum, cos_sum)
        R = np.sqrt(sin_sum**2 + cos_sum**2) / np.clip(N, 1, None)
        R_clipped = np.clip(R, 1e-8, 1.0)
        theta_std = np.sqrt(-2.0 * np.log(R_clipped))

    # Set all grid points with no samples to NaN
    mask_zero = N == 0
    for arr in (u_mean, v_mean, n_mean, n_std, theta_mean, theta_std):
        arr[mask_zero] = np.nan

    # Uncertainty conditions (relaxed appropriately):
    # 1) Only treat "sigma_n comparable to or larger than n_mean" as unstable when n_mean > amp_eps;
    cond1 = (n_mean > amp_eps) & (n_std >= sigma_n_factor * n_mean)
    # 2) Direction standard deviation threshold relaxed from 90° to theta_std_deg (default 120°);
    cond2 = theta_std >= np.deg2rad(theta_std_deg)
    # 3) Too few valid samples still considered unreliable, relaxed from 3 to min_samples (default 2);
    cond3 = N < min_samples

    bad_mask = cond1 | cond2 | cond3

    return {
        "u_mean": u_mean,
        "v_mean": v_mean,
        "n_mean": n_mean,
        "n_std": n_std,
        "theta_mean": theta_mean,
        "theta_std": theta_std,
        "N": N,
        "bad_mask": bad_mask,
    }
