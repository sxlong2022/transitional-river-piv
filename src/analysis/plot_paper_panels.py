import argparse
import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Iterable, Optional

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.analysis.quantitative_relationships import (
    analyze_trunk_level_relationships,
    analyze_link_level_relationships,
    dominant_wavelength,
    fft_spectrum,
)
from src.analysis.plot_preset import setup_preset, get_paper_figsize


def _normalize_site(site: str) -> str:
    s = str(site)
    if s == "YR-A":
        return "HuangHe-A"
    if s == "YR-B":
        return "HuangHe-B"
    return s


def _display_site(site: str) -> str:
    s = _normalize_site(site)
    if s == "HuangHe-A":
        return "YR-A"
    if s == "HuangHe-B":
        return "YR-B"
    if s == "Jurua-A":
        return "Jurua"
    return str(site)


def _default_sBCMn_npz_path(site: str, mask_level: int) -> Path:
    site_io = _normalize_site(site)
    p = _PROJECT_ROOT / "results" / "PostprocessedPIV" / site_io
    cand = p / f"{site_io}_mask{int(mask_level)}_link_sBCMn_flat_step20_metric_v2.npz"
    if cand.exists():
        return cand
    return p / f"{site_io}_mask{int(mask_level)}_link_sBCMn_flat.npz"


def _fill_nan_linear_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x
    m = np.isfinite(x)
    if np.all(m):
        return x
    out = x.copy()
    idx = np.arange(x.size, dtype=float)
    if int(np.sum(m)) >= 2:
        out[~m] = np.interp(idx[~m], idx[m], out[m])
        return out
    fill = float(np.nanmean(out[m])) if np.any(m) else 0.0
    out[~m] = fill
    return out


def _estimate_step_m_from_s(s: np.ndarray) -> float:
    s = np.asarray(s, dtype=float)
    s = s[np.isfinite(s)]
    if s.size < 2:
        return float("nan")
    ds = np.diff(s)
    ds = ds[np.isfinite(ds)]
    if ds.size == 0:
        return float("nan")
    return float(np.nanmedian(ds))


# --- Helper functions for Overlay (duplicated from plot_trunk_overlay.py for stability) ---

def _iter_lines_from_vector(path: Path) -> Iterable[Tuple[str, np.ndarray]]:
    try:
        import fiona
        from shapely.geometry import LineString, MultiLineString, shape
    except ImportError:
        return []

    with fiona.open(path) as src:
        for idx, feat in enumerate(src):
            geom = feat.get("geometry")
            if geom is None:
                continue
            shp = shape(geom)
            lines = []
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
                else (str(props.get("link_id")) if props.get("link_id") is not None else str(idx))
            )

            for li, line in enumerate(lines):
                lid = f"{link_id}_{li}" if li > 0 else link_id
                coords = np.asarray(line.coords, dtype=float)
                if coords.ndim == 2 and coords.shape[0] >= 2:
                    yield lid, coords

def _read_basemap(path: Path) -> Tuple[Optional[np.ndarray], Optional[Tuple[float, float, float, float]]]:
    try:
        import rasterio
    except ImportError:
        return None, None

    with rasterio.open(path) as src:
        arr = src.read()
        b = src.bounds
        extent = (float(b.left), float(b.right), float(b.bottom), float(b.top))

    if arr.ndim == 2:
        img = arr
    else:
        if arr.shape[0] >= 3:
            rgb = np.transpose(arr[:3, :, :], (1, 2, 0)).astype(float)
            q2 = np.nanpercentile(rgb, 2)
            q98 = np.nanpercentile(rgb, 98)
            if np.isfinite(q2) and np.isfinite(q98) and q98 > q2:
                rgb = (rgb - q2) / (q98 - q2)
            img = np.clip(rgb, 0.0, 1.0)
        else:
            img = arr[0]
    return img, extent

def _read_basemap_extent(path: Path) -> Optional[Tuple[float, float, float, float]]:
    try:
        import rasterio
    except ImportError:
        return None

    with rasterio.open(path) as src:
        b = src.bounds
        return (float(b.left), float(b.right), float(b.bottom), float(b.top))

def _plot_segments(ax, segments: List[np.ndarray], color: str, lw: float, alpha: float) -> None:
    from matplotlib.collections import LineCollection
    if not segments:
        return
    lc = LineCollection(segments, colors=color, linewidths=lw, alpha=alpha, clip_on=True)
    lc.set_clip_path(ax.patch)
    ax.add_collection(lc)

def _add_panel_label(ax: plt.Axes, text: str, *, pos: str = "upper-left") -> None:
    pos = str(pos)
    if pos == "upper-right":
        x, y, ha, va = 0.98, 0.98, "right", "top"
    elif pos == "lower-left":
        x, y, ha, va = 0.02, 0.02, "left", "bottom"
    elif pos == "lower-right":
        x, y, ha, va = 0.98, 0.02, "right", "bottom"
    else:
        x, y, ha, va = 0.02, 0.98, "left", "top"

    t = ax.text(
        float(x),
        float(y),
        str(text),
        transform=ax.transAxes,
        ha=str(ha),
        va=str(va),
        fontsize=9,
        color="k",
        zorder=20,
    )
    t.set_bbox(dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.25))
    t.set_path_effects([pe.Stroke(linewidth=2.0, foreground="w"), pe.Normal()])

def _add_scalebar(
    ax: plt.Axes,
    extent: Tuple[float, float, float, float],
    *,
    length_m: float,
    pos: str = "bottom-right",
    box_alpha: float = 0.75,
    x_offset_frac: float = 0.0,
    y_offset_frac: float = 0.0,
) -> None:
    if extent is None:
        return
    if float(length_m) <= 0:
        return

    xmin, xmax, ymin, ymax = extent
    dx = float(xmax - xmin)
    dy = float(ymax - ymin)
    if dx <= 0 or dy <= 0:
        return

    # If extent is in lon/lat degrees (WGS84), convert meters -> degrees of longitude at mid-latitude
    length_x = float(length_m)
    is_lonlat = (
        -180.0 <= float(xmin) <= 180.0
        and -180.0 <= float(xmax) <= 180.0
        and -90.0 <= float(ymin) <= 90.0
        and -90.0 <= float(ymax) <= 90.0
        and dx < 10.0
        and dy < 10.0
    )
    if is_lonlat:
        lat_c = 0.5 * (float(ymin) + float(ymax))
        meters_per_degree_lon = 111320.0 * math.cos(math.radians(float(lat_c)))
        if np.isfinite(meters_per_degree_lon) and meters_per_degree_lon > 0:
            length_x = float(length_m) / float(meters_per_degree_lon)

    x_off = float(x_offset_frac)
    y_off = float(y_offset_frac)

    if str(pos) == "bottom-left":
        x0 = xmin + (0.06 + x_off) * dx
        x1 = x0 + float(length_x)
        y0 = ymin + (0.10 + y_off) * dy
    elif str(pos) == "top-right":
        x1 = xmax - (0.06 + x_off) * dx
        x0 = x1 - float(length_x)
        y0 = ymin + (0.90 - y_off) * dy
    elif str(pos) == "top-left":
        x0 = xmin + (0.06 + x_off) * dx
        x1 = x0 + float(length_x)
        y0 = ymin + (0.90 - y_off) * dy
    else:
        x1 = xmax - (0.06 + x_off) * dx
        x0 = x1 - float(length_x)
        y0 = ymin + (0.10 + y_off) * dy

    xpad = 0.018 * dx
    ypad0 = 0.020 * dy
    ypad1 = 0.085 * dy
    x_box0 = float(x0) - xpad
    y_box0 = float(y0) - ypad0
    w_box = float(x1 - x0) + 2.0 * xpad
    h_box = ypad0 + ypad1
    ax.add_patch(
        FancyBboxPatch(
            (x_box0, y_box0),
            w_box,
            h_box,
            boxstyle="round,pad=0.01",
            transform=ax.transData,
            facecolor="white",
            edgecolor="none",
            alpha=float(box_alpha),
            zorder=18,
        )
    )

    ax.plot([x0, x1], [y0, y0], color="k", linewidth=2.0, solid_capstyle="butt", zorder=19)
    ax.plot([x0, x0], [y0, y0 + 0.01 * dy], color="k", linewidth=2.0, zorder=19)
    ax.plot([x1, x1], [y0, y0 + 0.01 * dy], color="k", linewidth=2.0, zorder=19)
    label_km = float(length_m) / 1000.0
    ax.text(
        0.5 * (x0 + x1),
        y0 + 0.02 * dy,
        f"{label_km:g} km",
        ha="center",
        va="bottom",
        color="k",
        fontsize=9,
        zorder=19,
    )

def _sorted_trunk_ids(trunk_links: Dict[str, List[str]]) -> List[str]:
    def key(tid: str) -> int:
        try:
            return int(str(tid).split("_")[-1])
        except Exception:
            return 10**9
    return sorted(list(trunk_links.keys()), key=key)

def _binned_stats(x: np.ndarray, y: np.ndarray, bin_edges: np.ndarray, min_count: int = 5):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    edges = np.asarray(bin_edges, dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mean = np.full(centers.shape, np.nan)
    p25 = np.full(centers.shape, np.nan)
    p75 = np.full(centers.shape, np.nan)
    count = np.zeros(centers.shape, dtype=int)

    for i in range(centers.size):
        lo, hi = edges[i], edges[i + 1]
        m = (x >= lo) & (x < hi) if i < centers.size - 1 else (x >= lo) & (x <= hi)
        v = np.isfinite(x) & np.isfinite(y) & m
        yi = y[v]
        count[i] = int(yi.size)
        if yi.size < min_count:
            continue
        mean[i] = float(np.mean(yi))
        p25[i] = float(np.percentile(yi, 25))
        p75[i] = float(np.percentile(yi, 75))
    
    return {"centers": centers, "mean": mean, "p25": p25, "p75": p75, "count": count}

def plot_scatter_2x2(
    sites: List[str],
    mask: int,
    min_samples: int,
    min_arc_length: float,
    n_bins: int,
    out_path: Path,
    dpi: int,
    show_titles: bool = True,
    log_clip_q_low: float = 10.0,
    log_clip_q_high: float = 99.0,
    preset: str = "",
):
    """Creates a 2x2 grid of link-scale scatter plots (rows: |Mn|-|C| and |Mn|-B; cols: sites)."""

    figsize = get_paper_figsize(190, 130)
    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True, sharex="row", sharey="row")

    cols_data: List[Dict[str, object]] = []
    for col, site in enumerate(sites[:2]):
        site_io = _normalize_site(site)
        npz_path = _PROJECT_ROOT / "results" / "PostprocessedPIV" / site_io / f"{site_io}_mask{mask}_link_sBCMn_flat_step20_metric_v2.npz"
        if not npz_path.exists():
            cols_data.append({"site_io": site_io, "ok": False})
            continue

        res = analyze_link_level_relationships(
            npz_path,
            use_abs_mn=True,
            min_samples=int(min_samples),
            min_arc_length=float(min_arc_length),
        )
        metrics = res.metrics

        xC = np.asarray(metrics.get("mean_abs_C", np.array([])), dtype=float)
        yM = np.asarray(metrics.get("mean_abs_Mn", np.array([])), dtype=float)
        xB = np.asarray(metrics.get("mean_B", np.array([])), dtype=float)
        n_samples_arr = np.asarray(metrics.get("n_samples", np.array([])), dtype=float)
        arc = np.asarray(metrics.get("arc_length", np.array([])), dtype=float)

        mC = np.isfinite(xC) & np.isfinite(yM)
        mC &= (n_samples_arr >= float(min_samples))
        mC &= np.isfinite(arc) & (arc >= float(min_arc_length))
        mC &= (xC > 0)

        mB = np.isfinite(xB) & np.isfinite(yM)
        mB &= (n_samples_arr >= float(min_samples))
        mB &= np.isfinite(arc) & (arc >= float(min_arc_length))

        cols_data.append(
            {
                "site_io": site_io,
                "ok": True,
                "xC": xC,
                "xB": xB,
                "yM": yM,
                "mC": mC,
                "mB": mB,
                "fits": res.fits,
            }
        )

    # Global, comparable bins across both sites
    _xC_parts = [
        np.asarray(d.get("xC"), dtype=float)[np.asarray(d.get("mC"), dtype=bool)]
        for d in cols_data
        if d.get("ok")
    ]
    _xB_parts = [
        np.asarray(d.get("xB"), dtype=float)[np.asarray(d.get("mB"), dtype=bool)]
        for d in cols_data
        if d.get("ok")
    ]
    _yM_parts = [
        np.asarray(d.get("yM"), dtype=float)[np.asarray(d.get("mC"), dtype=bool) | np.asarray(d.get("mB"), dtype=bool)]
        for d in cols_data
        if d.get("ok")
    ]

    xC_all = np.concatenate(_xC_parts) if len(_xC_parts) > 0 else np.array([], dtype=float)
    xB_all = np.concatenate(_xB_parts) if len(_xB_parts) > 0 else np.array([], dtype=float)
    yM_all = np.concatenate(_yM_parts) if len(_yM_parts) > 0 else np.array([], dtype=float)

    edgesC = None
    if xC_all.size > 0:
        qlo = float(log_clip_q_low)
        qhi = float(log_clip_q_high)
        qlo = max(0.0, min(qlo, 50.0))
        qhi = max(50.0, min(qhi, 100.0))
        lo = float(np.nanpercentile(xC_all, qlo))
        hi = float(np.nanpercentile(xC_all, qhi))
        lo = max(lo, float(np.nanmin(xC_all)))
        hi = min(hi, float(np.nanmax(xC_all)))
        if (not np.isfinite(lo)) or lo <= 0:
            lo = float(np.nanmin(xC_all[xC_all > 0]))
        if (not np.isfinite(hi)) or hi <= lo:
            hi = lo * 1.01
        edgesC = np.logspace(np.log10(lo), np.log10(hi), int(n_bins) + 1)

    edgesB = None
    if xB_all.size > 0:
        loB = float(np.nanmin(xB_all))
        hiB = float(np.nanmax(xB_all))
        if np.isfinite(loB) and np.isfinite(hiB) and hiB > loB:
            edgesB = np.linspace(loB, hiB, int(n_bins) + 1)

    if yM_all.size > 0 and np.any(np.isfinite(yM_all)):
        y0 = float(np.nanmin(yM_all))
        y1 = float(np.nanmax(yM_all))
        if np.isfinite(y0) and np.isfinite(y1) and y1 > y0:
            pad = 0.05 * (y1 - y0)
            ylims = (max(0.0, y0 - pad), y1 + pad)
            for rr in range(2):
                for cc in range(2):
                    axes[rr, cc].set_ylim(*ylims)

    for col, d in enumerate(cols_data[:2]):
        site_io = str(d.get("site_io", ""))
        if not bool(d.get("ok")):
            continue

        xC = np.asarray(d.get("xC"), dtype=float)
        xB = np.asarray(d.get("xB"), dtype=float)
        yM = np.asarray(d.get("yM"), dtype=float)
        mC = np.asarray(d.get("mC"), dtype=bool)
        mB = np.asarray(d.get("mB"), dtype=bool)

        # Row 0: |Mn| vs |C|
        ax0 = axes[0, col]
        ax0.scatter(xC[mC], yM[mC], s=6, alpha=0.15, color="0.2", edgecolors="none", rasterized=True)
        if edgesC is not None and np.any(mC):
            ax0.set_xscale("log")
            st = _binned_stats(xC[mC], yM[mC], edgesC, min_count=5)
            ax0.plot(st["centers"], st["mean"], color="tab:blue", linewidth=2.0)
            ax0.fill_between(st["centers"], st["p25"], st["p75"], color="tab:blue", alpha=0.16, linewidth=0)
            ax0.set_xlim(float(edgesC[0]), float(edgesC[-1]))
        if col == 0:
            ax0.set_ylabel(r"Mean $|M_{\mathrm{n}}|$ (m/yr)")
        ax0.set_xlabel(r"Mean $|C|$ (m$^{-1}$)")
        ax0.grid(True, linestyle=":", alpha=0.25, linewidth=0.6)
        if show_titles or str(preset) == "paper":
            ax0.set_title(_display_site(site_io))

        if str(preset) == "paper":
            n_used = int(np.sum(mC))
            # Get R² from fits
            fits = d.get("fits", {})
            r2_C = fits.get("Mn_C_linear_link")
            r2_val = r2_C.r2 if r2_C is not None and np.isfinite(r2_C.r2) else float("nan")
            label_text = f"$n$ = {n_used}\n$R^2$ = {r2_val:.3f}" if np.isfinite(r2_val) else f"$n$ = {n_used}"
            ax0.text(
                0.97,
                0.96,
                label_text,
                transform=ax0.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                linespacing=1.3,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.2),
            )

        # Row 1: |Mn| vs B
        ax1 = axes[1, col]
        ax1.scatter(xB[mB], yM[mB], s=6, alpha=0.15, color="0.2", edgecolors="none", rasterized=True)
        if edgesB is not None and np.any(mB):
            st = _binned_stats(xB[mB], yM[mB], edgesB, min_count=5)
            ax1.plot(st["centers"], st["mean"], color="tab:orange", linewidth=2.0)
            ax1.fill_between(st["centers"], st["p25"], st["p75"], color="tab:orange", alpha=0.16, linewidth=0)
            ax1.set_xlim(float(edgesB[0]), float(edgesB[-1]))
        if col == 0:
            ax1.set_ylabel(r"Mean $|M_{\mathrm{n}}|$ (m/yr)")
        ax1.set_xlabel(r"Mean $B$ (m)")
        ax1.grid(True, linestyle=":", alpha=0.25, linewidth=0.6)

        if str(preset) == "paper":
            n_used = int(np.sum(mB))
            # Get R² from fits
            fits = d.get("fits", {})
            r2_B = fits.get("Mn_B_linear_link")
            r2_val = r2_B.r2 if r2_B is not None and np.isfinite(r2_B.r2) else float("nan")
            label_text = f"$n$ = {n_used}\n$R^2$ = {r2_val:.3f}" if np.isfinite(r2_val) else f"$n$ = {n_used}"
            ax1.text(
                0.98,
                0.96,
                label_text,
                transform=ax1.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.2),
            )

        if str(preset) == "paper":
            _add_panel_label(ax0, f"({chr(ord('a') + int(0 * 2 + col))})", pos="upper-left")
            _add_panel_label(ax1, f"({chr(ord('a') + int(1 * 2 + col))})", pos="upper-left")

    fig.savefig(out_path, dpi=dpi)
    print(f"Saved scatter panel: {out_path}")


def plot_fft_spectra(
    sites: List[str],
    masks: List[int],
    k_trunks: int,
    min_trunk_length_m: float,
    endpoint_tol_m: float,
    weight_by: str,
    out_path: Path,
    dpi: int,
    show_titles: bool = True,
    preset: str = "",
) -> None:
    ncols = int(len(sites))
    if ncols <= 0:
        return

    if str(preset) == "paper":
        figsize = get_paper_figsize(190, 105)
    else:
        figsize = (14, 7)

    fig, axes = plt.subplots(2, ncols, figsize=figsize, constrained_layout=True, sharey="row")
    if ncols == 1:
        axes = np.asarray(axes).reshape(2, 1)

    def _smooth_1d(y: np.ndarray, win: int = 9) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if y.size < 5:
            return y
        w = int(win)
        w = max(3, w)
        if w % 2 == 0:
            w += 1
        w = min(w, y.size if (y.size % 2 == 1) else max(3, y.size - 1))
        if w < 3:
            return y
        k = np.ones(w, dtype=float) / float(w)
        return np.convolve(y, k, mode="same")

    def _dominant_wavelength_limited(
        x: np.ndarray,
        step_m: float,
        max_lambda_km: Optional[float] = None,
    ) -> Dict[str, float]:
        spec = fft_spectrum(x, step_m=float(step_m), detrend=True)
        freq = np.asarray(spec.get("freq", np.array([])), dtype=float)
        amp = np.asarray(spec.get("amp", np.array([])), dtype=float)
        phase = np.asarray(spec.get("phase", np.array([])), dtype=float)

        m = np.isfinite(freq) & np.isfinite(amp) & (freq > 0)
        freq = freq[m]
        amp = amp[m]
        phase = phase[m]
        if freq.size < 2:
            return {"lambda_m": float("nan"), "freq": float("nan"), "amp": float("nan"), "phase": float("nan")}

        if max_lambda_km is not None and np.isfinite(max_lambda_km) and float(max_lambda_km) > 0:
            lam_km = (1.0 / freq) / 1000.0
            mm = np.isfinite(lam_km) & (lam_km > 0) & (lam_km <= float(max_lambda_km))
            freq = freq[mm]
            amp = amp[mm]
            phase = phase[mm]
            if freq.size < 2:
                return {"lambda_m": float("nan"), "freq": float("nan"), "amp": float("nan"), "phase": float("nan")}

        idx = int(np.argmax(amp))
        f = float(freq[idx])
        lam_m = float("nan") if f == 0 else float(1.0 / f)
        return {"lambda_m": lam_m, "freq": f, "amp": float(amp[idx]), "phase": float(phase[idx])}

    def _lambda_spectrum(
        x: np.ndarray,
        step_m: float,
        max_lambda_km: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        spec = fft_spectrum(x, step_m=float(step_m), detrend=True)
        freq = np.asarray(spec.get("freq", np.array([])), dtype=float)
        amp = np.asarray(spec.get("amp", np.array([])), dtype=float)
        m = np.isfinite(freq) & np.isfinite(amp) & (freq > 0)
        freq = freq[m]
        amp = amp[m]
        if freq.size == 0:
            return np.array([]), np.array([])
        lam_km = (1.0 / freq) / 1000.0
        mm = np.isfinite(lam_km) & (lam_km > 0)
        if max_lambda_km is not None and np.isfinite(max_lambda_km) and float(max_lambda_km) > 0:
            mm &= lam_km <= float(max_lambda_km)
        lam_km = lam_km[mm]
        amp = amp[mm]
        if lam_km.size == 0:
            return np.array([]), np.array([])
        order = np.argsort(lam_km)
        lam_km = lam_km[order]
        amp = amp[order]
        amax = float(np.nanmax(amp)) if np.any(np.isfinite(amp)) else float("nan")
        amp_n = (amp / amax) if np.isfinite(amax) and amax > 0 else amp
        return lam_km, amp_n

    lamB_min, lamB_max = float("inf"), float("-inf")
    lamC_min, lamC_max = float("inf"), float("-inf")

    summary: Dict[str, object] = {
        "mode": "fft_spectra",
        "sites": [str(s) for s in sites],
        "masks": [int(m) for m in masks],
        "k_trunks": int(k_trunks),
        "min_trunk_length_m": float(min_trunk_length_m),
        "endpoint_tol_m": float(endpoint_tol_m),
        "weight_by": str(weight_by),
        "per_site": {},
    }

    for col, site in enumerate(list(sites)):
        mask = int(masks[col] if col < len(masks) else masks[-1])
        site_io = _normalize_site(site)
        npz_path = _default_sBCMn_npz_path(site, mask)

        axB = axes[0, col]
        axC = axes[1, col]
        for ax in (axB, axC):
            ax.grid(True, linestyle=":", alpha=0.25, linewidth=0.6)
            ax.set_ylim(0.0, 1.05)
            ax.set_xscale("log")

        if show_titles or str(preset) == "paper":
            axB.set_title(_display_site(site_io))

        if col == 0:
            axB.set_ylabel("Normalized amplitude")
            axC.set_ylabel("Normalized amplitude")
        axC.set_xlabel(r"Wavelength ($\mathrm{km}$)")

        if not npz_path.exists():
            axB.text(0.5, 0.5, "Missing data", transform=axB.transAxes, ha="center", va="center")
            axC.text(0.5, 0.5, "Missing data", transform=axC.transAxes, ha="center", va="center")
            summary["per_site"][site_io] = {"ok": False, "path": str(npz_path), "mask": int(mask)}
            continue

        res = analyze_trunk_level_relationships(
            npz_path,
            k_trunks=int(k_trunks),
            endpoint_tol_m=float(endpoint_tol_m),
            weight_by=str(weight_by),
            min_trunk_length_m=float(min_trunk_length_m),
        )
        if not res.trunks:
            axB.text(0.5, 0.5, "No trunks", transform=axB.transAxes, ha="center", va="center")
            axC.text(0.5, 0.5, "No trunks", transform=axC.transAxes, ha="center", va="center")
            summary["per_site"][site_io] = {"ok": False, "path": str(npz_path), "mask": int(mask), "reason": "no_trunks"}
            continue

        trunk_id = "trunk_1" if "trunk_1" in res.trunks else sorted(res.trunks.keys())[0]
        tr = res.trunks[trunk_id]
        s = np.asarray(tr.get("s", np.array([])), dtype=float)
        B = np.asarray(tr.get("B", np.array([])), dtype=float)
        C = np.asarray(tr.get("C", np.array([])), dtype=float)

        step_m = float(res.diagnostics.get("step_m", float("nan"))) if isinstance(res.diagnostics, dict) else float("nan")
        if (not np.isfinite(step_m)) or step_m <= 0:
            step_m = _estimate_step_m_from_s(s)

        B_f = _fill_nan_linear_1d(B)
        C_f = _fill_nan_linear_1d(np.abs(C))

        if not np.isfinite(step_m) or step_m <= 0:
            axB.text(0.5, 0.5, "Invalid step", transform=axB.transAxes, ha="center", va="center")
            axC.text(0.5, 0.5, "Invalid step", transform=axC.transAxes, ha="center", va="center")
            summary["per_site"][site_io] = {"ok": False, "path": str(npz_path), "mask": int(mask), "reason": "invalid_step_m"}
            continue

        trunk_len_km = float(np.nanmax(s) - np.nanmin(s)) / 1000.0 if np.isfinite(s).any() else float("nan")
        max_lambda_km = 0.9 * trunk_len_km if np.isfinite(trunk_len_km) and trunk_len_km > 0 else None

        lamB_km, ampB = _lambda_spectrum(B_f, step_m=float(step_m), max_lambda_km=max_lambda_km)
        lamC_km, ampC = _lambda_spectrum(C_f, step_m=float(step_m), max_lambda_km=max_lambda_km)

        ampB_plot = _smooth_1d(ampB, win=11)
        ampC_plot = _smooth_1d(ampC, win=11)

        if lamB_km.size > 0:
            axB.plot(lamB_km, ampB_plot, color="tab:blue", linewidth=2.0)
            lamB_min = min(lamB_min, float(np.nanmin(lamB_km)))
            lamB_max = max(lamB_max, float(np.nanmax(lamB_km)))
        if lamC_km.size > 0:
            axC.plot(lamC_km, ampC_plot, color="tab:green", linewidth=2.0)
            lamC_min = min(lamC_min, float(np.nanmin(lamC_km)))
            lamC_max = max(lamC_max, float(np.nanmax(lamC_km)))

        domB = _dominant_wavelength_limited(B_f, step_m=float(step_m), max_lambda_km=max_lambda_km)
        domC = _dominant_wavelength_limited(C_f, step_m=float(step_m), max_lambda_km=max_lambda_km)
        domB_km = float(domB.get("lambda_m", float("nan"))) / 1000.0 if isinstance(domB, dict) else float("nan")
        domC_km = float(domC.get("lambda_m", float("nan"))) / 1000.0 if isinstance(domC, dict) else float("nan")

        if np.isfinite(domB_km) and domB_km > 0:
            axB.axvline(domB_km, color="tab:blue", linestyle="--", alpha=0.8, linewidth=1.2)
            if str(preset) == "paper":
                axB.text(
                    0.98,
                    0.90,
                    f"$\\lambda_B = {domB_km:.2f}\\,\\mathrm{{km}}$",
                    transform=axB.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.2),
                )

        if np.isfinite(domC_km) and domC_km > 0:
            axC.axvline(domC_km, color="tab:green", linestyle="--", alpha=0.8, linewidth=1.2)
            if str(preset) == "paper":
                axC.text(
                    0.98,
                    0.90,
                    f"$\\lambda_{{|C|}} = {domC_km:.2f}\\,\\mathrm{{km}}$",
                    transform=axC.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.2),
                )

        if str(preset) == "paper":
            _add_panel_label(axB, f"({chr(ord('a') + int(col))})", pos="upper-left")
            _add_panel_label(axC, f"({chr(ord('a') + int(ncols + col))})", pos="upper-left")

        trunk_len_km = float(np.nanmax(s) - np.nanmin(s)) / 1000.0 if np.isfinite(s).any() else float("nan")
        summary["per_site"][site_io] = {
            "ok": True,
            "path": str(npz_path),
            "mask": int(mask),
            "trunk_id": str(trunk_id),
            "step_m": float(step_m),
            "trunk_length_km": float(trunk_len_km),
            "dominant_B": domB,
            "dominant_abs_C": domC,
        }

    if np.isfinite(lamB_min) and np.isfinite(lamB_max) and lamB_max > lamB_min:
        for col in range(ncols):
            axes[0, col].set_xlim(float(lamB_min), float(lamB_max))
    if np.isfinite(lamC_min) and np.isfinite(lamC_max) and lamC_max > lamC_min:
        for col in range(ncols):
            axes[1, col].set_xlim(float(lamC_min), float(lamC_max))

    fig.savefig(out_path, dpi=dpi)
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved FFT spectra panel: {out_path}")
    print(f"Saved summary: {summary_path}")


def plot_dimless_cb(
    sites: List[str],
    masks: List[int],
    min_samples: int,
    min_arc_length: float,
    n_bins: int,
    out_path: Path,
    dpi: int,
    show_titles: bool = True,
    log_clip_q_low: float = 10.0,
    log_clip_q_high: float = 99.0,
    preset: str = "",
) -> None:
    ncols = int(len(sites))
    if ncols <= 0:
        return

    if str(preset) == "paper":
        figsize = get_paper_figsize(190, 70)
    else:
        figsize = (14, 4)

    fig, axes = plt.subplots(1, ncols, figsize=figsize, constrained_layout=True, sharey=True)
    if ncols == 1:
        axes = np.asarray([axes])

    cols_data: List[Dict[str, object]] = []
    for col, site in enumerate(list(sites)):
        mask = int(masks[col] if col < len(masks) else masks[-1])
        site_io = _normalize_site(site)
        npz_path = _default_sBCMn_npz_path(site, mask)
        if not npz_path.exists():
            cols_data.append({"site_io": site_io, "mask": int(mask), "path": str(npz_path), "ok": False})
            continue

        res = analyze_link_level_relationships(
            npz_path,
            use_abs_mn=True,
            min_samples=int(min_samples),
            min_arc_length=float(min_arc_length),
        )
        metrics = res.metrics

        xC = np.asarray(metrics.get("mean_abs_C", np.array([])), dtype=float)
        xB = np.asarray(metrics.get("mean_B", np.array([])), dtype=float)
        yM = np.asarray(metrics.get("mean_abs_Mn", np.array([])), dtype=float)
        n_samples_arr = np.asarray(metrics.get("n_samples", np.array([])), dtype=float)
        arc = np.asarray(metrics.get("arc_length", np.array([])), dtype=float)

        m = np.isfinite(xC) & np.isfinite(xB) & np.isfinite(yM)
        m &= (n_samples_arr >= float(min_samples))
        m &= np.isfinite(arc) & (arc >= float(min_arc_length))
        m &= (xC > 0) & (xB > 0)

        x = xC[m] * xB[m]
        y = yM[m]
        cols_data.append(
            {
                "site_io": site_io,
                "mask": int(mask),
                "path": str(npz_path),
                "ok": True,
                "x": x,
                "y": y,
                "N": int(np.sum(m)),
            }
        )

    x_all_parts = [np.asarray(d.get("x"), dtype=float) for d in cols_data if d.get("ok")]
    y_all_parts = [np.asarray(d.get("y"), dtype=float) for d in cols_data if d.get("ok")]
    x_all = np.concatenate(x_all_parts) if len(x_all_parts) > 0 else np.array([], dtype=float)
    y_all = np.concatenate(y_all_parts) if len(y_all_parts) > 0 else np.array([], dtype=float)

    edgesX = None
    xlim_lo, xlim_hi = None, None
    if x_all.size > 0:
        # Robust log-x range: derive limits from per-site percentiles
        # (avoids near-zero numerical outliers stretching the axis)
        qlo = float(log_clip_q_low)
        qhi = float(log_clip_q_high)
        qlo = max(0.0, min(qlo, 50.0))
        qhi = max(50.0, min(qhi, 100.0))

        per_lo: List[float] = []
        per_hi: List[float] = []
        for d in cols_data:
            if not bool(d.get("ok")):
                continue
            xs = np.asarray(d.get("x"), dtype=float)
            xs = xs[np.isfinite(xs) & (xs > 0)]
            if xs.size == 0:
                continue
            per_lo.append(float(np.nanpercentile(xs, qlo)))
            per_hi.append(float(np.nanpercentile(xs, qhi)))

        x_pos = x_all[np.isfinite(x_all) & (x_all > 0)]
        lo_all = float(np.nanmin(x_pos)) if x_pos.size > 0 else float("nan")
        hi_all = float(np.nanmax(x_pos)) if x_pos.size > 0 else float("nan")

        if len(per_lo) > 0 and len(per_hi) > 0:
            xlim_lo = float(np.nanmin(per_lo))
            xlim_hi = float(np.nanmax(per_hi))
        else:
            xlim_lo = lo_all
            xlim_hi = hi_all

        if (not np.isfinite(xlim_lo)) or xlim_lo <= 0:
            xlim_lo = lo_all
        if (not np.isfinite(xlim_hi)) or xlim_hi <= xlim_lo:
            xlim_hi = hi_all
        if (not np.isfinite(xlim_hi)) or xlim_hi <= xlim_lo:
            xlim_hi = float(xlim_lo) * 1.01

        # Cap log-range for readability (near-zero curvature values can otherwise dominate the axis)
        if np.isfinite(xlim_lo) and np.isfinite(xlim_hi) and xlim_hi > xlim_lo:
            max_decades = 4.0 if str(preset) == "paper" else 6.0
            decades = float(np.log10(xlim_hi) - np.log10(xlim_lo))
            if np.isfinite(decades) and decades > max_decades:
                xlim_lo = float(xlim_hi) / float(10 ** max_decades)

        edgesX = np.logspace(np.log10(float(xlim_lo)), np.log10(float(xlim_hi)), int(n_bins) + 1)

    if y_all.size > 0 and np.any(np.isfinite(y_all)):
        y_pos = y_all[np.isfinite(y_all) & (y_all >= 0)]
        if y_pos.size > 0:
            y0 = 0.0
            y1 = float(np.nanpercentile(y_pos, 99.0 if str(preset) == "paper" else 99.5))
            if np.isfinite(y1) and y1 > y0:
                pad = 0.06 * (y1 - y0)
                ylims = (y0, y1 + pad)
                for ax in axes:
                    ax.set_ylim(*ylims)

    summary: Dict[str, object] = {
        "mode": "dimless_cb",
        "sites": [str(s) for s in sites],
        "masks": [int(m) for m in masks],
        "min_samples": int(min_samples),
        "min_arc_length": float(min_arc_length),
        "n_bins": int(n_bins),
        "per_site": {},
    }

    for col, d in enumerate(cols_data):
        site_io = str(d.get("site_io", ""))
        ax = axes[col]
        ax.grid(True, linestyle=":", alpha=0.25, linewidth=0.6)
        ax.set_xscale("log")
        if col == 0:
            ax.set_ylabel(r"Mean $|M_{\mathrm{n}}|$ (m/yr)")
        ax.set_xlabel(r"Mean $|C|\,B$ (-)")
        if show_titles or str(preset) == "paper":
            ax.set_title(_display_site(site_io))

        if not bool(d.get("ok")):
            ax.text(0.5, 0.5, "Missing data", transform=ax.transAxes, ha="center", va="center")
            summary["per_site"][site_io] = {"ok": False, "path": str(d.get("path", "")), "mask": int(d.get("mask", -1))}
            continue

        x = np.asarray(d.get("x"), dtype=float)
        y = np.asarray(d.get("y"), dtype=float)
        n_site = int(d.get("N", 0))
        s_scatter = 6 if n_site >= 40 else 10
        a_scatter = 0.15 if n_site >= 40 else 0.28
        ax.scatter(x, y, s=s_scatter, alpha=a_scatter, color="0.2", edgecolors="none", rasterized=True)
        if edgesX is not None and x.size > 0:
            st = _binned_stats(x, y, edgesX, min_count=5)
            valid = np.isfinite(st["mean"])
            n_valid = int(np.count_nonzero(valid))
            if n_valid >= 2:
                ax.plot(st["centers"], st["mean"], color="tab:purple", linewidth=2.0)
                ax.fill_between(st["centers"], st["p25"], st["p75"], color="tab:purple", alpha=0.16, linewidth=0)
            else:
                if n_valid == 1:
                    i0 = int(np.where(valid)[0][0])
                    xm = float(st["centers"][i0])
                    ym = float(st["mean"][i0])
                    y25 = float(st["p25"][i0])
                    y75 = float(st["p75"][i0])
                else:
                    xm = float(np.nanmedian(x))
                    ym = float(np.nanmedian(y))
                    y25 = float(np.nanpercentile(y, 25))
                    y75 = float(np.nanpercentile(y, 75))
                if np.isfinite(xm) and np.isfinite(ym) and np.isfinite(y25) and np.isfinite(y75):
                    yerr = np.array([[max(0.0, ym - y25)], [max(0.0, y75 - ym)]], dtype=float)
                    ax.errorbar(
                        [xm],
                        [ym],
                        yerr=yerr,
                        fmt="D",
                        color="tab:purple",
                        markersize=4.5,
                        elinewidth=1.4,
                        capsize=3.0,
                        alpha=0.95,
                        zorder=5,
                    )

        if xlim_lo is not None and xlim_hi is not None and xlim_hi > xlim_lo:
            ax.set_xlim(float(xlim_lo), float(xlim_hi))
        elif edgesX is not None:
            ax.set_xlim(float(edgesX[0]), float(edgesX[-1]))

        if str(preset) == "paper":
            ax.text(
                0.98,
                0.96,
                f"$n$ = {int(d.get('N', 0))}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.2),
            )
            _add_panel_label(ax, f"({chr(ord('a') + int(col))})", pos="upper-left")

        summary["per_site"][site_io] = {"ok": True, "path": str(d.get("path", "")), "mask": int(d.get("mask", -1)), "N": int(d.get("N", 0))}

    fig.savefig(out_path, dpi=dpi)
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved dimless C*B panel: {out_path}")
    print(f"Saved summary: {summary_path}")


# --- Panel Plotters ---

def plot_overlay_2x2(
    sites: List[str],
    masks: List[int],
    year: int,
    k_trunks: int,
    min_trunk_length_m: float,
    endpoint_tol_m: float,
    weight_by: str,
    out_path: Path,
    dpi: int,
    show_titles: bool = True,
    preset: str = "",
):
    """
    Creates a 2x2 grid of overlay plots.
    Rows: Sites
    Cols: Masks
    """
    row_aspects: List[float] = []
    for site in sites:
        ars: List[float] = []
        for mask in masks:
            basemap_tif = (
                _PROJECT_ROOT
                / "data"
                / "GEOTIFFS"
                / _normalize_site(site)
                / f"mask{mask}"
                / f"{_normalize_site(site)}_{year}_01-01_12-31_mask{mask}.tif"
            )
            if not basemap_tif.exists():
                continue
            extent = _read_basemap_extent(basemap_tif)
            if extent is None:
                continue
            dx = float(extent[1] - extent[0])
            dy = float(extent[3] - extent[2])
            if dx > 0 and dy > 0:
                ars.append(dy / dx)
        row_aspects.append(float(max(ars)) if ars else 1.0)

    # 2x2 layout, full width
    figsize = get_paper_figsize(190, 160) # Square-ish panels
    fig, axes = plt.subplots(
        len(sites),
        len(masks),
        figsize=figsize,
        constrained_layout=True,
        gridspec_kw={"height_ratios": row_aspects},
    )

    if str(preset) == "paper":
        fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.02)
    
    # Ensure axes is 2D array
    if len(sites) == 1 and len(masks) == 1:
        axes = np.array([[axes]])
    elif len(sites) == 1:
        axes = axes.reshape(1, -1)
    elif len(masks) == 1:
        axes = axes.reshape(-1, 1)

    for i, site in enumerate(sites):
        for j, mask in enumerate(masks):
            ax = axes[i, j]
            
            # Paths
            npz_path = _PROJECT_ROOT / "results" / "PostprocessedPIV" / _normalize_site(site) / f"{_normalize_site(site)}_mask{mask}_link_sBCMn_flat_step20_metric_v2.npz"
            links_shp = _PROJECT_ROOT / "results" / "RivGraph" / _normalize_site(site) / f"mask{mask}" / f"{_normalize_site(site)}_mask{mask}_links.shp"
            basemap_tif = _PROJECT_ROOT / "data" / "GEOTIFFS" / _normalize_site(site) / f"mask{mask}" / f"{_normalize_site(site)}_{year}_01-01_12-31_mask{mask}.tif"
            
            # Load Data
            all_segments = {}
            if links_shp.exists():
                for lid, coords in _iter_lines_from_vector(links_shp):
                    all_segments[str(lid)] = coords
            
            img, extent = None, None
            if basemap_tif.exists():
                img, extent = _read_basemap(basemap_tif)
            
            # Analyze Trunks
            if npz_path.exists():
                res = analyze_trunk_level_relationships(
                    npz_path,
                    k_trunks=k_trunks,
                    endpoint_tol_m=endpoint_tol_m,
                    weight_by=weight_by,
                    min_trunk_length_m=min_trunk_length_m
                )
            else:
                res = None

            # Plot Basemap
            if img is not None and extent is not None:
                if img.ndim == 3:
                    if str(preset) == "paper":
                        g = 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
                        ax.imshow(g, extent=extent, origin="upper", cmap="gray_r")
                    else:
                        ax.imshow(img, extent=extent, origin="upper")
                else:
                    cmap = "gray_r" if str(preset) == "paper" else "gray"
                    ax.imshow(img, extent=extent, origin="upper", cmap=cmap)
                ax.set_xlim(extent[0], extent[1])
                ax.set_ylim(extent[2], extent[3])
            
            # Plot All Links
            base_segs = list(all_segments.values())
            _plot_segments(ax, base_segs, color="0.6", lw=0.5, alpha=0.5)
            
            # Plot Trunks
            if res:
                trunk_ids = _sorted_trunk_ids(res.trunk_links)
                colors = ["tab:red", "tab:blue", "tab:green", "tab:orange"]
                for ti, tid in enumerate(trunk_ids):
                    lids = set([str(x) for x in res.trunk_links.get(tid, [])])
                    segs = [all_segments[l] for l in lids if l in all_segments]
                    _plot_segments(ax, segs, color=colors[ti % len(colors)], lw=2.0, alpha=0.9)
                
                # Annotate
                n_trunks = len(trunk_ids)
                critA = res.diagnostics.get("criterionA", {}) if isinstance(res.diagnostics, dict) else {}
                n_eff = float(critA.get("N_eff", np.nan)) if isinstance(critA, dict) else float("nan")
                ax.text(
                    0.98,
                    0.02,
                    f"N = {n_trunks}, $N_{{\\mathrm{{eff}}}}$ = {n_eff:.2f}",
                    transform=ax.transAxes,
                    va="bottom",
                    ha="right",
                    fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5),
                )

            # Labels
            ax.set_aspect("equal", adjustable="box")
            if str(preset) == "paper":
                # Reduce the apparent gap between the two rows under fixed aspect.
                # Push the top row down and the bottom row up within their grid cells.
                ax.set_anchor("S" if i == 0 else "N")
            if i == 0 and (show_titles or str(preset) == "paper"):
                ax.set_title(f"Mask {mask}")
            if j == 0:
                if str(preset) == "paper":
                    ax.set_ylabel(_display_site(site))
                else:
                    ax.set_ylabel(site)
            
            if str(preset) == "paper":
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.tick_params(axis="both", which="major", labelsize=6)
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

            if str(preset) == "paper":
                panel_idx = i * len(masks) + j
                panel_letter = chr(ord("a") + int(panel_idx))
                _add_panel_label(ax, f"({panel_letter})", pos="upper-left")
                if i == 0 and j == 0 and extent is not None:
                    # Place scalebar near the panel label (a), to its right.
                    _add_scalebar(
                        ax,
                        extent,
                        length_m=5000.0,
                        pos="top-left",
                        x_offset_frac=0.16,
                        y_offset_frac=0.015,
                    )

    if str(preset) == "paper":
        fig.savefig(out_path, dpi=dpi, bbox_inches=fig.bbox_inches)
    else:
        fig.savefig(out_path, dpi=dpi)
    print(f"Saved overlay panel: {out_path}")


def plot_profiles_2col(
    sites: List[str],
    masks: List[int],
    k_trunks: int,
    min_trunk_length_m: float,
    endpoint_tol_m: float,
    weight_by: str,
    out_path: Path,
    dpi: int,
    abs_mn: bool = True,
    abs_curv: bool = True,
    show_titles: bool = True,
    preset: str = "",
):
    """
    Creates a 3x2 grid of profiles.
    Cols: Site A vs Site B
    Rows: Width, Curvature, Migration Rate
    """
    # 2 columns (190mm total width), 3 rows (height approx 150-160mm)
    figsize = get_paper_figsize(190, 160)
    fig, axes = plt.subplots(3, 2, figsize=figsize, sharex="col", sharey="row", constrained_layout=True)
    
    # If only 1 site provided, we might fail or just plot one column? 
    # Logic assumes 2 sites for comparison. If 1 site, handle gracefully or error.
    if len(sites) < 2:
        print("Warning: profiles_2col intended for 2 sites comparison. Using same site twice if needed.")
        if len(sites) == 1:
            sites = [sites[0], sites[0]]
            if len(masks) == 1:
                masks = [masks[0], masks[0]]

    for col, site in enumerate(sites[:2]):
        site_io = _normalize_site(site)
        mask = masks[col] if col < len(masks) else masks[0]
        
        npz_path = _PROJECT_ROOT / "results" / "PostprocessedPIV" / site_io / f"{site_io}_mask{mask}_link_sBCMn_flat_step20_metric_v2.npz"
        
        if not npz_path.exists():
            continue
            
        res = analyze_trunk_level_relationships(
            npz_path,
            k_trunks=k_trunks,
            endpoint_tol_m=endpoint_tol_m,
            weight_by=weight_by,
            min_trunk_length_m=min_trunk_length_m
        )
        
        # Get dominant trunk
        if not res.trunks:
            continue
            
        # Usually we want the main trunk.
        # If k_trunks > 1, maybe we plot the top 1? Or multiple?
        # For clarity in comparison, let's plot trunk_1 (dominant).
        # We can support plotting multiple if needed, but for "Paper Figure" clean comparison, primary trunk is best.
        
        trunk_ids = _sorted_trunk_ids(res.trunk_links)
        main_tid = trunk_ids[0]
        data = res.trunks[main_tid]
        
        s = data["s"]
        s_km = s / 1000.0
        B = data["B"]
        C = data["C"]
        Mn = data["Mn"]

        def _interp_nans(y: np.ndarray) -> np.ndarray:
            y = np.asarray(y, dtype=float)
            if y.size == 0:
                return y
            x = np.arange(y.size, dtype=float)
            m = np.isfinite(y)
            if int(m.sum()) < 2:
                return y
            yy = y.copy()
            yy[~m] = np.interp(x[~m], x[m], y[m])
            return yy

        def _smooth(y: np.ndarray, window_pts: int) -> np.ndarray:
            if int(window_pts) <= 1:
                return np.asarray(y, dtype=float)
            y2 = _interp_nans(y)
            w = np.ones(int(window_pts), dtype=float) / float(window_pts)
            return np.convolve(y2, w, mode="same")

        step_m = float(np.nanmedian(np.diff(s))) if np.size(s) > 1 else 20.0
        if (not np.isfinite(step_m)) or step_m <= 0:
            step_m = 20.0
        smooth_window_m = 500.0
        window_pts = max(5, int(round(float(smooth_window_m) / float(step_m))))
        if window_pts % 2 == 0:
            window_pts += 1

        C_plot = np.abs(C) if abs_curv else C
        Mn_plot = np.abs(Mn) if abs_mn else Mn

        B_trend = _smooth(B, window_pts)
        C_trend_plot = _smooth(C_plot, window_pts)
        Mn_trend_plot = _smooth(Mn_plot, window_pts)
        
        # Row 0: Width
        ax0 = axes[0, col]
        ax0.plot(s_km, B, color="tab:blue", lw=0.8, alpha=0.35)
        ax0.plot(s_km, B_trend, color="tab:blue", lw=1.8, alpha=0.95)
        if str(preset) != "paper":
            ax0.axhline(np.nanmean(B), color='tab:blue', ls='--', alpha=0.5, lw=0.8)
        if col == 0:
            ax0.set_ylabel(r"Width $B$ ($\mathrm{m}$)")
        ax0.grid(True, ls=":", alpha=0.25, linewidth=0.6)
        if show_titles:
            ax0.set_title(f"{site_io} Mask {mask}\n({main_tid})")
        elif str(preset) == "paper":
            col_title = _display_site(site_io)
            ax0.set_title(col_title)
        if str(preset) == "paper":
            _add_panel_label(ax0, f"({chr(ord('a') + int(0 * 2 + col))})", pos="upper-left")
        
        # Row 1: Curvature
        ax1 = axes[1, col]
        ax1.plot(s_km, C_plot, color="tab:green", lw=0.8, alpha=0.35)
        ax1.plot(s_km, C_trend_plot, color="tab:green", lw=1.8, alpha=0.95)
        ax1.axhline(0, color='k', lw=0.5, alpha=0.5)
        if col == 0:
            ax1.set_ylabel("Curvature $|C|$ (m$^{-1}$)" if abs_curv else "Curvature $C$ (m$^{-1}$)")
        ax1.grid(True, ls=":", alpha=0.25, linewidth=0.6)
        if str(preset) == "paper":
            _add_panel_label(ax1, f"({chr(ord('a') + int(1 * 2 + col))})", pos="upper-left")
        
        # Row 2: Migration
        ax2 = axes[2, col]
        ax2.plot(s_km, Mn_plot, color="tab:red", lw=0.8, alpha=0.35)
        ax2.plot(s_km, Mn_trend_plot, color="tab:red", lw=1.8, alpha=0.95)
        ax2.axhline(0, color='k', lw=0.5, alpha=0.5)
        if col == 0:
            ax2.set_ylabel("Migration rate $|M_{\\mathrm{n}}|$ (m/yr)" if abs_mn else "Migration rate $M_{\\mathrm{n}}$ (m/yr)")
        ax2.set_xlabel(r"Streamwise distance $s$ ($\mathrm{km}$)")
        ax2.grid(True, ls=":", alpha=0.25, linewidth=0.6)
        if str(preset) == "paper":
            _add_panel_label(ax2, f"({chr(ord('a') + int(2 * 2 + col))})", pos="upper-left")

    fig.savefig(out_path, dpi=dpi)
    print(f"Saved profiles panel: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=["overlay_2x2", "scatter_2x2", "profiles_2col", "fft_spectra", "dimless_cb"],
    )
    parser.add_argument("--sites", nargs="+", default=["YR-A", "YR-B"])
    parser.add_argument("--masks", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--out", type=str, required=False)

    # Analysis params
    parser.add_argument("--k-trunks", type=int, default=2)
    parser.add_argument("--min-trunk-length-m", type=float, default=5000.0)
    parser.add_argument("--endpoint-tol-m", type=float, default=80.0)
    parser.add_argument("--weight-by", type=str, default="length_B")

    # Scatter params
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--min-arc-length", type=float, default=200.0)
    parser.add_argument("--n-bins", type=int, default=20)
    parser.add_argument("--log-clip-q-low", type=float, default=10.0)
    parser.add_argument("--log-clip-q-high", type=float, default=99.0)

    # Profile params
    parser.add_argument("--abs-mn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--abs-curv", action=argparse.BooleanOptionalAction, default=True)

    # Preset
    parser.add_argument("--preset", type=str, default="paper")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--titles", type=str, default="auto", choices=["auto", "on", "off"])

    args = parser.parse_args()
    setup_preset(args.preset, args.dpi)

    if args.titles == "on":
        show_titles = True
    elif args.titles == "off":
        show_titles = False
    else:
        show_titles = args.preset != "paper"

    out_dir = _PROJECT_ROOT / "results" / "figures" / "paper"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "overlay_2x2":
        if not args.out:
            if [_normalize_site(s) for s in args.sites[:2]] == ["HuangHe-A", "HuangHe-B"] and sorted(args.masks) == [2, 4]:
                fname = "Fig4_Overlay_AvsB.png"
            else:
                fname = f"overlay_{'_'.join(args.sites)}_mask{''.join(map(str, args.masks))}.png"
            out_path = out_dir / fname
        else:
            out_path = Path(args.out)

        plot_overlay_2x2(
            sites=args.sites,
            masks=args.masks,
            year=args.year,
            k_trunks=args.k_trunks,
            min_trunk_length_m=args.min_trunk_length_m,
            endpoint_tol_m=args.endpoint_tol_m,
            weight_by=args.weight_by,
            out_path=out_path,
            dpi=args.dpi,
            show_titles=show_titles,
            preset=str(args.preset),
        )

    elif args.mode == "scatter_2x2":
        # Usually implies comparison of sites for a SPECIFIC mask
        # So we use the first mask in the list
        mask = args.masks[0]
        if not args.out:
            if [_normalize_site(s) for s in args.sites[:2]] == ["HuangHe-A", "HuangHe-B"] and int(mask) == 4:
                fname = "Fig7_Scatter_AvsB.png"
            else:
                fname = f"scatter_{'_'.join(args.sites)}_mask{mask}.png"
            out_path = out_dir / fname
        else:
            out_path = Path(args.out)

        plot_scatter_2x2(
            sites=args.sites,
            mask=mask,
            min_samples=args.min_samples,
            min_arc_length=args.min_arc_length,
            n_bins=args.n_bins,
            out_path=out_path,
            dpi=args.dpi,
            show_titles=show_titles,
            log_clip_q_low=args.log_clip_q_low,
            log_clip_q_high=args.log_clip_q_high,
            preset=str(args.preset),
        )

    elif args.mode == "profiles_2col":
        if not args.out:
            if [_normalize_site(s) for s in args.sites[:2]] == ["HuangHe-A", "HuangHe-B"]:
                fname = "Fig5_Profiles_AvsB.png"
            else:
                fname = f"profiles_{'_'.join(args.sites)}_mask{''.join(map(str, args.masks))}.png"
            out_path = out_dir / fname
        else:
            out_path = Path(args.out)

        plot_profiles_2col(
            sites=args.sites,
            masks=args.masks,
            k_trunks=args.k_trunks,
            min_trunk_length_m=args.min_trunk_length_m,
            endpoint_tol_m=args.endpoint_tol_m,
            weight_by=args.weight_by,
            out_path=out_path,
            dpi=args.dpi,
            abs_mn=args.abs_mn,
            abs_curv=args.abs_curv,
            show_titles=show_titles,
            preset=str(args.preset),
        )

    elif args.mode == "fft_spectra":
        sites = list(args.sites)
        masks = list(args.masks)
        if len(masks) < len(sites) and len(masks) > 0:
            masks = masks + [int(masks[-1])] * (len(sites) - len(masks))

        if not args.out:
            norm_sites = [_normalize_site(s) for s in sites]
            if norm_sites[:3] == ["HuangHe-A", "HuangHe-B", "Jurua-A"] and [int(m) for m in masks[:3]] == [4, 4, 1]:
                fname = "FigS1_FFTSpectra_AvsB_Jurua.png"
            else:
                fname = f"fft_spectra_{'_'.join(norm_sites)}_mask{'_'.join(map(str, masks))}.png"
            out_path = out_dir / fname
        else:
            out_path = Path(args.out)

        plot_fft_spectra(
            sites=sites,
            masks=[int(m) for m in masks],
            k_trunks=args.k_trunks,
            min_trunk_length_m=args.min_trunk_length_m,
            endpoint_tol_m=args.endpoint_tol_m,
            weight_by=args.weight_by,
            out_path=out_path,
            dpi=args.dpi,
            show_titles=show_titles,
            preset=str(args.preset),
        )

    elif args.mode == "dimless_cb":
        sites = list(args.sites)
        masks = list(args.masks)
        if len(masks) < len(sites) and len(masks) > 0:
            masks = masks + [int(masks[-1])] * (len(sites) - len(masks))

        if not args.out:
            norm_sites = [_normalize_site(s) for s in sites]
            if norm_sites[:3] == ["HuangHe-A", "HuangHe-B", "Jurua-A"] and [int(m) for m in masks[:3]] == [4, 4, 1]:
                fname = "FigS2_DimlessCB_AvsB_Jurua.png"
            else:
                fname = f"dimless_cb_{'_'.join(norm_sites)}_mask{'_'.join(map(str, masks))}.png"
            out_path = out_dir / fname
        else:
            out_path = Path(args.out)

        plot_dimless_cb(
            sites=sites,
            masks=[int(m) for m in masks],
            min_samples=args.min_samples,
            min_arc_length=args.min_arc_length,
            n_bins=args.n_bins,
            out_path=out_path,
            dpi=args.dpi,
            show_titles=show_titles,
            log_clip_q_low=args.log_clip_q_low,
            log_clip_q_high=args.log_clip_q_high,
            preset=str(args.preset),
        )

if __name__ == "__main__":
    main()
