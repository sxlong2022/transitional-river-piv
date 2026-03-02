from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Patch, FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
import numpy as np
import rasterio
import rasterio.windows
from scipy.ndimage import binary_dilation, binary_erosion
from skimage.measure import find_contours

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.analysis.plot_preset import get_paper_figsize, setup_preset
from src.preprocessing.prepared_imagery import get_geotiffs_dir, get_prepared_imagery_dir


def _read_rgb_geotiff(path: Path, window: rasterio.windows.Window | None) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(path) as src:
        if src.transform.is_identity and src.crs is None and not src.gcps[0]:
            raise ValueError(
                f"背景影像缺少地理参考（无法与 PIV/Mask 对齐）: {path}. "
                "请优先使用 GEOTIFFS/image 下的 GeoTIFF，或用 --background 显式指定有地理参考的影像。"
            )

        arr = src.read(window=window)
        m = src.dataset_mask(window=window)

    if arr.ndim == 2:
        img = np.stack([arr, arr, arr], axis=-1).astype(float)
    else:
        if arr.shape[0] >= 3:
            img = np.transpose(arr[:3, :, :], (1, 2, 0)).astype(float)
        else:
            img = np.repeat(arr[0, :, :][:, :, None], 3, axis=2).astype(float)

    q2 = float(np.nanpercentile(img, 2))
    q98 = float(np.nanpercentile(img, 98))
    if np.isfinite(q2) and np.isfinite(q98) and q98 > q2:
        img = (img - q2) / (q98 - q2)
    return np.clip(img, 0.0, 1.0), m > 0


def _apply_bg_black_threshold(img: np.ndarray, valid: np.ndarray, black_threshold: float) -> np.ndarray:
    if float(black_threshold) < 0:
        return valid

    if img.ndim != 3 or img.shape[2] != 3:
        return valid

    brightness = np.nanmean(img, axis=2)
    return valid & np.isfinite(brightness) & (brightness > float(black_threshold))


def _apply_bg_style(img: np.ndarray, style: str, sat: float) -> np.ndarray:
    if img.ndim != 3 or img.shape[2] != 3:
        return img

    if style == "rgb":
        return img

    gray = (0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]).astype(float)
    gray3 = np.stack([gray, gray, gray], axis=2)

    if style == "gray":
        return gray3

    if style == "desat":
        a = float(np.clip(sat, 0.0, 1.0))
        return a * img + (1.0 - a) * gray3

    return img


def _nice_number_125(x: float) -> float:
    x = float(x)
    if not np.isfinite(x) or x <= 0:
        return 1.0

    p = np.floor(np.log10(x))
    base = 10.0 ** p
    f = x / base
    if f <= 1.0:
        n = 1.0
    elif f <= 2.0:
        n = 2.0
    elif f <= 5.0:
        n = 5.0
    else:
        n = 10.0
    return float(n * base)


def _read_extent(path: Path, window: rasterio.windows.Window | None) -> tuple[float, float, float, float]:
    with rasterio.open(path) as src:
        if window is None:
            b = src.bounds
            return (float(b.left), float(b.right), float(b.bottom), float(b.top))

        b = rasterio.windows.bounds(window, src.transform)
        return (float(b[0]), float(b[2]), float(b[1]), float(b[3]))


def _read_mask(path: Path, window: rasterio.windows.Window | None) -> tuple[np.ndarray, rasterio.Affine]:
    with rasterio.open(path) as src:
        if window is None:
            arr = src.read(1)
            transform = src.transform
        else:
            arr = src.read(1, window=window, boundless=True, fill_value=0)
            transform = src.window_transform(window)

    m = np.asarray(arr)
    if m.dtype.kind in {"f"}:
        m = np.isfinite(m) & (m > 0)
    else:
        m = m > 0
    return m.astype(bool), transform


def _default_mask_path(site: str, mask_level: int, year: int) -> Path:
    root = get_geotiffs_dir(site) / f"mask{int(mask_level)}"
    return root / f"{site}_{int(year)}_01-01_12-31_mask.tif"


def _default_background_path(site: str, year: int) -> Path | None:
    img = get_geotiffs_dir(site) / "image" / f"{site}_{int(year)}_01-01_12-31_full_image.tif"
    if img.exists():
        return img

    color = get_prepared_imagery_dir(site) / "Color" / f"{site}_{int(year)}_01-01_12-31_full_image_color.tif"
    if color.exists():
        return color

    return None


def _default_piv_npz(site: str, mask_level: int) -> Path:
    return (
        _PROJECT_ROOT
        / "results"
        / "PostprocessedPIV"
        / site
        / f"jurua_mask{int(mask_level)}_multitilt_georef_step4a_strict.npz"
    )


def _compute_crop_window(
    background_path: Path,
    margin_px: int,
) -> rasterio.windows.Window | None:
    with rasterio.open(background_path) as src:
        m = src.dataset_mask()
        valid = m > 0
        b1 = src.read(1)
        valid = valid & np.isfinite(b1) & (b1 != 0)

        rows, cols = np.where(valid)
        if rows.size == 0 or cols.size == 0:
            return None

        r0 = max(0, int(rows.min()) - int(margin_px))
        r1 = min(int(src.height) - 1, int(rows.max()) + int(margin_px))
        c0 = max(0, int(cols.min()) - int(margin_px))
        c1 = min(int(src.width) - 1, int(cols.max()) + int(margin_px))

        return rasterio.windows.Window(
            col_off=float(c0),
            row_off=float(r0),
            width=float(c1 - c0 + 1),
            height=float(r1 - r0 + 1),
        )


def _dataset_bounds(path: Path) -> tuple[float, float, float, float]:
    with rasterio.open(path) as src:
        b = src.bounds
    return (float(b.left), float(b.right), float(b.bottom), float(b.top))


def _clamp_bounds_to_limit(
    bounds: tuple[float, float, float, float],
    limit_bounds: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    xmin, xmax, ymin, ymax = (float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3]))
    lxmin, lxmax, lymin, lymax = (
        float(limit_bounds[0]),
        float(limit_bounds[1]),
        float(limit_bounds[2]),
        float(limit_bounds[3]),
    )

    w = float(xmax - xmin)
    h = float(ymax - ymin)
    if w <= 0 or h <= 0:
        return (xmin, xmax, ymin, ymax)

    if w >= (lxmax - lxmin):
        xmin, xmax = lxmin, lxmax
    else:
        if xmin < lxmin:
            xmin = lxmin
            xmax = xmin + w
        if xmax > lxmax:
            xmax = lxmax
            xmin = xmax - w

    if h >= (lymax - lymin):
        ymin, ymax = lymin, lymax
    else:
        if ymin < lymin:
            ymin = lymin
            ymax = ymin + h
        if ymax > lymax:
            ymax = lymax
            ymin = ymax - h

    return (float(xmin), float(xmax), float(ymin), float(ymax))


def _window_from_bounds(path: Path, bounds: tuple[float, float, float, float]) -> rasterio.windows.Window:
    xmin, xmax, ymin, ymax = bounds
    with rasterio.open(path) as src:
        w = rasterio.windows.from_bounds(xmin, ymin, xmax, ymax, transform=src.transform)
        w = w.round_offsets().round_lengths()

        col0 = max(0.0, float(w.col_off))
        row0 = max(0.0, float(w.row_off))
        col1 = min(float(src.width), float(w.col_off + w.width))
        row1 = min(float(src.height), float(w.row_off + w.height))
        width = max(0.0, col1 - col0)
        height = max(0.0, row1 - row0)

        if width <= 1 or height <= 1:
            raise ValueError(f"ROI 太小或超出栅格范围: {bounds}")

        return rasterio.windows.Window(col_off=col0, row_off=row0, width=width, height=height)


def _auto_select_roi_bounds(
    *,
    mask0_path: Path,
    mask1_path: Path,
    background_path: Path,
    base_bounds: tuple[float, float, float, float],
    piv_npz: Path | None,
    n: int,
    roi_w_km: float,
    roi_h_km: float,
    bg_black_threshold: float,
    min_sep_km: float,
) -> list[tuple[float, float, float, float]]:
    n = int(n)
    if n <= 0:
        return []

    w_base0 = _window_from_bounds(mask0_path, base_bounds)
    w_base1 = _window_from_bounds(mask1_path, base_bounds)
    w_bg = _window_from_bounds(background_path, base_bounds)

    mask0, tf0 = _read_mask(mask0_path, window=w_base0)
    mask1, tf1 = _read_mask(mask1_path, window=w_base1)
    if mask0.shape != mask1.shape:
        raise ValueError(f"Mask shapes differ: {mask0.shape} vs {mask1.shape}")
    if (tf0.a != tf1.a) or (tf0.e != tf1.e):
        raise ValueError("mask0 与 mask1 的分辨率不一致，无法自动选 ROI。")

    rgb, bg_valid = _read_rgb_geotiff(background_path, window=w_bg)
    bg_valid = _apply_bg_black_threshold(rgb, bg_valid, float(bg_black_threshold))

    change = (mask0 ^ mask1) & bg_valid
    change = binary_dilation(change, iterations=1)

    piv_ref_p90: float | None = None
    Xp: np.ndarray | None = None
    Yp: np.ndarray | None = None
    Mp: np.ndarray | None = None
    if piv_npz is not None and Path(piv_npz).exists():
        piv = np.load(Path(piv_npz))
        Xp = np.asarray(piv["X_geo"], dtype=float)
        Yp = np.asarray(piv["Y_geo"], dtype=float)
        Up = np.asarray(piv["u_m_per_year"], dtype=float)
        Vp = np.asarray(piv["v_m_per_year"], dtype=float)
        s = 4
        Xp = Xp[::s, ::s]
        Yp = Yp[::s, ::s]
        Mp = np.hypot(Up[::s, ::s], Vp[::s, ::s])

        xmin0, xmax0, ymin0, ymax0 = base_bounds
        inb0 = (Xp >= xmin0) & (Xp <= xmax0) & (Yp >= ymin0) & (Yp <= ymax0) & np.isfinite(Mp)
        if np.any(inb0):
            piv_ref_p90 = float(np.nanpercentile(Mp[inb0], 90))
            if not np.isfinite(piv_ref_p90) or piv_ref_p90 <= 0:
                piv_ref_p90 = None

    px_w = float(abs(tf0.a))
    px_h = float(abs(tf0.e))
    roi_w_m = float(roi_w_km) * 1000.0
    roi_h_m = float(roi_h_km) * 1000.0
    roi_w_px = max(20, int(round(roi_w_m / px_w)))
    roi_h_px = max(20, int(round(roi_h_m / px_h)))

    H, W = change.shape
    r_half = roi_h_px // 2
    c_half = roi_w_px // 2
    if H <= roi_h_px + 2 or W <= roi_w_px + 2:
        left, bottom, right, top = rasterio.windows.bounds(w_base0, tf0)
        return [(float(left), float(right), float(bottom), float(top))]

    if float(min_sep_km) > 0:
        min_sep_px = int(round((float(min_sep_km) * 1000.0) / max(px_w, px_h)))
    else:
        min_sep_px = int(round(0.8 * max(roi_w_px, roi_h_px)))

    step = max(30, int(round(min(roi_w_px, roi_h_px) / 3)))
    min_change_px = max(30, int(round(0.0008 * roi_w_px * roi_h_px)))

    candidates: list[tuple[float, int, int]] = []
    for r in range(r_half, H - r_half, step):
        r0 = r - r_half
        r1 = r + r_half
        for c in range(c_half, W - c_half, step):
            c0 = c - c_half
            c1 = c + c_half
            vfrac = float(bg_valid[r0:r1, c0:c1].mean())
            if vfrac < 0.92:
                continue
            score_change = int(change[r0:r1, c0:c1].sum())
            if score_change < min_change_px:
                continue
            score_eff = float(score_change)
            if piv_ref_p90 is not None and Xp is not None and Yp is not None and Mp is not None:
                win = rasterio.windows.Window(
                    col_off=float(c - c_half),
                    row_off=float(r - r_half),
                    width=float(roi_w_px),
                    height=float(roi_h_px),
                )
                left, bottom, right, top = rasterio.windows.bounds(win, tf0)
                inb = (Xp >= float(left)) & (Xp <= float(right)) & (Yp >= float(bottom)) & (Yp <= float(top)) & np.isfinite(Mp)
                if np.any(inb):
                    p90 = float(np.nanpercentile(Mp[inb], 90))
                    if np.isfinite(p90) and p90 > 0:
                        ratio = float(p90) / float(piv_ref_p90)
                        if ratio > 1.9:
                            continue
                        score_eff = score_eff / (1.0 + 0.8 * max(0.0, ratio - 1.0))

            candidates.append((score_eff, r, c))

    candidates.sort(reverse=True, key=lambda t: t[0])
    chosen_rc: list[tuple[int, int]] = []
    chosen_bounds: list[tuple[float, float, float, float]] = []

    for score, r, c in candidates:
        if len(chosen_rc) >= n:
            break
        ok = True
        for rr, cc in chosen_rc:
            if (r - rr) * (r - rr) + (c - cc) * (c - cc) < (min_sep_px * min_sep_px):
                ok = False
                break
        if not ok:
            continue

        win = rasterio.windows.Window(
            col_off=float(c - c_half),
            row_off=float(r - r_half),
            width=float(roi_w_px),
            height=float(roi_h_px),
        )
        left, bottom, right, top = rasterio.windows.bounds(win, tf0)
        chosen_rc.append((r, c))
        chosen_bounds.append((float(left), float(right), float(bottom), float(top)))

    if len(chosen_bounds) < n:
        left0, right0, bottom0, top0 = base_bounds
        xmid = 0.5 * (left0 + right0)
        ymid = 0.5 * (bottom0 + top0)
        chosen_bounds.append(
            (xmid - 0.5 * roi_w_m, xmid + 0.5 * roi_w_m, ymid - 0.5 * roi_h_m, ymid + 0.5 * roi_h_m)
        )

    return chosen_bounds[:n]


def _plot_bankline(
    ax: plt.Axes,
    mask: np.ndarray,
    transform: rasterio.Affine,
    color: str,
    lw: float,
    min_points: int,
) -> None:
    contours = find_contours(mask.astype(np.uint8), 0.5)
    for c in contours:
        if int(min_points) > 0 and int(c.shape[0]) < int(min_points):
            continue
        xs, ys = rasterio.transform.xy(transform, c[:, 0], c[:, 1], offset="center")
        ln = ax.plot(xs, ys, color=color, linewidth=float(lw), alpha=0.98, zorder=9)[0]
        ln.set_path_effects([pe.Stroke(linewidth=float(lw) + 1.4, foreground="k"), pe.Normal()])


def _contours_length_m(mask: np.ndarray, transform: rasterio.Affine, *, min_points: int) -> float:
    contours = find_contours(mask.astype(np.uint8), 0.5)
    L = 0.0
    for c in contours:
        if int(min_points) > 0 and int(c.shape[0]) < int(min_points):
            continue
        xs, ys = rasterio.transform.xy(transform, c[:, 0], c[:, 1], offset="center")
        x = np.asarray(xs, dtype=float)
        y = np.asarray(ys, dtype=float)
        if x.size < 2:
            continue
        dx = np.diff(x)
        dy = np.diff(y)
        L += float(np.nansum(np.hypot(dx, dy)))
    return float(L)


def _add_scalebar(
    ax: plt.Axes,
    extent: tuple[float, float, float, float],
    length_m: float,
    pos: str = "bottom-left",
    *,
    boxed: bool = False,
    box_alpha: float = 0.75,
) -> None:
    if float(length_m) <= 0:
        return

    xmin, xmax, ymin, ymax = extent
    if str(pos) == "bottom-right":
        x1 = xmax - 0.06 * (xmax - xmin)
        x0 = x1 - float(length_m)
        y0 = ymin + 0.10 * (ymax - ymin)
    elif str(pos) == "legend-below":
        x0 = xmin + 0.06 * (xmax - xmin)
        x1 = x0 + float(length_m)
        y0 = ymax - 0.30 * (ymax - ymin)
    elif str(pos) == "right-middle":
        x1 = xmax - 0.06 * (xmax - xmin)
        x0 = x1 - float(length_m)
        y0 = ymin + 0.50 * (ymax - ymin)
    elif str(pos) == "top-middle":
        x0 = xmin + 0.50 * (xmax - xmin) - 0.5 * float(length_m)
        x1 = x0 + float(length_m)
        y0 = ymin + 0.90 * (ymax - ymin)
    else:
        x0 = xmin + 0.06 * (xmax - xmin)
        y0 = ymin + 0.10 * (ymax - ymin)
        x1 = x0 + float(length_m)

    if bool(boxed):
        dx = float(xmax - xmin)
        dy = float(ymax - ymin)
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
                zorder=3,
            )
        )

    ln = ax.plot([x0, x1], [y0, y0], color="k", linewidth=2.0, solid_capstyle="butt", zorder=4)[0]
    ln0 = ax.plot([x0, x0], [y0, y0 + 0.01 * (ymax - ymin)], color="k", linewidth=2.0, zorder=4)[0]
    ln1 = ax.plot([x1, x1], [y0, y0 + 0.01 * (ymax - ymin)], color="k", linewidth=2.0, zorder=4)[0]
    if not bool(boxed):
        for l in (ln, ln0, ln1):
            l.set_path_effects([pe.Stroke(linewidth=4.0, foreground="w"), pe.Normal()])

    t = ax.text(
        0.5 * (x0 + x1),
        y0 + 0.02 * (ymax - ymin),
        f"{float(length_m) / 1000.0:g} km",
        ha="center",
        va="bottom",
        color="k",
        fontsize=9,
        zorder=4,
    )
    if not bool(boxed):
        t.set_bbox(dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.3))
        t.set_path_effects([pe.Stroke(linewidth=2.0, foreground="w"), pe.Normal()])


def _draw_piv_key(
    ax: plt.Axes,
    x: float,
    y: float,
    length_frac: float,
    label: str,
    *,
    color: str,
    boxed: bool = False,
    box_alpha: float = 0.75,
) -> None:
    length_frac = float(np.clip(float(length_frac), 0.05, 0.25))
    x0 = float(x) - 0.5 * length_frac
    x1 = float(x) + 0.5 * length_frac
    y0 = float(y)

    if bool(boxed):
        xpad = 0.035
        ypad_up = 0.025
        ypad_down = 0.070
        ax.add_patch(
            FancyBboxPatch(
                (float(x0) - xpad, float(y0) - ypad_down),
                float(x1 - x0) + 2.0 * xpad,
                ypad_up + ypad_down,
                boxstyle="round,pad=0.01",
                transform=ax.transAxes,
                facecolor="white",
                edgecolor="none",
                alpha=float(box_alpha),
                zorder=9,
            )
        )

    arr = FancyArrowPatch(
        (x0, y0),
        (x1, y0),
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.8,
        color=str(color),
        zorder=10,
    )
    arr.set_path_effects([pe.Stroke(linewidth=3.6, foreground="k"), pe.Normal()])
    ax.add_patch(arr)

    t = ax.text(
        float(x),
        float(y) - 0.035,
        str(label),
        transform=ax.transAxes,
        ha="center",
        va="top",
        color="k",
        fontsize=9,
        zorder=10,
    )
    if not bool(boxed):
        t.set_bbox(dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.25))
    t.set_path_effects([pe.Stroke(linewidth=2.0, foreground="w"), pe.Normal()])


def _add_panel_label(ax: plt.Axes, text: str, pos: str = "upper-right") -> None:
    pos = str(pos)
    if pos == "upper-left":
        x, y, ha, va = 0.02, 0.98, "left", "top"
    elif pos == "lower-left":
        x, y, ha, va = 0.02, 0.02, "left", "bottom"
    elif pos == "lower-right":
        x, y, ha, va = 0.98, 0.02, "right", "bottom"
    else:
        x, y, ha, va = 0.98, 0.98, "right", "top"

    t = ax.text(
        float(x),
        float(y),
        str(text),
        transform=ax.transAxes,
        ha=str(ha),
        va=str(va),
        fontsize=9,
        color="k",
        zorder=11,
    )
    t.set_bbox(dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.25))
    t.set_path_effects([pe.Stroke(linewidth=2.0, foreground="w"), pe.Normal()])


def plot_fig3a_bankline_change(
    site: str,
    mask_level: int,
    year0: int,
    year1: int,
    piv_npz: Path,
    mask0_path: Path,
    mask1_path: Path,
    background_path: Path | None,
    out_path: Path,
    preset: str,
    dpi: int,
    titles: str,
    piv_stride: int,
    piv_scale: float,
    piv_mag_clip_quantile: float,
    piv_mag_clip_max_factor: float,
    piv_max_len_frac: float,
    piv_max_per_panel: int,
    alpha_change: float,
    crop: bool,
    crop_margin_px: int,
    change_dilate_px: int,
    edge_buffer_px: int,
    show_banklines: bool,
    bankline_lw: float,
    bankline_min_points: int,
    legend: str,
    quiver_key_m_per_yr: float,
    quiver_key_quantile: float,
    quiver_key_each_panel: bool,
    quiver_key_pos: str,
    quiver_key_box: bool | None,
    piv_len_frac: float,
    piv_water_buffer_px: int,
    scalebar_km: float,
    scalebar_each_panel: bool,
    scalebar_pos: str,
    scalebar_box: bool | None,
    piv_width: float,
    piv_debug: bool,
    bg_alpha: float,
    bg_black_threshold: float,
    bg_style: str,
    bg_sat: float,
    layout: str,
    rois: list[tuple[float, float, float, float]] | None,
    roi_fracs: list[tuple[float, float, float, float]] | None,
    roi_labels: str,
    roi_label_pos: str,
    diagnostic: bool = False,
    diagnostic_out: str | None = None,
) -> None:
    setup_preset(preset, dpi)

    # --- Publication Grade Style ---
    import matplotlib as mpl
    mpl.rcParams["font.family"] = "Times New Roman"
    mpl.rcParams["mathtext.fontset"] = "stix"
    mpl.rcParams["font.size"] = 10 if preset == "paper" else 12
    # -----------------------------

    show_titles: bool
    if titles == "on":
        show_titles = True
    elif titles == "off":
        show_titles = False
    else:
        show_titles = preset != "paper"

    if bg_style == "auto":
        bg_style_eff = "desat" if preset == "paper" else "rgb"
    else:
        bg_style_eff = bg_style

    if background_path is None:
        raise FileNotFoundError("未找到可用背景影像（Color 或 GEOTIFFS/image）。请用 --background 指定。")

    base_bounds = _dataset_bounds(mask0_path)
    if crop:
        wbg = _compute_crop_window(Path(background_path), margin_px=int(crop_margin_px))
        if wbg is not None:
            with rasterio.open(Path(background_path)) as src_bg:
                b = rasterio.windows.bounds(wbg, src_bg.transform)
            base_bounds = (float(b[0]), float(b[2]), float(b[1]), float(b[3]))

    bounds_list: list[tuple[float, float, float, float]] = []
    if rois:
        bounds_list.extend(list(rois))

    if roi_fracs:
        xmin0, xmax0, ymin0, ymax0 = base_bounds
        for fx0, fx1, fy0, fy1 in roi_fracs:
            xmin = xmin0 + float(fx0) * (xmax0 - xmin0)
            xmax = xmin0 + float(fx1) * (xmax0 - xmin0)
            ymin = ymin0 + float(fy0) * (ymax0 - ymin0)
            ymax = ymin0 + float(fy1) * (ymax0 - ymin0)
            bounds_list.append((float(xmin), float(xmax), float(ymin), float(ymax)))

    if not bounds_list:
        bounds_list = [base_bounds]

    piv = np.load(piv_npz)
    X = np.asarray(piv["X_geo"], dtype=float)
    Y = np.asarray(piv["Y_geo"], dtype=float)
    U = np.asarray(piv["u_m_per_year"], dtype=float)
    V = np.asarray(piv["v_m_per_year"], dtype=float)

    ss = max(1, int(piv_stride))
    if ss <= 1:
        Xs, Ys, Us, Vs = X, Y, U, V
    else:
        Xs = X[::ss, ::ss]
        Ys = Y[::ss, ::ss]
        Us = U[::ss, ::ss]
        Vs = V[::ss, ::ss]

    n_panels = len(bounds_list)
    if n_panels > 4:
        raise ValueError("当前仅支持 1–4 个 ROI（--roi/--roi-frac/--roi-center）")

    if layout == "2x2":
        nrows, ncols = 2, 2
    else:
        if n_panels <= 3:
            nrows, ncols = 1, n_panels
        else:
            nrows, ncols = 2, 2

    if preset == "paper":
        height_mm = 75 if nrows == 1 else 150
        figsize = get_paper_figsize(190, height_mm)
    else:
        figsize = (12, 4.5 * nrows)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, constrained_layout=True)
    if isinstance(axes, np.ndarray):
        axes_list = list(axes.ravel())
    else:
        axes_list = [axes]

    key_eff = float(quiver_key_m_per_yr)
    if key_eff <= 0:
        qtile = float(quiver_key_quantile)
        if not np.isfinite(qtile):
            qtile = 0.7
        qtile = float(np.clip(qtile, 0.05, 0.95))
        mags_all: list[np.ndarray] = []
        for bounds in bounds_list:
            xmin, xmax, ymin, ymax = bounds
            inb = (Xs >= xmin) & (Xs <= xmax) & (Ys >= ymin) & (Ys <= ymax)
            mvec = inb & np.isfinite(Us) & np.isfinite(Vs)
            if np.any(mvec):
                mags_all.append(np.hypot(Us[mvec], Vs[mvec]).ravel())

        if mags_all:
            mcat = np.concatenate(mags_all)
            mcat = mcat[np.isfinite(mcat)]
            if mcat.size:
                p = float(np.nanpercentile(mcat, 100.0 * qtile))
                key_eff = _nice_number_125(max(p, 0.05))
            else:
                key_eff = 1.0
        else:
            key_eff = 1.0

        print(f"Auto quiver-key selected: {key_eff:g} m/yr (based on PIV magnitude quantile={qtile:g})")

    gained_color = "#ff1744"
    lost_color = "#2979ff"
    gained_cmap = ListedColormap([gained_color])
    gained_cmap.set_bad(alpha=0.0)
    lost_cmap = ListedColormap([lost_color])
    lost_cmap.set_bad(alpha=0.0)

    diagnostic_rois: list[dict] = []

    for i, bounds in enumerate(bounds_list):
        ax = axes_list[i]
        xmin, xmax, ymin, ymax = bounds
        w_m = float(xmax - xmin)

        w_mask0 = _window_from_bounds(mask0_path, bounds)
        w_mask1 = _window_from_bounds(mask1_path, bounds)
        w_bg = _window_from_bounds(Path(background_path), bounds)

        mask0, tf0 = _read_mask(mask0_path, window=w_mask0)
        mask1, tf1 = _read_mask(mask1_path, window=w_mask1)

        if mask0.shape != mask1.shape:
            raise ValueError(f"Mask shapes differ: {mask0.shape} vs {mask1.shape} (ROI={bounds})")

        if (tf0.a != tf1.a) or (tf0.e != tf1.e):
            raise ValueError("mask0 与 mask1 的分辨率不一致，无法直接差分/叠加。")

        extent = (float(xmin), float(xmax), float(ymin), float(ymax))

        rgb, bg_valid = _read_rgb_geotiff(Path(background_path), window=w_bg)
        rgb = _apply_bg_style(rgb, style=str(bg_style_eff), sat=float(bg_sat))
        bg_valid = _apply_bg_black_threshold(rgb, bg_valid, float(bg_black_threshold))

        gained = mask1 & (~mask0)
        lost = mask0 & (~mask1)

        water = (mask0 | mask1)
        if int(piv_water_buffer_px) > 0:
            water_buf = binary_dilation(water, iterations=int(piv_water_buffer_px))
        else:
            water_buf = water

        if int(change_dilate_px) > 0:
            gained_plot = binary_dilation(gained, iterations=int(change_dilate_px))
            lost_plot = binary_dilation(lost, iterations=int(change_dilate_px))
        else:
            gained_plot = gained
            lost_plot = lost

        if int(edge_buffer_px) > 0:
            e0 = binary_dilation(mask0, iterations=1) ^ binary_erosion(mask0, iterations=1)
            e1 = binary_dilation(mask1, iterations=1) ^ binary_erosion(mask1, iterations=1)
            buf = binary_dilation(e0 | e1, iterations=int(edge_buffer_px))
            gained_plot = gained_plot & buf
            lost_plot = lost_plot & buf

        if float(piv_scale) > 0:
            scale_val = float(piv_scale)
        else:
            if w_m <= 0:
                scale_val = 1.0
            else:
                frac = float(piv_len_frac)
                if not np.isfinite(frac) or frac <= 0:
                    frac = 0.10
                desired_len_m = frac * w_m
                scale_val = float(key_eff) / float(desired_len_m)

        stat_gained_px = int(np.sum(gained))
        stat_lost_px = int(np.sum(lost))
        stat_change_px = int(np.sum(gained | lost))
        px_area_m2 = float(abs(tf0.a * tf0.e))
        stat_gained_km2 = float(stat_gained_px) * px_area_m2 / 1e6
        stat_lost_km2 = float(stat_lost_px) * px_area_m2 / 1e6
        stat_change_km2 = float(stat_change_px) * px_area_m2 / 1e6
        valid_px = int(np.sum(bg_valid))
        valid_km2 = float(valid_px) * px_area_m2 / 1e6
        gained_frac_valid = float(stat_gained_px) / float(valid_px) if valid_px > 0 else 0.0
        lost_frac_valid = float(stat_lost_px) / float(valid_px) if valid_px > 0 else 0.0
        change_frac_valid = float(stat_change_px) / float(valid_px) if valid_px > 0 else 0.0

        len0_m = _contours_length_m(mask0, tf0, min_points=int(bankline_min_points))
        len1_m = _contours_length_m(mask1, tf1, min_points=int(bankline_min_points))
        len_m = 0.5 * (float(len0_m) + float(len1_m)) if (len0_m > 0 and len1_m > 0) else float(max(len0_m, len1_m))
        avg_disp_m = (float(stat_change_px) * px_area_m2) / (len_m if len_m > 0 else 1.0)

        ax.set_facecolor("white")
        ax.imshow(
            rgb,
            extent=extent,
            origin="upper",
            alpha=(bg_valid.astype(float) * float(bg_alpha)),
        )

        ax.imshow(
            np.where(gained_plot, 1.0, np.nan),
            extent=extent,
            origin="upper",
            cmap=gained_cmap,
            vmin=0,
            vmax=1,
            alpha=float(alpha_change),
            interpolation="nearest",
            zorder=2,
        )
        ax.imshow(
            np.where(lost_plot, 1.0, np.nan),
            extent=extent,
            origin="upper",
            cmap=lost_cmap,
            vmin=0,
            vmax=1,
            alpha=float(alpha_change),
            interpolation="nearest",
            zorder=2,
        )

        inb = (Xs >= xmin) & (Xs <= xmax) & (Ys >= ymin) & (Ys <= ymax)
        mvec = inb & np.isfinite(Us) & np.isfinite(Vs)
        width_val = float(piv_width) if float(piv_width) > 0 else (0.0036 if preset == "paper" else 0.0045)
        xq = Xs.ravel()[mvec.ravel()]
        yq = Ys.ravel()[mvec.ravel()]
        uq = Us.ravel()[mvec.ravel()]
        vq = Vs.ravel()[mvec.ravel()]

        if xq.size and int(piv_water_buffer_px) >= 0:
            H, W = water_buf.shape
            inv = ~tf0
            colf = inv.c + inv.a * xq + inv.b * yq
            rowf = inv.f + inv.d * xq + inv.e * yq
            cols = np.round(colf).astype(int)
            rows = np.round(rowf).astype(int)
            inside = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
            keep_w = np.zeros_like(inside, dtype=bool)
            keep_w[inside] = water_buf[rows[inside], cols[inside]]
            xq = xq[keep_w]
            yq = yq[keep_w]
            uq = uq[keep_w]
            vq = vq[keep_w]

        magq0 = np.hypot(uq, vq)
        piv_count_raw = int(magq0.size)
        piv_mean_raw = float(np.nanmean(magq0)) if magq0.size else 0.0
        piv_p10_raw = float(np.nanpercentile(magq0, 10)) if magq0.size else 0.0
        piv_p50_raw = float(np.nanpercentile(magq0, 50)) if magq0.size else 0.0
        piv_p90_raw = float(np.nanpercentile(magq0, 90)) if magq0.size else 0.0

        keep = np.isfinite(magq0)
        if np.any(keep):
            qclip = float(piv_mag_clip_quantile)
            if np.isfinite(qclip) and 0 < qclip < 1:
                thr = float(np.nanpercentile(magq0[keep], 100.0 * qclip))
                keep = keep & (magq0 <= thr)

            mf = float(piv_mag_clip_max_factor)
            if np.isfinite(mf) and mf > 0 and np.isfinite(key_eff) and key_eff > 0:
                keep = keep & (magq0 <= (mf * float(key_eff)))

        xq = xq[keep]
        yq = yq[keep]
        uq = uq[keep]
        vq = vq[keep]
        magq = np.hypot(uq, vq)

        max_len_frac = float(piv_max_len_frac)
        if np.isfinite(max_len_frac) and max_len_frac > 0 and np.isfinite(scale_val) and scale_val > 0 and w_m > 0:
            max_mag = float(max_len_frac) * w_m * float(scale_val)
            if np.isfinite(max_mag) and max_mag > 0:
                m = np.hypot(uq, vq)
                good = np.isfinite(m) & (m > 0)
                fac = np.ones_like(m)
                fac[good] = np.minimum(1.0, float(max_mag) / m[good])
                uq = uq * fac
                vq = vq * fac
                magq = np.hypot(uq, vq)

        piv_count_used = int(magq.size)
        piv_mean_used = float(np.nanmean(magq)) if magq.size else 0.0
        piv_p10_used = float(np.nanpercentile(magq, 10)) if magq.size else 0.0
        piv_p50_used = float(np.nanpercentile(magq, 50)) if magq.size else 0.0
        piv_p90_used = float(np.nanpercentile(magq, 90)) if magq.size else 0.0
        piv_keep_ratio = float(piv_count_used) / float(piv_count_raw) if piv_count_raw > 0 else 0.0

        piv_dir_R: float = 0.0
        if piv_count_used > 0:
            m = np.hypot(uq, vq)
            ok = np.isfinite(m) & (m > 0)
            if np.any(ok):
                ux = (uq[ok] / m[ok]).astype(float)
                uy = (vq[ok] / m[ok]).astype(float)
                piv_dir_R = float(np.hypot(np.nanmean(ux), np.nanmean(uy)))
        
        if diagnostic:
            roi_info = {
                "roi_id": int(i + 1),
                "bounds_utm": [float(xmin), float(xmax), float(ymin), float(ymax)],
                "valid_area_km2": float(valid_km2),
                "gained_km2": float(stat_gained_km2),
                "lost_km2": float(stat_lost_km2),
                "change_km2": float(stat_change_km2),
                "gained_frac_valid": float(gained_frac_valid),
                "lost_frac_valid": float(lost_frac_valid),
                "change_frac_valid": float(change_frac_valid),
                "bankline_length_m": float(len_m),
                "bankline_avg_displacement_m": float(avg_disp_m),
                "piv_raw_count": int(piv_count_raw),
                "piv_raw_mean_m_yr": float(piv_mean_raw),
                "piv_raw_p10_m_yr": float(piv_p10_raw),
                "piv_raw_p50_m_yr": float(piv_p50_raw),
                "piv_raw_p90_m_yr": float(piv_p90_raw),
                "piv_used_count": int(piv_count_used),
                "piv_used_keep_ratio": float(piv_keep_ratio),
                "piv_used_mean_m_yr": float(piv_mean_used),
                "piv_used_p10_m_yr": float(piv_p10_used),
                "piv_used_p50_m_yr": float(piv_p50_used),
                "piv_used_p90_m_yr": float(piv_p90_used),
                "piv_used_direction_R": float(piv_dir_R),
            }
            diagnostic_rois.append(roi_info)
            print(
                f"ROI {i+1} Diag: change={stat_change_km2:.3f} km2 ({100.0*change_frac_valid:.1f}%), "
                f"disp~{avg_disp_m:.1f} m, PIV used N={piv_count_used} (keep={piv_keep_ratio:.2f}) P90={piv_p90_used:.2f} m/yr, R={piv_dir_R:.2f}"
            )

        if int(piv_max_per_panel) > 0 and magq.size:
            h_m = float(ymax - ymin)
            if w_m > 0 and h_m > 0:
                aspect = w_m / h_m
                nbx = int(np.ceil(np.sqrt(float(piv_max_per_panel) * max(aspect, 1e-6))))
                nbx = max(2, nbx)
                nby = int(np.ceil(float(piv_max_per_panel) / float(nbx)))
                nby = max(2, nby)

                fx = (xq - float(xmin)) / float(w_m)
                fy = (yq - float(ymin)) / float(h_m)
                fx = np.clip(fx, 0.0, 0.999999)
                fy = np.clip(fy, 0.0, 0.999999)
                bx = np.floor(fx * float(nbx)).astype(int)
                by = np.floor(fy * float(nby)).astype(int)
                key = by * nbx + bx

                order = np.argsort(key)
                key_s = key[order]
                uniq, start = np.unique(key_s, return_index=True)
                end = np.r_[start[1:], np.array([order.size], dtype=int)]

                xs2: list[float] = []
                ys2: list[float] = []
                us2: list[float] = []
                vs2: list[float] = []
                for s, e in zip(start, end):
                    idxs = order[int(s) : int(e)]
                    if idxs.size == 0:
                        continue
                    xs2.append(float(np.nanmean(xq[idxs])))
                    ys2.append(float(np.nanmean(yq[idxs])))
                    us2.append(float(np.nanmean(uq[idxs])))
                    vs2.append(float(np.nanmean(vq[idxs])))

                xq = np.asarray(xs2, dtype=float)
                yq = np.asarray(ys2, dtype=float)
                uq = np.asarray(us2, dtype=float)
                vq = np.asarray(vs2, dtype=float)
                magq = np.hypot(uq, vq)

                if xq.size > int(piv_max_per_panel):
                    idx = np.linspace(0, int(xq.size) - 1, int(piv_max_per_panel), dtype=int)
                    xq = xq[idx]
                    yq = yq[idx]
                    uq = uq[idx]
                    vq = vq[idx]
                    magq = magq[idx]

        if bool(piv_debug):
            if magq0.size:
                p50_0 = float(np.nanpercentile(magq0, 50))
                p90_0 = float(np.nanpercentile(magq0, 90))
                p99_0 = float(np.nanpercentile(magq0, 99))
                print(
                    f"ROI{i+1}: PIV raw N={int(magq0.size)} | mag p50={p50_0:.2f} p90={p90_0:.2f} p99={p99_0:.2f} (m/yr)"
                )
            else:
                print(f"ROI{i+1}: PIV raw N=0")

            if magq.size:
                p50 = float(np.nanpercentile(magq, 50))
                p90 = float(np.nanpercentile(magq, 90))
                p99 = float(np.nanpercentile(magq, 99))
                print(f"ROI{i+1}: PIV used N={int(magq.size)} | mag p50={p50:.2f} p90={p90:.2f} p99={p99:.2f} (m/yr)")
            else:
                print(f"ROI{i+1}: PIV used N=0")

        ax.quiver(
            xq,
            yq,
            uq,
            vq,
            color="k",
            angles="xy",
            scale_units="xy",
            scale=float(scale_val),
            width=float(width_val) * 1.8,
            alpha=0.85,
            zorder=4,
            pivot="mid",
            headwidth=4.5 if preset == "paper" else 5.5,
            headlength=6.5 if preset == "paper" else 7.5,
            headaxislength=5.8 if preset == "paper" else 6.8,
        )
        q = ax.quiver(
            xq,
            yq,
            uq,
            vq,
            color="w",
            angles="xy",
            scale_units="xy",
            scale=float(scale_val),
            width=float(width_val),
            alpha=0.9,
            zorder=5,
            pivot="mid",
            headwidth=4.5 if preset == "paper" else 5.5,
            headlength=6.5 if preset == "paper" else 7.5,
            headaxislength=5.8 if preset == "paper" else 6.8,
        )

        if show_banklines:
            _plot_bankline(
                ax,
                mask0,
                tf0,
                color="white",
                lw=float(bankline_lw) * 1.00,
                min_points=int(bankline_min_points),
            )
            _plot_bankline(
                ax,
                mask1,
                tf1,
                color="#ffea00",
                lw=float(bankline_lw) * 1.20,
                min_points=int(bankline_min_points),
            )

        def _pos(name: str) -> tuple[float, float]:
            if name == "top-middle":
                return (0.50, 0.93)
            if name == "top-right":
                return (0.86, 0.93)
            if name == "bottom-left":
                return (0.14, 0.08)
            if name == "bottom-right":
                return (0.86, 0.08)
            return (0.50, 0.86)

        show_key = bool(quiver_key_each_panel) or (i == 0)
        if show_key and float(key_eff) > 0:
            kx, ky = _pos(str(quiver_key_pos))
            if quiver_key_box is None:
                key_boxed = preset == "paper"
            else:
                key_boxed = bool(quiver_key_box)
            _draw_piv_key(
                ax,
                x=float(kx),
                y=float(ky),
                length_frac=float(piv_len_frac),
                label=f"PIV {float(key_eff):g} m/yr",
                color="w",
                boxed=bool(key_boxed),
            )

        show_bar = bool(scalebar_each_panel) or (i == 0)
        if show_bar and float(scalebar_km) > 0:
            if scalebar_box is None:
                bar_boxed = preset == "paper"
            else:
                bar_boxed = bool(scalebar_box)
            _add_scalebar(
                ax,
                extent,
                length_m=1000.0 * float(scalebar_km),
                pos=str(scalebar_pos),
                boxed=bool(bar_boxed),
            )

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xticks([])
        ax.set_yticks([])

        if show_titles:
            ax.set_title(f"ROI {i + 1}")

        show_roi_labels: bool
        if str(roi_labels) == "on":
            show_roi_labels = True
        elif str(roi_labels) == "off":
            show_roi_labels = False
        else:
            show_roi_labels = preset == "paper"

        if show_roi_labels:
            _add_panel_label(ax, text=f"ROI-{i+1}", pos=str(roi_label_pos))

        if i == 0 and (legend == "on" or legend == "auto"):
            handles = [
                Patch(facecolor=gained_color, edgecolor="none", label="Gained water"),
                Patch(facecolor=lost_color, edgecolor="none", label="Lost water"),
            ]
            if show_banklines:
                handles.extend(
                    [
                        Line2D([0], [0], color="white", lw=float(bankline_lw), label=f"Bankline {int(year0)}"),
                        Line2D([0], [0], color="#ffd54f", lw=float(bankline_lw), label=f"Bankline {int(year1)}"),
                    ]
                )
            ax.legend(
                handles=handles,
                loc="upper left",
                frameon=True,
                framealpha=0.85,
                fontsize=9 if preset == "paper" else 10,
            )

    for j in range(n_panels, len(axes_list)):
        axes_list[j].axis("off")

    if diagnostic:
        payload = {
            "site": str(site),
            "mask_level": int(mask_level),
            "year0": int(year0),
            "year1": int(year1),
            "piv_stride": int(piv_stride),
            "piv_max_per_panel": int(piv_max_per_panel),
            "piv_water_buffer_px": int(piv_water_buffer_px),
            "alpha_change": float(alpha_change),
            "rois": diagnostic_rois,
        }

        out_json: Path | None
        if diagnostic_out:
            out_json = Path(diagnostic_out)
        else:
            out_json = out_path.with_suffix("").with_name(out_path.stem + "_roi_summary.json")

        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Diagnostic results saved to {out_json}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(dpi))
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fig.3a: PIV vectors vs bankline change from multi-temporal masks")
    parser.add_argument("--site", type=str, default="Jurua-A")
    parser.add_argument("--mask-level", type=int, default=1)
    parser.add_argument("--year0", type=int, default=1987)
    parser.add_argument("--year1", type=int, default=2021)

    parser.add_argument("--mask0", type=str, default="")
    parser.add_argument("--mask1", type=str, default="")
    parser.add_argument("--background", type=str, default="")
    parser.add_argument("--piv-npz", type=str, default="")

    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--preset", type=str, default="paper", choices=["", "paper"])
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--titles", type=str, default="auto", choices=["auto", "on", "off"])

    parser.add_argument("--piv-stride", type=int, default=1)
    parser.add_argument("--piv-scale", type=float, default=0.0)
    parser.add_argument("--piv-len-frac", type=float, default=0.10)
    parser.add_argument("--piv-max-len-frac", type=float, default=0.18)
    parser.add_argument("--piv-mag-clip-quantile", type=float, default=0.95)
    parser.add_argument("--piv-mag-clip-max-factor", type=float, default=6.0)
    parser.add_argument("--piv-max-per-panel", type=int, default=250)
    parser.add_argument(
        "--piv-water-buffer-px",
        type=int,
        default=0,
        help="Keep only PIV vectors whose locations fall within a dilated (mask0|mask1) buffer (pixels). 0 disables.",
    )
    parser.add_argument("--alpha-change", type=float, default=0.35)

    parser.add_argument("--crop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--crop-margin-px", type=int, default=80)
    parser.add_argument("--change-dilate-px", type=int, default=2)
    parser.add_argument("--edge-buffer-px", type=int, default=6)
    parser.add_argument("--banklines", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bankline-lw", type=float, default=0.7)
    parser.add_argument("--bankline-min-points", type=int, default=200)
    parser.add_argument("--legend", type=str, default="auto", choices=["auto", "on", "off"])
    parser.add_argument("--quiver-key", type=float, default=0.0)
    parser.add_argument("--quiver-key-quantile", type=float, default=0.70)
    parser.add_argument("--quiver-key-each-panel", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--quiver-key-pos",
        type=str,
        default="top-middle",
        choices=["top-middle", "top-right", "bottom-left", "bottom-right"],
    )
    parser.add_argument("--quiver-key-box", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--scalebar-km", type=float, default=2.0)
    parser.add_argument("--scalebar-each-panel", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--scalebar-pos",
        type=str,
        default="bottom-right",
        choices=["bottom-left", "bottom-right", "top-middle", "right-middle", "legend-below"],
    )
    parser.add_argument("--scalebar-box", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--roi-labels", type=str, default="auto", choices=["auto", "on", "off"])
    parser.add_argument("--diagnostic", action="store_true", help="Enable ROI diagnostic mode")
    parser.add_argument("--diagnostic-out", type=str, default=None, help="Path to save diagnostic JSON output")
    parser.add_argument(
        "--roi-label-pos",
        type=str,
        default="upper-right",
        choices=["upper-left", "upper-right", "lower-left", "lower-right"],
    )
    parser.add_argument("--piv-width", type=float, default=0.0)
    parser.add_argument("--piv-debug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bg-alpha", type=float, default=0.85)
    parser.add_argument("--bg-black-threshold", type=float, default=0.02)
    parser.add_argument("--bg-style", type=str, default="auto", choices=["auto", "rgb", "gray", "desat"])
    parser.add_argument("--bg-sat", type=float, default=0.20)
    parser.add_argument("--layout", type=str, default="auto", choices=["auto", "2x2"])

    parser.add_argument("--roi", type=float, nargs=4, action="append", default=[])
    parser.add_argument("--roi-frac", type=float, nargs=4, action="append", default=[])
    parser.add_argument("--roi-center", type=float, nargs=2, action="append", default=[])
    parser.add_argument("--roi-size-km", type=float, nargs="+", default=[12.0, 10.0])
    parser.add_argument(
        "--roi-shift",
        type=float,
        nargs=2,
        action="append",
        default=[],
        metavar=("DX_M", "DY_M"),
        help="Shift each ROI by dx,dy in meters (UTM). Repeat for each ROI; if provided once, apply to all.",
    )
    parser.add_argument("--auto-roi", type=int, default=0)
    parser.add_argument("--auto-min-sep-km", type=float, default=0.0)

    args = parser.parse_args()

    site = str(args.site)
    mask_level = int(args.mask_level)
    year0 = int(args.year0)
    year1 = int(args.year1)

    mask0_path = Path(args.mask0) if args.mask0 else _default_mask_path(site, mask_level, year0)
    mask1_path = Path(args.mask1) if args.mask1 else _default_mask_path(site, mask_level, year1)

    background_path = Path(args.background) if args.background else _default_background_path(site, year0)

    piv_npz = Path(args.piv_npz) if args.piv_npz else _default_piv_npz(site, mask_level)

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = _PROJECT_ROOT / "results" / "figures" / "paper" / f"Fig3a_{site}_bankline_change.png"

    if not mask0_path.exists():
        raise FileNotFoundError(mask0_path)
    if not mask1_path.exists():
        raise FileNotFoundError(mask1_path)
    if background_path is not None and not background_path.exists():
        background_path = None
    if not piv_npz.exists():
        raise FileNotFoundError(piv_npz)

    roi_list = [tuple(float(v) for v in r) for r in args.roi] if args.roi else []
    if args.roi_center:
        if len(args.roi_size_km) == 1:
            w_km = float(args.roi_size_km[0])
            h_km = float(args.roi_size_km[0])
        elif len(args.roi_size_km) >= 2:
            w_km = float(args.roi_size_km[0])
            h_km = float(args.roi_size_km[1])
        else:
            raise ValueError("--roi-size-km 参数无效")

        w_m = 1000.0 * w_km
        h_m = 1000.0 * h_km
        for cx, cy in args.roi_center:
            cx = float(cx)
            cy = float(cy)
            roi_list.append((cx - 0.5 * w_m, cx + 0.5 * w_m, cy - 0.5 * h_m, cy + 0.5 * h_m))

    roi_frac_list = [tuple(float(v) for v in r) for r in args.roi_frac] if args.roi_frac else []

    roi_shifts = [tuple(float(v) for v in s) for s in args.roi_shift] if args.roi_shift else []
    if roi_shifts and roi_frac_list:
        raise ValueError("--roi-shift 仅支持与 --roi/--roi-center/--auto-roi 搭配使用，不支持 --roi-frac。")

    if roi_shifts and roi_list:
        if len(roi_shifts) == 1 and len(roi_list) > 1:
            roi_shifts = roi_shifts * len(roi_list)
        if len(roi_shifts) != len(roi_list):
            raise ValueError(f"--roi-shift 次数({len(roi_shifts)})必须等于 ROI 数量({len(roi_list)})，或只提供一次用于全部 ROI。")

        lim = _dataset_bounds(mask0_path)
        shifted: list[tuple[float, float, float, float]] = []
        for (xmin, xmax, ymin, ymax), (dxm, dym) in zip(roi_list, roi_shifts):
            b = (float(xmin) + float(dxm), float(xmax) + float(dxm), float(ymin) + float(dym), float(ymax) + float(dym))
            shifted.append(_clamp_bounds_to_limit(b, lim))
        roi_list = shifted

    if int(args.auto_roi) > 0 and (not roi_list) and (not roi_frac_list):
        if background_path is None:
            raise FileNotFoundError("使用 --auto-roi 需要可用背景影像。")

        base_bounds = _dataset_bounds(mask0_path)
        if bool(args.crop):
            wbg = _compute_crop_window(Path(background_path), margin_px=int(args.crop_margin_px))
            if wbg is not None:
                with rasterio.open(Path(background_path)) as src_bg:
                    b = rasterio.windows.bounds(wbg, src_bg.transform)
                base_bounds = (float(b[0]), float(b[2]), float(b[1]), float(b[3]))

        if len(args.roi_size_km) == 1:
            roi_w_km = float(args.roi_size_km[0])
            roi_h_km = float(args.roi_size_km[0])
        else:
            roi_w_km = float(args.roi_size_km[0])
            roi_h_km = float(args.roi_size_km[1])

        roi_list = _auto_select_roi_bounds(
            mask0_path=mask0_path,
            mask1_path=mask1_path,
            background_path=Path(background_path),
            base_bounds=base_bounds,
            piv_npz=piv_npz,
            n=int(args.auto_roi),
            roi_w_km=float(roi_w_km),
            roi_h_km=float(roi_h_km),
            bg_black_threshold=float(args.bg_black_threshold),
            min_sep_km=float(args.auto_min_sep_km),
        )

        print("Auto ROI bounds (xmin, xmax, ymin, ymax) [UTM meters]:")
        for k, (xmin, xmax, ymin, ymax) in enumerate(roi_list, start=1):
            cx = 0.5 * (float(xmin) + float(xmax))
            cy = 0.5 * (float(ymin) + float(ymax))
            print(f"  ROI{k}: {xmin:.2f} {xmax:.2f} {ymin:.2f} {ymax:.2f} | center=({cx:.2f}, {cy:.2f})")

        print("Reproducible CLI args (copy/paste):")
        roi_args = " ".join([f"--roi {xmin:.2f} {xmax:.2f} {ymin:.2f} {ymax:.2f}" for (xmin, xmax, ymin, ymax) in roi_list])
        print(f"  {roi_args}")

    plot_fig3a_bankline_change(
        site=site,
        mask_level=mask_level,
        year0=year0,
        year1=year1,
        piv_npz=piv_npz,
        mask0_path=mask0_path,
        mask1_path=mask1_path,
        background_path=background_path,
        out_path=out_path,
        preset=str(args.preset),
        dpi=int(args.dpi),
        titles=str(args.titles),
        piv_stride=int(args.piv_stride),
        piv_scale=float(args.piv_scale),
        piv_mag_clip_quantile=float(args.piv_mag_clip_quantile),
        piv_mag_clip_max_factor=float(args.piv_mag_clip_max_factor),
        piv_max_len_frac=float(args.piv_max_len_frac),
        piv_max_per_panel=int(args.piv_max_per_panel),
        alpha_change=float(args.alpha_change),
        crop=bool(args.crop),
        crop_margin_px=int(args.crop_margin_px),
        change_dilate_px=int(args.change_dilate_px),
        edge_buffer_px=int(args.edge_buffer_px),
        show_banklines=bool(args.banklines),
        bankline_lw=float(args.bankline_lw),
        bankline_min_points=int(args.bankline_min_points),
        legend=str(args.legend),
        quiver_key_m_per_yr=float(args.quiver_key),
        quiver_key_quantile=float(args.quiver_key_quantile),
        quiver_key_each_panel=bool(args.quiver_key_each_panel),
        quiver_key_pos=str(args.quiver_key_pos),
        quiver_key_box=args.quiver_key_box,
        piv_len_frac=float(args.piv_len_frac),
        piv_water_buffer_px=int(args.piv_water_buffer_px),
        scalebar_km=float(args.scalebar_km),
        scalebar_each_panel=bool(args.scalebar_each_panel),
        scalebar_pos=str(args.scalebar_pos),
        scalebar_box=args.scalebar_box,
        piv_width=float(args.piv_width),
        piv_debug=bool(args.piv_debug),
        bg_alpha=float(args.bg_alpha),
        bg_black_threshold=float(args.bg_black_threshold),
        bg_style=str(args.bg_style),
        bg_sat=float(args.bg_sat),
        layout=str(args.layout),
        rois=roi_list if roi_list else None,
        roi_fracs=roi_frac_list if roi_frac_list else None,
        roi_labels=str(args.roi_labels),
        roi_label_pos=str(args.roi_label_pos),
        diagnostic=bool(args.diagnostic),
        diagnostic_out=args.diagnostic_out,
    )


if __name__ == "__main__":
    main()
