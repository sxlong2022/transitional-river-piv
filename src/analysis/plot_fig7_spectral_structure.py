from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.analysis.quantitative_relationships import (
    analyze_trunk_level_relationships,
    autocorr_length_scales,
    cross_correlation_lag,
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
    return str(site)


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


def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if win <= 1 or x.size == 0:
        return x
    x = _fill_nan_linear_1d(x)
    w = np.ones(int(win), dtype=float) / float(win)
    return np.convolve(x, w, mode="same")


def _acf_curve(x: np.ndarray, step_m: float, max_lag_m: float | None) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 4:
        return np.array([]), np.array([])

    x = x - float(np.mean(x))
    acf = np.correlate(x, x, mode="full")[x.size - 1 :]
    if float(acf[0]) != 0:
        acf = acf / float(acf[0])

    lags = np.arange(acf.size, dtype=float) * float(step_m)
    if max_lag_m is not None:
        m = lags <= float(max_lag_m)
        lags = lags[m]
        acf = acf[m]

    return lags, acf


def _ccf_curve(
    x: np.ndarray,
    y: np.ndarray,
    step_m: float,
    max_lag_m: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]

    if x.size < 4:
        return np.array([]), np.array([])

    x = x - float(np.mean(x))
    y = y - float(np.mean(y))

    ccf = np.correlate(x, y, mode="full")
    lags = np.arange(-x.size + 1, x.size, dtype=float) * float(step_m)

    denom = float(np.sqrt(np.sum(x**2) * np.sum(y**2)))
    if denom > 0:
        ccf = ccf / denom

    if max_lag_m is not None:
        mm = np.abs(lags) <= float(max_lag_m)
        lags = lags[mm]
        ccf = ccf[mm]

    return lags, ccf


def _default_npz_path(site: str, mask_level: int) -> Path:
    site_io = _normalize_site(site)
    return (
        _PROJECT_ROOT
        / "results"
        / "PostprocessedPIV"
        / site_io
        / f"{site_io}_mask{mask_level}_link_sBCMn_flat_step20_metric_v2.npz"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot trunk-level autocorrelation/cross-correlation curves and characteristic lags."
    )
    parser.add_argument("--site", type=str, required=True)
    parser.add_argument("--mask-level", type=int, required=True)
    parser.add_argument("--npz", type=str, default="", help="Override input NPZ path")
    parser.add_argument("--out", type=str, default="", help="Override output PNG path")

    parser.add_argument("--k-trunks", type=int, default=1)
    parser.add_argument("--min-trunk-length-m", type=float, default=5000.0)
    parser.add_argument("--endpoint-tol-m", type=float, default=80.0)
    parser.add_argument("--weight-by", type=str, default="length_B")

    parser.add_argument("--abs-mn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--abs-curv", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--max-lag-m", type=float, default=20000.0)
    parser.add_argument("--peak-search-max-lag-m", type=float, default=5000.0)
    parser.add_argument("--smooth-window-m", type=float, default=0.0)
    parser.add_argument("--summary-json", type=str, default="", help="Write summary JSON to this path")

    # Preset args
    parser.add_argument("--preset", type=str, default="", choices=["", "paper"], help="Style preset")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved figure")

    args = parser.parse_args()

    # Apply preset
    setup_preset(args.preset, args.dpi)

    site = str(args.site)
    site_io = _normalize_site(site)
    mask = int(args.mask_level)

    npz_path = Path(args.npz) if args.npz else _default_npz_path(site, mask)
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = _PROJECT_ROOT / "results" / "figures" / "spectral"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"Fig7_{site}_mask{mask}_trunk_spectral_structure.png"

    res = analyze_trunk_level_relationships(
        npz_path,
        k_trunks=int(args.k_trunks),
        endpoint_tol_m=float(args.endpoint_tol_m),
        weight_by=str(args.weight_by),
        min_trunk_length_m=float(args.min_trunk_length_m),
    )

    if not res.trunks:
        print("No trunks found with the given parameters.")
        return

    trunk_id = "trunk_1" if "trunk_1" in res.trunks else sorted(res.trunks.keys())[0]
    tr = res.trunks[trunk_id]

    s = np.asarray(tr["s"], dtype=float)
    B = np.asarray(tr["B"], dtype=float)
    C = np.asarray(tr["C"], dtype=float)
    Mn = np.asarray(tr["Mn"], dtype=float)

    C_plot = np.abs(C) if bool(args.abs_curv) else C
    Mn_plot = np.abs(Mn) if bool(args.abs_mn) else Mn

    step_m = float(res.diagnostics.get("step_m", float("nan")))
    if not np.isfinite(step_m):
        step_m = _estimate_step_m_from_s(s)

    max_lag_m = float(args.max_lag_m) if args.max_lag_m is not None else None

    smooth_window_m = float(args.smooth_window_m)
    if np.isfinite(smooth_window_m) and smooth_window_m > 0 and np.isfinite(step_m) and step_m > 0:
        win = int(round(smooth_window_m / float(step_m)))
    else:
        win = 1

    B_for_corr = _moving_average(B, win=win)
    C_for_corr = _moving_average(C_plot, win=win)
    Mn_for_corr = _moving_average(Mn_plot, win=win)

    acfB_lags, acfB = _acf_curve(B_for_corr, step_m=step_m, max_lag_m=max_lag_m)
    acfC_lags, acfC = _acf_curve(C_for_corr, step_m=step_m, max_lag_m=max_lag_m)

    ccfCM_lags, ccfCM = _ccf_curve(C_for_corr, Mn_for_corr, step_m=step_m, max_lag_m=max_lag_m)
    ccfBM_lags, ccfBM = _ccf_curve(B_for_corr, Mn_for_corr, step_m=step_m, max_lag_m=max_lag_m)

    scB = autocorr_length_scales(B_for_corr, step_m=step_m)
    scC = autocorr_length_scales(C_for_corr, step_m=step_m)
    peak_search_max_lag_m = float(args.peak_search_max_lag_m) if args.peak_search_max_lag_m is not None else None
    lagCM = cross_correlation_lag(C_for_corr, Mn_for_corr, step_m=step_m, max_lag_m=peak_search_max_lag_m)
    lagBM = cross_correlation_lag(B_for_corr, Mn_for_corr, step_m=step_m, max_lag_m=peak_search_max_lag_m)

    summary = {
        "path": str(npz_path),
        "site": str(site),
        "site_io": str(site_io),
        "site_display": str(_display_site(site)),
        "trunk_id": str(trunk_id),
        "step_m": float(step_m),
        "abs_curv": bool(args.abs_curv),
        "abs_mn": bool(args.abs_mn),
        "max_lag_m": float(max_lag_m) if max_lag_m is not None else None,
        "peak_search_max_lag_m": float(peak_search_max_lag_m) if peak_search_max_lag_m is not None else None,
        "smooth_window_m": float(smooth_window_m),
        "acf_B": scB,
        "acf_C": scC,
        "ccf_C_vs_Mn": lagCM,
        "ccf_B_vs_Mn": lagBM,
    }
    print("SPECTRAL_SUMMARY_JSON=", json.dumps(summary, ensure_ascii=False))

    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved summary: {summary_path}")

    # Determine figsize based on preset
    if args.preset == "paper":
        # Full width (190mm), height approx 120-130mm (2 rows)
        figsize = get_paper_figsize(190, 130)
    else:
        figsize = (12, 8)

    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)

    ccf_xlim_km = 10.0 if str(args.preset) == "paper" else 20.0

    def _annotate_peak(
        ax: plt.Axes,
        tau_km: float,
        rr: float,
        color: str,
        *,
        xoff: float | None = None,
        yoff: float | None = None,
    ) -> None:
        # Keep label close to the peak; place to the right by default (stable left alignment).
        if xoff is None:
            xoff = 22.0 if float(tau_km) < 0 else 12.0

        # Place label above/below based on the peak's normalized y-position.
        if yoff is None:
            y0, y1 = ax.get_ylim()
            if np.isfinite(y0) and np.isfinite(y1) and y1 != y0:
                yfrac = (float(rr) - float(y0)) / (float(y1) - float(y0))
            else:
                yfrac = 0.5
            # If peak is close to top, put label below; otherwise above.
            yoff = -16.0 if yfrac > 0.80 else 12.0

        ha = "left"
        va = "top" if float(yoff) < 0 else "bottom"

        # Avoid starting the string with '$' (can trigger whole-string math parsing).
        # NOTE: do NOT use raw strings here, otherwise '\n' becomes literal.
        zwsp = "\u200b"
        label = (
            zwsp
            + f"$\\tau = {tau_km:.2f}\\,\\mathrm{{km}}$\n"
            + f"$r = {rr:.2f}$"
        )

        ax.annotate(
            label,
            xy=(tau_km, rr),
            xycoords="data",
            xytext=(xoff, yoff),
            textcoords="offset points",
            ha=ha,
            va=va,
            multialignment="left",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5),
            arrowprops=dict(arrowstyle="-", color=color, lw=0.8, alpha=0.75, shrinkA=0, shrinkB=0),
            annotation_clip=True,
        )

    ax = axes[0, 0]
    ax.plot(acfB_lags / 1000.0, acfB, color="tab:blue", linewidth=2.0)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title(r"ACF of $B$")
    ax.set_xlabel(r"Lag ($\mathrm{km}$)")
    ax.set_ylabel("ACF")
    ax.grid(True, linestyle=":", alpha=0.25, linewidth=0.6)
    if np.isfinite(scB.get("e_folding_m", float("nan"))):
        le_km = float(scB["e_folding_m"]) / 1000.0
        ax.axvline(le_km, color="tab:blue", linestyle="--", alpha=0.6)
        ax.text(
            le_km,
            0.96,
            r"$L_{\mathrm{e}}$",
            transform=ax.get_xaxis_transform(),
            ha="left",
            va="top",
            fontsize=10,
            color="tab:blue",
        )

    ax = axes[0, 1]
    ax.plot(acfC_lags / 1000.0, acfC, color="tab:green", linewidth=2.0)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title(r"ACF of $|C|$" if bool(args.abs_curv) else r"ACF of $C$")
    ax.set_xlabel(r"Lag ($\mathrm{km}$)")
    ax.set_ylabel("ACF")
    ax.grid(True, linestyle=":", alpha=0.25, linewidth=0.6)
    if np.isfinite(scC.get("e_folding_m", float("nan"))):
        le_km = float(scC["e_folding_m"]) / 1000.0
        ax.axvline(le_km, color="tab:green", linestyle="--", alpha=0.6)
        ax.text(
            le_km,
            0.96,
            r"$L_{\mathrm{e}}$",
            transform=ax.get_xaxis_transform(),
            ha="left",
            va="top",
            fontsize=10,
            color="tab:green",
        )

    ax = axes[1, 0]
    ax.plot(ccfCM_lags / 1000.0, ccfCM, color="tab:red", linewidth=2.0)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.5)
    if bool(args.abs_curv) and bool(args.abs_mn):
        ax.set_title(r"CCF: $|C|$ vs $|M_{\mathrm{n}}|$")
    elif bool(args.abs_curv) and (not bool(args.abs_mn)):
        ax.set_title(r"CCF: $|C|$ vs $M_{\mathrm{n}}$")
    elif (not bool(args.abs_curv)) and bool(args.abs_mn):
        ax.set_title(r"CCF: $C$ vs $|M_{\mathrm{n}}|$")
    else:
        ax.set_title(r"CCF: $C$ vs $M_{\mathrm{n}}$")
    ax.set_xlabel(r"Lag ($\mathrm{km}$)")
    ax.set_ylabel("CCF")
    ax.grid(True, linestyle=":", alpha=0.25, linewidth=0.6)
    ax.set_xlim(-ccf_xlim_km, ccf_xlim_km)
    if np.isfinite(lagCM.get("lag_m", float("nan"))):
        tau_km = float(lagCM["lag_m"]) / 1000.0
        rr = float(lagCM["corr"])
        ax.plot(tau_km, rr, "o", color="tab:red")
        yoff_override = 18.0 if str(_display_site(site)) == "YR-A" else None
        _annotate_peak(ax, tau_km=tau_km, rr=rr, color="tab:red", yoff=yoff_override)

    ax = axes[1, 1]
    ax.plot(ccfBM_lags / 1000.0, ccfBM, color="tab:orange", linewidth=2.0)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title(r"CCF: $B$ vs $|M_{\mathrm{n}}|$" if bool(args.abs_mn) else r"CCF: $B$ vs $M_{\mathrm{n}}$")
    ax.set_xlabel(r"Lag ($\mathrm{km}$)")
    ax.set_ylabel("CCF")
    ax.grid(True, linestyle=":", alpha=0.25, linewidth=0.6)
    ax.set_xlim(-ccf_xlim_km, ccf_xlim_km)
    if np.isfinite(lagBM.get("lag_m", float("nan"))):
        tau_km = float(lagBM["lag_m"]) / 1000.0
        rr = float(lagBM["corr"])
        ax.plot(tau_km, rr, "o", color="tab:orange")
        xoff_override = 28.0 if str(_display_site(site)) == "YR-B" else None
        _annotate_peak(ax, tau_km=tau_km, rr=rr, color="tab:orange", xoff=xoff_override)

    if args.preset != "paper":
        fig.suptitle(
            f"{site} Mask {mask} | {trunk_id} | min_trunk_length={float(args.min_trunk_length_m):.0f} m",
            fontsize=12,
        )

    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
