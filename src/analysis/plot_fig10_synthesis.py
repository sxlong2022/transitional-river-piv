import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from src.analysis.plot_preset import get_paper_figsize, setup_preset
from src.analysis.quantitative_relationships import analyze_trunk_level_relationships


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
    cand2 = p / f"{site_io}_mask{int(mask_level)}_link_sBCMn_flat.npz"
    return cand2


def _pick_trunk_id(res) -> Optional[str]:
    if not getattr(res, "trunks", None):
        return None
    if "trunk_1" in res.trunks:
        return "trunk_1"
    try:
        return sorted(res.trunks.keys())[0]
    except Exception:
        return None


def _extract_metrics(res, trunk_id: str) -> Dict[str, float]:
    d = getattr(res, "diagnostics", {}) if res is not None else {}
    critA = d.get("criterionA", {}) if isinstance(d, dict) else {}
    n_eff = float(critA.get("N_eff", np.nan)) if isinstance(critA, dict) else float("nan")

    critB_all = d.get("criterionB", {}) if isinstance(d, dict) else {}
    critB = critB_all.get(trunk_id, {}) if isinstance(critB_all, dict) else {}

    acfB = critB.get("autocorr_B", {}) if isinstance(critB, dict) else {}
    acfC = critB.get("autocorr_abs_C", {}) if isinstance(critB, dict) else {}

    le_B_m = float(acfB.get("e_folding_m", np.nan)) if isinstance(acfB, dict) else float("nan")
    le_C_m = float(acfC.get("e_folding_m", np.nan)) if isinstance(acfC, dict) else float("nan")

    lag_BM_m = float(critB.get("lag_B_vs_absMn_m", np.nan)) if isinstance(critB, dict) else float("nan")
    corr_BM = float(critB.get("corr_B_vs_absMn", np.nan)) if isinstance(critB, dict) else float("nan")

    lag_CM_m = float(critB.get("lag_absC_vs_absMn_m", np.nan)) if isinstance(critB, dict) else float("nan")
    corr_CM = float(critB.get("corr_absC_vs_absMn", np.nan)) if isinstance(critB, dict) else float("nan")

    return {
        "N_eff": n_eff,
        "Le_B_m": le_B_m,
        "Le_absC_m": le_C_m,
        "lag_B_vs_absMn_m": lag_BM_m,
        "corr_B_vs_absMn": corr_BM,
        "lag_absC_vs_absMn_m": lag_CM_m,
        "corr_absC_vs_absMn": corr_CM,
    }


def _range_summary(values: List[float]) -> Dict[str, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"center": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    return {
        "center": float(np.nanmedian(x)),
        "lo": float(np.nanmin(x)),
        "hi": float(np.nanmax(x)),
        "n": int(x.size),
    }


def plot_fig10(
    out_path: Path,
    *,
    sites: List[str],
    yr_masks: List[int],
    jurua_mask: int,
    k_trunks_yr: int,
    k_trunks_jurua: int,
    min_trunk_length_m: float,
    endpoint_tol_m: float,
    weight_by: str,
    preset: str,
    dpi: int,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    setup_preset(str(preset), int(dpi))

    per_site: Dict[str, Dict[str, object]] = {}

    for site in sites:
        site_io = _normalize_site(site)
        if site_io == "Jurua-A":
            masks = [int(jurua_mask)]
            k_trunks = int(k_trunks_jurua)
        else:
            masks = [int(m) for m in yr_masks]
            k_trunks = int(k_trunks_yr)

        rows = []
        for mask in masks:
            npz_path = _default_sBCMn_npz_path(site_io, int(mask))
            if not npz_path.exists():
                rows.append({"mask": int(mask), "ok": False, "path": str(npz_path)})
                continue

            res = analyze_trunk_level_relationships(
                npz_path,
                k_trunks=int(k_trunks),
                endpoint_tol_m=float(endpoint_tol_m),
                weight_by=str(weight_by),
                min_trunk_length_m=float(min_trunk_length_m),
            )
            tid = _pick_trunk_id(res)
            if tid is None:
                rows.append({"mask": int(mask), "ok": False, "path": str(npz_path)})
                continue

            metrics = _extract_metrics(res, tid)
            rows.append({
                "mask": int(mask),
                "ok": True,
                "path": str(npz_path),
                "trunk_id": str(tid),
                "metrics": metrics,
            })

        per_site[site_io] = {
            "site": str(site),
            "site_io": str(site_io),
            "site_display": _display_site(site_io),
            "masks": masks,
            "rows": rows,
        }

    site_order = [_normalize_site(s) for s in sites]
    site_labels = [_display_site(s) for s in site_order]
    x_pos = np.arange(len(site_order), dtype=float)

    def _collect(site_io: str, key: str) -> List[float]:
        rows = per_site.get(site_io, {}).get("rows", [])
        out = []
        for r in rows:
            if not isinstance(r, dict) or (not bool(r.get("ok"))):
                continue
            m = r.get("metrics", {})
            if isinstance(m, dict):
                out.append(float(m.get(key, np.nan)))
        return out

    N_eff_s = [_range_summary(_collect(s, "N_eff")) for s in site_order]
    LeB_s = [_range_summary([v / 1000.0 for v in _collect(s, "Le_B_m")]) for s in site_order]
    LeC_s = [_range_summary([v / 1000.0 for v in _collect(s, "Le_absC_m")]) for s in site_order]

    lagBM_s = [_range_summary([v / 1000.0 for v in _collect(s, "lag_B_vs_absMn_m")]) for s in site_order]
    corrBM_s = [_range_summary(_collect(s, "corr_B_vs_absMn")) for s in site_order]

    lagCM_s = [_range_summary([v / 1000.0 for v in _collect(s, "lag_absC_vs_absMn_m")]) for s in site_order]
    corrCM_s = [_range_summary(_collect(s, "corr_absC_vs_absMn")) for s in site_order]

    summary = {
        "mode": "fig10_synthesis",
        "sites": site_order,
        "site_labels": site_labels,
        "yr_masks": [int(m) for m in yr_masks],
        "jurua_mask": int(jurua_mask),
        "params": {
            "k_trunks_yr": int(k_trunks_yr),
            "k_trunks_jurua": int(k_trunks_jurua),
            "min_trunk_length_m": float(min_trunk_length_m),
            "endpoint_tol_m": float(endpoint_tol_m),
            "weight_by": str(weight_by),
        },
        "per_site": per_site,
        "aggregated": {
            "N_eff": N_eff_s,
            "Le_B_km": LeB_s,
            "Le_absC_km": LeC_s,
            "ccf_B_vs_absMn": {"lag_km": lagBM_s, "corr": corrBM_s},
            "ccf_absC_vs_absMn": {"lag_km": lagCM_s, "corr": corrCM_s},
        },
    }

    fig, axes = plt.subplots(2, 2, figsize=get_paper_figsize(190, 130), constrained_layout=True)

    ax = axes[0, 0]
    y = np.array([d["center"] for d in N_eff_s], dtype=float)
    ylo = np.array([d["lo"] for d in N_eff_s], dtype=float)
    yhi = np.array([d["hi"] for d in N_eff_s], dtype=float)
    yerr = np.vstack([y - ylo, yhi - y])
    ax.errorbar(x_pos, y, yerr=yerr, fmt="o", color="k", capsize=3, elinewidth=1.0, markersize=4)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(site_labels)
    ax.set_ylabel(r"$N_{\mathrm{eff}}$")
    ax.set_title("Corridor-scale trunk organization")
    ax.grid(True, linestyle=":", alpha=0.25, linewidth=0.6)
    ax.set_ylim(0.8, max(1.2, float(np.nanmax(yhi)) * 1.05) if np.isfinite(yhi).any() else 1.2)
    ax.text(0.02, 0.98, "(a)", transform=ax.transAxes, ha="left", va="top")

    ax = axes[0, 1]
    off = 0.12
    yB = np.array([d["center"] for d in LeB_s], dtype=float)
    yBlo = np.array([d["lo"] for d in LeB_s], dtype=float)
    yBhi = np.array([d["hi"] for d in LeB_s], dtype=float)
    yBerr = np.vstack([yB - yBlo, yBhi - yB])

    yC = np.array([d["center"] for d in LeC_s], dtype=float)
    yClo = np.array([d["lo"] for d in LeC_s], dtype=float)
    yChi = np.array([d["hi"] for d in LeC_s], dtype=float)
    yCerr = np.vstack([yC - yClo, yChi - yC])

    ax.errorbar(x_pos - off, yB, yerr=yBerr, fmt="o", color="tab:blue", capsize=3, elinewidth=1.0, markersize=4, label=r"$L_e(B)$")
    ax.errorbar(x_pos + off, yC, yerr=yCerr, fmt="s", color="tab:green", capsize=3, elinewidth=1.0, markersize=4, label=r"$L_e(|C|)$")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(site_labels)
    ax.set_ylabel(r"E-folding scale $L_e$ (km)")
    ax.set_title("Characteristic organization scales")
    ax.grid(True, linestyle=":", alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False, loc="upper right")
    ymax = float("nan")
    if np.isfinite(yBhi).any() or np.isfinite(yChi).any():
        ymax = float(np.nanmax([float(np.nanmax(yBhi)), float(np.nanmax(yChi))]))
    if np.isfinite(ymax):
        ax.set_ylim(0.0, max(0.45, 1.05 * float(ymax)))
    else:
        ax.set_ylim(0.0, 0.45)
    ax.text(0.02, 0.98, "(b)", transform=ax.transAxes, ha="left", va="top")

    ax = axes[1, 0]
    xb = np.array([d["center"] for d in lagBM_s], dtype=float)
    xblo = np.array([d["lo"] for d in lagBM_s], dtype=float)
    xbhi = np.array([d["hi"] for d in lagBM_s], dtype=float)
    xberr = np.vstack([xb - xblo, xbhi - xb])

    yb = np.array([d["center"] for d in corrBM_s], dtype=float)
    yblo = np.array([d["lo"] for d in corrBM_s], dtype=float)
    ybhi = np.array([d["hi"] for d in corrBM_s], dtype=float)
    yberr = np.vstack([yb - yblo, ybhi - yb])

    ax.errorbar(xb, yb, xerr=xberr, yerr=yberr, fmt="o", color="tab:orange", capsize=3, elinewidth=1.0, markersize=4)
    for i, lab in enumerate(site_labels):
        ax.text(float(xb[i]) + 0.10, float(yb[i]), lab, fontsize=7, va="center")
    ax.axvline(0.0, color="0.2", linewidth=0.8, alpha=0.5)
    ax.axhline(0.0, color="0.2", linewidth=0.8, alpha=0.5)
    ax.set_xlabel(r"Lag $\tau$ (km)")
    ax.set_ylabel(r"Peak correlation $r$")
    ax.set_title(r"CCF peak: $B$ vs $|M_{\mathrm{n}}|$")
    ax.grid(True, linestyle=":", alpha=0.25, linewidth=0.6)
    ax.set_xlim(-6.0, 6.0)
    ax.set_ylim(-0.2, 1.0)
    ax.text(0.02, 0.98, "(c)", transform=ax.transAxes, ha="left", va="top")

    ax = axes[1, 1]
    xc = np.array([d["center"] for d in lagCM_s], dtype=float)
    xclo = np.array([d["lo"] for d in lagCM_s], dtype=float)
    xchi = np.array([d["hi"] for d in lagCM_s], dtype=float)
    xcerr = np.vstack([xc - xclo, xchi - xc])

    yc = np.array([d["center"] for d in corrCM_s], dtype=float)
    yclo = np.array([d["lo"] for d in corrCM_s], dtype=float)
    ychi = np.array([d["hi"] for d in corrCM_s], dtype=float)
    ycerr = np.vstack([yc - yclo, ychi - yc])

    # Visual jitter: avoid overlapping uncertainty envelopes for YR-A and YR-B in panel (d)
    yc_plot = yc.copy()
    yjit = np.zeros_like(yc_plot)
    for i, lab in enumerate(site_labels):
        if str(lab) == "YR-A":
            yjit[i] = 0.015
        elif str(lab) == "YR-B":
            yjit[i] = -0.015
    yc_plot = yc_plot + yjit

    ax.errorbar(xc, yc_plot, xerr=xcerr, yerr=ycerr, fmt="o", color="tab:red", capsize=3, elinewidth=1.0, markersize=4)
    for i, lab in enumerate(site_labels):
        ax.text(float(xc[i]) + 0.10, float(yc_plot[i]), lab, fontsize=7, va="center")
    ax.axvline(0.0, color="0.2", linewidth=0.8, alpha=0.5)
    ax.axhline(0.0, color="0.2", linewidth=0.8, alpha=0.5)
    ax.set_xlabel(r"Lag $\tau$ (km)")
    ax.set_ylabel(r"Peak correlation $r$")
    ax.set_title(r"CCF peak: $|C|$ vs $|M_{\mathrm{n}}|$")
    ax.grid(True, linestyle=":", alpha=0.25, linewidth=0.6)
    ax.set_xlim(-6.0, 6.0)
    ax.set_ylim(-0.2, 1.0)
    ax.text(0.02, 0.98, "(d)", transform=ax.transAxes, ha="left", va="top")

    fig.savefig(out_path, dpi=int(dpi))
    fig.savefig(out_path.with_suffix(".pdf"), format="pdf")
    plt.close(fig)

    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--sites", nargs="+", default=["Jurua-A", "YR-A", "YR-B"])
    parser.add_argument("--yr-masks", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--jurua-mask", type=int, default=1)

    parser.add_argument("--k-trunks-yr", type=int, default=2)
    parser.add_argument("--k-trunks-jurua", type=int, default=1)
    parser.add_argument("--min-trunk-length-m", type=float, default=5000.0)
    parser.add_argument("--endpoint-tol-m", type=float, default=80.0)
    parser.add_argument("--weight-by", type=str, default="length_B")

    parser.add_argument("--preset", type=str, default="paper")
    parser.add_argument("--dpi", type=int, default=600)

    args = parser.parse_args()

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = _PROJECT_ROOT / "results" / "figures" / "paper" / "Fig10_Synthesis.png"

    plot_fig10(
        out_path,
        sites=[str(s) for s in args.sites],
        yr_masks=[int(m) for m in args.yr_masks],
        jurua_mask=int(args.jurua_mask),
        k_trunks_yr=int(args.k_trunks_yr),
        k_trunks_jurua=int(args.k_trunks_jurua),
        min_trunk_length_m=float(args.min_trunk_length_m),
        endpoint_tol_m=float(args.endpoint_tol_m),
        weight_by=str(args.weight_by),
        preset=str(args.preset),
        dpi=int(args.dpi),
    )

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
