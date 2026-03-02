from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.analysis.quantitative_relationships import analyze_trunk_level_relationships
from src.analysis.plot_preset import setup_preset, get_paper_figsize


def plot_trunk_profiles(
    npz_path: Path,
    out_path: Path,
    site_name: str,
    mask_level: int,
    k_trunks: int = 1,
    min_trunk_length_m: float = 5000.0,
    endpoint_tol_m: float = 80.0,
    weight_by: str = "length_B",
    abs_mn: bool = False,
    abs_curv: bool = False,
    preset: str = "",
    dpi: int = 300,
    layout: str = "grid2x2",
) -> None:
    """
    Plots the streamwise profiles of Width (B), Curvature (C), and Migration Rate (Mn)
    for the extracted trunk(s).
    """
    print(f"Loading and analyzing: {npz_path}")
    res = analyze_trunk_level_relationships(
        npz_path,
        k_trunks=k_trunks,
        endpoint_tol_m=endpoint_tol_m,
        weight_by=weight_by,
        min_trunk_length_m=min_trunk_length_m,
    )

    if not res.trunks:
        print("No trunks found with the given parameters.")
        return

    try:
        summary = {
            tid: {
                "arc_length_m": float(res.trunk_metrics.get(tid, {}).get("arc_length_m", float("nan"))),
                "n_links": int(len(res.trunk_links.get(tid, []))),
                "n_samples": float(res.trunk_metrics.get(tid, {}).get("n_samples", float("nan"))),
            }
            for tid in res.trunks.keys()
        }
        print("TRUNK_SUMMARY=", summary)
    except Exception:
        pass

    # Sort trunk IDs to ensure consistent order (usually trunk_1 is the longest/dominant)
    trunk_ids = sorted(
        res.trunks.keys(),
        key=lambda x: int(x.split("_")[-1]) if "_" in x else 0
    )

    n_trunks_to_plot = len(trunk_ids)
    if n_trunks_to_plot == 0:
        print("No trunks to plot.")
        return

    for tid in trunk_ids:
        data = res.trunks[tid]
        s = data["s"]
        B = data["B"]
        C = data["C"]
        Mn = data["Mn"]

        C_plot = np.abs(C) if abs_curv else C
        Mn_plot = np.abs(Mn) if abs_mn else Mn
        
        # Convert s to km
        s_km = s / 1000.0

        trunk_metrics = res.trunk_metrics.get(tid, {}) if isinstance(res.trunk_metrics, dict) else {}
        arc_length_m = float(trunk_metrics.get("arc_length_m", float("nan"))) if isinstance(trunk_metrics, dict) else float("nan")
        arc_length_km = arc_length_m / 1000.0 if np.isfinite(arc_length_m) else float("nan")
        n_links = int(len(res.trunk_links.get(tid, []))) if isinstance(res.trunk_links, dict) else 0
        n_samples = float(trunk_metrics.get("n_samples", float("nan"))) if isinstance(trunk_metrics, dict) else float("nan")

        mean_B = float(np.nanmean(B)) if np.size(B) > 0 else float("nan")
        mean_C = float(np.nanmean(C_plot)) if np.size(C_plot) > 0 else float("nan")
        mean_Mn = float(np.nanmean(Mn_plot)) if np.size(Mn_plot) > 0 else float("nan")

        c_label = r"|C|" if bool(abs_curv) else r"C"
        mn_label = r"|M_{\mathrm{n}}|" if bool(abs_mn) else r"M_{\mathrm{n}}"
        
        # Labels (math): variables italic; descriptive subscript (normal-direction n) in roman
        b_ylabel = r"Width $B$ ($\mathrm{m}$)"
        c_ylabel = r"Curvature $|C|$ ($\mathrm{m^{-1}}$)" if abs_curv else r"Curvature $C$ ($\mathrm{m^{-1}}$)"
        m_ylabel = (
            r"Migration rate $|M_{\mathrm{n}}|$ ($\mathrm{m/yr}$)"
            if abs_mn
            else r"Migration rate $M_{\mathrm{n}}$ ($\mathrm{m/yr}$)"
        )

        # Determine figsize + layout
        if preset == "paper":
            if layout == "grid2x2":
                figsize = get_paper_figsize(190, 140)
            else:
                # Full width (190mm) for stacked 3 rows
                figsize = get_paper_figsize(190, 140)
        else:
            figsize = (12, 10) if layout != "grid2x2" else (12, 8)

        # Create figure
        if layout == "grid2x2":
            fig, axarr = plt.subplots(2, 2, figsize=figsize, constrained_layout=True, sharex=True)
            axB = axarr[0, 0]
            axC = axarr[0, 1]
            axM = axarr[1, 0]
            axInfo = axarr[1, 1]
            axInfo.axis("off")

            # Paper-style info panel (avoid an empty quadrant)
            info_lines = [
                (rf"$L = {arc_length_km:.2f}\,\mathrm{{km}}$" if np.isfinite(arc_length_km) else r"$L = \mathrm{n/a}$"),
                rf"$n_{{\mathrm{{links}}}} = {n_links}$",
                (rf"$n_{{\mathrm{{samples}}}} = {int(n_samples)}$" if np.isfinite(n_samples) else r"$n_{\mathrm{samples}} = \mathrm{n/a}$"),
                (rf"$\overline{{B}} = {mean_B:.0f}\,\mathrm{{m}}$" if np.isfinite(mean_B) else r"$\overline{B} = \mathrm{n/a}$"),
                (rf"$\overline{{{c_label}}} = {mean_C:.2e}\,\mathrm{{m^{{-1}}}}$" if np.isfinite(mean_C) else rf"$\overline{{{c_label}}} = \mathrm{{n/a}}$"),
                (rf"$\overline{{{mn_label}}} = {mean_Mn:.2f}\,\mathrm{{m/yr}}$" if np.isfinite(mean_Mn) else rf"$\overline{{{mn_label}}} = \mathrm{{n/a}}$"),
            ]
            axInfo.text(
                0.02,
                0.98,
                "\n".join(info_lines),
                transform=axInfo.transAxes,
                ha="left",
                va="top",
                fontsize=10 if preset == "paper" else 9,
                linespacing=1.5 if preset == "paper" else 1.2,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2.0),
            )
        else:
            fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True, constrained_layout=True)
            axB, axC, axM = axes[0], axes[1], axes[2]
            axInfo = None
        
        # 1. Width B(s)
        axB.plot(s_km, B, color='tab:blue', linewidth=1.0 if preset=="paper" else 1.5, label=r"$B$")
        axB.set_ylabel(b_ylabel)
        if preset != "paper":
            axB.set_title(f'{site_name} Mask {mask_level} - {tid} - Morphodynamics')

        grid_alpha = 0.25 if preset == "paper" else 0.6
        axB.grid(True, linestyle=":", alpha=grid_alpha, linewidth=0.6)
        
        # Add mean width line
        axB.axhline(mean_B, color='tab:blue', linestyle='--', alpha=0.5, label=rf"Mean $B$ = {mean_B:.0f}\,\mathrm{{m}}")
        if not (preset == "paper" and layout == "grid2x2"):
            axB.legend(loc='upper right')

        # 2. Curvature C(s)
        axC.plot(s_km, C_plot, color='tab:green', linewidth=1.0 if preset=="paper" else 1.5)
        axC.set_ylabel(c_ylabel)
        axC.grid(True, linestyle=":", alpha=grid_alpha, linewidth=0.6)
        axC.axhline(0, color='black', linewidth=0.8, alpha=0.5)

        # In 2×2 layout the top-right panel should also show the x-axis information.
        if layout == "grid2x2":
            axC.set_xlabel(r"Streamwise distance $s$ ($\mathrm{km}$)")
            axC.tick_params(axis="x", labelbottom=True)

        # 3. Migration Rate Mn(s)
        axM.plot(s_km, Mn_plot, color='tab:red', linewidth=1.0 if preset=="paper" else 1.5)
        axM.set_ylabel(m_ylabel)
        axM.set_xlabel(r"Streamwise distance $s$ ($\mathrm{km}$)")
        axM.grid(True, linestyle=":", alpha=grid_alpha, linewidth=0.6)
        axM.axhline(0, color='black', linewidth=0.8, alpha=0.5)

        # In paper mode, keep the 2x2 layout visually consistent with Fig.3c:
        # the bottom-right panel is intentionally left blank.
        # (Any explanatory text should go into the figure caption.)

        # Save
        if len(trunk_ids) > 1:
            save_name = out_path.stem + f"_{tid}" + out_path.suffix
            this_out = out_path.with_name(save_name)
        else:
            this_out = out_path
            
        fig.savefig(this_out, dpi=dpi)
        plt.close(fig)
        print(f"Saved plot to: {this_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot B, C, Mn profiles for river trunks.")
    parser.add_argument("--site", type=str, required=True, help="Site name (e.g., HuangHe-A)")
    parser.add_argument("--mask-level", type=int, required=True, help="Mask level (1-4)")
    parser.add_argument("--npz", type=str, default="", help="Path to input NPZ file (optional override)")
    parser.add_argument("--out", type=str, default="", help="Path to output PNG file (optional override)")
    parser.add_argument("--min-trunk-length-m", type=float, default=5000.0, help="Minimum trunk length in meters")
    parser.add_argument("--k-trunks", type=int, default=1, help="Number of trunks to extract and plot")
    parser.add_argument("--endpoint-tol-m", type=float, default=80.0, help="Endpoint clustering tolerance (m)")
    parser.add_argument("--weight-by", type=str, default="length_B", help="Trunk weight: length or length_B")
    parser.add_argument("--abs-mn", action="store_true", help="Plot absolute migration rate |Mn| instead of signed Mn")
    parser.add_argument("--abs-curv", action="store_true", help="Plot absolute curvature |C| instead of signed C")
    
    # Preset args
    parser.add_argument("--preset", type=str, default="", choices=["", "paper"], help="Style preset")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for saved figure")
    parser.add_argument(
        "--layout",
        type=str,
        default="grid2x2",
        choices=["stacked", "grid2x2"],
        help="Layout for profiles: stacked (3x1) or grid2x2 (2x2 with info panel)",
    )

    args = parser.parse_args()
    
    # Apply preset
    setup_preset(args.preset, args.dpi)

    site = args.site
    mask = args.mask_level
    
    # Default paths
    if not args.npz:

        # Assuming standard naming convention from run_huanghe_pipeline.bat
        # results/PostprocessedPIV/%SITE%/%SITE%_mask%MASK%_link_sBCMn_flat_step20_metric_v2.npz
        npz_path = _PROJECT_ROOT / "results" / "PostprocessedPIV" / site / f"{site}_mask{mask}_link_sBCMn_flat_step20_metric_v2.npz"
    else:
        npz_path = Path(args.npz)

    if not args.out:
        out_dir = _PROJECT_ROOT / "results" / "figures" / "profiles"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{site}_mask{mask}_trunk_profiles.png"
    else:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    if not npz_path.exists():
        print(f"Error: Input file not found: {npz_path}")
        sys.exit(1)

    plot_trunk_profiles(
        npz_path=npz_path,
        out_path=out_path,
        site_name=site,
        mask_level=mask,
        k_trunks=args.k_trunks,
        min_trunk_length_m=args.min_trunk_length_m,
        endpoint_tol_m=args.endpoint_tol_m,
        weight_by=args.weight_by,
        abs_mn=bool(args.abs_mn),
        abs_curv=bool(args.abs_curv),
        preset=str(args.preset),
        dpi=int(args.dpi),
        layout=str(args.layout),
    )


if __name__ == "__main__":
    main()
