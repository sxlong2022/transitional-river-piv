"""One-click pipeline for PIV riverbank migration at the Jurua site.

Functional Overview:
- For a given set of mask levels and tilt angles at a site, sequentially runs:
  1) Multi-angle time-series PIV fusion (pixel coordinates, pixel/yr);
  2) Step 4A: Strict affine georeferencing + conversion to m/yr;
  3) Step 4B: Centerline normal projection to obtain bank migration profiles;
  4) (Optional) Superposition of Mn profiles across multiple masks to calculate
     mean and standard deviation, representing uncertainty from DSWE confidence levels.

This script serves as a "scheduler"; core algorithms are implemented in sub-modules:
- src.piv_analysis.jurua_multitilt.run_multitilt_jurua
- src.morphodynamics.jurua_georef_multitilt.georef_multitilt_jurua
- src.morphodynamics.jurua_centerline_profile.build_centerline_profile

Execution Examples (from project root):

    # Full pipeline for a single mask (Mask 1)
    python -m src.pipeline.jurua_pipeline \
        --site Jurua-A --mask-levels 1 --tilts 0 15 30 45 --step-m 100 --ref-year 1987

    # Full pipeline for multi-mask (Mask 1–4) + uncertainty envelope
    python -m src.pipeline.jurua_pipeline \
        --site Jurua-A --mask-levels 1 2 3 4 --tilts 0 15 30 45 --step-m 100 --ref-year 1987
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import argparse
import numpy as np
import matplotlib.pyplot as plt

from src.config import PROJECT_ROOT
from src.piv_analysis.jurua_multitilt import TILTS_DEG, run_multitilt_jurua
from src.morphodynamics.jurua_georef_multitilt import georef_multitilt_jurua
from src.morphodynamics.jurua_centerline_profile import build_centerline_profile
from src.postprocessing.postprocess import get_postprocessed_dir
from src.visualization.quicklook import describe_output_root


def run_full_pipeline(
    site: str = "Jurua-A",
    mask_levels: Iterable[int] = (1,),
    tilts: Iterable[int] = TILTS_DEG,
    step_m: float = 100.0,
    ref_year: int = 1987,
) -> Dict[int, Dict[str, Any]]:
    """Runs the full pipeline for a set of mask levels and returns centerline profile results.

    Returns profiles[mask_level] as the dictionary from build_centerline_profile.
    """

    mask_levels = tuple(int(m) for m in mask_levels)
    tilts = tuple(int(t) for t in tilts)

    out_root = describe_output_root(PROJECT_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)

    profiles: Dict[int, Dict[str, Any]] = {}

    for mask_level in mask_levels:
        print("=" * 80)
        print(f"Running PIV riverbank migration pipeline for {site} Mask{mask_level}")
        print("=" * 80)

        # ------------------------------------------------------------------
        # Step 1–2: Multi-tilt time-series PIV fusion (pixel coordinates)
        # ------------------------------------------------------------------
        x, y, stats_tilt, per_tilt = run_multitilt_jurua(
            site=site,
            mask_level=mask_level,
            tilt_degs=tilts,
        )

        u_mean = stats_tilt["u_mean"]
        v_mean = stats_tilt["v_mean"]
        bad = stats_tilt["bad_mask"]

        print("Participating tilts:", sorted(per_tilt.keys()))

        # Plot: Time-averaged vector field (unfiltered / filtered) after multi-tilt fusion
        # Filenames consistent with jurua_multitilt.main
        out_png_all = out_root / f"jurua_multitilt_mask{mask_level}_all.png"
        fig, ax = plt.subplots(figsize=(8, 6))
        mag = np.hypot(u_mean, v_mean)
        q = ax.quiver(x, y, u_mean, v_mean, mag, cmap="viridis", scale=50)
        plt.colorbar(q, ax=ax, label="|v_mean| (multi-tilt, pixel/yr)")
        ax.set_aspect("equal")
        ax.set_title(f"{site} Mask{mask_level} multi-tilt preferred PIV (all)")
        fig.savefig(out_png_all, dpi=200)
        plt.close(fig)

        out_png_filt = out_root / f"jurua_multitilt_mask{mask_level}_filtered.png"
        u_f = u_mean.copy()
        v_f = v_mean.copy()
        u_f[bad] = np.nan
        v_f[bad] = np.nan

        fig, ax = plt.subplots(figsize=(8, 6))
        mag_f = np.hypot(u_f, v_f)
        q = ax.quiver(x, y, u_f, v_f, mag_f, cmap="viridis", scale=50)
        plt.colorbar(q, ax=ax, label="|v_mean| (multi-tilt, filtered, pixel/yr)")
        ax.set_aspect("equal")
        ax.set_title(f"{site} Mask{mask_level} multi-tilt preferred PIV (filtered)")
        fig.savefig(out_png_filt, dpi=200)
        plt.close(fig)

        print("Saved multi-tilt fusion vector field (all) to:", out_png_all)
        print("Saved multi-tilt fusion vector field (filtered) to:", out_png_filt)

        # ------------------------------------------------------------------
        # Step 3 / Step 4A: Strict affine georeferencing + m/yr conversion
        # ------------------------------------------------------------------
        geo = georef_multitilt_jurua(site=site, mask_level=mask_level, ref_year=ref_year)

        X_geo = geo["X_geo"]
        Y_geo = geo["Y_geo"]
        u_m_per_year = geo["u_m_per_year"]
        v_m_per_year = geo["v_m_per_year"]
        dt_mean_years = geo["dt_mean_years"]
        ref_path = geo["ref_path"]

        print("Reference mask (Step 4A):", ref_path)
        print("Estimated mean year interval dt_mean_years:", dt_mean_years)

        out_dir = get_postprocessed_dir(PROJECT_ROOT, site)
        out_dir.mkdir(parents=True, exist_ok=True)

        out_npz_4a = out_dir / f"jurua_mask{mask_level}_multitilt_georef_step4a_strict.npz"
        np.savez(
            out_npz_4a,
            X_geo=X_geo,
            Y_geo=Y_geo,
            u_m_per_year=u_m_per_year,
            v_m_per_year=v_m_per_year,
            dt_mean_years=dt_mean_years,
        )
        print("Saved strict m/yr vector field to:", out_npz_4a)

        out_png_4a = out_root / f"jurua_multitilt_mask{mask_level}_georef_m_per_year_strict.png"
        fig, ax = plt.subplots(figsize=(8, 6))
        mag_m = np.hypot(u_m_per_year, v_m_per_year)
        q = ax.quiver(X_geo, Y_geo, u_m_per_year, v_m_per_year, mag_m, cmap="viridis", scale=50_000)
        plt.colorbar(q, ax=ax, label="|v| (m/yr, strict affine)")
        ax.set_aspect("equal")
        ax.set_title(f"{site} Mask{mask_level} multi-tilt preferred PIV (m/yr, Step 4A strict)")
        fig.savefig(out_png_4a, dpi=200)
        plt.close(fig)
        print("Saved physical scale vector plot to:", out_png_4a)

        # ------------------------------------------------------------------
        # Step 4 / Step 4B: Centerline normal projection, obtain Mn profile
        # ------------------------------------------------------------------
        profile = build_centerline_profile(
            site=site,
            mask_level=mask_level,
            step_m=step_m,
            ref_year=ref_year,
        )

        s = profile["s"]
        xs = profile["xs"]
        ys = profile["ys"]
        Mn = profile["Mn"]

        profiles[mask_level] = profile

        out_npz_4b = out_dir / f"jurua_mask{mask_level}_multitilt_centerline_profile_step4b.npz"
        np.savez(out_npz_4b, s=s, xs=xs, ys=ys, Mn=Mn)
        print("Saved centerline normal migration profile to:", out_npz_4b)

        out_png_map = out_root / f"jurua_mask{mask_level}_multitilt_centerline_Mn_map.png"
        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(xs, ys, c=Mn, cmap="RdBu_r", s=10)
        plt.colorbar(sc, ax=ax, label="Normal migration rate Mn (m/yr)")
        ax.set_aspect("equal")
        ax.set_title(f"{site} Mask{mask_level} multi-tilt: centerline normal migration (Step 4B)")
        fig.savefig(out_png_map, dpi=200)
        plt.close(fig)

        out_png_prof = out_root / f"jurua_mask{mask_level}_multitilt_centerline_Mn_profile.png"
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(s / 1000.0, Mn, "k-")
        ax.set_xlabel("Centerline distance s (km)")
        ax.set_ylabel("Normal migration rate Mn (m/yr)")
        ax.set_title(f"{site} Mask{mask_level} multi-tilt: centerline normal migration profile")
        ax.grid(True, alpha=0.3)
        fig.savefig(out_png_prof, dpi=200)
        plt.close(fig)

        print("Saved centerline normal migration map to:", out_png_map)
        print("Saved centerline normal migration profile to:", out_png_prof)

    # ----------------------------------------------------------------------
    # Multi-mask uncertainty: Superimpose Mn profiles across masks
    # ----------------------------------------------------------------------
    if len(mask_levels) > 1:
        mask_list: List[int] = []
        Mn_list: List[np.ndarray] = []
        s_ref = None
        xs_ref = None
        ys_ref = None

        for m in mask_levels:
            prof = profiles.get(m)
            if prof is None:
                continue
            if s_ref is None:
                s_ref = prof["s"]
                xs_ref = prof["xs"]
                ys_ref = prof["ys"]
            mask_list.append(m)
            Mn_list.append(prof["Mn"])

        if len(Mn_list) >= 2 and s_ref is not None:
            Mn_stack = np.stack(Mn_list, axis=0)
            Mn_mean = np.nanmean(Mn_stack, axis=0)
            Mn_std = np.nanstd(Mn_stack, axis=0)

            out_dir = get_postprocessed_dir(PROJECT_ROOT, site)
            out_dir.mkdir(parents=True, exist_ok=True)

            out_npz_multi = out_dir / "jurua_multimask_multitilt_centerline_profiles_step4b.npz"
            np.savez(
                out_npz_multi,
                mask_levels=np.array(mask_list, dtype=int),
                s=s_ref,
                xs=xs_ref,
                ys=ys_ref,
                Mn_stack=Mn_stack,
                Mn_mean=Mn_mean,
                Mn_std=Mn_std,
            )
            print("Saved multi-mask Mn profile stack results to:", out_npz_multi)

            out_png_multi = out_root / "jurua_multimask_multitilt_centerline_Mn_profile.png"
            fig, ax = plt.subplots(figsize=(8, 4))
            s_km = s_ref / 1000.0

            for m, Mn_m in zip(mask_list, Mn_stack):
                ax.plot(s_km, Mn_m, lw=0.6, alpha=0.4, label=f"Mask{m}")

            ax.plot(s_km, Mn_mean, "k-", lw=1.5, label="Mean across masks")
            ax.fill_between(
                s_km,
                Mn_mean - Mn_std,
                Mn_mean + Mn_std,
                color="k",
                alpha=0.15,
                linewidth=0,
                label="±1σ",
            )

            ax.set_xlabel("Centerline distance s (km)")
            ax.set_ylabel("Normal migration rate Mn (m/yr)")
            ax.set_title(f"{site} multi-mask multi-tilt: centerline normal migration profile")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=8)

            fig.savefig(out_png_multi, dpi=200)
            plt.close(fig)
            print("Saved multi-mask Mn profile plot to:", out_png_multi)

    return profiles


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full Jurua PIV riverbank migration pipeline "
            "(multi-tilt PIV → georeferencing → centerline normal migration)."
        ),
    )

    parser.add_argument("--site", default="Jurua-A", help="Site name, e.g., 'Jurua-A'")
    parser.add_argument(
        "--mask-levels",
        type=int,
        nargs="+",
        default=[1],
        help="List of mask levels to process, e.g., 1 2 3 4",
    )
    parser.add_argument(
        "--tilts",
        type=int,
        nargs="+",
        default=list(TILTS_DEG),
        help="List of tilts to involve in fusion, e.g., 0 15 30 45",
    )
    parser.add_argument("--step-m", type=float, default=100.0, help="Centerline sampling interval (meters)")
    parser.add_argument("--ref-year", type=int, default=1987, help="Reference mask priority year for Step 4A")

    args = parser.parse_args()

    run_full_pipeline(
        site=args.site,
        mask_levels=args.mask_levels,
        tilts=args.tilts,
        step_m=args.step_m,
        ref_year=args.ref_year,
    )


if __name__ == "__main__":
    main()
