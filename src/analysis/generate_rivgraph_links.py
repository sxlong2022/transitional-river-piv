from __future__ import annotations

from pathlib import Path

import argparse

from rivgraph.classes import river

from src.config import PROJECT_ROOT
from src.morphodynamics.jurua_georef_multitilt import _choose_reference_mask


def build_rivgraph_links(
    site: str = "Jurua-A",
    mask_level: int = 1,
    ref_year: int = 1987,
    exit_sides: str = "SN",
    mask_raster: str | None = None,
    out_dir: str | None = None,
    name: str | None = None,
) -> Path:
    exit_sides = exit_sides.upper()
    if len(exit_sides) != 2 or any(ch not in "NESW" for ch in exit_sides):
        raise ValueError(
            "exit_sides must be a 2-character string composed of N/E/S/W, e.g., 'NS', 'EW', 'NW'.",
        )
    if mask_raster is None:
        mask_path = _choose_reference_mask(site=site, mask_level=mask_level, year=ref_year)
    else:
        mask_path = Path(mask_raster)
        if not mask_path.exists():
            raise FileNotFoundError(f"Specified mask raster does not exist: {mask_path}")

    if out_dir is None:
        out_dir_path = PROJECT_ROOT / "results" / "RivGraph" / site / f"mask{mask_level}"
    else:
        out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    if name is None:
        name = f"{site}_mask{mask_level}"

    print("Using mask raster:", mask_path)
    print("RivGraph output directory:", out_dir_path)
    print("exit_sides:", exit_sides)

    net = river(
        name=name,
        path_to_mask=str(mask_path),
        results_folder=str(out_dir_path),
        exit_sides=exit_sides,
        verbose=False,
    )

    net.compute_network()

    # Network pruning can be done as needed; we skip automatic prune here
    # so users can handle it later in their RivGraph workflow if desired.
    net.to_geovectors(export="network", ftype="shp")

    links_path = Path(net.paths.get("links", out_dir_path))
    nodes_path = Path(net.paths.get("nodes", out_dir_path))

    print("Exported links to:", links_path)
    print("Exported nodes to:", nodes_path)

    return links_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate RivGraph river network (links/nodes) from a DSWE river mask "
            "consistent with the PIV georeferencing step."
        ),
    )
    parser.add_argument("--site", default="Jurua-A", help="Site name, e.g., 'Jurua-A'")
    parser.add_argument("--mask-level", type=int, default=1, help="Mask level, e.g., 1 for Mask1")
    parser.add_argument(
        "--ref-year",
        type=int,
        default=None,
        help="Preferred year for selecting reference mask (optional; if omitted, first .tif is selected automatically)",
    )
    parser.add_argument(
        "--exit-sides",
        default="SN",
        help="Upstream and downstream exit boundaries on the mask image, e.g., 'SN', 'NS' (upstream first).",
    )
    parser.add_argument(
        "--mask-raster",
        default=None,
        help="Optional: explicitly specify mask raster path; if omitted, _choose_reference_mask will be called.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="RivGraph output directory, defaults to results/RivGraph/<site>/mask<mask_level>.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="RivGraph RiverNetwork name identifier, defaults to '<site>_mask<mask_level>'.",
    )

    args = parser.parse_args()

    out_dir_path = build_rivgraph_links(
        site=args.site,
        mask_level=args.mask_level,
        ref_year=args.ref_year,
        exit_sides=args.exit_sides,
        mask_raster=args.mask_raster,
        out_dir=args.out_dir,
        name=args.name,
    )

    print("RivGraph network generated, output directory:", out_dir_path)
    print("Please find links/nodes vector files (e.g., links.shp/nodes.shp or corresponding gpkg) in this directory.")


if __name__ == "__main__":
    main()
