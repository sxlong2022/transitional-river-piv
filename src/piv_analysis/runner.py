"""PIV analysis placeholder module: Full OpenPIV workflow will be implemented here."""

from pathlib import Path

from src.preprocessing.prepared_imagery import get_prepared_imagery_dir


def describe_piv_inputs(site: str, mask_level: int = 1, tilt_deg: int = 0) -> Path:
    """Returns the input image directory Path for a given river segment, mask level, and tilt.

    Consistent with the Data/PreparedImagery/{Site}/MaskX_TiltYY structure in the paper.
    """
    base = get_prepared_imagery_dir(site)
    sub = f"Mask{mask_level}_Tilt{abs(tilt_deg):02d}"
    return base / sub


if __name__ == "__main__":
    site = "Jurua-A"
    path = describe_piv_inputs(site, mask_level=1, tilt_deg=0)
    print(f"PIV input dir for {site}, mask1, tilt0: {path} (exists={path.exists()})")
