"""Image preprocessing layer: Module interfacing with the PreparedImagery directory.

Supports two data sources:
- Jurua-A and other original paper sites: located at 文献/.../Data_and_code/Data/PreparedImagery/
- HuangHe-A/B and other custom sites: located at project root data/PreparedImagery/
"""

from pathlib import Path

from src.config import DATA_DIR, PROJECT_ROOT

# Custom data directory under project root
LOCAL_DATA_DIR = PROJECT_ROOT / "data"

# Custom site list (e.g., Yellow River sites)
LOCAL_SITES = {"HuangHe-A", "HuangHe-B"}


def get_prepared_imagery_dir(site: str) -> Path:
    """Returns the PreparedImagery directory Path for the specified river site.

    Parameters
    ----------
    site : str
        Site name, e.g., "Jurua-A" or "HuangHe-A".

    Returns
    -------
    Path
        Path to the corresponding PreparedImagery directory (existence not guaranteed).
        - Jurua-A and other paper sites → 文献/.../Data_and_code/Data/PreparedImagery/{site}
        - HuangHe-A/B → project root/data/PreparedImagery/{site}
    """
    if site in LOCAL_SITES:
        return LOCAL_DATA_DIR / "PreparedImagery" / site
    else:
        return DATA_DIR / "PreparedImagery" / site


def get_geotiffs_dir(site: str) -> Path:
    """Returns the GEOTIFFS directory Path for the specified river site.

    Parameters
    ----------
    site : str
        Site name, e.g., "Jurua-A" or "HuangHe-A".
    """
    if site in LOCAL_SITES:
        return LOCAL_DATA_DIR / "GEOTIFFS" / site
    else:
        return DATA_DIR / "GEOTIFFS" / site


def get_gis_dir(site: str) -> Path:
    if site in LOCAL_SITES:
        return LOCAL_DATA_DIR / "GIS"
    else:
        return DATA_DIR / "GIS"


if __name__ == "__main__":
    for example in ["Jurua-A", "HuangHe-A", "HuangHe-B"]:
        p = get_prepared_imagery_dir(example)
        g = get_geotiffs_dir(example)
        print(f"{example}:")
        print(f"  PreparedImagery: {p} (exists={p.exists()})")
        print(f"  GEOTIFFS:        {g} (exists={g.exists()})")
