import os
from pathlib import Path

# Project root = two levels up from this file (i.e., parent of src)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data directory can be overridden via RIVERPIV_DATA_ROOT env var; defaults to data folder under project root
_env_data_root = os.environ.get("RIVERPIV_DATA_ROOT")
if _env_data_root:
    DATA_DIR = Path(_env_data_root)
else:
    DATA_DIR = PROJECT_ROOT / "data"


def summarize_paths() -> str:
    """Returns an overview of key path existence for quick validation of data coupling."""
    lines = [
        f"PROJECT_ROOT      = {PROJECT_ROOT}",
        f"DATA_DIR          = {DATA_DIR}  (exists={DATA_DIR.exists()})",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(summarize_paths())
