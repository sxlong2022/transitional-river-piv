"""Data acquisition layer: Module interfacing with GEE and local data.

Currently provides only path return functions. Future implementations may include:
- Using earthengine-api + geemap to download DSWE annual masks from GEE
- Uniformly writing downloaded results to Data/GEOTIFFS or custom data directory
"""

from pathlib import Path

from src.config import DATA_DIR


def get_data_root() -> Path:
    """Returns the data root directory convention for the current project."""
    return DATA_DIR


if __name__ == "__main__":
    print("DATA_DIR:", get_data_root(), "exists=", get_data_root().exists())
