from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np


def export_link_sBCMn_to_npz(
    link_profiles: Dict[str, Dict[str, np.ndarray]],
    site: str,
    mask_level: int,
    step_m: float,
    out_path: str | Path,
) -> None:
    """Export per-link s-B-C-Mn profiles as flattened sample .npz.

    Parameters
    ------
    link_profiles : {link_id: {"s","x","y","B","C","Mn"}}
        For example, output from add_Mn_to_link_profiles.
    site : str
        Site name, e.g., "Jurua-A".
    mask_level : int
        Mask level (for metadata recording).
    step_m : float
        Nominal sampling interval along the curve (for metadata recording).
    out_path : str or Path
        Output .npz path.
    """

    out_path = Path(out_path)

    link_ids = sorted(link_profiles.keys())
    if not link_ids:
        raise ValueError("link_profiles is empty, cannot export.")

    link_index_list: list[np.ndarray] = []
    sample_in_link_list: list[np.ndarray] = []
    s_list: list[np.ndarray] = []
    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    B_list: list[np.ndarray] = []
    C_list: list[np.ndarray] = []
    Mn_list: list[np.ndarray] = []

    for idx, link_id in enumerate(link_ids):
        prof = link_profiles[link_id]

        s = np.asarray(prof["s"], dtype=float)
        x = np.asarray(prof["x"], dtype=float)
        y = np.asarray(prof["y"], dtype=float)
        B = np.asarray(prof["B"], dtype=float)
        C = np.asarray(prof["C"], dtype=float)
        Mn = np.asarray(prof["Mn"], dtype=float)

        n = s.size
        if not (x.size == y.size == B.size == C.size == Mn.size == n):
            raise ValueError(f"link {link_id} profile array lengths are inconsistent, cannot export.")

        link_index_list.append(np.full(n, idx, dtype=int))
        sample_in_link_list.append(np.arange(n, dtype=int))

        s_list.append(s)
        x_list.append(x)
        y_list.append(y)
        B_list.append(B)
        C_list.append(C)
        Mn_list.append(Mn)

    link_index = np.concatenate(link_index_list)
    sample_in_link = np.concatenate(sample_in_link_list)
    s_all = np.concatenate(s_list)
    x_all = np.concatenate(x_list)
    y_all = np.concatenate(y_list)
    B_all = np.concatenate(B_list)
    C_all = np.concatenate(C_list)
    Mn_all = np.concatenate(Mn_list)

    np.savez(
        out_path,
        link_ids=np.array(link_ids),
        link_index=link_index,
        sample_in_link=sample_in_link,
        s=s_all,
        x=x_all,
        y=y_all,
        B=B_all,
        C=C_all,
        Mn=Mn_all,
        site=str(site),
        mask_level=int(mask_level),
        step_m=float(step_m),
    )


def load_link_sBCMn_npz(path: str | Path) -> Dict[str, np.ndarray]:
    """Read flattened sample .npz and return all arrays as dict."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}
