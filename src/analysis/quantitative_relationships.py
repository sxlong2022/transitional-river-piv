from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Tuple

import numpy as np

from src.analysis.link_sBCMn_io import load_link_sBCMn_npz


try:
    from scipy.optimize import curve_fit  # type: ignore

    _HAS_SCIPY = True
except Exception:
    curve_fit = None
    _HAS_SCIPY = False


@dataclass(frozen=True)
class RegressionResult:
    model: str
    params: Dict[str, float]
    r2: float
    n: int


@dataclass(frozen=True)
class LinkLevelResult:
    metrics: Dict[str, np.ndarray]
    fits: Dict[str, RegressionResult]
    diagnostics: Dict[str, Any]


@dataclass(frozen=True)
class TrunkAggregationResult:
    trunks: Dict[str, Dict[str, np.ndarray]]
    trunk_links: Dict[str, list[str]]
    trunk_metrics: Dict[str, Dict[str, float]]
    fits: Dict[str, RegressionResult]
    diagnostics: Dict[str, Any]


def _as_1d_float(a: np.ndarray) -> np.ndarray:
    x = np.asarray(a, dtype=float).ravel()
    return x


def _valid_mask(*arrs: np.ndarray) -> np.ndarray:
    if not arrs:
        raise ValueError("arrs cannot be empty")

    m = np.ones_like(_as_1d_float(arrs[0]), dtype=bool)
    for a in arrs:
        x = _as_1d_float(a)
        m &= np.isfinite(x)
    return m


def _meters_per_degree(lat_deg: float) -> tuple[float, float]:
    lat_rad = np.deg2rad(float(lat_deg))
    m_per_deg_lat = (
        111132.92
        - 559.82 * np.cos(2 * lat_rad)
        + 1.175 * np.cos(4 * lat_rad)
        - 0.0023 * np.cos(6 * lat_rad)
    )
    m_per_deg_lon = (
        111412.84 * np.cos(lat_rad)
        - 93.5 * np.cos(3 * lat_rad)
        + 0.118 * np.cos(5 * lat_rad)
    )
    return float(m_per_deg_lon), float(m_per_deg_lat)


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
            return
        if self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
            return
        self.parent[rb] = ra
        self.rank[ra] += 1


def _cluster_points_xy_m(
    x_m: np.ndarray,
    y_m: np.ndarray,
    tol_m: float,
) -> np.ndarray:
    x_m = _as_1d_float(x_m)
    y_m = _as_1d_float(y_m)
    if x_m.size != y_m.size:
        raise ValueError("x_m and y_m must have the same length")
    n = int(x_m.size)
    if n == 0:
        return np.zeros(0, dtype=int)

    cell = float(tol_m)
    if (not np.isfinite(cell)) or cell <= 0:
        return np.arange(n, dtype=int)

    uf = _UnionFind(n)
    grid: Dict[tuple[int, int], list[int]] = {}

    xm_safe = np.where(np.isfinite(x_m), x_m, 0.0)
    ym_safe = np.where(np.isfinite(y_m), y_m, 0.0)
    rx = xm_safe / cell
    ry = ym_safe / cell
    rx = np.where(np.isfinite(rx), rx, 0.0)
    ry = np.where(np.isfinite(ry), ry, 0.0)
    fx = np.floor(rx)
    fy = np.floor(ry)
    fx = np.where(np.isfinite(fx), fx, 0.0)
    fy = np.where(np.isfinite(fy), fy, 0.0)
    with np.errstate(invalid="ignore"):
        xk = fx.astype(int)
        yk = fy.astype(int)
    tol2 = float(tol_m) ** 2

    for i in range(n):
        key = (int(xk[i]), int(yk[i]))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nb = (key[0] + dx, key[1] + dy)
                if nb not in grid:
                    continue
                for j in grid[nb]:
                    dd = float((x_m[i] - x_m[j]) ** 2 + (y_m[i] - y_m[j]) ** 2)
                    if dd <= tol2:
                        uf.union(i, j)

        grid.setdefault(key, []).append(i)

    roots = np.array([uf.find(i) for i in range(n)], dtype=int)
    uniq, inv = np.unique(roots, return_inverse=True)
    _ = uniq
    return inv


def _infer_west_east_node_sets(node_x_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    node_x_m = _as_1d_float(node_x_m)
    v = node_x_m[np.isfinite(node_x_m)]
    if v.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    # node_x_m represents the projected coordinate along the main axis (meters) during trunk aggregation.
    # We use percentiles here instead of min/max to avoid outliers expanding the span and selecting isolated endpoints.
    west_thr = float(np.nanpercentile(v, 5))
    east_thr = float(np.nanpercentile(v, 95))
    west = np.where(node_x_m <= west_thr)[0]
    east = np.where(node_x_m >= east_thr)[0]
    return west.astype(int), east.astype(int)


def _longest_path_dag(
    n_nodes: int,
    edges: list[tuple[int, int, float, str]],
    node_x_m: np.ndarray,
    sources: np.ndarray,
    targets: np.ndarray,
) -> list[str]:
    if n_nodes <= 0:
        return []
    if len(edges) == 0:
        return []
    # 若 source/target 为空，则退化为“全图最长路”（对分裂网络更稳健）
    if sources.size == 0:
        sources = np.arange(n_nodes, dtype=int)
    if targets.size == 0:
        targets = np.arange(n_nodes, dtype=int)

    adj: list[list[tuple[int, float, str]]] = [[] for _ in range(n_nodes)]
    for u, v, w, lid in edges:
        if 0 <= u < n_nodes and 0 <= v < n_nodes:
            adj[u].append((v, float(w), str(lid)))

    order = np.argsort(np.asarray(node_x_m, dtype=float))
    neg_inf = -1e300
    best = np.full(n_nodes, neg_inf, dtype=float)
    pred_node = np.full(n_nodes, -1, dtype=int)
    pred_edge = np.full(n_nodes, "", dtype=object)

    for s in sources.tolist():
        if 0 <= int(s) < n_nodes:
            best[int(s)] = 0.0

    for u in order.tolist():
        bu = float(best[u])
        if bu <= neg_inf / 2:
            continue
        for v, w, lid in adj[u]:
            cand = bu + float(w)
            if cand > float(best[v]):
                best[v] = cand
                pred_node[v] = int(u)
                pred_edge[v] = str(lid)

    t_best = -1
    best_val = neg_inf
    for t in targets.tolist():
        tt = int(t)
        if 0 <= tt < n_nodes and float(best[tt]) > best_val:
            best_val = float(best[tt])
            t_best = tt

    # If no target is reachable, fall back to the node with the maximum 'best' value reachable from sources.
    if t_best < 0 or best_val <= neg_inf / 2:
        candidates = np.where(pred_node >= 0)[0]
        if candidates.size == 0:
            return []
        t_best = int(candidates[int(np.nanargmax(best[candidates]))])

    path_edges: list[str] = []
    cur = t_best
    while True:
        e = str(pred_edge[cur])
        pn = int(pred_node[cur])
        if pn < 0 or e == "":
            break
        path_edges.append(e)
        cur = pn

    path_edges.reverse()
    return path_edges


def aggregate_trunks_from_flat_npz(
    data: Dict[str, np.ndarray],
    k_trunks: int = 4,
    endpoint_tol_m: float = 80.0,
    weight_by: str = "length",
    min_trunk_length_m: float = 0.0,
) -> TrunkAggregationResult:
    link_ids = np.asarray(data["link_ids"], dtype=object).astype(str)
    link_index = np.asarray(data["link_index"], dtype=int)
    s_all = _as_1d_float(data["s"])
    x_all = _as_1d_float(data["x"])
    y_all = _as_1d_float(data["y"])
    B_all = _as_1d_float(data["B"])
    C_all = _as_1d_float(data["C"])
    Mn_all = _as_1d_float(data["Mn"])
    step_m = float(data.get("step_m", np.nan))

    n_links = int(link_ids.size)
    if n_links == 0:
        return TrunkAggregationResult(trunks={}, trunk_links={}, trunk_metrics={}, fits={}, diagnostics={})

    x0 = np.full(n_links, np.nan, dtype=float)
    y0 = np.full(n_links, np.nan, dtype=float)
    x1 = np.full(n_links, np.nan, dtype=float)
    y1 = np.full(n_links, np.nan, dtype=float)
    link_len = np.full(n_links, np.nan, dtype=float)
    link_meanB = np.full(n_links, np.nan, dtype=float)

    for i in range(n_links):
        m = link_index == i
        if not np.any(m):
            continue
        xs = x_all[m]
        ys = y_all[m]
        ss = s_all[m]
        Bs = B_all[m]
        x0[i] = float(xs[0])
        y0[i] = float(ys[0])
        x1[i] = float(xs[-1])
        y1[i] = float(ys[-1])
        if np.isfinite(ss).any():
            link_len[i] = float(np.nanmax(ss) - np.nanmin(ss))
        if np.isfinite(Bs).any():
            link_meanB[i] = float(np.nanmean(Bs[np.isfinite(Bs)]))

    lat_vals = np.concatenate([y0[np.isfinite(y0)], y1[np.isfinite(y1)]])
    lat_ref = float(np.nanmean(lat_vals)) if lat_vals.size > 0 else 0.0
    m_per_deg_lon0, m_per_deg_lat0 = _meters_per_degree(lat_ref)

    pts_x_m = np.concatenate([x0, x1]) * m_per_deg_lon0
    pts_y_m = np.concatenate([y0, y1]) * m_per_deg_lat0

    valid_pts = np.isfinite(pts_x_m) & np.isfinite(pts_y_m)
    if not np.all(valid_pts):
        pts_x_m = np.where(valid_pts, pts_x_m, np.nan)
        pts_y_m = np.where(valid_pts, pts_y_m, np.nan)

    ids = _cluster_points_xy_m(
        x_m=np.nan_to_num(pts_x_m, nan=1e30),
        y_m=np.nan_to_num(pts_y_m, nan=1e30),
        tol_m=float(endpoint_tol_m),
    )
    start_node = ids[:n_links]
    end_node = ids[n_links:]
    n_nodes = int(np.max(ids)) + 1 if ids.size else 0

    node_x_m = np.full(n_nodes, np.nan, dtype=float)
    node_y_m = np.full(n_nodes, np.nan, dtype=float)
    for nid in range(n_nodes):
        mm = ids == nid
        if not np.any(mm):
            continue
        node_x_m[nid] = float(np.nanmean(pts_x_m[mm]))
        node_y_m[nid] = float(np.nanmean(pts_y_m[mm]))

    # Use PCA main axis as the 'along-channel' direction to avoid assuming a fixed West->East (x-axis) orientation.
    finite_nodes = np.isfinite(node_x_m) & np.isfinite(node_y_m)
    if np.sum(finite_nodes) >= 2:
        XY = np.vstack([node_x_m[finite_nodes], node_y_m[finite_nodes]]).T
        mu = np.mean(XY, axis=0)
        C = np.cov((XY - mu).T)
        evals, evecs = np.linalg.eigh(C)
        axis = evecs[:, int(np.argmax(evals))]
        node_p_m = (np.vstack([node_x_m - mu[0], node_y_m - mu[1]]).T @ axis).astype(float)
    else:
        axis = np.array([1.0, 0.0], dtype=float)
        node_p_m = node_x_m.copy()

    # Before building directed edges, place source/target placeholders; 
    # these will be combined with node degrees for a more robust selection later.
    west, east = _infer_west_east_node_sets(node_p_m)

    def _edge_weight(i: int) -> float:
        L = float(link_len[i]) if np.isfinite(link_len[i]) else 0.0
        if weight_by == "length":
            return L
        if weight_by == "length_B":
            Bm = float(link_meanB[i]) if np.isfinite(link_meanB[i]) else 0.0
            return L * Bm
        return L

    edges: list[tuple[int, int, float, str]] = []
    edge_dir: Dict[str, tuple[int, int]] = {}
    out_deg = np.zeros(n_nodes, dtype=int)
    in_deg = np.zeros(n_nodes, dtype=int)
    for i, lid in enumerate(link_ids.tolist()):
        u = int(start_node[i])
        v = int(end_node[i])
        if n_nodes == 0:
            continue
        pu = float(node_p_m[u])
        pv = float(node_p_m[v])
        if not (np.isfinite(pu) and np.isfinite(pv)):
            continue
        if abs(pv - pu) < 1e-9:
            continue
        if pu < pv:
            uu, vv = u, v
        else:
            uu, vv = v, u
        w = _edge_weight(i)
        edges.append((uu, vv, float(w), str(lid)))
        edge_dir[str(lid)] = (uu, vv)
        out_deg[int(uu)] += 1
        in_deg[int(vv)] += 1

    # Filter source/target based on degree: source must have outgoing edges, target must have incoming edges.
    finite = np.isfinite(node_p_m)
    cand_w = finite & (out_deg > 0)
    cand_e = finite & (in_deg > 0)
    if np.any(cand_w):
        vw = node_p_m[cand_w]
        west_thr = float(np.nanpercentile(vw, 5))
        west = np.where(cand_w & (node_p_m <= west_thr))[0].astype(int)
        if west.size == 0:
            west = np.array([int(np.nanargmin(np.where(cand_w, node_p_m, np.nan)))], dtype=int)
    if np.any(cand_e):
        ve = node_p_m[cand_e]
        east_thr = float(np.nanpercentile(ve, 95))
        east = np.where(cand_e & (node_p_m >= east_thr))[0].astype(int)
        if east.size == 0:
            east = np.array([int(np.nanargmax(np.where(cand_e, node_p_m, np.nan)))], dtype=int)

    remaining = list(edges)
    trunks: Dict[str, Dict[str, np.ndarray]] = {}
    trunk_links: Dict[str, list[str]] = {}
    trunk_metrics: Dict[str, Dict[str, float]] = {}
    fits: Dict[str, RegressionResult] = {}
    criterionB: Dict[str, Any] = {}

    for k in range(int(k_trunks)):
        path = _longest_path_dag(
            n_nodes=n_nodes,
            edges=remaining,
            node_x_m=node_p_m,
            sources=west,
            targets=east,
        )
        if not path:
            break

        used = set(path)
        remaining = [e for e in remaining if str(e[3]) not in used]

        s_cat: list[np.ndarray] = []
        B_cat: list[np.ndarray] = []
        C_cat: list[np.ndarray] = []
        Mn_cat: list[np.ndarray] = []
        links_in_trunk: list[str] = []

        s_offset = 0.0
        for lid in path:
            idx = int(np.where(link_ids.astype(str) == str(lid))[0][0])
            m = link_index == idx
            ss = s_all[m]
            xs = x_all[m]
            ys = y_all[m]
            Bs = B_all[m]
            Cs = C_all[m]
            Ms = Mn_all[m]

            if ss.size == 0:
                continue
            ss0 = ss - float(np.nanmin(ss))

            u_edge, v_edge = edge_dir.get(str(lid), (None, None))
            reverse = False
            if u_edge is not None and v_edge is not None:
                su = int(start_node[idx])
                eu = int(end_node[idx])
                if not (su == u_edge and eu == v_edge):
                    if su == v_edge and eu == u_edge:
                        reverse = True

            if reverse:
                ss0 = float(np.nanmax(ss0)) - ss0
                xs = xs[::-1]
                ys = ys[::-1]
                Bs = Bs[::-1]
                Cs = (-Cs[::-1])
                Ms = (-Ms[::-1])
                ss0 = ss0[::-1]

            ss0 = ss0 + s_offset

            if s_cat:
                ss0 = ss0[1:]
                Bs = Bs[1:]
                Cs = Cs[1:]
                Ms = Ms[1:]

            s_cat.append(np.asarray(ss0, dtype=float))
            B_cat.append(np.asarray(Bs, dtype=float))
            C_cat.append(np.asarray(Cs, dtype=float))
            Mn_cat.append(np.asarray(Ms, dtype=float))
            links_in_trunk.append(str(lid))
            s_offset = float(np.nanmax(ss0)) if np.isfinite(ss0).any() else s_offset

        if not s_cat:
            continue

        s_tr = np.concatenate(s_cat)
        B_tr = np.concatenate(B_cat)
        C_tr = np.concatenate(C_cat)
        Mn_tr = np.concatenate(Mn_cat)

        trunk_len = float(np.nanmax(s_tr) - np.nanmin(s_tr)) if np.isfinite(s_tr).any() else float("nan")
        if np.isfinite(trunk_len) and float(trunk_len) < float(min_trunk_length_m):
            continue

        trunk_id = f"trunk_{k+1}"
        trunks[trunk_id] = {"s": s_tr, "B": B_tr, "C": C_tr, "Mn": Mn_tr}
        trunk_links[trunk_id] = links_in_trunk

        m_valid = np.isfinite(s_tr) & np.isfinite(B_tr) & np.isfinite(C_tr) & np.isfinite(Mn_tr)
        trunk_metrics[trunk_id] = {
            "n_samples": float(np.sum(np.isfinite(s_tr))),
            "arc_length_m": trunk_len,
            "mean_B": float(np.nanmean(B_tr[np.isfinite(B_tr)])) if np.isfinite(B_tr).any() else float("nan"),
            "mean_abs_C": float(np.nanmean(np.abs(C_tr[np.isfinite(C_tr)]))) if np.isfinite(C_tr).any() else float("nan"),
            "mean_abs_Mn": float(np.nanmean(np.abs(Mn_tr[np.isfinite(Mn_tr)]))) if np.isfinite(Mn_tr).any() else float("nan"),
            "valid_frac": float(np.sum(m_valid)) / float(s_tr.size) if s_tr.size > 0 else float("nan"),
        }

        # Sample-level fitting within each trunk (corresponds to per-channel analysis)
        xC = np.abs(C_tr)
        yM = np.abs(Mn_tr)
        fits[f"Mn_C_linear_{trunk_id}"] = fit_linear(xC, yM, with_intercept=True)
        fits[f"Mn_B_linear_{trunk_id}"] = fit_linear(B_tr, yM, with_intercept=True)

        if np.isfinite(step_m) and float(step_m) > 0:
            B_f = _fill_nan_linear(B_tr)
            C_f = _fill_nan_linear(np.abs(C_tr))
            M_f = _fill_nan_linear(yM)
            domB = dominant_wavelength(B_f, step_m=float(step_m))
            domC = dominant_wavelength(C_f, step_m=float(step_m))
            if np.isfinite(domC.get("freq", np.nan)):
                dphi_B_minus_C = phase_difference_at_frequency(C_f, B_f, step_m=float(step_m), freq=float(domC["freq"]))
            else:
                dphi_B_minus_C = float("nan")
            lag_BM = cross_correlation_lag(B_f, M_f, step_m=float(step_m))
            lag_CM = cross_correlation_lag(C_f, M_f, step_m=float(step_m))
            criterionB[trunk_id] = {
                "autocorr_B": autocorr_length_scales(B_f, step_m=float(step_m)),
                "autocorr_abs_C": autocorr_length_scales(C_f, step_m=float(step_m)),
                "dominant_B": domB,
                "dominant_abs_C": domC,
                "delta_phi_B_minus_absC_deg_at_absCdom": float(dphi_B_minus_C),
                "lag_B_vs_absMn_m": float(lag_BM["lag_m"]),
                "corr_B_vs_absMn": float(lag_BM["corr"]),
                "lag_absC_vs_absMn_m": float(lag_CM["lag_m"]),
                "corr_absC_vs_absMn": float(lag_CM["corr"]),
            }
        else:
            criterionB[trunk_id] = {}

    criterionA: Dict[str, Any] = {}
    across_linear_n = 0
    across_power_n = 0
    if trunk_metrics:
        trunk_ids = list(trunk_metrics.keys())
        arc = np.array([trunk_metrics[t]["arc_length_m"] for t in trunk_ids], dtype=float)
        mb = np.array([trunk_metrics[t]["mean_B"] for t in trunk_ids], dtype=float)
        if str(weight_by) == "length_B":
            w = arc * mb
        else:
            w = arc
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
        wsum = float(np.sum(w))
        if wsum > 0:
            p = w / wsum
            order = np.argsort(p)[::-1]
            p_sorted = p[order]
            trunks_sorted = [trunk_ids[int(i)] for i in order.tolist()]
            denom = float(np.sum(p_sorted**2))
            n_eff = 1.0 / denom if denom > 0 else float("nan")
            weights = {trunks_sorted[i]: float(w[int(order[i])]) for i in range(len(trunks_sorted))}
            fractions = {trunks_sorted[i]: float(p_sorted[i]) for i in range(len(trunks_sorted))}
            criterionA = {
                "weight_by": str(weight_by),
                "trunks_sorted": trunks_sorted,
                "weights": weights,
                "fractions": fractions,
                "N_eff": float(n_eff),
                "p_top1": float(p_sorted[0]) if p_sorted.size >= 1 else float("nan"),
                "p_top2": float(np.sum(p_sorted[:2])) if p_sorted.size >= 2 else float("nan"),
                "p_top3": float(np.sum(p_sorted[:3])) if p_sorted.size >= 3 else float("nan"),
            }

        xC_tr = np.array([trunk_metrics[t]["mean_abs_C"] for t in trunk_ids], dtype=float)
        yM_tr = np.array([trunk_metrics[t]["mean_abs_Mn"] for t in trunk_ids], dtype=float)
        m = np.isfinite(xC_tr) & np.isfinite(yM_tr)
        across_linear_n = int(np.sum(m))
        if across_linear_n >= 3:
            fits["Mn_C_linear_across_trunks"] = fit_linear(xC_tr[m], yM_tr[m], with_intercept=True)
        else:
            fits["Mn_C_linear_across_trunks"] = RegressionResult(model="linear", params={}, r2=float("nan"), n=int(across_linear_n))

        mp = m & (xC_tr > 0) & (yM_tr > 0)
        across_power_n = int(np.sum(mp))
        if across_power_n >= 3:
            fits["Mn_C_powerlaw_across_trunks"] = fit_powerlaw(xC_tr[mp], yM_tr[mp])
        else:
            fits["Mn_C_powerlaw_across_trunks"] = RegressionResult(model="powerlaw", params={}, r2=float("nan"), n=int(across_power_n))

    diagnostics: Dict[str, Any] = {
        "n_links": int(n_links),
        "n_nodes": int(n_nodes),
        "n_edges": int(len(edges)),
        "k_trunks": int(len(trunks)),
        "endpoint_tol_m": float(endpoint_tol_m),
        "weight_by": str(weight_by),
        "min_trunk_length_m": float(min_trunk_length_m),
        "step_m": float(step_m),
        "n_west_nodes": int(west.size),
        "n_east_nodes": int(east.size),
        "axis_px": float(axis[0]),
        "axis_py": float(axis[1]),
        "criterionA": criterionA,
        "criterionB": criterionB,
        "across_trunks_linear_n": int(across_linear_n),
        "across_trunks_powerlaw_n": int(across_power_n),
    }

    return TrunkAggregationResult(
        trunks=trunks,
        trunk_links=trunk_links,
        trunk_metrics=trunk_metrics,
        fits=fits,
        diagnostics=diagnostics,
    )


def analyze_trunk_level_relationships(
    sBCMn_npz_path: str | Path,
    k_trunks: int = 4,
    endpoint_tol_m: float = 80.0,
    weight_by: str = "length",
    min_trunk_length_m: float = 0.0,
) -> TrunkAggregationResult:
    p = Path(sBCMn_npz_path)
    data = load_link_sBCMn_npz(p)
    return aggregate_trunks_from_flat_npz(
        data=data,
        k_trunks=k_trunks,
        endpoint_tol_m=endpoint_tol_m,
        weight_by=weight_by,
        min_trunk_length_m=min_trunk_length_m,
    )


def scan_trunk_length_thresholds(
    sBCMn_npz_path: str | Path,
    thresholds_m: Iterable[float],
    k_trunks: int = 4,
    endpoint_tol_m: float = 80.0,
    weight_by: str = "length",
) -> Dict[str, Any]:
    p = Path(sBCMn_npz_path)
    data = load_link_sBCMn_npz(p)

    out: Dict[str, Any] = {
        "path": str(p),
        "thresholds_m": [float(t) for t in thresholds_m],
        "results": [],
    }

    for t in out["thresholds_m"]:
        r = aggregate_trunks_from_flat_npz(
            data=data,
            k_trunks=k_trunks,
            endpoint_tol_m=endpoint_tol_m,
            weight_by=weight_by,
            min_trunk_length_m=float(t),
        )
        critA = r.diagnostics.get("criterionA", {})
        out["results"].append(
            {
                "min_trunk_length_m": float(t),
                "k_trunks": int(r.diagnostics.get("k_trunks", 0)),
                "N_eff": float(critA.get("N_eff", float("nan"))) if isinstance(critA, dict) else float("nan"),
                "p_top2": float(critA.get("p_top2", float("nan"))) if isinstance(critA, dict) else float("nan"),
                "trunk_metrics": r.trunk_metrics,
            }
        )

    return out


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    y = _as_1d_float(y)
    yhat = _as_1d_float(yhat)
    m = _valid_mask(y, yhat)
    y = y[m]
    yhat = yhat[m]
    if y.size < 2:
        return float("nan")
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    if ss_tot == 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def fit_linear(x: np.ndarray, y: np.ndarray, with_intercept: bool = True) -> RegressionResult:
    x = _as_1d_float(x)
    y = _as_1d_float(y)
    m = _valid_mask(x, y)
    x = x[m]
    y = y[m]

    if x.size < 2:
        return RegressionResult(model="linear", params={}, r2=float("nan"), n=int(x.size))

    if with_intercept:
        A = np.vstack([x, np.ones_like(x)]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        a = float(coef[0])
        b = float(coef[1])
        yhat = a * x + b
        return RegressionResult(
            model="linear",
            params={"a": a, "b": b},
            r2=_r2(y, yhat),
            n=int(x.size),
        )

    denom = float(np.sum(x**2))
    if denom == 0:
        return RegressionResult(model="linear_no_intercept", params={}, r2=float("nan"), n=int(x.size))

    a = float(np.sum(x * y) / denom)
    yhat = a * x
    return RegressionResult(
        model="linear_no_intercept",
        params={"a": a},
        r2=_r2(y, yhat),
        n=int(x.size),
    )


def fit_powerlaw(x: np.ndarray, y: np.ndarray) -> RegressionResult:
    x = _as_1d_float(x)
    y = _as_1d_float(y)

    m = _valid_mask(x, y)
    m &= x > 0
    m &= y > 0

    x = x[m]
    y = y[m]

    if x.size < 2:
        return RegressionResult(model="powerlaw", params={}, r2=float("nan"), n=int(x.size))

    lx = np.log(x)
    ly = np.log(y)

    res = fit_linear(lx, ly, with_intercept=True)
    if not res.params:
        return RegressionResult(model="powerlaw", params={}, r2=float("nan"), n=int(x.size))

    beta = res.params["a"]
    ln_alpha = res.params["b"]
    alpha = float(np.exp(ln_alpha))

    yhat = alpha * (x**beta)
    return RegressionResult(
        model="powerlaw",
        params={"alpha": float(alpha), "beta": float(beta)},
        r2=_r2(y, yhat),
        n=int(x.size),
    )


def _hickin_nanson(C: np.ndarray, alpha: float, C_star: float) -> np.ndarray:
    C = _as_1d_float(C)
    C_star = float(C_star)
    if C_star == 0:
        return np.full_like(C, np.nan)
    return float(alpha) * (C / C_star) * np.exp(1.0 - C / C_star)


def fit_hickin_nanson(C: np.ndarray, Mn: np.ndarray) -> RegressionResult:
    if not _HAS_SCIPY or curve_fit is None:
        raise RuntimeError("SciPy missing, cannot fit Hickin-Nanson model.")

    C = _as_1d_float(C)
    Mn = _as_1d_float(Mn)

    m = _valid_mask(C, Mn)
    m &= C > 0
    m &= Mn > 0

    C = C[m]
    Mn = Mn[m]

    if C.size < 5:
        return RegressionResult(model="hickin_nanson", params={}, r2=float("nan"), n=int(C.size))

    alpha0 = float(np.nanpercentile(Mn, 95)) if np.isfinite(Mn).any() else 1.0
    Cstar0 = float(np.nanpercentile(C, 75)) if np.isfinite(C).any() else 1.0
    Cstar0 = max(Cstar0, 1e-12)

    popt, _pcov = curve_fit(
        _hickin_nanson,
        C,
        Mn,
        p0=[alpha0, Cstar0],
        maxfev=20000,
    )

    alpha, C_star = float(popt[0]), float(popt[1])
    yhat = _hickin_nanson(C, alpha=alpha, C_star=C_star)
    return RegressionResult(
        model="hickin_nanson",
        params={"alpha": alpha, "C_star": C_star},
        r2=_r2(Mn, yhat),
        n=int(C.size),
    )


def fft_spectrum(x: np.ndarray, step_m: float, detrend: bool = True) -> Dict[str, np.ndarray]:
    x = _as_1d_float(x)
    m = np.isfinite(x)
    x = x[m]
    if x.size < 4:
        return {"freq": np.array([]), "amp": np.array([]), "phase": np.array([])}

    if detrend:
        x = x - float(np.mean(x))

    X = np.fft.rfft(x)
    freq = np.fft.rfftfreq(x.size, d=float(step_m))
    amp = np.abs(X)
    phase = np.angle(X)

    return {"freq": freq, "amp": amp, "phase": phase}


def dominant_wavelength(x: np.ndarray, step_m: float) -> Dict[str, float]:
    spec = fft_spectrum(x, step_m=step_m, detrend=True)
    freq = spec["freq"]
    amp = spec["amp"]
    phase = spec["phase"]

    if freq.size < 3:
        return {"lambda_m": float("nan"), "freq": float("nan"), "amp": float("nan"), "phase": float("nan")}

    idx = int(np.argmax(amp[1:]) + 1)
    f = float(freq[idx])
    if f == 0:
        lam = float("nan")
    else:
        lam = float(1.0 / f)

    return {"lambda_m": lam, "freq": f, "amp": float(amp[idx]), "phase": float(phase[idx])}


def phase_difference_at_frequency(
    x: np.ndarray,
    y: np.ndarray,
    step_m: float,
    freq: float,
) -> float:
    sx = fft_spectrum(x, step_m=step_m, detrend=True)
    sy = fft_spectrum(y, step_m=step_m, detrend=True)

    fx = sx["freq"]
    if fx.size == 0:
        return float("nan")

    idx = int(np.argmin(np.abs(fx - float(freq))))
    phix = float(sx["phase"][idx])
    phiy = float(sy["phase"][idx])

    dphi = phiy - phix
    dphi = (dphi + np.pi) % (2.0 * np.pi) - np.pi
    return float(np.degrees(dphi))


def cross_correlation_lag(
    x: np.ndarray,
    y: np.ndarray,
    step_m: float,
    max_lag_m: float | None = None,
) -> Dict[str, float]:
    x = _as_1d_float(x)
    y = _as_1d_float(y)

    m = _valid_mask(x, y)
    x = x[m]
    y = y[m]

    if x.size < 4:
        return {"lag_m": float("nan"), "corr": float("nan")}

    x = x - float(np.mean(x))
    y = y - float(np.mean(y))

    ccf = np.correlate(x, y, mode="full")
    lags = np.arange(-x.size + 1, x.size) * float(step_m)

    if max_lag_m is not None:
        mm = np.abs(lags) <= float(max_lag_m)
        ccf = ccf[mm]
        lags = lags[mm]

    denom = float(np.sqrt(np.sum(x**2) * np.sum(y**2)))
    if denom > 0:
        ccf_norm = ccf / denom
    else:
        ccf_norm = ccf

    idx = int(np.argmax(ccf_norm))
    return {"lag_m": float(lags[idx]), "corr": float(ccf_norm[idx])}


def _fill_nan_linear(x: np.ndarray) -> np.ndarray:
    x = _as_1d_float(x)
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


def autocorr_length_scales(x: np.ndarray, step_m: float) -> Dict[str, float]:
    x = _as_1d_float(x)
    x = x[np.isfinite(x)]
    if x.size < 4:
        return {"e_folding_m": float("nan"), "integral_scale_m": float("nan"), "n": int(x.size)}

    x = x - float(np.mean(x))
    denom0 = float(np.sum(x**2))
    if denom0 <= 0:
        return {"e_folding_m": float("nan"), "integral_scale_m": float("nan"), "n": int(x.size)}

    acf = np.correlate(x, x, mode="full")[x.size - 1 :]
    acf = acf / float(acf[0])

    thr = float(np.exp(-1.0))
    below = np.where(acf <= thr)[0]
    if below.size == 0:
        e_fold = float("nan")
    else:
        e_fold = float(below[0]) * float(step_m)

    neg = np.where(acf < 0)[0]
    kmax = int(neg[0]) if neg.size > 0 else int(acf.size)
    if kmax <= 1:
        integral = float("nan")
    else:
        integral = float(np.sum(acf[1:kmax])) * float(step_m)

    return {"e_folding_m": float(e_fold), "integral_scale_m": float(integral), "n": int(x.size)}


def iter_links_from_flat_npz(data: Dict[str, np.ndarray]) -> Iterator[Tuple[str, np.ndarray]]:
    link_ids = data.get("link_ids")
    link_index = data.get("link_index")
    if link_ids is None or link_index is None:
        raise ValueError("Missing link_ids or link_index in NPZ file")

    link_ids = np.asarray(link_ids)
    link_index = np.asarray(link_index, dtype=int)

    for idx, lid in enumerate(link_ids.tolist()):
        mask = link_index == idx
        yield str(lid), mask


def analyze_C_B_Mn_relationships(
    sBCMn_npz_path: str | Path,
    use_abs_mn: bool = True,
    use_abs_c_for_global_fit: bool = True,
    per_link: bool = True,
    min_samples_per_link: int = 64,
) -> Dict[str, Any]:
    p = Path(sBCMn_npz_path)
    data = load_link_sBCMn_npz(p)

    C = _as_1d_float(data["C"])
    B = _as_1d_float(data["B"])
    Mn = _as_1d_float(data["Mn"])
    step_m = float(data.get("step_m", np.nan))

    C_for_global = np.abs(C) if use_abs_c_for_global_fit else C

    if use_abs_mn:
        Mn_for_fit = np.abs(Mn)
    else:
        Mn_for_fit = Mn

    out: Dict[str, Any] = {
        "path": str(p),
        "site": str(data.get("site", "")),
        "mask_level": int(data.get("mask_level", -1)),
        "step_m": step_m,
        "diagnostics": {},
        "global": {},
        "per_link": {},
    }

    m = _valid_mask(C_for_global, Mn_for_fit)
    out["global"]["Mn_C_linear"] = fit_linear(
        C_for_global[m],
        Mn_for_fit[m],
        with_intercept=True,
    )

    mp = m & (C_for_global > 0) & (Mn_for_fit > 0)
    out["global"]["Mn_C_powerlaw"] = fit_powerlaw(C_for_global[mp], Mn_for_fit[mp])

    if _HAS_SCIPY:
        try:
            out["global"]["Mn_C_hickin_nanson"] = fit_hickin_nanson(C_for_global, Mn_for_fit)
        except Exception:
            out["global"]["Mn_C_hickin_nanson"] = RegressionResult(
                model="hickin_nanson",
                params={},
                r2=float("nan"),
                n=int(np.sum(mp)),
            )

    # Diagnostics: help explain empty fits / near-zero slopes
    def _q(x: np.ndarray, qs: Iterable[float]) -> Dict[str, float]:
        x = _as_1d_float(x)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return {f"p{int(qq)}": float("nan") for qq in qs}
        outq: Dict[str, float] = {}
        for qq in qs:
            outq[f"p{int(qq)}"] = float(np.nanpercentile(x, qq))
        return outq

    out["diagnostics"] = {
        "n_total": int(C.size),
        "n_valid_global": int(np.sum(m)),
        "n_valid_powerlaw": int(np.sum(mp)),
        "use_abs_mn": bool(use_abs_mn),
        "use_abs_c_for_global_fit": bool(use_abs_c_for_global_fit),
        "C_global_percentiles": _q(C_for_global[m], [0, 5, 25, 50, 75, 95, 100]),
        "Mn_percentiles": _q(Mn_for_fit[m], [0, 5, 25, 50, 75, 95, 100]),
        "n_C_global_gt_0": int(np.sum((C_for_global > 0) & m)),
        "n_Mn_gt_0": int(np.sum((Mn_for_fit > 0) & m)),
    }

    if not per_link:
        return out

    for link_id, lm in iter_links_from_flat_npz(data):
        if int(np.sum(lm)) < int(min_samples_per_link):
            continue

        C_i = C[lm]
        B_i = B[lm]
        Mn_i = Mn[lm]

        domC = dominant_wavelength(C_i, step_m=step_m)
        domB = dominant_wavelength(B_i, step_m=step_m)

        if np.isfinite(domC["freq"]):
            dphi_CB = phase_difference_at_frequency(C_i, B_i, step_m=step_m, freq=domC["freq"])
        else:
            dphi_CB = float("nan")

        lag_CM = cross_correlation_lag(C_i, np.abs(Mn_i) if use_abs_mn else Mn_i, step_m=step_m)

        out["per_link"][link_id] = {
            "dominant_C": domC,
            "dominant_B": domB,
            "delta_phi_B_minus_C_deg_at_Cdom": float(dphi_CB),
            "lag_C_vs_Mn_m": float(lag_CM["lag_m"]),
            "corr_C_vs_Mn": float(lag_CM["corr"]),
        }

    return out


def compute_link_level_metrics_from_flat_npz(
    data: Dict[str, np.ndarray],
    use_abs_mn: bool = True,
) -> Dict[str, np.ndarray]:
    """Aggregates flat sample .npz into link-level metrics.

    Notes
    -----
    - The goal is to support statistical modeling at the link level rather than spectral analysis.
    - Output arrays all have length n_links.
    """

    link_ids = np.asarray(data["link_ids"], dtype=object)
    link_index = np.asarray(data["link_index"], dtype=int)

    s = _as_1d_float(data["s"])
    B = _as_1d_float(data["B"])
    C = _as_1d_float(data["C"])
    Mn = _as_1d_float(data["Mn"])

    n_links = int(link_ids.size)

    arc_length = np.full(n_links, np.nan, dtype=float)
    n_samples = np.zeros(n_links, dtype=int)

    mean_B = np.full(n_links, np.nan, dtype=float)
    median_B = np.full(n_links, np.nan, dtype=float)

    mean_abs_C = np.full(n_links, np.nan, dtype=float)
    median_abs_C = np.full(n_links, np.nan, dtype=float)

    mean_Mn = np.full(n_links, np.nan, dtype=float)
    mean_abs_Mn = np.full(n_links, np.nan, dtype=float)
    pos_Mn_frac = np.full(n_links, np.nan, dtype=float)

    valid_B_frac = np.full(n_links, np.nan, dtype=float)
    valid_C_frac = np.full(n_links, np.nan, dtype=float)
    valid_Mn_frac = np.full(n_links, np.nan, dtype=float)

    for i in range(n_links):
        m = link_index == i
        n = int(np.sum(m))
        n_samples[i] = n
        if n == 0:
            continue

        s_i = s[m]
        B_i = B[m]
        C_i = C[m]
        Mn_i = Mn[m]

        # Arc length (same units as input coordinates)
        if np.isfinite(s_i).any():
            arc_length[i] = float(np.nanmax(s_i) - np.nanmin(s_i))

        vb = np.isfinite(B_i)
        vc = np.isfinite(C_i)
        vm = np.isfinite(Mn_i)
        valid_B_frac[i] = float(np.sum(vb)) / n
        valid_C_frac[i] = float(np.sum(vc)) / n
        valid_Mn_frac[i] = float(np.sum(vm)) / n

        if np.any(vb):
            mean_B[i] = float(np.nanmean(B_i[vb]))
            median_B[i] = float(np.nanmedian(B_i[vb]))

        if np.any(vc):
            absC = np.abs(C_i[vc])
            mean_abs_C[i] = float(np.nanmean(absC))
            median_abs_C[i] = float(np.nanmedian(absC))

        if np.any(vm):
            mn_valid = Mn_i[vm]
            mean_Mn[i] = float(np.nanmean(mn_valid))
            if use_abs_mn:
                mean_abs_Mn[i] = float(np.nanmean(np.abs(mn_valid)))
            else:
                mean_abs_Mn[i] = float(np.nanmean(mn_valid))
            pos_Mn_frac[i] = float(np.sum(mn_valid > 0)) / float(mn_valid.size)

    return {
        "link_id": link_ids.astype(str),
        "n_samples": n_samples,
        "arc_length": arc_length,
        "mean_B": mean_B,
        "median_B": median_B,
        "mean_abs_C": mean_abs_C,
        "median_abs_C": median_abs_C,
        "mean_Mn": mean_Mn,
        "mean_abs_Mn": mean_abs_Mn,
        "pos_Mn_frac": pos_Mn_frac,
        "valid_B_frac": valid_B_frac,
        "valid_C_frac": valid_C_frac,
        "valid_Mn_frac": valid_Mn_frac,
    }


def analyze_link_level_relationships(
    sBCMn_npz_path: str | Path,
    use_abs_mn: bool = True,
    min_samples: int = 2,
    min_arc_length: float | None = None,
) -> LinkLevelResult:
    """Performs link-level regressions based on aggregated metrics for each link 
    (avoids the higher complexity of per-link sequence spectral analysis)."""

    p = Path(sBCMn_npz_path)
    data = load_link_sBCMn_npz(p)

    metrics = compute_link_level_metrics_from_flat_npz(data, use_abs_mn=use_abs_mn)

    # Filter criteria
    m = np.isfinite(metrics["mean_abs_C"]) & np.isfinite(metrics["mean_abs_Mn"])
    ms = metrics["n_samples"] >= int(min_samples)
    m &= ms
    if min_arc_length is not None:
        ml = np.isfinite(metrics["arc_length"]) & (metrics["arc_length"] >= float(min_arc_length))
        m &= ml
    else:
        ml = np.isfinite(metrics["arc_length"])

    xC = np.asarray(metrics["mean_abs_C"], dtype=float)
    yM = np.asarray(metrics["mean_abs_Mn"], dtype=float)

    fits: Dict[str, RegressionResult] = {}
    fits["Mn_C_linear_link"] = fit_linear(xC[m], yM[m], with_intercept=True)

    mp = m & (xC > 0) & (yM > 0)
    fits["Mn_C_powerlaw_link"] = fit_powerlaw(xC[mp], yM[mp])

    # B–Mn
    xB = np.asarray(metrics["mean_B"], dtype=float)
    mb = m & np.isfinite(xB)
    fits["Mn_B_linear_link"] = fit_linear(xB[mb], yM[mb], with_intercept=True)

    diagnostics: Dict[str, Any] = {
        "path": str(p),
        "site": str(data.get("site", "")),
        "mask_level": int(data.get("mask_level", -1)),
        "step_m": float(data.get("step_m", np.nan)),
        "use_abs_mn": bool(use_abs_mn),
        "n_links_total": int(metrics["link_id"].size),
        "n_links_used_C": int(np.sum(m)),
        "n_links_used_C_powerlaw": int(np.sum(mp)),
        "n_links_used_B": int(np.sum(mb)),
        "min_samples": int(min_samples),
        "min_arc_length": float(min_arc_length) if min_arc_length is not None else None,
        "n_links_ge_min_samples": int(np.sum(ms)),
        "n_links_ge_min_arc_length": int(np.sum(ml)) if min_arc_length is not None else int(np.sum(np.isfinite(metrics["arc_length"]))),
        "mean_abs_C_percentiles": {
            "p0": float(np.nanpercentile(xC, 0)) if np.isfinite(xC).any() else float("nan"),
            "p50": float(np.nanpercentile(xC, 50)) if np.isfinite(xC).any() else float("nan"),
            "p95": float(np.nanpercentile(xC, 95)) if np.isfinite(xC).any() else float("nan"),
            "p100": float(np.nanpercentile(xC, 100)) if np.isfinite(xC).any() else float("nan"),
        },
        "n_samples_percentiles": {
            "p0": float(np.nanpercentile(metrics["n_samples"], 0)) if metrics["n_samples"].size > 0 else float("nan"),
            "p50": float(np.nanpercentile(metrics["n_samples"], 50)) if metrics["n_samples"].size > 0 else float("nan"),
            "p95": float(np.nanpercentile(metrics["n_samples"], 95)) if metrics["n_samples"].size > 0 else float("nan"),
            "p100": float(np.nanpercentile(metrics["n_samples"], 100)) if metrics["n_samples"].size > 0 else float("nan"),
        },
        "arc_length_percentiles": {
            "p0": float(np.nanpercentile(metrics["arc_length"], 0)) if np.isfinite(metrics["arc_length"]).any() else float("nan"),
            "p50": float(np.nanpercentile(metrics["arc_length"], 50)) if np.isfinite(metrics["arc_length"]).any() else float("nan"),
            "p95": float(np.nanpercentile(metrics["arc_length"], 95)) if np.isfinite(metrics["arc_length"]).any() else float("nan"),
            "p100": float(np.nanpercentile(metrics["arc_length"], 100)) if np.isfinite(metrics["arc_length"]).any() else float("nan"),
        },
    }

    return LinkLevelResult(metrics=metrics, fits=fits, diagnostics=diagnostics)
