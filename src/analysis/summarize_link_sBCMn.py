"""为每条 RivGraph link 计算汇总指标，并输出 CSV 与可选的带属性矢量文件。

本模块读取 `link_sBCMn_io.export_link_sBCMn_to_npz` 导出的扁平 .npz，
对每条 link 聚合计算以下指标：

- arc_length : 弧长范围（m）
- n_samples : 采样点数量
- valid_B_frac : B 有效（非 NaN）比例
- valid_Mn_frac : Mn 有效（非 NaN）比例
- mean_B, median_B, std_B : 宽度统计
- mean_Mn, median_Mn, std_Mn : 法向迁移率统计
- mean_abs_Mn, max_abs_Mn : 迁移率绝对值统计
- pos_Mn_frac : Mn > 0 的比例（正向迁移占比）
- mean_abs_C : 曲率绝对值均值

典型用法（在项目根目录下）：

    python -m src.analysis.summarize_link_sBCMn \
        --input-npz results/PostprocessedPIV/Jurua-A/Jurua-A_mask1_link_sBCMn_flat.npz \
        --output-csv results/PostprocessedPIV/Jurua-A/Jurua-A_mask1_link_summary.csv

若要同时输出带属性的矢量文件（需提供原 links 矢量路径）：

    python -m src.analysis.summarize_link_sBCMn \
        --input-npz ... \
        --output-csv ... \
        --links-vector results/RivGraph/Jurua-A/mask1/Jurua-A_mask1_links.shp \
        --output-gpkg results/PostprocessedPIV/Jurua-A/Jurua-A_mask1_links_with_stats.gpkg
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd


def summarize_link_sBCMn(npz_path: str | Path) -> pd.DataFrame:
    """读取扁平 .npz 并为每条 link 计算汇总指标。

    返回
    ------
    df : pd.DataFrame
        每行对应一条 link，列为各统计指标。
    """

    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    data = np.load(npz_path, allow_pickle=True)

    link_ids = data["link_ids"]
    link_index = data["link_index"]
    s = data["s"]
    B = data["B"]
    C = data["C"]
    Mn = data["Mn"]

    records: list[Dict[str, Any]] = []

    for idx, lid in enumerate(link_ids):
        mask = link_index == idx
        s_link = s[mask]
        B_link = B[mask]
        C_link = C[mask]
        Mn_link = Mn[mask]

        n_samples = int(mask.sum())
        arc_length = float(s_link.max() - s_link.min()) if n_samples > 0 else 0.0

        valid_B = ~np.isnan(B_link)
        valid_Mn = ~np.isnan(Mn_link)

        valid_B_frac = float(valid_B.sum()) / n_samples if n_samples > 0 else 0.0
        valid_Mn_frac = float(valid_Mn.sum()) / n_samples if n_samples > 0 else 0.0

        B_valid = B_link[valid_B]
        Mn_valid = Mn_link[valid_Mn]
        C_valid = C_link[~np.isnan(C_link)]

        mean_B = float(np.nanmean(B_valid)) if B_valid.size > 0 else np.nan
        median_B = float(np.nanmedian(B_valid)) if B_valid.size > 0 else np.nan
        std_B = float(np.nanstd(B_valid)) if B_valid.size > 0 else np.nan

        mean_Mn = float(np.nanmean(Mn_valid)) if Mn_valid.size > 0 else np.nan
        median_Mn = float(np.nanmedian(Mn_valid)) if Mn_valid.size > 0 else np.nan
        std_Mn = float(np.nanstd(Mn_valid)) if Mn_valid.size > 0 else np.nan
        mean_abs_Mn = float(np.nanmean(np.abs(Mn_valid))) if Mn_valid.size > 0 else np.nan
        max_abs_Mn = float(np.nanmax(np.abs(Mn_valid))) if Mn_valid.size > 0 else np.nan

        pos_Mn_frac = float((Mn_valid > 0).sum()) / Mn_valid.size if Mn_valid.size > 0 else np.nan

        mean_abs_C = float(np.nanmean(np.abs(C_valid))) if C_valid.size > 0 else np.nan

        records.append({
            "link_id": str(lid),
            "arc_length": arc_length,
            "n_samples": n_samples,
            "valid_B_frac": valid_B_frac,
            "valid_Mn_frac": valid_Mn_frac,
            "mean_B": mean_B,
            "median_B": median_B,
            "std_B": std_B,
            "mean_Mn": mean_Mn,
            "median_Mn": median_Mn,
            "std_Mn": std_Mn,
            "mean_abs_Mn": mean_abs_Mn,
            "max_abs_Mn": max_abs_Mn,
            "pos_Mn_frac": pos_Mn_frac,
            "mean_abs_C": mean_abs_C,
        })

    return pd.DataFrame(records)


def join_stats_to_links(
    df_stats: pd.DataFrame,
    links_vector_path: str | Path,
    output_gpkg_path: str | Path,
) -> None:
    """将统计表 join 到原 links 矢量文件，并输出为 GeoPackage。

    需要 geopandas。
    """

    import geopandas as gpd

    links_vector_path = Path(links_vector_path)
    output_gpkg_path = Path(output_gpkg_path)

    gdf = gpd.read_file(links_vector_path)

    # 尝试识别 link_id 字段
    id_col = None
    for col in ["id", "link_id", "ID", "LINK_ID"]:
        if col in gdf.columns:
            id_col = col
            break

    if id_col is None:
        # 使用行索引作为 link_id
        gdf["link_id"] = gdf.index.astype(str)
        id_col = "link_id"
    else:
        gdf["link_id"] = gdf[id_col].astype(str)

    # 合并
    gdf_merged = gdf.merge(df_stats, on="link_id", how="left")

    output_gpkg_path.parent.mkdir(parents=True, exist_ok=True)
    gdf_merged.to_file(output_gpkg_path, driver="GPKG")
    print(f"已输出带属性的矢量文件: {output_gpkg_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="为每条 RivGraph link 计算汇总指标，并输出 CSV 与可选的带属性矢量文件。",
    )
    parser.add_argument(
        "--input-npz",
        required=True,
        help="扁平 link_sBCMn .npz 路径",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="输出 CSV 路径",
    )
    parser.add_argument(
        "--links-vector",
        default=None,
        help="原 RivGraph links 矢量文件路径（可选，用于生成带属性的矢量文件）",
    )
    parser.add_argument(
        "--output-gpkg",
        default=None,
        help="输出带属性的 GeoPackage 路径（可选，需同时提供 --links-vector）",
    )

    args = parser.parse_args()

    df = summarize_link_sBCMn(args.input_npz)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"已输出 link 汇总统计 CSV: {output_csv}")

    if args.links_vector and args.output_gpkg:
        join_stats_to_links(df, args.links_vector, args.output_gpkg)
    elif args.output_gpkg and not args.links_vector:
        print("警告：指定了 --output-gpkg 但未提供 --links-vector，跳过矢量输出。")


if __name__ == "__main__":
    main()
