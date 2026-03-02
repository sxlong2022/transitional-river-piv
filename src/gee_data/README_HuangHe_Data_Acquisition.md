# 黄河目标河段遥感数据获取方案

本文档说明如何为黄河两个局部河段获取与 Jurua-A 相同结构的年度水体掩膜序列，以便复用现有 PIV–RivGraph 管线。

---

## 1. 选站与时段设计

### 1.1 选站策略

| 站点代号 | 位置描述 | 特征 |
|---------|---------|------|
| **HuangHe-A** | 待定（如宁蒙河段某弯曲段） | 相对自然或弱工程化，弯曲度较高 |
| **HuangHe-B** | 待定（如下游游荡–弯曲过渡段） | 强工程化、堤防约束、调水调沙影响显著 |

> **用户任务**：准备两个河段的 `.shp` 或 `.gpkg` 边界多边形文件，以站点名命名（如 `HuangHe-A.shp` 或 `HuangHe-A.gpkg`），并放入项目根目录下的 `data/GIS/`。

### 1.2 时段设计

- **时间范围**：1985–2023（与 Landsat 5/7/8/9 时间覆盖对齐）
- **年度合成窗口**：
  - 默认：01-01 至 12-31（全年合成）
  - 备选：06-01 至 10-31（汛期合成，减少冰雪干扰）
- **掩膜等级**：Jones DSWE water_level 1–4（与 Jurua 一致）

---

## 2. 数据源与工具

### 2.1 数据源

- **Landsat Collection 2 Level-2**（Surface Reflectance）
  - Landsat 5 TM：1984–2012
  - Landsat 7 ETM+：1999–至今（SLC-off 后有条带）
  - Landsat 8/9 OLI：2013–至今
- **DSWE 方法**：Jones et al. (2019) 的 Dynamic Surface Water Extent 算法
  - water_level 1：High confidence water
  - water_level 2：Moderate confidence water
  - water_level 3：Potential wetland
  - water_level 4：Low confidence water / partial surface water

### 2.2 工具选择

我们有两种方式获取数据：

#### 方式 A：使用 GEE_watermasks Python 包（推荐）

基于 `evan-greenbrg/GEE_watermasks` 包，已在 `文献/.../附属GEE_watermasks_v1.0.0/` 目录下。

**优点**：
- 与 Chadwick (2023) 原始工作流一致
- 支持 Jones DSWE 方法和多 water_level
- 自动处理河道网络过滤（GRWL / largest）

**使用步骤**：
1. 安装依赖：`pip install earthengine-api geemap rasterio geopandas scikit-image`
2. 认证 GEE：`earthengine authenticate`
3. 运行脚本（见下文）

#### 方式 B：直接在 GEE Code Editor 中操作

如果 Python-GEE 环境有问题，可以在 [GEE Code Editor](https://code.earthengine.google.com/) 中：
1. 上传边界多边形
2. 使用 DSWE 算法脚本
3. 导出到 Google Drive
4. 下载到本地

---

## 3. 目录结构设计

为黄河站点在**项目根目录**下建立与 Jurua 相似、但相互独立的目录结构，例如：

```
data/
├── GIS/
│   ├── HuangHe-A.shp / .gpkg   # 用户提供
│   └── HuangHe-B.shp / .gpkg   # 用户提供
├── GEOTIFFS/
│   ├── HuangHe-A/
│   │   ├── image/
│   │   ├── mask1/
│   │   ├── mask2/
│   │   ├── mask3/
│   │   └── mask4/
│   └── HuangHe-B/
│       ├── image/
│       ├── mask1/
│       ├── mask2/
│       ├── mask3/
│       └── mask4/
└── PreparedImagery/
    ├── HuangHe-A/
    │   ├── Color/
    │   └── MaskX_TiltYY/
    └── HuangHe-B/
        ├── Color/
        └── MaskX_TiltYY/
```

> 说明：Jurua-A 仍使用 Dryad 原始数据包下的 `文献/.../Data_and_code/Data` 结构；黄河站点使用项目根目录下独立的 `data/` 结构，避免混用。

---

## 4. 数据获取脚本

### 4.1 Python-GEE 脚本（基于 GEE_watermasks）

在项目根目录下创建 `src/gee_data/pull_huanghe_masks.py`：

```python
"""
黄河目标河段年度水体掩膜获取脚本

使用方法：
    python -m src.gee_data.pull_huanghe_masks --site HuangHe-A --start-year 1985 --end-year 2023

依赖：
    - earthengine-api
    - geemap
    - GEE_watermasks 包（需添加到 PYTHONPATH）
"""

import sys
from pathlib import Path

# 添加 GEE_watermasks 到路径
GEE_WATERMASKS_PATH = Path(__file__).parent.parent.parent / "文献" / \
    "Remote Sensing of Riverbank Migration Using Particle Image Velocimetry" / \
    "Data_and_code" / "Codes" / "附属GEE_watermasks_v1.0.0" / \
    "evan-greenbrg-GEE_watermasks-eb93edc" / "GEE_watermasks"
sys.path.insert(0, str(GEE_WATERMASKS_PATH))

import argparse
import ee
from main import main as gee_main

# 初始化 GEE
ee.Initialize()

def pull_huanghe_masks(
    site: str,
    start_year: int = 1985,
    end_year: int = 2023,
    water_levels: list = [1, 2, 3, 4],
):
    """为黄河站点获取多 water_level 的年度掩膜。"""
    
    # 路径配置
    data_root = Path(__file__).parent.parent.parent / "文献" / \
        "Remote Sensing of Riverbank Migration Using Particle Image Velocimetry" / \
        "Data_and_code" / "Data"
    
    poly_path = data_root / "GIS" / f"{site}.gpkg"
    out_root = data_root / "GEOTIFFS" / site
    
    if not poly_path.exists():
        raise FileNotFoundError(f"边界多边形文件不存在: {poly_path}")
    
    for wl in water_levels:
        print(f"\n{'='*60}")
        print(f"正在获取 {site} water_level={wl} 的掩膜...")
        print(f"{'='*60}")
        
        gee_main(
            poly=str(poly_path),
            masks="true",
            images="true",
            dataset="landsat",
            water_level=str(wl),
            mask_method="Jones",
            network_method="largest",  # 或 "grwl"
            network_path=None,
            start="01-01",
            end="12-31",
            start_year=str(start_year),
            end_year=str(end_year),
            out=str(out_root / f"mask{wl}"),
            river=site,
        )
    
    print(f"\n完成！掩膜已保存到: {out_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="黄河目标河段年度水体掩膜获取")
    parser.add_argument("--site", required=True, help="站点名称，如 HuangHe-A")
    parser.add_argument("--start-year", type=int, default=1985)
    parser.add_argument("--end-year", type=int, default=2023)
    args = parser.parse_args()
    
    pull_huanghe_masks(args.site, args.start_year, args.end_year)
```

### 4.2 GEE Code Editor 脚本（备选）

如果 Python-GEE 环境有问题，可以在 GEE Code Editor 中使用以下脚本：

```javascript
// 黄河年度 DSWE 掩膜导出脚本
// 使用方法：
// 1. 上传边界多边形到 Assets
// 2. 修改下方参数
// 3. 运行并导出到 Google Drive

// ========== 参数配置 ==========
var site = 'HuangHe-A';
var roi = ee.FeatureCollection('users/YOUR_USERNAME/' + site);
var startYear = 1985;
var endYear = 2023;
var waterLevel = 2;  // 1-4，分别运行

// ========== DSWE 函数 ==========
function calcDSWE(image) {
  var mndwi = image.normalizedDifference(['SR_B3', 'SR_B5']).rename('mndwi');
  var ndvi = image.normalizedDifference(['SR_B4', 'SR_B3']).rename('ndvi');
  var swir1 = image.select('SR_B5');
  var nir = image.select('SR_B4');
  var blue = image.select('SR_B1');
  var swir2 = image.select('SR_B7');
  
  // Jones et al. (2019) DSWE tests
  var t1 = mndwi.gt(0.124);
  var t2 = image.select('SR_B3').add(image.select('SR_B4'))
           .gt(image.select('SR_B4').add(image.select('SR_B5')));
  // ... (完整 DSWE 逻辑)
  
  return image.addBands(mndwi);
}

// ========== 年度循环 ==========
for (var year = startYear; year <= endYear; year++) {
  var startDate = year + '-01-01';
  var endDate = year + '-12-31';
  
  var collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterBounds(roi)
    .filterDate(startDate, endDate)
    .map(calcDSWE);
  
  var composite = collection.median().clip(roi);
  
  Export.image.toDrive({
    image: composite.select('mndwi').gt(0),
    description: site + '_' + year + '_mask' + waterLevel,
    folder: site + '_masks',
    region: roi.geometry(),
    scale: 30,
    maxPixels: 1e13
  });
}
```

---

## 5. 后处理与 PIV 准备

获取掩膜后，需要进行以下后处理：

### 5.1 文件重命名

确保文件名格式与 Jurua 一致：
```
{site}_{year}_01-01_12-31_mask.tif
```

### 5.2 生成 PreparedImagery

使用现有的 `prepare_imagery.py`（如果有）或手动：
1. 将掩膜复制到 `PreparedImagery/{site}/MaskX_TiltYY/`
2. 应用倾角变换（如需要）
3. 生成阈值化掩膜 `*_mask_thresh.tif`

### 5.3 运行 PIV 管线

```bash
# Step 1-3: 时间序列 PIV + 多倾角融合
python -m src.piv_analysis.huanghe_timeseries --site HuangHe-A --mask-level 1

# Step 4A: 严格仿射 georef
python -m src.piv_analysis.huanghe_georef_multitilt --site HuangHe-A --mask-level 1

# RivGraph links 生成
python -m src.analysis.generate_rivgraph_links --site HuangHe-A --mask-level 1

# s–B–C–Mn 计算
python -m src.morphodynamics.link_sBCMn_pipeline \
    --site HuangHe-A \
    --mask-level 1 \
    --links-vector results/RivGraph/HuangHe-A/mask1/HuangHe-A_mask1_links.shp \
    --piv-npz results/PostprocessedPIV/HuangHe-A/huanghe-a_mask1_multitilt_georef_step4a_strict.npz \
    --export-npz results/PostprocessedPIV/HuangHe-A/HuangHe-A_mask1_link_sBCMn_flat.npz
```

---

## 6. 检查清单

- [ ] 准备黄河两个河段的边界多边形 `.gpkg` 文件
- [ ] 确认 GEE 账户已认证
- [ ] 运行 `pull_huanghe_masks.py` 获取 water_level 1-4 的掩膜
- [ ] 检查下载的掩膜质量（云覆盖、冰雪、SLC-off 条带）
- [ ] 重命名文件并组织目录结构
- [ ] 生成 PreparedImagery
- [ ] 运行 PIV 管线
- [ ] 生成 s–B–C–Mn 剖面
- [ ] 与 Jurua-A 结果对比

---

## 7. 常见问题

### Q1: GEE 认证失败
```bash
earthengine authenticate
```
按提示在浏览器中完成认证。

### Q2: 掩膜质量差（云多、冰雪）
- 尝试使用汛期窗口（06-01 至 10-31）
- 使用更严格的云掩膜
- 考虑使用 Sentinel-2（2017 年后）

### Q3: SLC-off 条带（Landsat 7）
- 优先使用 Landsat 5（1984-2012）和 Landsat 8/9（2013-至今）
- 或使用 gap-filling 算法

### Q4: 河道网络过滤失败
- 尝试 `network_method="largest"` 而非 `"grwl"`
- 检查边界多边形是否完整覆盖目标河段
