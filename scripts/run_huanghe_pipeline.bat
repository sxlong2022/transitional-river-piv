@echo off
REM ============================================================================
REM 黄河站点 PIV 完整管线批处理脚本
REM ============================================================================
REM 用法：
REM   run_huanghe_pipeline.bat HuangHe-A 2
REM   run_huanghe_pipeline.bat HuangHe-B 1
REM
REM 参数：
REM   %1 = 站点名称 (HuangHe-A 或 HuangHe-B)
REM   %2 = mask 等级 (1-4)
REM ============================================================================

setlocal enabledelayedexpansion

set SITE=%1
set MASK=%2

if "%SITE%"=="" (
    echo 用法: run_huanghe_pipeline.bat ^<站点名称^> ^<mask等级^>
    echo 示例: run_huanghe_pipeline.bat HuangHe-A 2
    exit /b 1
)

if "%MASK%"=="" (
    set MASK=2
    echo 未指定 mask 等级，使用默认值 2
)

echo ============================================================================
echo 站点: %SITE%
echo Mask 等级: %MASK%
echo ============================================================================

REM Step 1-3: 多倾角时间序列 PIV + 融合
echo.
echo [Step 1-3] 多倾角时间序列 PIV + 融合...
python -m src.piv_analysis.jurua_multitilt --site %SITE% --mask-level %MASK%
if errorlevel 1 (
    echo [错误] Step 1-3 失败！
    exit /b 1
)

REM Step 4A: 严格仿射 georef（映射到地理坐标 + m/yr）
echo.
echo [Step 4A] 地理坐标映射与物理量纲转换...
python -m src.morphodynamics.jurua_georef_multitilt --site %SITE% --mask-level %MASK%
if errorlevel 1 (
    echo [错误] Step 4A 失败！
    exit /b 1
)

REM RivGraph links 生成
echo.
echo [RivGraph] 生成河道网络 links...
python -m src.analysis.generate_rivgraph_links --site %SITE% --mask-level %MASK% --exit-sides WE
if errorlevel 1 (
    echo [错误] RivGraph links 生成失败！
    exit /b 1
)

REM s-B-C-Mn 计算
echo.
echo [s-B-C-Mn] 计算沿河剖面...
set LINKS_PATH=results\RivGraph\%SITE%\mask%MASK%\%SITE%_mask%MASK%_links.shp
set PIV_NPZ=results\PostprocessedPIV\%SITE%\jurua_mask%MASK%_multitilt_georef_step4a_strict.npz
set STEP_M=20
set OUT_NPZ=results\PostprocessedPIV\%SITE%\%SITE%_mask%MASK%_link_sBCMn_flat_step%STEP_M%_metric_v2.npz

python -m src.morphodynamics.link_sBCMn_pipeline ^
    --site %SITE% ^
    --mask-level %MASK% ^
    --links-vector %LINKS_PATH% ^
    --piv-npz %PIV_NPZ% ^
    --step-m %STEP_M% ^
    --export-npz %OUT_NPZ%
if errorlevel 1 (
    echo [错误] s-B-C-Mn 计算失败！
    exit /b 1
)

echo.
echo ============================================================================
echo 完成！%SITE% mask%MASK% 管线运行成功
echo ============================================================================
echo 输出文件:
echo   - PIV 矢量场: %PIV_NPZ%
echo   - RivGraph links: %LINKS_PATH%
echo   - s-B-C-Mn 剖面: %OUT_NPZ%
echo ============================================================================

endlocal
