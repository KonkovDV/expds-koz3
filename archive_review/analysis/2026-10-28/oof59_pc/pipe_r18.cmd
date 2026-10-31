@echo off
set OMP_NUM_THREADS=6
set MKL_NUM_THREADS=6
set NUMEXPR_NUM_THREADS=6
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set R=C:\Sloi\expds-koz3_v0.2.0\expds-koz3
set O=C:\Sloi\oof59
set PY=%R%\.venv\Scripts\python.exe
set EXPDS_METRIC=%R%\materials\compute_metrics.py
cd /d %R%
echo PIPE START %DATE% %TIME% > %O%\pipe.log
:wait
findstr /b /c:"EXIT " %O%\r18\run.log >nul 2>&1
if errorlevel 1 ( timeout /t 30 /nobreak >nul & goto wait )
findstr /b /c:"EXIT 0 " %O%\r18\run.log >nul 2>&1
if errorlevel 1 ( echo OOF FAILED, stop >> %O%\pipe.log & goto end )
if not exist %O%\r18\li_r18_hard2_59_oof.geojson ( echo NO OOF OUTPUT >> %O%\pipe.log & goto end )
echo FUSE %TIME% >> %O%\pipe.log
"%PY%" -u %O%\fuse_r18.py %O%\r18\li_r18_hard2_59_oof.geojson %O%\r18\li_r18_hard2_59_fused.geojson >> %O%\pipe.log 2>&1
if errorlevel 1 ( echo FUSE FAILED >> %O%\pipe.log & goto end )
echo DIAG %TIME% >> %O%\pipe.log
"%PY%" -u %O%\diag_r18.py F:\expds3\work\fusion\li_hard2_fused.geojson %O%\r18\li_r18_hard2_59_fused.geojson %O%\diag_r18.json >> %O%\pipe.log 2>&1
echo SCOREBOARD %TIME% >> %O%\pipe.log
"%PY%" -u %O%\sb_wrap.py %O%\sb_r18_extra.json --work F:\expds3\work --train F:\2\train-expds-3 --honest 10 --base-add F:\expds3\work\ae_fort_mask_oof.geojson:ae_fort:fortifikatsii:ae --base-add F:\expds3\work\li_fort_mask_oof.geojson:li_fort:fortifikatsii:li --base-add F:\expds3\work\spor_focal_wide_opt.geojson:sp_focal2 --base-add F:\expds3\work\gor_phys.geojson:gor_phys --base-add F:\expds3\work\surf_arch_sp.geojson:sp_arch:arkhitektury:sp_or --base-add F:\expds3\work\surf_arch_or.geojson:or:arkhitektury:or --fused %O%\r18\li_r18_hard2_59_fused.geojson --out %O%\sb_r18.json > %O%\sb_r18.log 2>&1
echo SB EXIT %ERRORLEVEL% %TIME% >> %O%\pipe.log
:end
echo PIPE END %DATE% %TIME% >> %O%\pipe.log
