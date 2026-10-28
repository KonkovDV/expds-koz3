$t=@'
"""Копии кругов лидара в аэрофото тем же правилом, что F:\\expds3\\work\\fuse_li59.py (27.10).
Отличие только в путях: вход и выход — аргументы, пишем под C:\\Sloi\\oof59."""
import json, sys
from collections import defaultdict
from pathlib import Path

REPO = Path(r"C:\Sloi\expds-koz3_v0.2.0\expds-koz3")
sys.path.insert(0, str(REPO))
from shapely.geometry import shape
from expds import writer
from expds.catalog import read_utm_json, scan_root
from expds.fusion import FusionParams, li_to_ae
from expds.pipeline import _params, load_config

ROOT = Path(r"F:\2\train-expds-3")
CFG = REPO / "configs" / "default.toml"


def main(src: Path, dst: Path) -> None:
    fp = _params(FusionParams, load_config(CFG).get("fusion"))
    raw = json.loads(src.read_text(encoding="utf-8"))
    by = defaultdict(list)
    for feature in raw["features"]:
        props = feature.get("properties") or {}
        by[str(props.get("region_name"))].append({
            "geom": shape(feature["geometry"]),
            "class_name": props.get("class_name"),
            "confidence": float(props.get("confidence") or 0),
            "region_name": props.get("region_name"),
            "markup_type": props.get("markup_type") or "li",
            "original_crs": props.get("original_crs"),
            "source": props.get("source"),
        })
    del raw
    out, n_copy, n_ae = [], 0, 0
    for reg in scan_root(ROOT):
        feats = by.pop(reg.name, [])
        if not feats:
            continue
        out.extend(feats)
        ae = reg.rasters.get("ae") or []
        if not ae:
            continue
        n_ae += 1
        copies = li_to_ae(feats, ae, fp, fallback_crs=read_utm_json(reg.utm_json))
        n_copy += len(copies)
        out.extend(copies)
    for feats in by.values():
        out.extend(feats)
    dst.write_text(json.dumps(writer.to_feature_collection(out), ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"features": len(out), "copies": n_copy, "regions_with_ae": n_ae,
                      "unmatched_regions": len(by)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
'@
[IO.File]::WriteAllText('C:\Sloi\oof59\fuse_r18.py', ($t -replace "`r?`n","`r`n"), (New-Object Text.UTF8Encoding $false))
$t=@'
"""Сравнение распределений: сданный li_hard2_fused против нового fused (курганы, по группам и семьям)."""
import json, sys, gc
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

REPO = Path(r"C:\Sloi\expds-koz3_v0.2.0\expds-koz3")
sys.path.insert(0, str(REPO))
from expds.naming import site_family

KUR = {"kurgany_povrezhdennye", "kurgany_tselye"}


def summarize(path: Path) -> dict:
    gc.disable()
    fc = json.loads(path.read_bytes())
    feats = fc["features"]
    del fc
    keys = Counter()
    conf = defaultdict(list)
    fam = defaultdict(Counter)
    regions = set()
    for f in feats:
        p = f.get("properties") or {}
        cls = p.get("class_name")
        mk = p.get("markup_type")
        keys[f"{mk}|{p.get('threshold_group')}|{cls}"] += 1
        if mk == "li":
            regions.add(p.get("region_name"))
        if cls in KUR and mk == "li":
            conf[cls].append(float(p.get("confidence") or 0))
            fam[cls][site_family(str(p.get("region_name")))] += 1
    del feats
    gc.enable(); gc.collect()
    q = [0.5, 0.9, 0.95, 0.99, 0.999]
    out = {"keys": dict(keys), "li_regions": len(regions), "regions": sorted(r for r in regions if r)}
    for cls, v in conf.items():
        a = np.asarray(v)
        out[cls] = {"n": int(a.size), "quantiles": dict(zip(map(str, q), np.quantile(a, q).round(5).tolist())),
                    "n_ge": {str(t): int((a >= t).sum()) for t in (0.05, 0.0641, 0.1, 0.1942, 0.3, 0.5, 0.7, 0.9)},
                    "by_family": dict(fam[cls])}
    return out


if __name__ == "__main__":
    res = {}
    for name, p in (("old", sys.argv[1]), ("new", sys.argv[2])):
        print("читаю", p, flush=True)
        res[name] = summarize(Path(p))
    Path(sys.argv[3]).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("ok", flush=True)
'@
[IO.File]::WriteAllText('C:\Sloi\oof59\diag_r18.py', ($t -replace "`r?`n","`r`n"), (New-Object Text.UTF8Encoding $false))
$t=@'
"""Обёртка над scripts/scoreboard.py: тот же расчёт, плюс честные F1 по классам и по семьям."""
import json, sys
from pathlib import Path

REPO = Path(r"C:\Sloi\expds-koz3_v0.2.0\expds-koz3")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import scoreboard as sb
from expds.metric import score_counts
from expds.naming import site_family

EXTRA = Path(sys.argv.pop(1))
CAP = {"eval": {}, "honest": {}}
_ev, _ho = sb.evaluate, sb.honest


def evaluate(name, feats, gt, new, optical, metric):
    r = _ev(name, feats, gt, new, optical, metric)
    CAP["eval"][name] = {"scores": r[0], "per_class": r[3]}
    CAP["sets"] = {"new": sorted(new), "optical": sorted(optical)}
    return r


def honest(name, new, optical, folds_k, by="family"):
    r = _ho(name, new, optical, folds_k, by)
    CAP["honest"][name] = {"scores": r[0], "counts": r[1]}
    return r


sb.evaluate, sb.honest = evaluate, honest
sb.main()
out = {"eval": CAP["eval"], "honest": {}}
for name, v in CAP["honest"].items():
    counts = v["counts"]
    regs = {r for by in counts.values() for r in by}
    fams = {}
    for r in regs:
        fams.setdefault(site_family(r), set()).add(r)
    out["honest"][name] = {
        "scores": v["scores"],
        "per_class": score_counts(counts)[1],
        "per_class_new": score_counts(counts, set(CAP["sets"]["new"]))[1],
        "per_family": {f: score_counts(counts, rs)[0] for f, rs in sorted(fams.items())},
        "per_family_class": {f: {c: d for c, d in score_counts(counts, rs)[1].items() if d["n_gt"] or d["tp"] or d["fp"]}
                             for f, rs in sorted(fams.items())},
    }
EXTRA.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print("extra ->", EXTRA, flush=True)
import os
sys.stdout.flush(); os._exit(0)
'@
[IO.File]::WriteAllText('C:\Sloi\oof59\sb_wrap.py', ($t -replace "`r?`n","`r`n"), (New-Object Text.UTF8Encoding $false))
$t=@'
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
'@
[IO.File]::WriteAllText('C:\Sloi\oof59\pipe_r18.cmd', ($t -replace "`r?`n","`r`n"), (New-Object Text.UTF8Encoding $false))
Get-ChildItem C:\Sloi\oof59 -File | Select-Object Name,Length | Format-Table -AutoSize | Out-String
$v='C:\Sloi\expds-koz3_v0.2.0\expds-koz3\.venv\Scripts\python.exe'
& $v -m py_compile C:\Sloi\oof59\fuse_r18.py C:\Sloi\oof59\diag_r18.py C:\Sloi\oof59\sb_wrap.py; "py_compile rc=$LASTEXITCODE"
Remove-Item C:\Sloi\oof59\__pycache__ -Recurse -Force -ErrorAction SilentlyContinue
$r=Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='cmd.exe /c "C:\Sloi\oof59\pipe_r18.cmd"'; CurrentDirectory='C:\Sloi\oof59'}
"pipe rc=$($r.ReturnValue) pid=$($r.ProcessId) at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Start-Sleep 5
[Console]::OutputEncoding=[Text.Encoding]::UTF8
Get-Content C:\Sloi\oof59\pipe.log -Encoding UTF8
"r18 alive: " + [bool](Get-Process -Id 23560 -ErrorAction SilentlyContinue)
Get-Content C:\Sloi\oof59\r18\run.log -Tail 3 -Encoding UTF8
