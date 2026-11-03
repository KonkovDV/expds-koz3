"""Отчёт о среде и опись входа. На объекты и пороги не влияет."""
from __future__ import annotations

import logging
import os
import platform
import sys
from collections import Counter
from pathlib import Path

log = logging.getLogger(__name__)


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _meminfo() -> dict[str, float]:
    out = {}
    text = _read("/proc/meminfo") or ""
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        if key in ("MemTotal", "MemAvailable"):
            out[key] = round(int(rest.split()[0]) / 2 ** 20, 1)
    return out


def _writable(path: Path, create: bool = True) -> bool:
    try:
        if create:
            path.mkdir(parents=True, exist_ok=True)
        probe = path / ".expds_probe"
        probe.write_text("x", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def runtime_report(output: Path) -> dict:
    rep: dict = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": os.cpu_count(),
        "mem_gb": _meminfo(),
        "cgroup_mem_max": _read("/sys/fs/cgroup/memory.max") or _read("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
        "uid": os.getuid() if hasattr(os, "getuid") else None,
        "output_writable": _writable(output.parent),
    }
    try:
        import torch

        rep["torch"] = torch.__version__
        rep["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            rep["gpu"] = props.name
            rep["gpu_mem_gb"] = round(props.total_memory / 2 ** 30, 1)
    except Exception as exc:
        rep["torch_error"] = type(exc).__name__
    try:
        from . import accel

        rep["accel"] = accel.backend()
    except Exception as exc:
        rep["accel_error"] = type(exc).__name__
    return rep


def model_report(cfg: dict, root: Path) -> dict:
    from .pipeline import model_files, model_specs

    out = {}
    for markup in (cfg.get("models") or {}):
        for path, _classes in model_specs(cfg["models"], markup):
            for rel in model_files(path):
                p = Path(rel) if Path(rel).is_absolute() else root / rel
                out[f"{markup}:{Path(rel).name}"] = round(p.stat().st_size / 2 ** 20, 1) if p.exists() else "НЕТ ФАЙЛА"
    lgbm = (cfg.get("candidates") or {}).get("lgbm_model")
    if lgbm:
        p = root / lgbm
        out[f"lgbm:{Path(lgbm).name}"] = round(p.stat().st_size / 2 ** 20, 1) if p.exists() else "НЕТ ФАЙЛА"
    return out


def log_inventory(regions: list, cfg: dict, root: Path) -> dict:
    from .crs import to_3857_ok
    from .pipeline import region_crs
    from .raster import needs_warp

    by_mt: Counter = Counter()
    las_n = 0
    crs_kinds: Counter = Counter()
    warp: Counter = Counter()
    for reg in regions:
        for mt, files in (getattr(reg, "rasters", None) or {}).items():
            by_mt[mt] += len(files)
            for f in files:
                try:
                    import rasterio

                    with rasterio.open(f) as ds:
                        declared = ds.crs.to_string() if ds.crs else None
                        if not to_3857_ok(declared):
                            warp["crs_не_переводится"] += 1
                        warp[needs_warp(ds) or "ok"] += 1
                except Exception:
                    warp["не открылся"] += 1
        if getattr(reg, "las", None):
            las_n += len(reg.las)
            crs_kinds[region_crs(reg, cfg) or "НЕТ CRS"] += 1
    rep = {
        "regions": len(regions),
        "rasters_by_modality": dict(by_mt),
        "las_files": las_n,
        "las_region_crs": dict(crs_kinds),
        "raster_read": dict(warp),
        "models": model_report(cfg, root),
    }
    log.info("опись входа: %s", rep)
    missing = [k for k, v in rep["models"].items() if v == "НЕТ ФАЙЛА"]
    if missing:
        log.error("нет файлов моделей: %s", missing)
    return rep
