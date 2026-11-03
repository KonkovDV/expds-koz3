"""Дисковый кэш DEM между прогонами (неинвазивный).

По умолчанию ВЫКЛЮЧЕН. Включается каталогом:
  EXPDS_DEM_DISK_CACHE=/path   — писать/читать сетки сюда;
  EXPDS_DEM_DISK_CACHE=0|off   — явно выкл.;
или флагом [speed] dem_disk_cache / EXPDS_SPEED_DEM_DISK_CACHE=1
(тогда каталог по умолчанию — /dev/shm/expds_dem_cache либо tempfile).

Ключ учитывает путь LAS, mtime/size, параметры сетки/заливки и отпечаток
кода lidar/accel/demcache. Попадание → побайтная копия сетки; промах →
обычная сборка и атомарная запись. Ошибки I/O молча игнорируются.
Предел каталога: EXPDS_DEM_DISK_CACHE_MB (по умолчанию 8192).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import struct
import tempfile
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

MAGIC = b"EXDEM001"
FORMAT = 1
SUFFIX = ".demcache"
_CODE_FP: str | None = None


def default_dir() -> Path:
    if Path("/dev/shm").is_dir():
        return Path("/dev/shm/expds_dem_cache")
    return Path(tempfile.gettempdir()) / "expds_dem_cache"


def resolve_dir() -> Path | None:
    env = os.environ.get("EXPDS_DEM_DISK_CACHE")
    if env is not None and env.strip() != "":
        if env.strip().lower() in ("0", "off", "none", "false", "no"):
            return None
        return Path(env)
    # флаг [speed] dem_disk_cache выставляет этот env в speed.configure
    return None


def _code_fp() -> str:
    global _CODE_FP
    if _CODE_FP is not None:
        return _CODE_FP
    here = Path(__file__).resolve().parent
    parts = []
    for name in ("demcache.py", "lidar.py", "accel.py"):
        p = here / name
        try:
            parts.append(hashlib.sha256(p.read_bytes()).hexdigest()[:16])
        except OSError:
            parts.append("?")
    try:
        from . import accel
        parts.append(str(accel.backend()))
    except Exception:
        parts.append("?")
    _CODE_FP = "|".join(parts)
    return _CODE_FP


def invalidate_code_fp() -> None:
    """Сброс отпечатка кода (тесты)."""
    global _CODE_FP
    _CODE_FP = None


def las_stamp(las_path: str | Path) -> dict:
    p = Path(las_path).resolve()
    st = p.stat()
    return {"path": str(p), "size": int(st.st_size), "mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))}


def key_for(kind: str, las_path: str | Path, **params) -> str:
    h = hashlib.sha256()
    h.update(f"fmt={FORMAT}|code={_code_fp()}|kind={kind}|".encode())
    stamp = las_stamp(las_path)
    h.update(json.dumps(stamp, sort_keys=True, separators=(",", ":")).encode())
    clean = {k: params[k] for k in sorted(params)}
    h.update(json.dumps(clean, sort_keys=True, separators=(",", ":"), default=str).encode())
    return h.hexdigest()


def _budget_bytes() -> int:
    try:
        return max(64, int(os.environ.get("EXPDS_DEM_DISK_CACHE_MB", "8192"))) * 1024 ** 2
    except ValueError:
        return 8192 * 1024 ** 2


def _prune(cache_dir: Path) -> None:
    try:
        files = sorted(cache_dir.glob(f"*{SUFFIX}"), key=lambda p: p.stat().st_mtime)
    except OSError:
        return
    total = sum(p.stat().st_size for p in files)
    budget = _budget_bytes()
    while total > budget and files:
        old = files.pop(0)
        try:
            sz = old.stat().st_size
            old.unlink(missing_ok=True)
            total -= sz
        except OSError:
            break


def load(cache_dir: Path | None, key: str):
    """(dem float32 HxW, meta dict) или None."""
    if cache_dir is None:
        return None
    path = cache_dir / f"{key}{SUFFIX}"
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if len(raw) < 16 or raw[:8] != MAGIC:
        return None
    hdr_len = struct.unpack_from("<I", raw, 8)[0]
    body_off = 12 + hdr_len
    if body_off > len(raw):
        return None
    try:
        hdr = json.loads(raw[12:12 + hdr_len])
    except json.JSONDecodeError:
        return None
    if hdr.get("key") != key or hdr.get("format") != FORMAT:
        return None
    h, w = int(hdr["h"]), int(hdr["w"])
    need = h * w * 4
    payload = raw[body_off:body_off + need]
    if len(payload) != need:
        return None
    digest = raw[body_off + need:body_off + need + 16]
    if hashlib.blake2b(raw[:body_off + need], digest_size=16).digest() != digest:
        return None
    dem = np.frombuffer(payload, dtype="<f4").reshape(h, w).copy()
    meta = dict(hdr.get("meta") or {})
    log.info("DEM-кэш hit %s (%dx%d)", key[:12], w, h)
    return dem, meta


def save(cache_dir: Path | None, key: str, dem: np.ndarray, meta: dict) -> None:
    if cache_dir is None:
        return
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log.debug("DEM-кэш mkdir: %s", e)
        return
    arr = np.ascontiguousarray(dem, dtype=np.float32)
    hdr = {
        "format": FORMAT,
        "key": key,
        "h": int(arr.shape[0]),
        "w": int(arr.shape[1]),
        "dtype": "float32",
        "meta": {k: meta[k] for k in meta if _jsonable(meta[k])},
        "saved_unix": time.time(),
    }
    hdr_bytes = json.dumps(hdr, ensure_ascii=False, separators=(",", ":"), default=str).encode()
    payload = arr.astype("<f4", copy=False).tobytes(order="C")
    blob = MAGIC + struct.pack("<I", len(hdr_bytes)) + hdr_bytes + payload
    blob += hashlib.blake2b(blob, digest_size=16).digest()
    tmp = cache_dir / f".{key}.{os.getpid()}.{time.time_ns()}.tmp"
    try:
        tmp.write_bytes(blob)
        os.replace(tmp, cache_dir / f"{key}{SUFFIX}")
        _prune(cache_dir)
        log.info("DEM-кэш save %s (%dx%d, %.1f МиБ)", key[:12], arr.shape[1], arr.shape[0],
                 len(blob) / 1024 ** 2)
    except OSError as e:
        log.debug("DEM-кэш save: %s", e)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _jsonable(v) -> bool:
    return isinstance(v, (str, int, float, bool, type(None))) or (
        isinstance(v, (list, tuple)) and all(_jsonable(x) for x in v)
    )


def wrap_build(kind: str, las_path, params: dict, builder, dem_cls):
    """builder() → Dem; при попадании в кэш собирает Dem без пересчёта."""
    cache_dir = resolve_dir()
    if cache_dir is None:
        return builder()
    try:
        key = key_for(kind, las_path, **params)
    except OSError:
        return builder()
    hit = load(cache_dir, key)
    if hit is not None:
        dem_arr, meta = hit
        left = float(meta["left"])
        top = float(meta["top"])
        res = float(meta["res"])
        crs = meta.get("crs")
        info = dict(meta.get("info") or {})
        info["dem_cache"] = "hit"
        return dem_cls(dem=dem_arr, left=left, top=top, res=res, crs=crs, info=info)
    dem = builder()
    meta = {
        "left": float(dem.left),
        "top": float(dem.top),
        "res": float(dem.res),
        "crs": dem.crs,
        "info": {k: dem.info[k] for k in dem.info if _jsonable(dem.info[k])},
    }
    save(cache_dir, key, dem.dem, meta)
    dem.info = dict(dem.info, dem_cache="miss")
    return dem
