"""Файловый кэш геометрий метрики (WKB) между запусками: Linux и Windows.

Что кэшируется: результат `_collect` для пачки новых объектов — уже извлечённые
(`_extract_geometry`), пересчитанные в UTM и (для предсказаний-точек) превращённые
в круги геометрии. Повторный запуск scoreboard на тех же файлах читает их одним
`shapely.from_wkb` вместо разбора GeoJSON, PROJ и построения кругов.

Ключ — sha256 от ПОЛНОГО содержимого геометрий пачки (все координаты, не выборка)
и от всего, что влияет на результат: версия формата и хэш исходников
(metric.py, fastgeom.py, geomcache.py), sha256 официального compute_metrics.py,
режим CRS, r_tolerance, circle_points, версии shapely/GEOS/pyproj/PROJ.
Любое изменение → другой ключ → промах (а не неверный ответ).

Формат файла `<ключ>.geomcache` (версия 2):
    MAGIC(8) | u32 длина заголовка | JSON-заголовок (ключ, размеры, список массивов) |
    массивы little-endian: valid_idx, kind (3 — полигон, 0 — точка, 9 — WKB), координаты
    и смещения колец/полигонов (как shapely.to_ragged_array), смещения WKB | WKB подряд |
    blake2b-16 от всего предыдущего.
При записи каждый объект проверяется: восстановленный из координат даёт тот же WKB.
Запись: во временный файл с уникальным именем + os.replace (атомарно и на NTFS).
Чтение: файл целиком; неверные магия/ключ/контрольная сумма/размеры → промах.
Параллельные процессы безопасны: читатель видит либо старый, либо новый целый
файл; ошибки записи (в т.ч. PermissionError на Windows, пока файл открыт другим) молча
пропускаются — кэш лишь ускоряет.

Настройка (переменные окружения):
    EXPDS_GEOM_CACHE      каталог кэша; 0/off/none/false — выключить.
                          По умолчанию: metric._SHM_CACHE_DIR, т.е. /dev/shm/expds_utm_cache,
                          если есть /dev/shm, иначе <tempfile.gettempdir()>/expds_geom_cache.
    EXPDS_GEOM_CACHE_MB   предел размера каталога, МБ (по умолчанию 1024); старые файлы удаляются.
    EXPDS_GEOM_CACHE_MIN  минимальный размер пачки для кэша (по умолчанию 200 объектов).
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
import threading
from pathlib import Path

import numpy as np

MAGIC = b"EXGC0002"
FORMAT_VERSION = 2
SUFFIX = ".geomcache"
_DIGEST = 16


def default_dir() -> Path:
    if Path("/dev/shm").is_dir():
        return Path("/dev/shm/expds_utm_cache")
    return Path(tempfile.gettempdir()) / "expds_geom_cache"


def resolve_dir(fallback: Path | None) -> Path | None:
    """Каталог кэша или None (выключен)."""
    env = os.environ.get("EXPDS_GEOM_CACHE")
    if env is not None and env.strip() != "":
        if env.strip().lower() in ("0", "off", "none", "false", "no"):
            return None
        return Path(env)
    return Path(fallback) if fallback is not None else default_dir()


def min_batch() -> int:
    try:
        return max(1, int(os.environ.get("EXPDS_GEOM_CACHE_MIN", "200")))
    except ValueError:
        return 200


def _file_sha(path) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return "?"


_ENV_KEY: dict = {}


def env_fingerprint(mod) -> str:
    """Всё, что кроме содержимого объектов влияет на геометрии (кэшируется на модуль метрики)."""
    k = id(mod)
    hit = _ENV_KEY.get(k)
    if hit is not None and hit[0] is mod:
        return hit[1]
    import shapely
    try:
        import pyproj
        pv = f"{pyproj.__version__}/{pyproj.proj_version_str}"
    except Exception:  # pragma: no cover
        pv = "?"
    here = Path(__file__).resolve().parent
    parts = [
        f"fmt={FORMAT_VERSION}",
        "code=" + ",".join(_file_sha(here / n) for n in ("metric.py", "fastgeom.py", "geomcache.py")),
        "metric=" + _file_sha(getattr(mod, "__file__", "") or ""),
        f"shapely={shapely.__version__}/{shapely.geos_version_string}",
        f"pyproj={pv}",
        f"utm_thr={getattr(mod, '_UTM_ABS_X_THRESHOLD', None)!r}",
    ]
    fp = "|".join(parts)
    _ENV_KEY[k] = (mod, fp)
    return fp


def batch_key(mod, mode: str, is_gt: bool, r_tol, c_pts, gdicts: list, flat=None) -> str:
    """sha256 полного содержимого геометрий пачки.

    flat — результат expds_core.flatten_polygons(gdicts) (coords, ring_off, geom_off, taken):
    для разобранных Rust простых полигонов хэшируются их координаты (точные биты float64) и
    смещения колец, для остальных объектов — repr(geometry) целиком. Без flat — repr у всех.
    """
    h = hashlib.sha256()
    h.update(env_fingerprint(mod).encode())
    h.update(f"|mode={mode}|gt={bool(is_gt)}|r={r_tol!r}|c={c_pts!r}|n={len(gdicts)}|".encode())
    if flat is not None:
        coords, ring_off, geom_off, taken = flat
        taken = np.asarray(taken, dtype=np.int64)
        h.update(b"F")
        for arr, dt in ((coords, np.float64), (ring_off, np.int64), (geom_off, np.int64), (taken, np.int64)):
            a = np.ascontiguousarray(arr, dtype=dt)
            h.update(struct.pack("<q", a.size))
            h.update(a.tobytes())
        mask = np.ones(len(gdicts), dtype=bool)
        mask[taken] = False
        rest = np.flatnonzero(mask).tolist()
    else:
        h.update(b"R")
        rest = range(len(gdicts))
    for i in rest:
        h.update(f"{i}:{gdicts[i]!r};".encode())
    return h.hexdigest()


def path_for(cache_dir: Path, key: str) -> Path:
    return Path(cache_dir) / f"{key}{SUFFIX}"


def _encode(geoms) -> tuple[list[tuple[str, np.ndarray]], bytes]:
    """Геометрии → плоские массивы: полигоны и точки — координатами (from_ragged_array в разы быстрее
    from_wkb), прочие типы — WKB. Если хоть один объект после обратного преобразования даёт
    другой WKB, вся пачка пишется как WKB."""
    import shapely

    g = np.asarray(geoms, dtype=object)
    n = len(g)
    kind = np.full(n, 9, dtype=np.int8)
    tid = shapely.get_type_id(g)
    ok2d = ~shapely.has_z(g) & ~shapely.is_empty(g)
    arrays: list[tuple[str, np.ndarray]] = []
    for t, name in ((3, "poly"), (0, "pt")):
        idx = np.flatnonzero((tid == t) & ok2d)
        if idx.size:
            if t == 3:
                _, coords, (ring_off, geom_off) = shapely.to_ragged_array(g[idx])
                back = shapely.from_ragged_array(shapely.GeometryType.POLYGON, coords, (ring_off, geom_off))
            else:
                _, coords, _ = shapely.to_ragged_array(g[idx])
                back = shapely.from_ragged_array(shapely.GeometryType.POINT, coords)
            same = shapely.to_wkb(back) == shapely.to_wkb(g[idx])
            if not same.all():
                return _encode_all_wkb(g)
            kind[idx] = t
            arrays.append((name + "_coords", np.ascontiguousarray(coords, dtype="<f8")))
            if t == 3:
                arrays.append(("poly_ring", np.asarray(ring_off, dtype="<i8")))
                arrays.append(("poly_geom", np.asarray(geom_off, dtype="<i8")))
    rest = np.flatnonzero(kind == 9)
    wk = shapely.to_wkb(g[rest]) if rest.size else []
    lens = np.fromiter((len(w) for w in wk), dtype=np.int64, count=len(wk))
    offs = np.zeros(len(wk) + 1, dtype="<i8")
    np.cumsum(lens, out=offs[1:])
    arrays.insert(0, ("kind", kind))
    arrays.append(("wkb_off", offs))
    return arrays, b"".join(wk)


def _encode_all_wkb(g) -> tuple[list[tuple[str, np.ndarray]], bytes]:
    """Запасной путь: всё как WKB (если хоть один объект не восстановился координатами побайтно)."""
    wk = shapely_to_wkb(g)
    lens = np.fromiter((len(w) for w in wk), dtype=np.int64, count=len(wk))
    offs = np.zeros(len(wk) + 1, dtype="<i8")
    np.cumsum(lens, out=offs[1:])
    return [("kind", np.full(len(g), 9, dtype=np.int8)), ("wkb_off", offs)], b"".join(wk)


def shapely_to_wkb(g):
    import shapely
    return shapely.to_wkb(np.asarray(g, dtype=object)) if len(g) else []


def _decode(arrs: dict, blob: bytes, n: int):
    import shapely

    kind = arrs["kind"]
    if kind.shape != (n,):
        return None
    out = np.empty(n, dtype=object)
    idx = np.flatnonzero(kind == 3)
    if idx.size:
        polys = shapely.from_ragged_array(shapely.GeometryType.POLYGON, arrs["poly_coords"].reshape(-1, 2),
                                          (arrs["poly_ring"], arrs["poly_geom"]))
        if len(polys) != idx.size:
            return None
        out[idx] = polys
    idx = np.flatnonzero(kind == 0)
    if idx.size:
        pts = shapely.from_ragged_array(shapely.GeometryType.POINT, arrs["pt_coords"].reshape(-1, 2))
        if len(pts) != idx.size:
            return None
        out[idx] = pts
    idx = np.flatnonzero(kind == 9)
    offs = arrs["wkb_off"]
    if len(offs) != idx.size + 1 or offs[0] != 0 or offs[-1] != len(blob) or np.any(np.diff(offs) <= 0):
        return None
    if idx.size:
        mv = memoryview(blob)
        o = offs.tolist()
        wk = np.array([bytes(mv[o[i]:o[i + 1]]) for i in range(idx.size)], dtype=object)
        out[idx] = shapely.from_wkb(wk)
    if np.any(~np.isin(kind, (0, 3, 9))) or shapely.is_missing(out).any():
        return None
    return out


def load(cache_dir: Path, key: str, n_todo: int):
    """(valid_idx list[int], geoms ndarray) или None при промахе/повреждении."""
    p = path_for(cache_dir, key)
    try:
        with open(p, "rb") as fh:
            buf = fh.read()
    except OSError:
        return None
    try:
        if len(buf) < len(MAGIC) + 4 + _DIGEST or buf[:8] != MAGIC:
            return None
        body, dig = memoryview(buf)[:-_DIGEST], buf[-_DIGEST:]
        if hashlib.blake2b(body, digest_size=_DIGEST).digest() != dig:
            return None
        (hlen,) = struct.unpack_from("<I", body, 8)
        pos = 12
        hdr = json.loads(bytes(body[pos:pos + hlen]).decode("utf-8"))
        pos += hlen
        if hdr.get("key") != key or hdr.get("n_todo") != n_todo or hdr.get("fmt") != FORMAT_VERSION:
            return None
        n = int(hdr["n_valid"])
        arrs: dict = {}
        for name, dt, count in hdr["arrays"]:
            a = np.frombuffer(body, dtype=dt, count=count, offset=pos)
            pos += a.nbytes
            arrs[name] = a
        blob = bytes(body[pos:])
        valid_idx = arrs.pop("valid_idx")
        if valid_idx.shape != (n,) or (n and (valid_idx.min() < 0 or valid_idx.max() >= n_todo
                                             or np.any(np.diff(valid_idx) <= 0))):
            return None
        geoms = _decode(arrs, blob, n)
        if geoms is None:
            return None
        return valid_idx.astype(np.int64).tolist(), geoms
    except Exception:
        return None


def save(cache_dir: Path, key: str, n_todo: int, valid_idx: list, geoms: list) -> bool:
    try:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        arrays, blob = _encode(geoms)
        arrays.insert(0, ("valid_idx", np.asarray(valid_idx, dtype="<i8")))
        hdr = json.dumps({"key": key, "n_todo": int(n_todo), "n_valid": len(valid_idx), "fmt": FORMAT_VERSION,
                          "arrays": [[nm, a.dtype.str, int(a.size)] for nm, a in arrays]},
                         sort_keys=True).encode("utf-8")
        body = b"".join([MAGIC, struct.pack("<I", len(hdr)), hdr] + [a.tobytes() for _, a in arrays] + [blob])
        data = body + hashlib.blake2b(body, digest_size=_DIGEST).digest()
        final = path_for(cache_dir, key)
        tmp = cache_dir / f".{key}.{os.getpid()}.{threading.get_ident()}.{os.urandom(4).hex()}.tmp"
        try:
            with open(tmp, "wb") as fh:
                fh.write(data)
            os.replace(tmp, final)
        finally:
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
        prune(cache_dir, keep=final)
        return final.exists()
    except Exception:
        return False


def prune(cache_dir: Path, keep: Path | None = None) -> None:
    """Держит каталог в пределах EXPDS_GEOM_CACHE_MB: удаляет самые старые *.geomcache."""
    try:
        limit = float(os.environ.get("EXPDS_GEOM_CACHE_MB", "1024")) * 1024 * 1024
    except ValueError:
        limit = 1024.0 * 1024 * 1024
    try:
        files = []
        for p in Path(cache_dir).glob("*" + SUFFIX):
            try:
                st = p.stat()
                files.append((st.st_mtime, st.st_size, p))
            except OSError:
                pass
        total = sum(s for _, s, _ in files)
        for _, size, p in sorted(files, key=lambda t: t[0]):
            if total <= limit:
                break
            if keep is not None and p == keep:
                continue
            try:
                p.unlink()
                total -= size
            except OSError:
                pass
    except Exception:
        pass
