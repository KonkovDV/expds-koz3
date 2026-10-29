"""bot-8: ускорение build_curves (векторное качество пар, память пар, дерево эталонов) и файловый
кэш геометрий expds/geomcache.py. Всё сравнивается побайтно с build_curves_reference +
_collect_reference (код до ускорения) в обеих областях сопоставления (global и site)."""
from __future__ import annotations

import importlib.util
import json
import math
import multiprocessing as mp
import os
import random
import threading
from pathlib import Path

import numpy as np
import pytest
import shapely

from expds import geomcache, metric

ROOT = Path(__file__).resolve().parents[1]
METRIC = os.environ.get("EXPDS_METRIC") or str(ROOT / "materials" / "compute_metrics.py")
B8 = Path("/workspace/koz3_shared/bot-8")
REAL = [(B8 / "data" / "stack_sub.geojson", B8 / "data" / "gt_sub.geojson"),
        (B8 / "data" / "stack_cross.geojson", B8 / "data" / "gt_cross.geojson")]
pytestmark = pytest.mark.skipif(not Path(METRIC).exists(), reason="нет compute_metrics.py")
X0, Y0 = 3_545_000.0, 7_310_000.0


@pytest.fixture(autouse=True)
def _iso(tmp_path, monkeypatch):
    monkeypatch.setattr(metric, "_SHM_CACHE_DIR", tmp_path / "shm")
    monkeypatch.setenv("EXPDS_GEOM_CACHE", str(tmp_path / "gc"))
    monkeypatch.setenv("EXPDS_GEOM_CACHE_MIN", "1")
    monkeypatch.delenv("EXPDS_MATCH_SCOPE", raising=False)
    monkeypatch.delenv("EXPDS_FAST_MATCH", raising=False)
    metric._UTM_GEOM_CACHE.clear(); metric._PAIR_MEMO.clear(); metric._GT_TREES.clear()
    yield
    metric._UTM_GEOM_CACHE.clear(); metric._PAIR_MEMO.clear(); metric._GT_TREES.clear()


def ring(cx, cy, r, n=8, rng=None):
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n
        rr = r * (rng.uniform(0.6, 1.0) if rng else 1.0)
        pts.append([cx + rr * math.cos(a), cy + rr * math.sin(a)])
    return pts + [pts[0]]


def synth(seed, n, gt=False, sites=3):
    rng = random.Random(seed)
    classes = ["kurgany_tselye", "kurgany_povrezhdennye", "gorodishcha", "fortifikatsii", "dorogi"]
    feats = []
    for _ in range(n):
        s = rng.randrange(sites)
        cx, cy = X0 + s * 120 + rng.uniform(0, 150), Y0 + rng.uniform(0, 150)
        k = rng.random()
        if k < 0.65:
            g = {"type": "Polygon", "coordinates": [ring(cx, cy, rng.uniform(1, 9), rng.randint(4, 16), rng)]}
        elif k < 0.85:
            g = {"type": "Point", "coordinates": [cx, cy]}
        elif k < 0.93:
            g = {"type": "MultiPolygon", "coordinates": [[ring(cx, cy, 3)], [ring(cx + 15, cy, 2)]]}
        else:   # самопересекающийся «бантик» — невалидный полигон
            g = {"type": "Polygon", "coordinates": [[[cx, cy], [cx + 5, cy + 5], [cx + 5, cy], [cx, cy + 5], [cx, cy]]]}
        p = {"class_name": rng.choice(classes), "region_name": f"region_{s:04d}",
             "markup_type": rng.choice(["li", "ae", None])}
        if not gt:
            p["confidence"] = rng.choice([0.5, 0.25, 0.9, rng.random(), rng.random(), 1])
        feats.append({"type": "Feature", "geometry": g, "properties": p})
    return {"type": "FeatureCollection", "features": feats}


def blob(c):
    return [(k, v.conf.tobytes(), v.hit.tobytes(), v.pred_region, v.n_gt, list(v.gt_regions.items()), v.pred_group)
            for k, v in c.items()]


def reference(pred, gt, scope):
    metric._UTM_GEOM_CACHE.clear()
    orig = metric._collect
    old = os.environ.get("EXPDS_GEOM_CACHE")
    os.environ["EXPDS_GEOM_CACHE"] = "off"
    try:
        metric._collect = metric._collect_reference
        return metric.build_curves_reference(pred, gt, METRIC, scope=scope)
    finally:
        metric._collect = orig
        os.environ["EXPDS_GEOM_CACHE"] = old
        metric._UTM_GEOM_CACHE.clear()


@pytest.mark.parametrize("scope", ["global", "site"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_build_curves_equal_reference_synthetic(seed, scope):
    pred, gt = synth(seed, 700), synth(seed + 50, 350, gt=True)
    ref = blob(reference(pred, gt, scope))
    first = blob(metric.build_curves(pred, gt, METRIC, scope=scope))       # холодный файловый кэш
    metric._UTM_GEOM_CACHE.clear()
    second = blob(metric.build_curves(pred, gt, METRIC, scope=scope))      # из файла + память пар
    metric._UTM_GEOM_CACHE.clear(); metric._PAIR_MEMO.clear(); metric._GT_TREES.clear()
    third = blob(metric.build_curves(pred, gt, METRIC, scope=scope))       # из файла, без памяти
    assert first == ref and second == ref and third == ref


@pytest.mark.parametrize("scope", ["global", "site"])
def test_fast_match_off_equal(scope, monkeypatch):
    pred, gt = synth(9, 400), synth(10, 200, gt=True)
    ref = blob(reference(pred, gt, scope))
    monkeypatch.setenv("EXPDS_FAST_MATCH", "0")
    monkeypatch.setenv("EXPDS_PAIR_MEMO", "0")
    monkeypatch.setenv("EXPDS_GEOM_CACHE", "off")
    assert blob(metric.build_curves(pred, gt, METRIC, scope=scope)) == ref


@pytest.mark.parametrize("scope", ["global", "site"])
@pytest.mark.parametrize("pair", REAL, ids=["sub", "cross"])
def test_build_curves_equal_reference_real(pair, scope):
    if not (pair[0].exists() and pair[1].exists()):
        pytest.skip("нет реальной выборки бота 8")
    pred, gt = (json.loads(p.read_bytes()) for p in pair)
    ref = blob(reference(pred, gt, scope))
    assert blob(metric.build_curves(pred, gt, METRIC, scope=scope)) == ref
    metric._UTM_GEOM_CACHE.clear()
    assert blob(metric.build_curves(pred, gt, METRIC, scope=scope)) == ref


def test_mutated_input_is_not_served_from_cache():
    pred, gt = synth(3, 300), synth(4, 150, gt=True)
    metric.build_curves(pred, gt, METRIC)
    gt["features"][0]["geometry"] = {"type": "Polygon", "coordinates": [ring(X0 + 1, Y0 + 1, 4)]}
    pred["features"][1]["geometry"]["coordinates"] = [X0 + 2, Y0 + 2] \
        if pred["features"][1]["geometry"]["type"] == "Point" else [ring(X0 + 2, Y0 + 2, 5)]
    metric._UTM_GEOM_CACHE.clear()
    assert blob(metric.build_curves(pred, gt, METRIC)) == blob(reference(pred, gt, "global"))


# ---------- geomcache ----------

def _geoms(rng):
    out = [shapely.Polygon(ring(rng.uniform(0, 1e5), rng.uniform(0, 1e5), rng.uniform(1, 20), rng.randint(3, 30), rng))
           for _ in range(200)]
    out.append(shapely.Polygon([(0, 0), (10, 0), (10, 10), (0, 10)], [[(2, 2), (3, 2), (3, 3), (2, 2)]]))
    out += [shapely.Point(rng.random(), rng.random()) for _ in range(20)]
    return out


def _wkb(gs):
    return [shapely.to_wkb(g) for g in gs]


@pytest.mark.parametrize("extra", [[], ["multi"], ["z"], ["line", "multi"]])
def test_roundtrip_mixed_types(tmp_path, extra):
    rng = random.Random(1)
    gs = _geoms(rng)
    if "multi" in extra:
        gs.insert(5, shapely.MultiPolygon([shapely.box(0, 0, 1, 1), shapely.box(3, 3, 4, 4)]))
    if "z" in extra:
        gs.insert(0, shapely.Polygon([(0, 0, 1), (1, 0, 2), (1, 1, 3), (0, 0, 1)]))
    if "line" in extra:
        gs.append(shapely.LineString([(0, 0), (1, 1)]))
    idx = list(range(0, 2 * len(gs), 2))
    assert geomcache.save(tmp_path, "k" * 64, 2 * len(gs), idx, gs)
    vi, back = geomcache.load(tmp_path, "k" * 64, 2 * len(gs))
    assert vi == idx and _wkb(back) == _wkb(gs)
    assert [type(g) for g in back] == [type(g) for g in gs]


def test_empty_batch_and_miss(tmp_path):
    assert geomcache.load(tmp_path, "nokey", 5) is None
    assert geomcache.save(tmp_path, "e" * 64, 3, [], [])
    vi, back = geomcache.load(tmp_path, "e" * 64, 3)
    assert vi == [] and len(back) == 0
    assert geomcache.load(tmp_path, "e" * 64, 4) is None     # другой n_todo → промах


def test_corrupt_and_truncated_are_misses(tmp_path):
    gs = _geoms(random.Random(2))
    key = "c" * 64
    assert geomcache.save(tmp_path, key, len(gs), list(range(len(gs))), gs)
    p = geomcache.path_for(tmp_path, key)
    good = p.read_bytes()
    for bad in (good[:-1], good[: len(good) // 2], good[:5], b"",
                good[:100] + bytes([good[100] ^ 1]) + good[101:], b"EXGC0001" + good[8:]):
        p.write_bytes(bad)
        assert geomcache.load(tmp_path, key, len(gs)) is None
    p.write_bytes(good)
    assert geomcache.load(tmp_path, key, len(gs)) is not None
    other = geomcache.path_for(tmp_path, "d" * 64)
    other.write_bytes(good)                                     # чужой ключ в заголовке
    assert geomcache.load(tmp_path, "d" * 64, len(gs)) is None


def test_corrupt_cache_file_does_not_change_curves(tmp_path):
    pred, gt = synth(11, 300), synth(12, 150, gt=True)
    ref = blob(reference(pred, gt, "global"))
    assert blob(metric.build_curves(pred, gt, METRIC)) == ref
    files = list((tmp_path / "gc").glob("*" + geomcache.SUFFIX))
    assert len(files) == 2
    for f in files:
        b = bytearray(f.read_bytes()); b[len(b) // 2] ^= 0xFF; f.write_bytes(bytes(b))
    metric._UTM_GEOM_CACHE.clear()
    assert blob(metric.build_curves(pred, gt, METRIC)) == ref


def test_key_depends_on_content_and_params():
    m = metric.load_official(METRIC)
    fc = synth(5, 50)
    gd = [f["geometry"] for f in fc["features"]]
    from expds import fastgeom
    flat = fastgeom.flatten(gd)
    if flat is not None:    # путь через Rust flatten: ключ тот же для того же содержимого, иной — для другого
        kf = geomcache.batch_key(m, "utm", False, 5.0, 32, gd, flat)
        gdm = json.loads(json.dumps(gd))
        j = next(i for i, g in enumerate(gdm) if g["type"] == "Polygon")
        gdm[j]["coordinates"][0][0][0] += 1e-9
        assert kf == geomcache.batch_key(m, "utm", False, 5.0, 32, json.loads(json.dumps(gd)),
                                         fastgeom.flatten(json.loads(json.dumps(gd))))
        assert kf != geomcache.batch_key(m, "utm", False, 5.0, 32, gdm, fastgeom.flatten(gdm))
    k = geomcache.batch_key(m, "utm", False, 5.0, 32, gd)
    assert k == geomcache.batch_key(m, "utm", False, 5.0, 32, [dict(g) for g in gd])
    gd2 = json.loads(json.dumps(gd))
    c = gd2[7]["coordinates"]
    if gd2[7]["type"] == "Point":
        c[0] += 1e-9
    elif gd2[7]["type"] == "Polygon":
        c[0][1][0] += 1e-9
    else:
        c[0][0][1][0] += 1e-9
    keys = {k, geomcache.batch_key(m, "utm", False, 5.0, 32, gd2), geomcache.batch_key(m, "utm", True, 5.0, 32, gd),
            geomcache.batch_key(m, "utm", False, 5.5, 32, gd), geomcache.batch_key(m, "utm", False, 5.0, 64, gd),
            geomcache.batch_key(m, "raw", False, 5.0, 32, gd), geomcache.batch_key(m, "utm", False, 5.0, 32, gd[:-1])}
    assert len(keys) == 7


def test_disabled(monkeypatch, tmp_path):
    for v in ("off", "0", "none", "false"):
        monkeypatch.setenv("EXPDS_GEOM_CACHE", v)
        assert geomcache.resolve_dir(tmp_path) is None
    monkeypatch.setenv("EXPDS_GEOM_CACHE", "off")
    pred, gt = synth(13, 300), synth(14, 150, gt=True)
    ref = blob(reference(pred, gt, "global"))
    monkeypatch.setenv("EXPDS_GEOM_CACHE", "off")
    before = sorted(tmp_path.rglob("*"))
    assert blob(metric.build_curves(pred, gt, METRIC)) == ref
    assert sorted(tmp_path.rglob("*")) == before and not (tmp_path / "gc").exists()
    monkeypatch.delenv("EXPDS_GEOM_CACHE")
    assert geomcache.resolve_dir(tmp_path / "x") == tmp_path / "x"


def test_prune(tmp_path, monkeypatch):
    gs = _geoms(random.Random(3))
    monkeypatch.setenv("EXPDS_GEOM_CACHE_MB", "0.000001")
    for i in range(4):
        assert geomcache.save(tmp_path, f"{i}" * 64, len(gs), list(range(len(gs))), gs)
    left = list(tmp_path.glob("*" + geomcache.SUFFIX))
    assert [p.name for p in left] == [f"{'3' * 64}{geomcache.SUFFIX}"]   # только последний


def test_concurrent_threads(tmp_path):
    gs = _geoms(random.Random(4))
    key, n = "t" * 64, len(gs)
    errs = []

    def work():
        try:
            for _ in range(5):
                geomcache.save(tmp_path, key, n, list(range(n)), gs)
                r = geomcache.load(tmp_path, key, n)
                assert r is None or _wkb(r[1]) == _wkb(gs)
        except Exception as e:          # pragma: no cover
            errs.append(e)
    ts = [threading.Thread(target=work) for _ in range(4)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert not errs
    assert _wkb(geomcache.load(tmp_path, key, n)[1]) == _wkb(gs)
    assert not list(tmp_path.glob("*.tmp"))


def _proc_worker(d, seed, q):
    gs = _geoms(random.Random(seed))
    ok = True
    for _ in range(5):
        geomcache.save(Path(d), "p" * 64, len(gs), list(range(len(gs))), gs)
        r = geomcache.load(Path(d), "p" * 64, len(gs))
        ok &= r is None or _wkb(r[1]) == _wkb(gs)
    q.put(ok)


def test_concurrent_processes(tmp_path):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=_proc_worker, args=(str(tmp_path), 6, q)) for _ in range(2)]
    [p.start() for p in ps]; [p.join(120) for p in ps]
    assert all(p.exitcode == 0 for p in ps) and q.get() and q.get()
    assert not list(tmp_path.glob("*.tmp"))


# ---------- orjson / has_digit_run ----------

def _scoreboard():
    spec = importlib.util.spec_from_file_location("bot8_scoreboard", ROOT / "scripts" / "scoreboard.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.parametrize("text", [
    '{"a": 1, "b": [1.5, -0.0, 1e-300, 3.141592653589793], "c": "привет", "d": null, "e": true}',
    '{"big": 123456789012345678901234567890, "x": 1}',
    '{"i64": 9223372036854775807, "neg": -9223372036854775808}',
    '{"u": 18446744073709551615}', '{"n": NaN}', '{"inf": Infinity}', '\ufeff{"a": 1}',
    '{"s": "\\ud800"}', '{"dup": 1, "dup": 2}', '[1.0, 2, 1E5, 0.1]', '{"k": "1234567890123456789012"}'])
def test_load_json_bytes_same_as_json(text, monkeypatch):
    sb = _scoreboard()
    data = text.encode("utf-8")
    try:
        want = json.loads(data)
    except ValueError:
        with pytest.raises(ValueError):
            sb.load_json_bytes(data)
        return
    got = sb.load_json_bytes(data)
    assert repr(got) == repr(want) and type(got) is type(want)
    monkeypatch.setenv("EXPDS_ORJSON", "0")
    assert repr(sb.load_json_bytes(data)) == repr(want)


def test_load_json_real_file_same():
    p = B8 / "data" / "gt_sub.geojson"
    if not p.exists():
        pytest.skip("нет файла")
    data = p.read_bytes()
    assert repr(_scoreboard().load_json_bytes(data)) == repr(json.loads(data))


def test_has_digit_run():
    core = pytest.importorskip("expds_core")
    if not hasattr(core, "has_digit_run"):
        pytest.skip("старый expds_core")
    sb = _scoreboard()
    for s, n, want in ((b"", 19, False), (b"1" * 18, 19, False), (b"1" * 19, 19, True), (b"a" + b"9" * 19 + b"a", 19, True),
                       (b"12345678901234567.8901", 19, False), (b"x" * 1000 + b"0" * 19, 19, True), (b"123", 3, True)):
        assert core.has_digit_run(s, n) is want
        if n == 19:
            assert bool(sb._long_digit_run(s)) is want
