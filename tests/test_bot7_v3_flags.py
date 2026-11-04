"""bot-7 (кандидат v3): D-FINE (sp_dfine) и шлюз городищ (gor_gate) — флаги выкл., перевод C/E, drop_groups."""
import math
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import Point

from expds import gor_gate, sp_dfine
from conftest import requires_submission_files  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
Y55 = 7361866.0  # ~55° с.ш. в EPSG:3857


def _cfg():
    return tomllib.loads((ROOT / "configs" / "default.toml").read_text(encoding="utf-8"))


def _det(x, y, conf, cls="kurgany_tselye", reg="r1", model=0, half=3.0):
    return {"geom": Point(x, y).buffer(half), "class_name": cls, "confidence": conf, "region_name": reg,
            "model": model, "source": "a.tif", "original_crs": "EPSG:32636"}


def test_default_flags_off():
    cfg = _cfg()
    assert cfg["sp_dfine"] == {"enabled": False}
    assert cfg["gor_gate"] == {"enabled": False}
    assert not sp_dfine.enabled(cfg) and not gor_gate.enabled(cfg)
    assert "sp_dfine" not in cfg.get("thresholds_by_markup", {})
    st = sp_dfine.settings(cfg)
    assert st["drop_groups"] == [] and st["convert"] == "circle" and st["ensemble"] == "max"
    assert st["drop_min_det"] == 1 and st["drop_cover_m"] == 12.0


def test_cluster_circles_variant_c():
    s = sp_dfine.ground_scale(Y55)
    dets = [_det(0, Y55, 0.6), _det(2 / s, Y55, 0.9), _det(20 / s, Y55, 0.3),
            _det(0, Y55, 0.5, cls="kurgany_povrezhdennye")]
    out = sp_dfine.cluster_circles(dets)
    assert len(out) == 3
    top = max(out, key=lambda f: f["confidence"])
    assert top["confidence"] == 0.9 and top["views"] == 2
    assert abs(top["geom"].centroid.x - 2 / s) < 1e-6           # центр самого уверенного
    r_ground = math.sqrt(top["geom"].area / math.pi) * s
    assert 3.6 < r_ground < 4.0                                    # 16-угольник r=4 м на земле
    for f in out:
        assert f["markup_type"] == "sp_or" and f["threshold_group"] == "sp_dfine"


def test_cluster_cover_merges_large_boxes():
    s = sp_dfine.ground_scale(Y55)
    a = _det(0, Y55, 0.8, cls="gorodishcha", half=60 / s)
    b = _det(10 / s, Y55, 0.7, cls="gorodishcha", half=20 / s)   # центр 10 м, но внутри большого
    assert len(sp_dfine.cluster_circles([a, b])) == 1


def test_ensemble_mean_vs_max():
    dets = [_det(0, Y55, 0.8, model=0), _det(0.5, Y55, 0.6, model=1), _det(500, Y55, 0.4, model=1)]
    mx = sorted(f["confidence"] for f in sp_dfine.cluster_circles(dets, ensemble="max", n_models=2))
    mn = sorted(f["confidence"] for f in sp_dfine.cluster_circles(dets, ensemble="mean", n_models=2))
    assert mx == [0.4, 0.8]
    assert mn == pytest.approx([0.2, 0.7])


def test_ellipse_variant_e():
    st = dict(sp_dfine.DEFAULTS, convert="ellipse")
    out = sp_dfine.as_features([_det(0, Y55, 0.5)], st, 1)
    assert len(out) == 1 and "model" not in out[0]
    assert out[0]["threshold_group"] == "sp_dfine" and out[0]["markup_type"] == "sp_or"


def test_drop_groups_variant_r():
    feats = [
        {"markup_type": "sp_or", "class_name": "gorodishcha", "region_name": "a"},
        {"markup_type": "sp_or", "class_name": "kurgany_tselye", "region_name": "a"},
        {"markup_type": "sp_or", "class_name": "arkhitektury", "region_name": "a"},
        {"markup_type": "sp_or", "class_name": "kurgany_tselye", "region_name": "a", "threshold_group": "sp_focal2"},
        {"markup_type": "sp_or", "class_name": "kurgany_tselye", "region_name": "a", "threshold_group": "sp_dfine"},
        {"markup_type": "li", "class_name": "kurgany_tselye", "region_name": "a"},
        {"markup_type": "sp_or", "class_name": "gorodishcha", "region_name": "b"},
    ]
    assert sp_dfine.drop_groups(feats, [], ["a"]) == feats
    out = sp_dfine.drop_groups(feats, ["sp_or", "sp_focal", "sp_arch", "sp_focal2"], ["a"])
    assert [f.get("threshold_group") or f["markup_type"] for f in out] == ["sp_dfine", "li", "sp_or"]
    assert out[-1]["region_name"] == "b"
    only_arch = sp_dfine.drop_groups(feats, ["sp_arch"], ["a"])
    assert len(only_arch) == len(feats) - 1


def test_no_thresholds_no_model_load(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise AssertionError("модель не должна грузиться без порогов")

    monkeypatch.setattr(sp_dfine, "load_models", boom)
    cfg = {"sp_dfine": {"enabled": True, "weights": ["x"], "strict": False}}
    reg = SimpleNamespace(name="r", rasters={"sp_or": [tmp_path / "a.tif"]})
    assert sp_dfine.region_feats(reg, cfg, None, {}, tmp_path) == ([], False)
    # Murder Board 29.10 (bot-10): по умолчанию strict — без порогов не тихий v2, а ошибка
    with pytest.raises(sp_dfine.DFineUnavailable):
        sp_dfine.region_feats(reg, {"sp_dfine": {"enabled": True, "weights": ["x"]}}, None, {}, tmp_path)
    ok, missing = sp_dfine.thresholds_ready(cfg)
    assert not ok and len(missing) == 5


def test_windows_grid():
    assert sp_dfine.windows(100, 100, 640, 160) == [(0, 0)]
    w = sp_dfine.windows(1500, 700, 640, 160)
    assert [x for x, y in w if y == 0] == [0, 480, 960] and sorted({y for _, y in w}) == [0, 480]


class _FakeScene:
    W = H = 640
    def __init__(self, img):
        self.img = img
    def read(self, x, y, n):
        return self.img


def _fake_model(torch, cx=0.25, cy=0.5, w=0.1, h=0.05, cls=2, score=4.0):
    class M:
        def __call__(self, pixel_values):
            b = pixel_values.shape[0]
            logits = torch.full((b, 3, 5), -10.0)
            logits[:, 0, cls] = score
            # картинка несимметрична: на отражённой половина кадра пустая — бокс в зеркальном месте
            flipped = bool((pixel_values[:, :, :, :8].sum() == 0).item())   # слева пусто только на отражённой
            boxes = torch.tensor([[1 - cx if flipped else cx, cy, w, h]] * 3).repeat(b, 1, 1)
            return SimpleNamespace(logits=logits, pred_boxes=boxes)
    return M()


def test_detect_scene_flip_and_nms():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    import numpy as np

    img = np.zeros((640, 640, 3), np.uint8)
    img[:, :320] = 200                                            # левая половина не пустая
    st = dict(sp_dfine.DEFAULTS)
    res, done, total = sp_dfine.detect_scene([_fake_model(torch)], _FakeScene(img), st, torch.device("cpu"))
    assert (done, total) == (1, 1)
    B, L, C = res[0]
    assert len(L) == 1 and int(C[0]) == 2                       # оригинал и отражение слились NMS
    assert B[0] == pytest.approx([0.2 * 640, 0.475 * 640, 0.3 * 640, 0.525 * 640], abs=1e-3)


def test_detect_scene_skips_empty_and_deadline():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    import numpy as np

    st = dict(sp_dfine.DEFAULTS)
    empty = np.zeros((640, 640, 3), np.uint8)
    res, done, total = sp_dfine.detect_scene([_fake_model(torch)], _FakeScene(empty), st, torch.device("cpu"))
    assert len(res[0][1]) == 0 and done == 1
    res, done, total = sp_dfine.detect_scene([_fake_model(torch)], _FakeScene(empty + 9), st, torch.device("cpu"),
                                             keep_going=lambda: False)
    assert done == 0 and total == 1 and len(res[0][1]) == 0


def test_region_feats_end_to_end_fake_model(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    path = tmp_path / "sp.tif"
    with rasterio.open(path, "w", driver="GTiff", width=400, height=400, count=3, dtype="uint8",
                       crs="EPSG:32636", transform=from_origin(500000, 6100000, 0.5, 0.5)) as ds:
        a = np.zeros((3, 400, 400), np.uint8)
        a[:, :, :200] = 150
        ds.write(a)
    monkeypatch.setattr(sp_dfine, "load_models", lambda st, root, dev: [_fake_model(torch, cx=0.1, cy=0.1)])
    th = {c: 0.1 for c in sp_dfine.CLASSES}
    cfg = {"sp_dfine": {"enabled": True, "weights": ["fake"], "device": "cpu"}, "thresholds_by_markup": {"sp_dfine": th}}
    reg = SimpleNamespace(name="r", rasters={"sp_or": [path]})
    feats, complete = sp_dfine.region_feats(reg, cfg, None, {}, tmp_path, crs_for=lambda r: "EPSG:32636")
    assert complete and len(feats) == 1
    f = feats[0]
    assert f["class_name"] == "gorodishcha" and f["threshold_group"] == "sp_dfine" and f["region_name"] == "r"
    from pyproj import Transformer
    x, y = Transformer.from_crs("EPSG:32636", "EPSG:3857", always_xy=True).transform(500000 + 32, 6100000 - 32)
    assert f["geom"].centroid.distance(Point(x, y)) < 1.0          # 64 px × 0,5 м от угла


def test_pipeline_phase_drop_only_complete(monkeypatch):
    from expds import pipeline

    class St:
        def __init__(self, feats):
            self.feats = list(feats)
        def add(self, f):
            self.feats += f
        def snapshot(self):
            return list(self.feats)
        def replace(self, f):
            self.feats = list(f)

    class G:
        soft_ceiling = None
        def ok(self, s=0):
            return True
        def elapsed(self):
            return 0.0

    def fake(reg, cfg, guard, cache, root, crs_for=None, soft_ok=None):
        cache["models"] = ["m"]
        f = {"markup_type": "sp_or", "class_name": "kurgany_tselye", "region_name": reg.name,
             "threshold_group": "sp_dfine", "confidence": 0.9}
        return [f], reg.name == "a"

    monkeypatch.setattr(sp_dfine, "region_feats", fake)
    old = [{"markup_type": "sp_or", "class_name": "gorodishcha", "region_name": n} for n in ("a", "b")]
    st = St(old)
    regs = [SimpleNamespace(name="a", rasters={"sp_or": ["x"]}), SimpleNamespace(name="b", rasters={"sp_or": ["y"]}),
            SimpleNamespace(name="c", rasters={})]
    cfg = {"sp_dfine": {"enabled": True, "drop_groups": ["sp_or"], "drop_cover_m": 0}}
    assert pipeline._sp_dfine_phase(regs, cfg, G(), st, ROOT) == 2
    kept = [(f["region_name"], f.get("threshold_group")) for f in st.feats]
    assert ("a", None) not in kept and ("b", None) in kept      # b прерван — старые объекты остаются
    assert kept.count(("a", "sp_dfine")) == 1 and kept.count(("b", "sp_dfine")) == 1




def test_pipeline_phase_drop_skips_regions_without_dfine(monkeypatch):
    """bot-2 safety valve: complete + 0 sp_dfine → v2 SpOr на участке сохраняются."""
    from expds import pipeline

    class St:
        def __init__(self, feats):
            self.feats = list(feats)
        def add(self, f):
            self.feats += f
        def snapshot(self):
            return list(self.feats)
        def replace(self, f):
            self.feats = list(f)

    class G:
        soft_ceiling = None
        def ok(self, s=0):
            return True
        def elapsed(self):
            return 0.0

    def fake(reg, cfg, guard, cache, root, crs_for=None, soft_ok=None):
        cache["models"] = ["m"]
        if reg.name == "empty":
            return [], True
        f = {"markup_type": "sp_or", "class_name": "kurgany_tselye", "region_name": reg.name,
             "threshold_group": "sp_dfine", "confidence": 0.9}
        return [f], True

    monkeypatch.setattr(sp_dfine, "region_feats", fake)
    old = [
        {"markup_type": "sp_or", "class_name": "gorodishcha", "region_name": "empty"},
        {"markup_type": "sp_or", "class_name": "gorodishcha", "region_name": "hit"},
    ]
    st = St(old)
    regs = [
        SimpleNamespace(name="empty", rasters={"sp_or": ["x"]}),
        SimpleNamespace(name="hit", rasters={"sp_or": ["y"]}),
    ]
    cfg = {"sp_dfine": {"enabled": True, "drop_groups": ["sp_or"], "drop_min_det": 1, "drop_cover_m": 0}}
    assert pipeline._sp_dfine_phase(regs, cfg, G(), st, ROOT) == 1
    kept = [(f["region_name"], f.get("threshold_group")) for f in st.feats]
    assert ("empty", None) in kept, kept  # v2 на empty сохранён
    assert ("hit", None) not in kept, kept  # v2 на hit снят
    assert ("hit", "sp_dfine") in kept
    assert ("empty", "sp_dfine") not in kept


def test_pipeline_phase_drop_min_det_zero_allows_empty(monkeypatch):
    """drop_min_det=0 — старое поведение R даже при 0 детекций (opt-out от багфикса)."""
    from expds import pipeline

    class St:
        def __init__(self, feats):
            self.feats = list(feats)
        def add(self, f):
            self.feats += f
        def snapshot(self):
            return list(self.feats)
        def replace(self, f):
            self.feats = list(f)

    class G:
        soft_ceiling = None
        def ok(self, s=0):
            return True
        def elapsed(self):
            return 0.0

    def fake(reg, cfg, guard, cache, root, crs_for=None, soft_ok=None):
        cache["models"] = ["m"]
        return [], True

    monkeypatch.setattr(sp_dfine, "region_feats", fake)
    st = St([{"markup_type": "sp_or", "class_name": "gorodishcha", "region_name": "a"}])
    regs = [SimpleNamespace(name="a", rasters={"sp_or": ["x"]})]
    cfg = {"sp_dfine": {"enabled": True, "drop_groups": ["sp_or"], "drop_min_det": 0, "drop_cover_m": 0}}
    assert pipeline._sp_dfine_phase(regs, cfg, G(), st, ROOT) == 0
    assert st.feats == []


def test_run_flag_off_never_touches_sp_dfine(monkeypatch):
    import inspect
    from expds import pipeline

    src = inspect.getsource(pipeline.run)
    assert 'if (cfg.get("sp_dfine") or {}).get("enabled"):' in src
    assert 'if (cfg.get("gor_gate") or {}).get("enabled"):' in src


def test_gor_gate_stub(tmp_path):
    feats = [{"markup_type": "sp_or", "class_name": "gorodishcha", "region_name": "a", "confidence": 0.99},
             {"markup_type": "sp_or", "class_name": "gorodishcha", "region_name": "a", "confidence": 0.5,
              "threshold_group": "sp_focal2"},
             {"markup_type": "or", "class_name": "gorodishcha", "region_name": "a", "confidence": 0.9}]
    assert gor_gate.apply(feats, {}, {}, tmp_path) is feats
    cfg = {"gor_gate": {"enabled": True}}
    miss = gor_gate.missing_artifacts(cfg, tmp_path)
    assert len(miss) == 4 and any("sp_or_gated" in m for m in miss)
    assert gor_gate.apply(feats, {}, cfg, tmp_path) == feats
    g = gor_gate.gate_confidences(feats, {"a": 0.5})
    assert g[0]["confidence"] == pytest.approx(0.495) and g[0]["threshold_group"] == "sp_or_gated"
    assert g[1] == feats[1] and g[2] == feats[2]
    assert gor_gate.gate_confidences(feats, {"z": 0.1}) == feats


def test_gor_gate_head_and_md5(tmp_path):
    """голова bot-5: p = σ([1, ((F−mu)/sd)·W/qsd]·w); неверный md5 — отказ загрузки."""
    import hashlib

    rng = np.random.default_rng(0)
    H = dict(mu=rng.normal(size=6), sd=rng.uniform(0.5, 2, 6), W=rng.normal(size=(6, 3)), qsd=rng.uniform(0.5, 2, 3),
             w=rng.normal(size=4), feat_dim=np.array(6))
    p = tmp_path / "h.npz"
    np.savez(p, **H)
    md5 = hashlib.md5(p.read_bytes()).hexdigest()
    h = gor_gate.load_head(p, md5)
    F = rng.normal(size=(5, 6))
    z = ((F - H["mu"]) / H["sd"]) @ H["W"] / H["qsd"]
    ref = 1 / (1 + np.exp(-(H["w"][0] + z @ H["w"][1:])))
    assert np.allclose(gor_gate.head_p(h, F), ref)
    with pytest.raises(Exception):
        gor_gate.load_head(p, "0" * 32)


def test_gor_gate_no_targets_skips_backbone(tmp_path, monkeypatch):
    """под вариантом R (sp_or выброшен) целей нет — бэкбон не грузится, объекты без изменений."""
    feats = [{"markup_type": "sp_dfine", "class_name": "gorodishcha", "region_name": "a", "confidence": 0.9}]
    cfg = {"gor_gate": {"enabled": True}, "thresholds_by_markup": {"sp_or_gated": {"gorodishcha": 0.99}}}
    monkeypatch.setattr(gor_gate, "missing_artifacts", lambda c, r: [])
    monkeypatch.setattr(gor_gate, "load_backbone", lambda *a: (_ for _ in ()).throw(AssertionError("загружен")))
    assert gor_gate.apply(feats, {"a": object()}, cfg, tmp_path) == feats


def test_train_only_modules_unreachable_and_not_packaged(monkeypatch):
    """Red team bot-4 В6: gt/blobs/coarse/det_boxes не нужны инференсу и не идут в архив."""
    import ast
    import importlib.util
    from conftest import allow_missing_submission_models
    allow_missing_submission_models(monkeypatch)

    mods = {p.stem: p for p in (ROOT / "expds").glob("*.py")}

    def deps(p):
        out = set()
        for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if isinstance(n, ast.ImportFrom):
                m = n.module or ""
                if n.level >= 1 and not m:
                    out |= {a.name for a in n.names}
                elif m.startswith("expds."):
                    out.add(m.split(".")[1])
                elif n.level >= 1:
                    out.add(m.split(".")[0])
                elif m == "expds":
                    out |= {a.name for a in n.names}
            elif isinstance(n, ast.Import):
                out |= {a.name.split(".")[1] for a in n.names if a.name.startswith("expds.")}
        return {d for d in out if d in mods}

    seen, todo = set(), deps(ROOT / "inference" / "main.py") | {"__init__"}
    while todo:
        m = todo.pop()
        if m not in seen:
            seen.add(m)
            todo |= deps(mods[m]) - seen
    assert {"gt", "blobs", "coarse", "det_boxes"}.isdisjoint(seen)
    assert {"pipeline", "sp_dfine", "gor_gate"} <= seen
    spec = importlib.util.spec_from_file_location("b7_build", ROOT / "scripts" / "build_submission.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.TRAIN_ONLY == {"gt.py", "blobs.py", "coarse.py", "det_boxes.py"}
    names = {p.name for p in mod.collect() if p.parent == ROOT / "expds"}
    assert names.isdisjoint(mod.TRAIN_ONLY) and "pipeline.py" in names


def test_no_personal_paths_in_packaged_code():
    import re

    pat = re.compile(r"[A-Z]:/|/workspace/|/home/|/Users/")
    bad = [f"{p.name}:{i}" for p in list((ROOT / "expds").glob("*.py")) + [ROOT / "configs" / "default.toml"]
           for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1) if pat.search(line)]
    assert not bad, bad


def test_kurgan_in_gor_replaced_by_audit_gormask():
    """наш kurgan_in_gor (−0,000848) удалён; правило — kurgans_outside_hillforts аудита, по умолчанию выкл."""
    import tomllib
    from expds import post

    cfg = tomllib.loads((ROOT / "configs" / "default.toml").read_text(encoding="utf-8"))
    assert "kurgan_in_gor" not in cfg["post"] and not cfg["post"].get("kurgans_outside_hillforts")
    assert not hasattr(post, "drop_kurgans_in_hillforts")


def test_flag_modules_not_packaged_when_off(monkeypatch):
    """v3: при выключенных [sp_dfine]/[gor_gate] их модули не идут в архив (граф импортов без transformers),
    pipeline.py импортирует их лениво только под флагом; вендор и веса D-FINE тоже не кладутся."""
    import importlib.util
    import tomllib
    from conftest import allow_missing_submission_models

    allow_missing_submission_models(monkeypatch)
    cfg = tomllib.loads((ROOT / "configs" / "default.toml").read_text(encoding="utf-8"))
    assert not cfg["sp_dfine"]["enabled"] and not cfg["gor_gate"]["enabled"]
    spec = importlib.util.spec_from_file_location("b7_build", ROOT / "scripts" / "build_submission.py")
    bs = importlib.util.module_from_spec(spec); spec.loader.exec_module(bs)
    rel = {p.relative_to(ROOT).as_posix() for p in bs.collect()}
    assert "expds/sp_dfine.py" not in rel and "expds/gor_gate.py" not in rel
    assert not any(r.startswith("inference/lib/vendor") or "dfine" in r for r in rel)
    src = (ROOT / "expds" / "pipeline.py").read_text(encoding="utf-8")
    top = [l for l in src.splitlines() if l.startswith(("from .", "import ")) and ("sp_dfine" in l or "gor_gate" in l)]
    assert not top


def test_time_protection_on_by_default_and_order_restored():
    """v3: [time] optics_order = small_first и optics_s_per_gb > 0 по умолчанию; после прохода оптики объекты
    стоят в исходном порядке участков (выход побайтно тот же), а при чужой записи в состояние — не трогаются."""
    import tomllib
    from types import SimpleNamespace
    from expds import pipeline
    from expds.watchdog import RunState

    cfg = tomllib.loads((ROOT / "configs" / "default.toml").read_text(encoding="utf-8"))
    assert cfg["time"]["optics_order"] == "small_first" and 0 < cfg["time"]["optics_s_per_gb"] <= 64
    regions = [SimpleNamespace(name=n) for n in ("a", "b", "c")]
    st = RunState(Path("/nonexistent/out.geojson"), {})
    st.add([{"id": "geo"}])
    by = {}
    for n in ("c", "a", "b"):            # порядок small_first
        f = [{"id": f"{n}{i}"} for i in range(2)]
        st.add(f)
        by[n] = f
    pipeline._restore_optics_order(st, regions, 1, by)
    assert [f["id"] for f in st.snapshot()] == ["geo", "a0", "a1", "b0", "b1", "c0", "c1"]
    # оптика оборвалась на «a» (не дошла до b): объекты пройденных участков сохранены в исходном порядке
    st2 = RunState(Path("/nonexistent/out.geojson"), {})
    st2.add([{"id": "c0"}, {"id": "a0"}])
    pipeline._restore_optics_order(st2, regions, 0, {"c": [{"id": "c0"}], "a": [{"id": "a0"}]})
    assert [f["id"] for f in st2.snapshot()] == ["a0", "c0"]
    # состояние изменилось во время прохода — перестановки нет
    st3 = RunState(Path("/nonexistent/out.geojson"), {})
    st3.add([{"id": "c0"}, {"id": "x"}])
    pipeline._restore_optics_order(st3, regions, 0, {"c": [{"id": "c0"}]})
    assert [f["id"] for f in st3.snapshot()] == ["c0", "x"]
