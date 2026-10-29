"""bot-5 (ветка bot5-gor-dense-v3): плотный локализатор городищ gor_dense (PFnoD бота 10) — флаг выкл., группа/порог,
noD, запись, EPSG:3857, пики. Медленный тест воспроизведения peaks.json — при наличии артефактов бокса bot-5."""
import json
import math
import re
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import Point, shape
from shapely import wkt as shp_wkt

from expds import gor_dense as GD
from expds import post, writer

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("/workspace/data/train-expds-3")
B5 = Path("/workspace/koz3_shared/bot-5")
# Эталон городища sp_or 081_Костомаровское в UTM 37N (gor_dense_inputs.json, упрощён на 2 м) и top-1 пик PFnoD
# (fam10 фолд 4, 'all+cp', s = 250; analysis/gor_dense/peaks.json) — пик внутри эталона.
EPSG_081 = 32637
GT_081_UTM = ("POLYGON ((554767.79 5616067.49, 554729.98 5616053.49, 554701.98 5616029.69, 554658.57 5615980.68, "
              "554644.57 5615952.68, 554637.57 5615882.67, 554638.97 5615850.46, 554678.17 5615811.26, 554735.58 5615773.45, "
              "554868.60 5615749.65, 554889.60 5615742.64, 554902.21 5615749.65, 554909.21 5615783.25, 554905.01 5615816.86, "
              "554907.81 5615840.66, 554938.61 5615909.27, 554927.41 5615945.68, 554879.80 5616011.48, 554851.80 5616036.69, "
              "554822.39 5616054.89, 554797.19 5616068.89, 554767.79 5616067.49))")
PEAK_081 = (0.8452816348642953, 554702.976618, 5615990.064747368)


def _cfg():
    return tomllib.loads((ROOT / "configs" / "default.toml").read_text(encoding="utf-8"))


def _enable_block() -> dict:
    """Закомментированный блок включения [gor_dense] из default.toml → dict (ключи `#   key = value`)."""
    txt = (ROOT / "configs" / "default.toml").read_text(encoding="utf-8")
    head = txt[txt.index("# bot-5 (ветка bot5-gor-dense-v3)"):txt.index("[gor_dense]\nenabled = false")]
    lines = [m.group(1) for m in re.finditer(r"^#   ([a-z0-9_]+ = .+?)(?:[ \t]+#.*)?$", head, re.M)]
    sec = tomllib.loads("\n".join(lines))
    thr = re.search(r"\[thresholds_by_markup\.gor_dense\] gorodishcha = ([0-9.]+)", head)
    return {"gor_dense": sec, "thresholds_by_markup": {"gor_dense": {"gorodishcha": float(thr.group(1))}}}


def _pt(reg="r1", conf=0.9, x=554702.976618, y=5615990.064747368):
    return GD.point_features(reg, [(conf, x, y)], 0.8, EPSG_081, GD.settings({}), "a.tif")[0]


def _dfine(cls, reg="r1", conf=0.95, x=4427737.0, y=6567300.0):
    return {"geom": Point(x, y).buffer(30), "class_name": cls, "confidence": conf, "region_name": reg,
            "markup_type": "sp_or", "threshold_group": "sp_dfine", "source": "a.tif"}


# ---------------------------------------------------------------- флаг и конфиг
def test_default_flag_off_and_noop():
    cfg = _cfg()
    assert cfg["gor_dense"] == {"enabled": False}
    assert not GD.enabled(cfg)
    assert "gor_dense" not in cfg.get("thresholds_by_markup", {})
    feats = [_dfine("gorodishcha")]
    out, n = GD.apply(feats, {}, cfg, ROOT)
    assert out is feats and n == 0


def test_enable_block_parses_and_is_complete():
    blk = _enable_block()
    assert blk["gor_dense"]["enabled"] is True
    assert blk["gor_dense"]["head_md5"] and blk["gor_dense"]["centre_s"] > 0
    st = GD.settings(blk)
    assert st["drop_dfine_classes"] == ["gorodishcha"] and st["k"] == 2 and st["suppress_m"] == 150.0
    assert st["gsds"] == [2.0, 4.0] and st["tile"] == 518 and st["stride"] == 392 and st["circle_r_m"] == 10.0
    # параметры есть; в корне репо артефактов нет → только файлы в списке недостающего
    miss = GD.missing_artifacts(blk, Path("/nonexistent"))
    assert miss and all("=" in m for m in miss)


def test_missing_params_unavailable_noop(caplog):
    cfg = {"gor_dense": {"enabled": True}}           # нет centre_s и порога gor_dense
    miss = GD.missing_artifacts(cfg, ROOT)
    assert "[gor_dense] centre_s" in miss and "[thresholds_by_markup.gor_dense] gorodishcha" in miss
    feats = [_dfine("gorodishcha")]
    reg = SimpleNamespace(name="r1", rasters={"sp_or": [Path("x.tif")]}, utm_json=None)
    out, n = GD.apply(feats, {"r1": reg}, cfg, ROOT)
    assert out is feats and n == 0 and GD.UNAVAILABLE in caplog.text


# ---------------------------------------------------------------- группа, порог, noD, merge, запись
def test_feature_fields_and_conf():
    f = _pt(conf=0.9)
    assert f["class_name"] == "gorodishcha" and f["markup_type"] == "gor_dense" and f["threshold_group"] == "gor_dense"
    assert f["out_markup_type"] == "sp_or" and f["region_name"] == "r1" and f["original_crs"] == "EPSG:32637"
    assert f["confidence"] == pytest.approx(0.5 * (0.8 + 0.9))      # conf = 0,5·(p_scene + пик), без умножения гейтом


def test_threshold_by_markup_gor_dense():
    by = {"gor_dense": {"gorodishcha": 0.824066}, "sp_or": {"gorodishcha": 0.10}, "sp_dfine": {"gorodishcha": 0.5}}
    thr = {"gorodishcha": 0.05}
    hi, lo = _pt(conf=0.86), _pt(conf=0.84)                        # conf 0.83 / 0.82
    fn = post.threshold_fn(thr, 0.15, by)
    assert fn(hi) == pytest.approx(0.824066) and fn(lo) == pytest.approx(0.824066)
    kept = post.apply_thresholds([hi, lo], thr, 0.15, by)
    assert kept == [hi]
    # без своей таблицы — НЕ падает на порог sp_or: идёт общий порог класса
    assert post.threshold_fn(thr, 0.15, {"sp_or": {"gorodishcha": 0.10}})(hi) == pytest.approx(0.05)


def test_nod_only_completed_regions_and_only_dfine_gorodishcha():
    a = _dfine("gorodishcha", "r1")
    b = _dfine("kurgany_tselye", "r1")
    c = _dfine("gorodishcha", "r2")                                 # участок не пройден gor_dense → не трогается
    d = dict(_dfine("gorodishcha", "r1"), threshold_group=None)     # городище sp_or не из D-FINE → остаётся
    e = dict(_dfine("gorodishcha", "r1"), threshold_group="sp_dfine_ae")   # подгруппы sp_dfine_* тоже noD
    pts = {"r1": [_pt("r1")], "r3": []}
    out = GD.apply_points([a, b, c, d, e], pts, ["gorodishcha"])
    assert a not in out and e not in out and b in out and c in out and d in out
    assert pts["r1"][0] in out and len(out) == 4
    assert GD.apply_points([a], pts, []) == [a, pts["r1"][0]]       # пустой список классов — без noD


def test_merge_keeps_gor_dense_separate_from_sp_or():
    p = _pt("r1")
    g = p["geom"]
    sp = {"geom": g.buffer(20), "class_name": "gorodishcha", "confidence": 0.5, "region_name": "r1",
          "markup_type": "sp_or", "source": "b.tif"}
    out = post.merge_within_markup([p, sp])
    assert len(out) == 2 and {f["markup_type"] for f in out} == {"gor_dense", "sp_or"}


def test_writer_out_markup_type_in_tz_list():
    fc = writer.to_feature_collection([_pt("r1")])
    pr = fc["features"][0]["properties"]
    assert pr["markup_type"] == "sp_or" and pr["class_name"] == "gorodishcha" and pr["crs"] == "EPSG:3857"
    assert not any("вне списка" in w for w in writer.validate(fc)["warnings"])
    # объекты без out_markup_type пишутся как раньше
    f = _dfine("gorodishcha")
    assert writer.to_feature_collection([f])["features"][0]["properties"]["markup_type"] == "sp_or"


# ---------------------------------------------------------------- EPSG:3857
def test_circle_3857_roundtrip_utm():
    from pyproj import Transformer
    g = GD.circle_3857(PEAK_081[1], PEAK_081[2], EPSG_081, 10.0, 32)
    assert 4.0e6 < g.centroid.x < 4.5e6 and 6.5e6 < g.centroid.y < 6.6e6      # 3857 (≈39,8° в.д., 50,7° с.ш.)
    back = Transformer.from_crs(3857, EPSG_081, always_xy=True)
    ux, uy = back.transform(*np.asarray(g.exterior.coords).T)
    r = np.hypot(ux - PEAK_081[1], uy - PEAK_081[2])
    assert np.allclose(r, 10.0, atol=1e-3)                            # радиус 10 м на земле (UTM), не 10 ед. 3857
    assert len(g.exterior.coords) == 4 * 32 + 1


def test_known_hillfort_lands_in_gt_3857():
    """Пик PFnoD 081_Костомаровское → круг 3857 внутри эталона, переведённого в 3857 (и внутри исходного эталона 3857)."""
    from pyproj import Transformer
    import shapely
    g = GD.circle_3857(PEAK_081[1], PEAK_081[2], EPSG_081)
    tr = Transformer.from_crs(EPSG_081, 3857, always_xy=True)
    gt3857 = shapely.transform(shp_wkt.loads(GT_081_UTM), lambda x, y: tr.transform(x, y), interleaved=False)
    assert gt3857.contains(g)
    exp = Point(*tr.transform(PEAK_081[1], PEAK_081[2]))
    assert g.centroid.distance(exp) < 0.05                            # центр круга = пик в 3857 (≤5 см)
    src = DATA / "081_Костомаровское" / "06_Костомаровское_разметка" / "Костомаровское_SpOr_городища.geojson"
    if src.exists():
        fc = json.loads(src.read_text(encoding="utf-8"))
        assert "3857" in fc["crs"]["properties"]["name"]
        gt = shape(fc["features"][0]["geometry"])
        assert gt.contains(g) and gt.hausdorff_distance(gt3857) < 5.0   # упрощение 2 м × 1/cos φ ≈ 3,2 ед. 3857


# ---------------------------------------------------------------- голова, сглаживание, пики
def test_head_p_formula():
    rng = np.random.default_rng(0)
    H = {"mu": rng.normal(size=4).astype(np.float32), "sd": np.ones(4, np.float32),
         "W": rng.normal(size=(4, 2)).astype(np.float32), "qsd": np.array([2.0, 0.5], np.float32),
         "w": np.array([0.1, 1.0, -1.0]), "feat_dim": np.int64(4)}
    F = rng.normal(size=(5, 4)).astype(np.float16)
    Q = ((F.astype(np.float32) - H["mu"]) @ H["W"]) / H["qsd"]
    exp = 1 / (1 + np.exp(-(0.1 + Q[:, 0] - Q[:, 1])))
    assert np.allclose(GD.head_p(H, F), exp, atol=1e-6)
    with pytest.raises(AssertionError):
        GD.head_p(H, F[:, :3])


def _grid(H=20, W=20, step=28.0):
    rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    return dict(H=H, W=W, row=rr.ravel(), col=cc.ravel(), ux=cc.ravel() * step, uy=-rr.ravel() * step,
                vf=np.ones(H * W, np.float32))


def test_file_candidates_prior_and_validity():
    tok = _grid()
    p = np.full(len(tok["row"]), 0.1)
    edge = 2 * 20 + 2
    centre = 10 * 20 + 10
    p[edge] = 0.9
    p[centre] = 0.7
    no_prior = GD.file_candidates(p, tok, float("inf"))
    assert (no_prior[0][1], no_prior[0][2]) == (tok["ux"][edge], tok["uy"][edge])
    with_prior = GD.file_candidates(p, tok, 100.0)                     # краевой пик ≈ 317 м от центра гасится
    assert (with_prior[0][1], with_prior[0][2]) == (tok["ux"][centre], tok["uy"][centre])
    tok2 = dict(tok, vf=np.where(np.arange(len(p)) == edge, 0.4, 1.0).astype(np.float32))   # vf < 0,5 — не кандидат
    c2 = GD.file_candidates(p, tok2, float("inf"))
    assert all((x, y) != (tok["ux"][edge], tok["uy"][edge]) for _, x, y in c2) and len(c2) == 200


def test_region_peaks_k2_suppression():
    c = [(0.9, 0, 0), (0.85, 100, 0), (0.8, 151, 0), (0.7, 1000, 0)]
    assert GD.region_peaks(c, 2, 150.0) == [(0.9, 0, 0), (0.8, 151, 0)]
    assert GD.region_peaks([], 2, 150.0) == []


# ---------------------------------------------------------------- воспроизведение бота 10 / bot-5 (медленный)
_RT = B5 / "work" / "gd" / "root"
_SLOW = (_RT / "models" / "gor_dense" / "fam10_f4_head.npz").exists() and (DATA / "084_Булан_Соготту").exists() \
    and (B5 / "analysis" / "gor_dense" / "peaks.json").exists()


@pytest.mark.skipif(not _SLOW, reason="нет артефактов бокса bot-5 (work/gd/root, analysis/gor_dense)")
def test_reproduces_peaks_json_fold4():
    import torch
    from expds import catalog, gor_gate
    torch.set_num_threads(2)
    pk = json.loads((B5 / "analysis" / "gor_dense" / "peaks.json").read_text(encoding="utf-8"))["layouts"]["fam10"][4]
    st = GD.settings({"gor_dense": {"centre_s": pk["cp_s"]["all"]}})
    model = gor_gate.load_backbone(st, _RT, torch.device("cpu"))
    H = GD.load_dense_head(_RT / "models" / "gor_dense" / "fam10_f4_head.npz", None)
    for r in ("084_Булан_Соготту", "081_Костомаровское"):
        reg = catalog.scan_region(DATA / r)
        epsg = gor_gate._epsg(reg)
        cands = []
        for p in sorted(reg.rasters["sp_or"]):
            tok = GD.dense_tokens(model, Path(p), epsg, torch.device("cpu"))
            cands += GD.file_candidates(GD.head_p(H, tok["F"]), tok, float(st["centre_s"]))
        got = GD.region_peaks(cands)
        ref = pk["all+cp"][r]
        assert len(got) == len(ref)
        for a, b in zip(got, ref):
            assert abs(a[0] - b[0]) < 1e-6 and math.hypot(a[1] - b[1], a[2] - b[2]) < 0.5


def _build_mod():
    import importlib.util
    spec = importlib.util.spec_from_file_location("b5_build", ROOT / "scripts" / "build_submission.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_package_flag_off_excludes_gor_dense():
    """Флаг ВЫКЛ.: gor_dense.py и его артефакты в архив не идут (набор файлов как v3 + ничего)."""
    mod = _build_mod()
    names = {p.name for p in mod.collect() if p.parent == ROOT / "expds"}
    assert "gor_dense.py" not in names
    assert not any("gor_dense" in str(p) for p in mod.collect())


def test_package_flag_on_ships_module_backbone_heads_vendor(tmp_path, monkeypatch):
    """Флаг ВКЛ.: gor_dense.py + gor_gate.py + sp_dfine.py (import_dfine), DINOv2-S, обе головы, вендор."""
    r = tmp_path / "root"
    (r / "inference" / "lib" / "vendor" / "transformers").mkdir(parents=True)
    (r / "inference" / "lib" / "vendor" / "transformers" / "__init__.py").write_text("")
    (r / "inference" / "main.py").symlink_to(ROOT / "inference" / "main.py")
    (r / "metadata.json").symlink_to(ROOT / "metadata.json")
    (r / "expds").symlink_to(ROOT / "expds")
    (r / "configs").mkdir()
    (r / "configs" / "default.toml").write_text((ROOT / "configs" / "default.toml").read_text(encoding="utf-8").replace(
        "[gor_dense]\nenabled = false", "[gor_dense]\nenabled = true\ncentre_s = 250.0"), encoding="utf-8")
    (r / "models").mkdir()
    for p in (ROOT / "models").iterdir():
        (r / "models" / p.name).symlink_to(p)
    for rel in ("dinov2-small/config.json", "dinov2-small/model.safetensors", "gor_gate/gor_v3_head.npz",
                "gor_dense/gor_dense_head.npz"):
        q = r / "models" / rel
        if q.parent.is_symlink():
            q.parent.unlink()
        q.parent.mkdir(parents=True, exist_ok=True)
        q.write_bytes(b"x")
    mod = _build_mod()
    monkeypatch.setattr(mod, "ROOT", r)
    files = mod.collect()
    names = {p.name for p in files if p.parent == r / "expds"}
    assert {"gor_dense.py", "gor_gate.py", "sp_dfine.py"} <= names
    rels = {str(p.relative_to(r)) for p in files}
    assert {"models/dinov2-small/config.json", "models/dinov2-small/model.safetensors", "models/gor_gate/gor_v3_head.npz",
            "models/gor_dense/gor_dense_head.npz", "inference/lib/vendor/transformers/__init__.py"} <= rels


# ---------------------------------------------------------------- порядок фаз, шлюз, сборка/проверка
def test_phase_order_after_sp_dfine_before_gor_gate():
    src = (ROOT / "expds" / "pipeline.py").read_text(encoding="utf-8")
    i_sd = src.index("n_phase = _sp_dfine_phase(regions, cfg, guard, state, root)")
    i_gd = src.index('if (cfg.get("gor_dense") or {}).get("enabled"):')
    i_gg = src.index('if (cfg.get("gor_gate") or {}).get("enabled"):\n            try:\n                from . import gor_gate')
    i_pf = src.index("if guard.partial:\n            state.note")
    assert i_sd < i_gd < i_gg < i_pf
    assert "GOR_DENSE_UNAVAILABLE" in src[i_gd:i_gg]


def test_gor_gate_never_multiplies_gor_dense_points():
    from expds import gor_gate
    assert not gor_gate.is_gated_target(_pt())               # своя группа порога → шлюз sp_or её не трогает


def test_sp_dfine_gorodishcha_101_block_kills_dfine_gorodishcha():
    """Закомментированный блок [thresholds_by_markup.sp_dfine] gorodishcha = 1.01: conf ≤ 1 никогда не проходит."""
    by = {"sp_dfine": {"gorodishcha": 1.01}, "gor_dense": {"gorodishcha": 0.824066}}
    d = _dfine("gorodishcha", conf=1.0)
    keep = post.apply_thresholds([d, _dfine("kurgany", conf=0.99), _pt(conf=0.9)], {"kurgany": 0.5}, 0.15, by)
    assert d not in keep and len(keep) == 2
    txt = (ROOT / "configs" / "default.toml").read_text(encoding="utf-8")
    assert "#   + [thresholds_by_markup.sp_dfine]\n#     gorodishcha = 1.01" in txt


def test_check_submission_knows_gor_dense():
    src = (ROOT / "scripts" / "check_submission.py").read_text(encoding="utf-8")
    assert "GOR_DENSE_UNAVAILABLE" in src and '"gor_dense.backbone"' in src and '"expds/gor_dense.py", dense' in src
