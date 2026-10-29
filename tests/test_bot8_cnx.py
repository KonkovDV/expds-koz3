"""bot-8: лидарные модели ConvNeXt bot-9 (in_kind cnx3/cnx5, [li_cnx]) — диспетчеризация, паритет подготовки, откат.

Сеть в тестах — крошечная CnxUnetCenter на tu-convnext_atto со случайными весами (контракт как у bot-9);
настоящие веса не нужны. Тест паритета подготовки — только если рядом лежит LAZ и эталон bot-9.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from conftest import ROOT, UTM, write_las

torch = pytest.importorskip("torch")
pytest.importorskip("segmentation_models_pytorch")
pytest.importorskip("timm")

from expds import lidar_cnx, pipeline  # noqa: E402
from expds.pipeline import TimeGuard  # noqa: E402

VENDOR = ROOT / "inference" / "cnx" / "vendor"
OUT_NAMES = ["hm_kurgany_tselye", "hm_kurgany_povrezhdennye", "wh_log_r_px", "off_x", "off_y",
             "mask_gorodishcha", "mask_fortifikatsii", "mask_arkhitektury"]
TAGS = {"cnx3": ["lrm_s", "lrm_l", "slope"], "cnx5": ["lrm_s", "lrm_l", "slope", "svf_r", "opp_r"]}


def _vendor_sha() -> dict:
    files = ["expds/__init__.py", "expds/accel.py", "expds/candidates.py", "expds/crs.py", "expds/lidar.py",
             "expds_core.abi3.so", "relief_ext.py"]
    return {f: hashlib.sha256((VENDOR / f).read_bytes()).hexdigest() for f in files}


def _contract(kind: str, seed: int = 0) -> dict:
    from expds.cnx_contract import build_from_arch

    torch.manual_seed(seed)
    ch = lidar_cnx.KINDS[kind]
    arch = {"name": "cnx_contract.CnxUnetCenter",
            "kwargs": {"encoder_name": "tu-convnext_atto", "encoder_weights": None, "in_channels": ch, "classes": 3,
                       "stride": 4, "feat_block": 2, "feat_ch": 64, "n_cls": 2, "head_ch": 8}}
    net = build_from_arch(arch)
    sd = {k: v.half() for k, v in net.state_dict().items()}      # как у bot-9: fp16 в файле
    vs = _vendor_sha()
    return {"format": "koz3-weights-contract-bot8", "contract_version": 1, "arch": arch, "state_dict": sd,
            "in_kind": kind, "in_channels": ch, "in_tags": TAGS[kind], "n_out": 8, "out_names": OUT_NAMES,
            "out_stride": [4, 4, 4, 4, 4, 1, 1, 1], "out_logits": True, "cell_m": 0.5,
            "prep_code_sha": hashlib.sha256(json.dumps(vs, sort_keys=True).encode()).hexdigest(),
            "prep_code_files": vs, "oof_thresholds": None}


@pytest.fixture(scope="module")
def weights(tmp_path_factory):
    d = tmp_path_factory.mktemp("cnxw")
    out = {}
    for kind in ("cnx3", "cnx5"):
        p = d / f"weights_cnx_{kind}_f0.pt"
        torch.save(_contract(kind), p)
        out[kind] = p
    p = d / "weights_cnx_cnx3_f1.pt"
    torch.save(_contract("cnx3", seed=1), p)
    out["cnx3_f1"] = p
    return out


def _cfg(*paths, **kw):
    kw.setdefault("tta", 1)       # по умолчанию 8 (D8); здесь — один проход, сверка с ручным forward
    return {"li_cnx": {"enabled": True, "weights": [str(p) for p in paths], **kw}}


# ------------------------------------------------------------ конфиг
def test_default_config_off():
    import tomllib
    cfg = tomllib.loads((ROOT / "configs" / "default.toml").read_text(encoding="utf-8"))
    assert cfg.get("li_cnx", {}).get("enabled") is False
    assert not lidar_cnx.enabled(cfg)
    assert not lidar_cnx.enabled({"li_cnx": {"enabled": True}})          # без весов — выкл.
    assert not lidar_cnx.enabled({})


# ------------------------------------------------------------ диспетчеризация / сеть
@pytest.mark.parametrize("kind", ["cnx3", "cnx5"])
def test_dispatch_shapes_classes_cpu_fp32(weights, kind):
    seg = lidar_cnx.load(_cfg(weights[kind]), Path("/"), device="cpu")
    assert seg.kind == kind and seg.in_channels == lidar_cnx.KINDS[kind]
    assert seg.classes == ["kurgany_tselye", "kurgany_povrezhdennye", "gorodishcha", "fortifikatsii", "arkhitektury"]
    assert seg.meta["li_source"] == "cnx" and seg.meta["li_tags"] == TAGS[kind]
    assert seg.amp is False and seg.device.type == "cpu"                # CPU — FP32
    assert (seg.tile, seg.overlap) == (512, 128)
    x = np.random.default_rng(0).uniform(-1, 1, (seg.in_channels, 70, 101)).astype(np.float32)  # не кратно 32
    p = seg.predict(x)
    assert p.shape == (5, 70, 101) and p.dtype == np.float32
    assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()


def test_padding_equals_manual(weights):
    """Дополнение до кратного 32 — значением u8 0 (−1,0079), как в обучении bot-9."""
    seg = lidar_cnx.load(_cfg(weights["cnx3"]), Path("/"), device="cpu")
    assert lidar_cnx.PAD_VALUE == pytest.approx(-128 / 127)
    x = np.random.default_rng(1).uniform(-1, 1, (3, 50, 40)).astype(np.float32)
    xp = np.full((3, 64, 64), lidar_cnx.PAD_VALUE, np.float32)
    xp[:, :50, :40] = x
    with torch.inference_mode():
        z = seg.nets[0](torch.from_numpy(xp[None])).numpy()[0]
    ref = 1 / (1 + np.exp(-z[:, :50, :40]))
    np.testing.assert_allclose(seg.predict(x), ref, atol=1e-5)


def test_tiling_single_window_equals_whole(weights):
    seg = lidar_cnx.load(_cfg(weights["cnx3"], tile_cpu=128, overlap_cpu=32), Path("/"), device="cpu")
    x = np.random.default_rng(2).uniform(-1, 1, (3, 96, 128)).astype(np.float32)
    with torch.inference_mode():
        whole = 1 / (1 + np.exp(-seg.nets[0](torch.from_numpy(x[None])).numpy()[0]))
    np.testing.assert_allclose(seg.predict(x), whole, atol=1e-5)     # 96×128 помещается в одно окно 128
    big = np.random.default_rng(3).uniform(-1, 1, (3, 300, 260)).astype(np.float32)
    p = seg.predict(big)                                             # несколько окон, края, стыки
    assert p.shape == (5, 300, 260) and np.isfinite(p).all()


def test_ensemble_and_tta_average_logits(weights):
    a = lidar_cnx.load(_cfg(weights["cnx3"]), Path("/"), device="cpu")
    b = lidar_cnx.load(_cfg(weights["cnx3_f1"]), Path("/"), device="cpu")
    ab = lidar_cnx.load(_cfg(weights["cnx3"], weights["cnx3_f1"]), Path("/"), device="cpu")
    x = np.random.default_rng(4).uniform(-1, 1, (3, 64, 64)).astype(np.float32)
    lg = lambda p: np.log(p) - np.log1p(-p)   # noqa: E731
    np.testing.assert_allclose(lg(ab.predict(x)), (lg(a.predict(x)) + lg(b.predict(x))) / 2, atol=2e-4)
    t8 = lidar_cnx.load(_cfg(weights["cnx3"], tta=8), Path("/"), device="cpu")
    assert t8.tta == 8
    p8 = t8.predict(x)
    assert p8.shape == (5, 64, 64) and np.isfinite(p8).all()


def test_contract_checks(weights, tmp_path):
    with pytest.raises(lidar_cnx.CnxError):                          # разные in_kind в ансамбле
        lidar_cnx.load(_cfg(weights["cnx3"], weights["cnx5"]), Path("/"), device="cpu")
    with pytest.raises(lidar_cnx.CnxError):                          # нет файла
        lidar_cnx.load(_cfg(tmp_path / "nope.pt"), Path("/"), device="cpu")
    c = _contract("cnx3")
    c["prep_code_files"]["expds/lidar.py"] = "0" * 64                 # вендор не тот, на котором учились
    p = tmp_path / "bad_vendor.pt"
    torch.save(c, p)
    with pytest.raises(lidar_cnx.CnxError, match="вендор"):
        lidar_cnx.load(_cfg(p), Path("/"), device="cpu")
    c = _contract("cnx3")
    c["in_kind"] = "las"
    torch.save(c, p)
    with pytest.raises(lidar_cnx.CnxError):
        lidar_cnx.load(_cfg(p), Path("/"), device="cpu")
    bad = tmp_path / "pickle.pt"                                       # weights_only=True: произвольный pickle не грузится
    torch.save({"format": "koz3-weights-contract-bot8", "x": Path("/")}, bad)
    with pytest.raises(Exception):
        lidar_cnx.load(_cfg(bad), Path("/"), device="cpu")


def test_nan_fp16_falls_back_to_fp32(weights, monkeypatch):
    seg = lidar_cnx.load(_cfg(weights["cnx3"]), Path("/"), device="cpu")
    calls = []
    real = seg._logits

    def fake(xb, amp, tta=None, nets=None):
        calls.append(amp)
        z = real(xb, False, tta, nets)
        return z * float("nan") if amp else z
    seg.amp = True
    seg.tta = 1
    monkeypatch.setattr(seg, "_logits", fake)
    p = seg.predict(np.zeros((3, 32, 32), np.float32))
    assert calls == [True, False] and np.isfinite(p).all() and seg.nan_fallbacks == 1
    monkeypatch.setattr(seg, "_logits", lambda xb, amp, tta=None, nets=None: real(xb, False, tta) * float("nan"))
    with pytest.raises(lidar_cnx.CnxError):
        seg.predict(np.zeros((3, 32, 32), np.float32))


def test_deadline_between_windows(weights):
    seg = lidar_cnx.load(_cfg(weights["cnx3"], tile_cpu=64, overlap_cpu=0), Path("/"), device="cpu")
    with pytest.raises(lidar_cnx.CnxError, match="время"):
        seg.predict(np.zeros((3, 128, 128), np.float32), deadline=lambda: True)


# ------------------------------------------------------------ подготовка (подпроцесс, вендор bot-9)
@pytest.mark.parametrize("ch", [3, 5])
def test_run_prep_synthetic_and_cleanup(tmp_path, ch):
    las = write_las(tmp_path / "a.las", n=20_000)
    tmp_root = tmp_path / "t"
    x, geo = lidar_cnx.run_prep(las, ch, UTM, 120, tmp_root=str(tmp_root))
    assert x.dtype == np.float32 and x.shape[0] == ch and x.shape[1] > 150 and x.shape[2] > 150
    assert geo["tags"] == TAGS[f"cnx{ch}"] and geo["step"] == 0.5
    assert list(tmp_root.iterdir()) == []                              # временный npz удалён
    assert x.min() >= -128 / 127 - 1e-6 and x.max() <= 127 / 127 + 1e-6


def test_run_prep_failure_is_cnx_error(tmp_path):
    with pytest.raises(lidar_cnx.CnxError):
        lidar_cnx.run_prep(tmp_path / "missing.las", 3, UTM, 60, tmp_root=str(tmp_path / "t"))
    with pytest.raises(lidar_cnx.CnxError):
        lidar_cnx.run_prep(tmp_path / "missing.las", 3, UTM, 0)
    with pytest.raises(lidar_cnx.CnxError):
        lidar_cnx.run_prep(tmp_path / "missing.las", 3, UTM, 60, prep_dir=tmp_path)   # нет вендора


LAZ = Path("/workspace/data/train-expds-3/191_Осечки_1.3км/01_Осечки_1.3км_La_точки/01_Осечки_1.3км_La_точки.laz")
REF = Path(os.environ.get("CNX_PREP_REF", "/workspace/koz3_shared/bot-9/for_bot8/ours_191_Осечки_1.3км.npz"))


@pytest.mark.skipif(not (LAZ.is_file() and REF.is_file()), reason="нет LAZ Осечки или эталона подготовки bot-9")
def test_prep_parity_with_bot9_reference(tmp_path):
    """Вендор в inference/cnx даёт побайтно тот же u8, что эталон bot-9 (без --crs, как эталон)."""
    with np.load(REF, allow_pickle=False) as z:
        ref, rgeo = z["u8"], json.loads(str(z["geo"]))
    x, geo = lidar_cnx.run_prep(LAZ, ref.shape[0], None, 300, tmp_root=str(tmp_path / "t"))
    u8 = np.rint(x * 127 + 128).astype(np.int16)
    assert u8.shape == ref.shape
    assert int(np.abs(u8 - ref.astype(np.int16)).max()) == 0
    for k in ("transform", "tags", "step", "dem_res", "dem_factor"):
        assert geo[k] == rgeo[k], k


# ------------------------------------------------------------ пайплайн: включение и откат на li
@pytest.fixture
def region(tmp_path):
    from expds.catalog import scan_root

    r = tmp_path / "input" / "region_0001"
    write_las(r / "01_region_0001_La_точки" / "01_region_0001_La_точки.las", n=30_000)
    (r / "UTM.json").write_text(json.dumps({"utm": "36N"}), encoding="utf-8")
    return next(x for x in scan_root(tmp_path / "input") if x.las)


def _key(fs):
    """Объекты в сравнимом виде: все поля, геометрия — WKT."""
    return sorted(json.dumps({k: (v.wkt if hasattr(v, "wkt") else v) for k, v in f.items()}, sort_keys=True,
                             default=str) for f in fs)


KUR = ["kurgany_povrezhdennye", "kurgany_tselye"]
THR = {"li_cnx": {"kurgany_povrezhdennye": 0.5, "kurgany_tselye": 0.5}}   # тестовые, не OOF


def _cfg_thr(*paths, **kw):
    c = _cfg(*paths, **kw)
    c["thresholds_by_markup"] = json.loads(json.dumps(THR))
    return c


# ------------------------------------------------------------ классы, пороги, TTA, точность
def test_default_classes_and_tta(weights):
    assert lidar_cnx.DEFAULTS["classes"] == KUR and lidar_cnx.DEFAULTS["tta"] == 8
    assert lidar_cnx.DEFAULTS["threshold_group"] == "li_cnx"
    seg = lidar_cnx.load({"li_cnx": {"enabled": True, "weights": [str(weights["cnx3"])]}}, Path("/"), device="cpu")
    assert seg.keep == set(KUR) and seg.tta == 8                       # gor/fort — только явно в конфиге
    seg = lidar_cnx.load(_cfg(weights["cnx3"], classes=KUR + ["gorodishcha"]), Path("/"), device="cpu")
    assert seg.keep == set(KUR) | {"gorodishcha"}
    for bad in (0, 3, 4, 16, "x", None, True):
        with pytest.raises(lidar_cnx.CnxError, match="tta"):
            lidar_cnx.load(_cfg(weights["cnx3"], tta=bad), Path("/"), device="cpu")


def test_threshold_problem():
    tp = lidar_cnx.threshold_problem
    assert "нет [thresholds_by_markup.li_cnx]" in tp({"li_cnx": {"enabled": True}})
    assert tp({"thresholds_by_markup": THR}) is None
    assert "kurgany_tselye" in tp({"thresholds_by_markup": {"li_cnx": {"kurgany_povrezhdennye": 0.5}}})
    assert tp({"thresholds_by_markup": THR}, KUR + ["gorodishcha"]) is not None   # класс без порога
    for g in ("li", "li_coarse", "li_mosaic", "li_cpu"):              # пороги li к cnx не применимы
        cfg = {"li_cnx": {"threshold_group": g}, "thresholds_by_markup": {g: THR["li_cnx"]}}
        assert "другой модели" in tp(cfg)
    cfg = {"li_cnx": {"threshold_group": "cnx_oof"}, "thresholds_by_markup": {"cnx_oof": THR["li_cnx"]}}
    assert tp(cfg) is None


def test_tta2_is_flip_w_average(weights):
    seg = lidar_cnx.load(_cfg(weights["cnx3"], tta=2), Path("/"), device="cpu")
    x = np.random.default_rng(5).uniform(-1, 1, (3, 64, 64)).astype(np.float32)
    with torch.inference_mode():
        net = seg.nets[0]
        z0 = net(torch.from_numpy(x[None])).numpy()[0]
        z1 = net(torch.from_numpy(np.ascontiguousarray(x[None, :, :, ::-1]))).numpy()[0][:, :, ::-1]
    ref = 1 / (1 + np.exp(-(z0 + z1) / 2))
    np.testing.assert_allclose(seg.predict(x), ref, atol=1e-5)


def test_tta_downgrade_under_budget(weights, caplog):
    seg = lidar_cnx.load(_cfg(weights["cnx3"], tta=8), Path("/"), device="cpu")
    assert seg.choose_tta(10, 1.0, 1000) == 8
    with caplog.at_level("WARNING"):
        assert seg.choose_tta(10, 1.0, 50) == 2                        # 80 с > 50 ≥ 20 с
        assert seg.choose_tta(10, 1.0, 15) == 1
        assert seg.choose_tta(10, 1.0, 1) == 1
    assert sum("TTA 8 → 2" in r.message for r in caplog.records) == 1
    x = np.random.default_rng(6).uniform(-1, 1, (3, 64, 64)).astype(np.float32)
    p = seg.predict(x, budget_s=0.0)                                   # запаса нет — tta 1
    assert seg.tta_used == 1
    one = lidar_cnx.load(_cfg(weights["cnx3"], tta=1), Path("/"), device="cpu")
    np.testing.assert_allclose(p, one.predict(x), atol=1e-6)
    seg.predict(x, budget_s=1e9)
    assert seg.tta_used == 8


def test_amp_follows_infer_config(weights, monkeypatch, caplog):
    from expds import infer_nn

    monkeypatch.setattr(infer_nn, "fp16_ok", lambda dev: True)          # как будто CUDA
    monkeypatch.setitem(infer_nn._RUNTIME, "amp", False)
    with caplog.at_level("WARNING"):
        seg = lidar_cnx.load(_cfg(weights["cnx3"], amp=True), Path("/"), device="cpu")
    assert seg.amp is False                                            # li_cnx.amp не читается
    assert any("li_cnx.amp не читается" in r.message for r in caplog.records)
    monkeypatch.setitem(infer_nn._RUNTIME, "amp", True)
    assert lidar_cnx.load(_cfg(weights["cnx3"]), Path("/"), device="cpu").amp is True
    monkeypatch.setattr(infer_nn, "fp16_ok", lambda dev: False)          # CPU / GTX 16xx — FP32
    assert lidar_cnx.load(_cfg(weights["cnx3"]), Path("/"), device="cpu").amp is False


# ------------------------------------------------------------ участок из нескольких LAS
def _shifted_las(src: Path, dst: Path, dx: float, scale: float = 0.001) -> Path:
    import laspy

    las = laspy.read(str(src))
    hdr = laspy.LasHeader(point_format=las.header.point_format.id, version=las.header.version)
    hdr.scales = np.array([scale] * 3)
    hdr.offsets = las.header.offsets + np.array([dx, 0, 0])
    hdr.vlrs.extend(las.header.vlrs)
    out = laspy.LasData(hdr)
    out.x, out.y, out.z = np.asarray(las.x) + dx, np.asarray(las.y), np.asarray(las.z)
    out.classification = las.classification
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.write(str(dst))
    return dst


def test_merge_las_grid_from_all_points(tmp_path):
    import laspy

    a = write_las(tmp_path / "a.las", n=5_000)
    b = _shifted_las(a, tmp_path / "b.las", 100.0, scale=0.01)
    m = lidar_cnx.merge_las([a, b], tmp_path / "m.las", chunk=1_234)   # мелкие куски — память
    la, lb, lm = (laspy.read(str(p)) for p in (a, b, m))
    assert len(lm.points) == len(la.points) + len(lb.points)
    np.testing.assert_allclose(lm.header.mins, np.minimum(la.header.mins, lb.header.mins), atol=0.01)
    np.testing.assert_allclose(lm.header.maxs, np.maximum(la.header.maxs, lb.header.maxs), atol=0.01)
    assert list(lm.header.scales) == [0.001] * 3                        # наименьший шаг
    np.testing.assert_array_equal(np.asarray(lm.classification),
                                  np.concatenate([la.classification, lb.classification]))
    np.testing.assert_allclose(np.asarray(lm.x), np.concatenate([la.x, lb.x]), atol=0.006)
    assert lm.header.parse_crs() is not None and lm.header.parse_crs() == la.header.parse_crs()


def test_bind_las_multi_las_one_prep(tmp_path, weights, monkeypatch):
    """Несколько LAS — одна подготовка на слитом файле (сетка по всем точкам), временный LAS удалён."""
    monkeypatch.setenv("EXPDS_TMP", str(tmp_path / "t"))
    a = write_las(tmp_path / "a.las", n=8_000)
    b = _shifted_las(a, tmp_path / "b.las", 60.0)
    seg = lidar_cnx.load(_cfg(weights["cnx3"], tta=1), Path("/"), device="cpu")
    one = lidar_cnx.bind_las(seg, [a], UTM, lidar_cnx.cnx_cfg({}))
    both = lidar_cnx.bind_las(seg, [a, b], UTM, lidar_cnx.cnx_cfg({}))
    assert both.probs.shape[2] > one.probs.shape[2] + 60               # 60 м = 120 ячеек по 0,5 м
    assert both.probs.shape[1] == one.probs.shape[1]
    assert list((tmp_path / "t").iterdir()) == []


# ------------------------------------------------------------ пайплайн: добавочная фаза cnx
def _fake_bind(probs):
    def fake(seg, las, crs, opts, **k):
        from rasterio.transform import from_origin
        return lidar_cnx.CnxBound(seg, probs, from_origin(500_000.0, 6_100_100.0, 0.5, 0.5), {"crs": UTM})
    return fake


def test_lidar_feats_never_touched_by_cnx(region, weights, monkeypatch):
    """li работает как без cnx: lidar_feats не зовёт cnx при любом [li_cnx]."""
    base = pipeline.lidar_feats(region, {}, TimeGuard(1000), {}, ROOT)
    assert base
    monkeypatch.setattr(lidar_cnx, "bind_las", lambda *a, **k: 1 / 0)
    monkeypatch.setattr(lidar_cnx, "load", lambda *a, **k: 1 / 0)
    assert pipeline.lidar_feats(region, _cfg_thr(weights["cnx3"]), TimeGuard(1000), {}, ROOT) == base


def test_cnx_without_own_thresholds_refuses(region, weights, monkeypatch, capsys):
    called = []
    monkeypatch.setattr(lidar_cnx, "load", lambda *a, **k: called.append(1))
    cache: dict = {}
    for cfg in (_cfg(weights["cnx3"]),                                  # нет порогов cnx
                dict(_cfg(weights["cnx3"]), thresholds_by_markup={"li": THR["li_cnx"]}),
                dict(_cfg(weights["cnx3"], threshold_group="li"), thresholds_by_markup={"li": THR["li_cnx"]})):
        assert pipeline.cnx_site_feats(region, cfg, TimeGuard(1000), cache, ROOT) == []
    assert not called
    assert "объекты cnx НЕ выдаются" in capsys.readouterr().err


def test_cnx_fallbacks_empty(region, weights, monkeypatch):
    g = lambda: TimeGuard(1000)   # noqa: E731
    cache: dict = {}
    assert pipeline.cnx_site_feats(region, _cfg_thr(Path("/nonexistent/w.pt")), g(), cache, ROOT) == []
    assert cache["li_cnx"] is None
    monkeypatch.setattr(lidar_cnx, "run_prep", lambda *a, **k: (_ for _ in ()).throw(lidar_cnx.CnxError("boom")))
    cache = {}
    assert pipeline.cnx_site_feats(region, _cfg_thr(weights["cnx3"]), g(), cache, ROOT) == []
    assert cache["li_cnx"] is not None
    monkeypatch.undo()
    monkeypatch.setattr(lidar_cnx.CnxSegmenter, "predict", lambda *a, **k: 1 / 0)
    assert pipeline.cnx_site_feats(region, _cfg_thr(weights["cnx3"]), g(), {}, ROOT) == []
    monkeypatch.undo()
    monkeypatch.setattr(lidar_cnx, "bind_las", _fake_bind(np.full((5, 200, 200), 0.9, np.float32)))
    monkeypatch.setattr(pipeline, "_li_with_model", lambda *a, **k: 1 / 0)
    assert pipeline.cnx_site_feats(region, _cfg_thr(weights["cnx3"]), g(), {}, ROOT) == []


def test_cnx_soft_ceiling_skips(region, weights, monkeypatch):
    g = TimeGuard(1000)
    g.set_soft(soft_ceiling_s=50)            # soft_left ≈ 50 < soft_need_s 120
    called = []
    monkeypatch.setattr(lidar_cnx, "bind_las", lambda *a, **k: called.append(1))
    assert pipeline.cnx_site_feats(region, _cfg_thr(weights["cnx3"]), g, {}, ROOT) == []
    assert not called and any(s["phase"] == "cnx" for s in g.soft_skipped)


def test_cnx_default_classes_and_group(region, weights, monkeypatch):
    monkeypatch.setattr(lidar_cnx, "bind_las", _fake_bind(np.full((5, 200, 200), 0.9, np.float32)))
    out = pipeline.cnx_site_feats(region, _cfg_thr(weights["cnx3"]), TimeGuard(1000), {}, ROOT)
    assert out and {f["class_name"] for f in out} <= set(KUR)          # gor/fort/arch не выдаются
    assert {f.get("threshold_group") for f in out} == {"li_cnx"}
    cfg = _cfg(weights["cnx3"], classes=["kurgany_tselye"], threshold_group="cnx_test")
    cfg["thresholds_by_markup"] = {"cnx_test": {"kurgany_tselye": 0.5}}
    out = pipeline.cnx_site_feats(region, cfg, TimeGuard(1000), {}, ROOT)
    assert out and {f.get("threshold_group") for f in out} == {"cnx_test"}
    assert {f["class_name"] for f in out} == {"kurgany_tselye"}
    cfg = _cfg(weights["cnx3"], classes=KUR + ["gorodishcha"])          # явно включённый класс-маска
    cfg["thresholds_by_markup"] = {"li_cnx": dict(THR["li_cnx"], gorodishcha=0.5)}
    out = pipeline.cnx_site_feats(region, cfg, TimeGuard(1000), {}, ROOT)
    assert "gorodishcha" in {f["class_name"] for f in out}


@pytest.mark.parametrize("kind", ["cnx3", "cnx5"])
def test_cnx_end_to_end(region, weights, kind, monkeypatch, tmp_path):
    """Настоящая подготовка подпроцессом + крошечная сеть: курганы по кандидатам ЦМР с вероятностью cnx."""
    monkeypatch.setenv("EXPDS_TMP", str(tmp_path / "t"))
    seen = []
    real = lidar_cnx.bind_las
    monkeypatch.setattr(lidar_cnx, "bind_las", lambda *a, **k: seen.append(real(*a, **k)) or seen[-1])
    out = pipeline.cnx_site_feats(region, _cfg_thr(weights[kind], tta=2), TimeGuard(1000), {}, ROOT)
    assert len(seen) == 1 and seen[0].probs.shape[0] == 5 and seen[0].meta["in_kind"] == kind
    assert out and all(f["class_name"] in KUR for f in out)
    assert all(f.get("markup_type") == "li" and f["threshold_group"] == "li_cnx" for f in out)
    assert list((tmp_path / "t").iterdir()) == []


def _kur(x, y, cls="kurgany_tselye", conf=0.9, group=None, region="r"):
    from shapely.geometry import Point
    f = {"geom": Point(x, y).buffer(5), "class_name": cls, "confidence": conf, "region_name": region,
         "markup_type": "li"}
    if group:
        f["threshold_group"] = group
    return f


def test_dedupe_cnx_against_kept_detections():
    from shapely.geometry import box

    cfg = {"thresholds_by_markup": {"li_coarse": {"kurgany_tselye": 0.4}, **THR}}
    pool = [_kur(0, 0),                                          # li, проходит KEPT_KURGAN_CONF
            _kur(100, 0, conf=0.01),                             # li ниже порога — не якорь
            _kur(200, 0, group="li_coarse", conf=0.5),           # группа вне таблицы — порог из конфига
            _kur(300, 0, group="li_coarse", conf=0.1),
            _kur(400, 0, cls="kurgany_povrezhdennye"),
            _kur(500, 0, region="other"),
            {"geom": box(1000, 0, 1100, 100), "class_name": "gorodishcha", "confidence": 0.99,
             "region_name": "r", "markup_type": "li"}]
    fresh = [_kur(8, 0, group="li_cnx"),                          # 8 м от якоря li — дубль
             _kur(20, 0, group="li_cnx"),                         # 20 м — новый
             _kur(105, 0, group="li_cnx"),                        # рядом слабый li — новый
             _kur(203, 0, group="li_cnx"),                        # дубль li_coarse
             _kur(303, 0, group="li_cnx"),                        # li_coarse ниже порога — новый
             _kur(402, 0, group="li_cnx"),                        # другой класс — новый
             _kur(500, 0, group="li_cnx"),                        # якорь в другом участке — новый
             {"geom": box(1050, 50, 1150, 150), "class_name": "gorodishcha", "confidence": 0.9,
              "region_name": "r", "markup_type": "li", "threshold_group": "li_cnx"},   # пересекается — дубль
             {"geom": box(2000, 0, 2100, 100), "class_name": "gorodishcha", "confidence": 0.9,
              "region_name": "r", "markup_type": "li", "threshold_group": "li_cnx"}]
    kept = pipeline.dedupe_cnx(fresh, pool, cfg)
    xs = sorted(round(f["geom"].centroid.x) for f in kept)
    assert xs == [20, 105, 303, 402, 500, 2050]
    assert pipeline.dedupe_cnx(fresh, [], cfg) == fresh
    assert pipeline.dedupe_cnx([], pool, cfg) == []


def test_vendor_manifest_matches_files():
    """inference/cnx/MANIFEST.json — sha256 всех файлов вендора (их сверяет check_submission.py)."""
    d = ROOT / "inference" / "cnx"
    m = json.loads((d / "MANIFEST.json").read_text(encoding="utf-8"))["sha256"]
    have = {p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file()
            and p.name not in ("MANIFEST.json", "WEIGHTS_MANIFEST.json") and "__pycache__" not in p.parts}
    assert set(m) == have
    for rel, sha in m.items():
        assert hashlib.sha256((d / rel).read_bytes()).hexdigest() == sha, rel
    assert m["cnx_prep.py"].startswith("77d30749")
    assert m["wheels/rvt_py-2.2.1-py3-none-any.whl"].startswith("f3b171e0")


# ---------- веса [li_cnx]: только по WEIGHTS_MANIFEST.json, не в git ----------

def test_weights_manifest_shape_and_not_in_git():
    import subprocess
    m = json.loads((ROOT / "inference" / "cnx" / "WEIGHTS_MANIFEST.json").read_text(encoding="utf-8"))
    paths = [e["path"] for e in m["weights"]]
    assert paths and all(p.startswith("models/cnx/") and p.endswith(".pt") for p in paths)
    assert all(len(e["sha256"]) == 64 and e["size"] > 0 for e in m["weights"])
    never = {e["sha256"] for e in m["never_pack"]}
    assert never and not never & {e["sha256"] for e in m["weights"]}
    assert "9b6b4c98f64f19f1" in "".join(never)            # c5 smoke bot-9
    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "models/cnx", "*.pt"], capture_output=True,
                             text=True).stdout.split()
    assert not [t for t in tracked if "cnx" in t]
    assert "models/cnx/" in (ROOT / ".gitignore").read_text(encoding="utf-8")


def _load_script(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"_b8_{name}", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_cnx_weights_verifies_sha_size_never_pack(tmp_path):
    bs = _load_script("build_submission_git")
    good, smoke, bad = tmp_path / "g.pt", tmp_path / "s.pt", tmp_path / "b.pt"
    good.write_bytes(b"good-weights")
    smoke.write_bytes(b"smoke")
    bad.write_bytes(b"good-weightZ")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    wman = {"weights": [{"path": "models/cnx/a.pt", "sha256": sha(good), "size": good.stat().st_size,
                         "source": str(good)},
                        {"path": "models/cnx/s.pt", "sha256": sha(smoke), "size": 5, "source": str(smoke)}],
            "never_pack": [{"sha256": sha(smoke)}]}
    cfg = lambda *w: {"li_cnx": {"enabled": True, "weights": list(w)}}
    assert bs.cnx_weights(cfg("models/cnx/a.pt"), wman, {}, set()) == {"models/cnx/a.pt": good}
    for c, ov, tracked, msg in ((cfg("models/cnx/x.pt"), {}, set(), "нет в"),
                                (cfg("models/cnx/a.pt"), {"models/cnx/a.pt": bad}, set(), "не совпадает"),
                                (cfg("models/cnx/s.pt"), {}, set(), "never_pack"),
                                (cfg("models/cnx/a.pt"), {}, {"models/cnx/a.pt"}, "git")):
        with pytest.raises(SystemExit, match=msg):
            bs.cnx_weights(c, wman, ov, tracked)
    assert "models/cnx/a.pt" not in bs.model_refs(cfg("models/cnx/a.pt"))


def test_enabled_requires_bool(caplog):
    """Повторное ревью bot-4 5bc6ad1, п. c: enabled = "false" (строка) — cnx ВЫКЛЮЧЕН с WARN, а не включён."""
    lidar_cnx._ENABLED_WARNED.clear()
    with caplog.at_level("WARNING", logger=lidar_cnx.__name__):
        for v in ("false", "true", 1, 0):
            assert not lidar_cnx.enabled({"li_cnx": {"enabled": v, "weights": ["w.pt"]}})
        assert not lidar_cnx.enabled({"li_cnx": {"enabled": "false", "weights": ["w.pt"]}})   # второй раз — без WARN
    msgs = [r.getMessage() for r in caplog.records if "не bool" in r.getMessage()]
    assert len(msgs) == 4 and "cnx ВЫКЛЮЧЕН" in msgs[0], msgs
    assert lidar_cnx.enabled({"li_cnx": {"enabled": True, "weights": ["w.pt"]}})
