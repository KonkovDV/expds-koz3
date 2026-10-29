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

    def fake(xb, amp):
        calls.append(amp)
        z = real(xb, False)
        return z * float("nan") if amp else z
    seg.amp = True
    monkeypatch.setattr(seg, "_logits", fake)
    p = seg.predict(np.zeros((3, 32, 32), np.float32))
    assert calls == [True, False] and np.isfinite(p).all() and seg.nan_fallbacks == 1
    monkeypatch.setattr(seg, "_logits", lambda xb, amp: real(xb, False) * float("nan"))
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


def test_lidar_feats_fallbacks_equal_cnx_off(region, weights, monkeypatch):
    base = pipeline.lidar_feats(region, {}, TimeGuard(1000), {}, ROOT)
    assert base, "запасной детектор должен дать курганы на синтетике"
    # 1) веса не грузятся
    cfg = _cfg(Path("/nonexistent/w.pt"))
    cache: dict = {}
    assert _key(pipeline.lidar_feats(region, cfg, TimeGuard(1000), cache, ROOT)) == _key(base)
    assert cache["li_cnx"] is None
    # 2) подготовка падает
    monkeypatch.setattr(lidar_cnx, "run_prep", lambda *a, **k: (_ for _ in ()).throw(lidar_cnx.CnxError("boom")))
    cache = {}
    assert _key(pipeline.lidar_feats(region, _cfg(weights["cnx3"]), TimeGuard(1000), cache, ROOT)) == _key(base)
    assert cache["li_cnx"] is not None
    monkeypatch.undo()
    # 3) инференс падает произвольным исключением
    monkeypatch.setattr(lidar_cnx.CnxSegmenter, "predict", lambda *a, **k: 1 / 0)
    assert _key(pipeline.lidar_feats(region, _cfg(weights["cnx3"]), TimeGuard(1000), {}, ROOT)) == _key(base)
    monkeypatch.undo()
    # 4) постобработка (rescore) падает на cnx — откат
    real = pipeline._li_with_model

    def li_with(seg, *a, **k):
        if seg.meta.get("li_source") == "cnx":
            raise ValueError("post")
        return real(seg, *a, **k)
    monkeypatch.setattr(pipeline, "_li_with_model", li_with)
    assert _key(pipeline.lidar_feats(region, _cfg(weights["cnx3"]), TimeGuard(1000), {}, ROOT)) == _key(base)


def test_lidar_feats_soft_ceiling_skips_cnx(region, weights, monkeypatch):
    base = pipeline.lidar_feats(region, {}, TimeGuard(1000), {}, ROOT)
    g = TimeGuard(1000)
    g.set_soft(soft_ceiling_s=50)            # soft_left ≈ 50 < soft_need_s 120
    called = []
    monkeypatch.setattr(lidar_cnx, "bind_las", lambda *a, **k: called.append(1))
    out = pipeline.lidar_feats(region, _cfg(weights["cnx3"]), g, {}, ROOT)
    assert not called and _key(out) == _key(base)
    assert any(s["phase"] == "cnx" for s in g.soft_skipped)


@pytest.mark.parametrize("kind", ["cnx3", "cnx5"])
def test_lidar_feats_cnx_end_to_end(region, weights, kind, monkeypatch, tmp_path):
    """Настоящая подготовка подпроцессом + крошечная сеть: курганы по кандидатам ЦМР с вероятностью cnx."""
    monkeypatch.setenv("EXPDS_TMP", str(tmp_path / "t"))
    cache: dict = {}
    seen = []
    real = lidar_cnx.bind_las
    monkeypatch.setattr(lidar_cnx, "bind_las", lambda *a, **k: seen.append(real(*a, **k)) or seen[-1])
    out = pipeline.lidar_feats(region, _cfg(weights[kind], classes=["kurgany_tselye", "kurgany_povrezhdennye"]),
                               TimeGuard(1000), cache, ROOT)
    assert len(seen) == 1 and seen[0].probs.shape[0] == 5
    assert seen[0].meta["in_kind"] == kind
    assert out and all(f["class_name"] in ("kurgany_tselye", "kurgany_povrezhdennye") for f in out)
    assert all(f.get("markup_type") == "li" for f in out)
    assert list((tmp_path / "t").iterdir()) == []


def test_lidar_feats_threshold_group_option(region, weights, monkeypatch):
    probs = np.full((5, 200, 200), 0.9, np.float32)

    def fake_bind(seg, las, crs, opts, **k):
        from rasterio.transform import from_origin
        return lidar_cnx.CnxBound(seg, probs, from_origin(500_000.0, 6_100_100.0, 0.5, 0.5), {"crs": UTM})
    monkeypatch.setattr(lidar_cnx, "bind_las", fake_bind)
    cfg = _cfg(weights["cnx3"], classes=["kurgany_tselye"], threshold_group="li_cnx_test")
    out = pipeline.lidar_feats(region, cfg, TimeGuard(1000), {}, ROOT)
    assert out and {f.get("threshold_group") for f in out} == {"li_cnx_test"}


def test_vendor_manifest_matches_files():
    """inference/cnx/MANIFEST.json — sha256 всех файлов вендора (их сверяет check_submission.py)."""
    d = ROOT / "inference" / "cnx"
    m = json.loads((d / "MANIFEST.json").read_text(encoding="utf-8"))["sha256"]
    have = {p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file() and p.name != "MANIFEST.json"
            and "__pycache__" not in p.parts}
    assert set(m) == have
    for rel, sha in m.items():
        assert hashlib.sha256((d / rel).read_bytes()).hexdigest() == sha, rel
    assert m["cnx_prep.py"].startswith("77d30749")
    assert m["wheels/rvt_py-2.2.1-py3-none-any.whl"].startswith("f3b171e0")
