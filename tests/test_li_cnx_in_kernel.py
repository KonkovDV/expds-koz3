"""Сборка ядра c5-in без S3 (kaggle/2026-11-05/li_cnx_in/make_kernel.py): правки, метаданные, разметка из файла."""
from __future__ import annotations

import collections
import glob
import importlib.util
import json
import os
import re
import unicodedata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MK = ROOT / "kaggle" / "2026-11-05" / "li_cnx_in" / "make_kernel.py"


def _mod():
    spec = importlib.util.spec_from_file_location("make_kernel_li_cnx_in", MK)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mk():
    m = _mod()
    if not m.SRC.exists():
        pytest.skip("снимок ядра bot-9 не в этом клоне")
    return m


@pytest.fixture(scope="module")
def built(mk):
    return mk.build("koz3-li-cnx-in-a", "c5-in:0,c5-in:1", 12000)


def test_kernel_compiles_and_carries_every_patch(built):
    text, _ = built
    compile(text, "koz3-li-cnx-in-a.py", "exec")
    assert "CNX_JOBS_DEFAULT = 'c5-in:0,c5-in:1'\n" in text
    assert 'iters=E("CNX_ITERS", 10 if SMOKE else 12000)' in text
    assert "GT_FILE = _find_gt_file()" in text and "def _load_gt_file(path):" in text
    assert "INV = {r: dict(region=r, las=None, marks=[], utm=None) for r in POOL}" in text
    assert "for g, c in GT_FEATS.get(_nfc(region), []):" in text
    # S3-путь снимка цел: без файла разметки ядро ведёт себя как раньше
    assert "CLI = s3_client()" in text and 'log("S3 OK")' in text
    # порядок: помощники объявлены до использования, CLASSES — до помощников
    assert text.index("CLASSES = [") < text.index("GT_CLS = ") < text.index("GT_FILE = _find_gt_file()")
    assert text.index("GT_FILE = _find_gt_file()") < text.index("if CLI is not None:") < text.index("def load_gt(region):")


def test_metadata_points_to_gt_dataset_and_prep_kernel(built):
    _, meta = built
    assert meta["id"] == "dmitrijkv/koz3-li-cnx-in-a" and meta["code_file"] == "koz3-li-cnx-in-a.py"
    assert meta["is_private"] is True and meta["enable_gpu"] is True and meta["enable_internet"] is True
    assert meta["dataset_sources"] == ["dmitrijkv/koz3-gate-box-20261104"]
    assert meta["kernel_sources"] == ["dmitrijkv/koz3-cnx-li-prep"]
    assert "koz3-s3-cred" not in json.dumps(meta)


def test_bad_jobs_and_changed_snapshot_are_refused(mk, tmp_path):
    with pytest.raises(SystemExit):
        mk.build("koz3-li-cnx-in-a", "c7-in:0", 8000)
    fake = tmp_path / "k.py"
    fake.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        mk.build("koz3-li-cnx-in-a", "c5-in:0", 8000, src=fake)


def _gt_file(tmp_path: Path) -> Path:
    nfd = unicodedata.normalize("NFD", "045_Тёсово")          # имя в NFD, как бывает на macOS
    sq = [[[3.0e6, 6.0e6], [3.0e6 + 10, 6.0e6], [3.0e6 + 10, 6.0e6 + 10], [3.0e6, 6.0e6 + 10], [3.0e6, 6.0e6]]]
    feats = [
        dict(type="Feature", geometry=dict(type="Polygon", coordinates=sq),
             properties=dict(class_name="kurgany_povrezhdennye", region_name=nfd, markup_type="li")),
        dict(type="Feature", geometry=dict(type="Polygon", coordinates=sq),
             properties=dict(class_name="kurgany_tselye", region_name=nfd, markup_type="sp_or")),   # не li
        dict(type="Feature", geometry=dict(type="Polygon", coordinates=sq),
             properties=dict(class_name="structure.poly", region_name=nfd, markup_type="li")),       # не класс сети
        dict(type="Feature", geometry=None,
             properties=dict(class_name="gorodishcha", region_name=nfd, markup_type="li")),
    ]
    p = tmp_path / "gt_train_v2.geojson"
    p.write_text(json.dumps(dict(type="FeatureCollection",
                                 crs=dict(type="name", properties=dict(name="urn:ogc:def:crs:EPSG::3857")),
                                 features=feats), ensure_ascii=False), encoding="utf-8")
    return p


def test_gt_file_loader_and_load_gt_branch(mk, built, tmp_path, monkeypatch):
    from pyproj import Transformer
    from shapely.ops import transform as stf

    text, _ = built
    classes = ["kurgany_tselye", "kurgany_povrezhdennye", "gorodishcha", "fortifikatsii", "arkhitektury"]
    ns = dict(os=os, glob=glob, Path=Path, json=json, collections=collections, CLASSES=classes)
    exec(mk.HELPERS, ns)
    p = _gt_file(tmp_path)
    monkeypatch.setenv("CNX_GT_FILE", str(p))
    assert ns["_find_gt_file"]() == p
    feats, skipped = ns["_load_gt_file"](p)
    key = unicodedata.normalize("NFC", "045_Тёсово")
    assert list(feats) == [key] and [c for _, c in feats[key]] == [1]
    assert skipped == {"structure.poly": 1}
    # настоящий load_gt из собранного ядра, ветка без S3
    m = re.search(r"\ndef load_gt\(region\):\n.*?(?=\ndef region_targets)", text, re.S)
    assert m
    ns2 = dict(META={unicodedata.normalize("NFD", "045_Тёсово"): dict(crs="EPSG:32637")}, GT_FEATS=feats, CLI=None,
               Transformer=Transformer, stf=stf, json=json, re=re, _nfc=ns["_nfc"])
    exec(m.group(0), ns2)
    out = ns2["load_gt"](unicodedata.normalize("NFD", "045_Тёсово"))
    assert len(out) == 1 and out[0][1] == 1
    back = Transformer.from_crs("EPSG:32637", "EPSG:3857", always_xy=True).transform
    g3857 = stf(back, out[0][0])
    assert abs(g3857.area - 100.0) < 1e-3 and abs(g3857.bounds[0] - 3.0e6) < 1e-3


def test_cli_writes_kernel_dir(mk, tmp_path):
    assert mk.main(["--slug", "koz3-li-cnx-in-a", "--jobs", "c5-in:0,c5-in:1", "--out", str(tmp_path)]) == 0
    d = tmp_path / "koz3-li-cnx-in-a"
    assert (d / "koz3-li-cnx-in-a.py").exists()
    assert json.loads((d / "kernel-metadata.json").read_text(encoding="utf-8"))["title"] == "koz3-li-cnx-in-a"
    with pytest.raises(SystemExit):
        mk.main(["--slug", "Bad_Slug", "--jobs", "c5-in:0", "--out", str(tmp_path)])


def _gate():
    p = ROOT / "kaggle" / "2026-11-05" / "li_cnx_in" / "gate_cnxin_ctx.py"
    spec = importlib.util.spec_from_file_location("gate_cnxin_ctx", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _oof_tree(root: Path, place=None, families=None):
    regions = {f"0{f}0_Участок_{f}": f for f in range(5)}
    fam = families or {r: r.split("_")[1] + str(f) for r, f in regions.items()}
    for k, kern in enumerate(("a", "b")):
        d = root / f"dmitrijkv_koz3-li-cnx-in-{kern}"
        d.mkdir(parents=True)
        (d / "region_fold.json").write_text(json.dumps(dict(region_fold=regions, region_family=fam), ensure_ascii=False),
                                            encoding="utf-8")
    for r, f in regions.items():
        f_out = (place or {}).get(r, f)
        d = root / ("dmitrijkv_koz3-li-cnx-in-a" if f_out < 3 else "dmitrijkv_koz3-li-cnx-in-b")
        feats = [dict(type="Feature", geometry=None, properties=dict(region_name=r, class_name=c, confidence=0.5))
                 for c in ("kurgany_povrezhdennye", "gorodishcha")]
        p = d / f"oof_cnx_c5-in_f{f_out}.geojson"
        old = json.loads(p.read_text(encoding="utf-8"))["features"] if p.exists() else []
        p.write_text(json.dumps(dict(type="FeatureCollection", features=old + feats), ensure_ascii=False), encoding="utf-8")
    return regions


def test_gate_oof_check_accepts_clean_folds(tmp_path):
    g = _gate()
    _oof_tree(tmp_path)
    rep = g.check_oof(tmp_path)
    assert rep["rows"] == {0: 2, 1: 2, 2: 2, 3: 2, 4: 2} and rep["classes"]["kurgany_povrezhdennye"] == 5


def test_gate_oof_check_refuses_leaks_and_missing_folds(tmp_path):
    g = _gate()
    _oof_tree(tmp_path / "wrong", place={"010_Участок_1": 0})          # участок фолда 1 в OOF фолда 0, фолда 1 нет
    with pytest.raises(SystemExit):
        g.check_oof(tmp_path / "wrong")
    regions = {f"0{f}0_Участок_{f}": f for f in range(5)}
    fam = {r: "одна_семья" if f in (0, 1) else r for r, f in regions.items()}
    _oof_tree(tmp_path / "split", families=fam)                         # одна семья в двух фолдах
    with pytest.raises(SystemExit):
        g.check_oof(tmp_path / "split")


def test_smoke_build_sets_mode_and_regions_only_for_smoke(mk):
    text, _ = mk.build("koz3-li-cnx-in-smoke", "c5-in:1", 12000, mode="smoke", regions="036_Тимерево,181_Гнёздово_Лесная_3")
    compile(text, "smoke.py", "exec")
    assert "CNX_MODE_DEFAULT = 'smoke'\n" in text and "CNX_REGIONS_DEFAULT = '036_Тимерево,181_Гнёздово_Лесная_3'\n" in text
    full, _ = mk.build("koz3-li-cnx-in-a", "c5-in:0", 12000)
    assert "CNX_MODE_DEFAULT = 'full'\n" in full and "CNX_REGIONS_DEFAULT = ''\n" in full
    with pytest.raises(SystemExit):
        mk.build("koz3-li-cnx-in-a", "c5-in:0", 12000, regions="036_Тимерево")      # подмножество — только smoke
    with pytest.raises(SystemExit):
        mk.build("koz3-li-cnx-in-a", "c5-in:0", 12000, mode="prep")


def test_c12_build_reads_prep_dataset_and_keeps_c5_bytes(mk):
    """c12: 12 каналов из приватного датасета; сборка c5 от этой правки не меняется (ядро f4 05.11 — sha 96a49e0a…)."""
    import hashlib
    c5, _ = mk.build("koz3-li-cnx-in-c", "c5-in:4", 12000)
    assert hashlib.sha256(c5.encode("utf-8")).hexdigest().startswith("96a49e0a4a8439bb")
    assert "TAGS5 = (\"lrm_s\", \"lrm_l\", \"slope\", \"svf_r\", \"opp_r\")" in c5      # c5 — прежние теги
    text, meta = mk.build("koz3-li-c12-in-a", "c12-in:0,c12-in:1", 12000, prep_dataset="dmitrijkv/koz3-cnx-li-prep12")
    compile(text, "c12.py", "exec")
    assert "CNX_JOBS_DEFAULT = 'c12-in:0,c12-in:1'\n" in text
    assert 'assert ch in ("c3", "c5", "c12")' in text and "ch=int(ch[1:])" in text
    assert f"TAGS5 = {mk.C12_TAGS!r}" in text and mk.C12_TAGS[:5] == ("lrm_s", "lrm_l", "slope", "svf_r", "opp_r")
    assert "(0, 1, 3, 4, 5, 6, 7, 8, 9)" in text                                       # высота — все каналы рельефа
    assert meta["dataset_sources"] == ["dmitrijkv/koz3-gate-box-20261104", "dmitrijkv/koz3-cnx-li-prep12"]
    assert meta["kernel_sources"] == [] and meta["is_private"] is True
    assert mk.FINDPREP_NEW in text and mk.FINDPREP_OLD not in text and mk.FINDPREP_OLD in c5


def test_c12_find_prep_takes_flat_dataset_by_tags(mk, tmp_path, monkeypatch):
    """05.11: Kaggle распаковал zip датасета без папки cnx_relief — c12 находит manifest по тегам, чужие manifest — нет."""
    import glob as _glob
    import json as _json
    from pathlib import Path as _Path
    text, _ = mk.build("koz3-li-c12-in-a", "c12-in:0", 12000, prep_dataset="dmitrijkv/koz3-cnx-li-prep12")
    i = text.index("def find_prep():")
    src = text[i:text.index("\n\n", i)]
    flat = tmp_path / "input" / "datasets" / "dmitrijkv" / "koz3-cnx-li-prep12"
    other = tmp_path / "input" / "datasets" / "dmitrijkv" / "koz3-gate-box-20261104"
    flat.mkdir(parents=True)
    other.mkdir(parents=True)
    (other / "manifest.json").write_text(_json.dumps({"tags": ["lrm_s"], "regions": ["x"]}), encoding="utf-8")
    (flat / "manifest.json").write_text(_json.dumps({"tags": list(mk.C12_TAGS), "regions": ["001"]}), encoding="utf-8")
    real_glob = _glob.glob
    ns = {"glob": type("G", (), {"glob": staticmethod(
        lambda pat, recursive=False: real_glob(pat.replace("/kaggle/input", str(tmp_path / "input")), recursive=recursive))}),
        "json": _json, "Path": _Path, "os": __import__("os"), "TAGS5": mk.C12_TAGS}
    exec(src, ns)
    monkeypatch.delenv("CNX_PREP_DIR", raising=False)
    assert ns["find_prep"]() == flat
    (flat / "manifest.json").unlink()
    assert ns["find_prep"]() is None


def test_c12_guards(mk):
    with pytest.raises(SystemExit):
        mk.build("koz3-li-c12-in-a", "c12-in:0", 12000)                                       # нет датасета каналов
    with pytest.raises(SystemExit):
        mk.build("koz3-li-c12-in-a", "c12-in:0,c5-in:1", 12000, prep_dataset="dmitrijkv/x")   # смешение входов
    with pytest.raises(SystemExit):
        mk.build("koz3-li-cnx-in-a", "c5-in:0", 12000, prep_dataset="dmitrijkv/x")           # датасет — только c12
