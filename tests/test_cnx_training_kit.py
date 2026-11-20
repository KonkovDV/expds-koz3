"""Обучение ConvNeXt-лидара воспроизводимо из репозитория: вендор, фолды и сборщик ядра (c5-in, c12-in).

Десять сетей курганов лидара финальной версии обучены ядрами, собранными 05.11 из обучающего ядра 01.11
правками 0–9. Здесь проверяется, что training/lidar_convnext собирает те же ядра из файлов репозитория: вендор
совпадает со своим манифестом (vendor_manifest.json), фолды семей не пересекаются, правки c12 применяются ровно
там, где нужно, а теги каналов обучения совпадают с тегами инференса.
"""
from __future__ import annotations

import ast
import base64
import hashlib
import importlib.util
import io
import json
import re
import tarfile

import pytest

from conftest import ROOT

KIT = ROOT / "training" / "lidar_convnext"


def _load(name: str, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mk():
    return _load("cnx_make_kernel", KIT / "make_kernel.py")


def _vendor_files(text: str) -> dict[str, bytes]:
    m = re.search(r"^VENDOR_B64 = '([A-Za-z0-9+/=]+)'\n", text, re.M)
    assert m, "VENDOR_B64 не встроен"
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(m.group(1))), mode="r:gz") as tf:
        return {x.name: tf.extractfile(x).read() for x in tf.getmembers() if x.isfile()}


def _folds(text: str) -> dict:
    m = re.search(r"^FOLDS = __import__\('json'\)\.loads\(r'''(.*?)'''\)\n", text, re.M | re.S)
    assert m, "FOLDS не встроены"
    return json.loads(m.group(1))


def _parse_jobs(text: str):
    """parse_jobs собранного ядра — без запуска остального кода ядра."""
    fn = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == "parse_jobs")
    ns: dict = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<parse_jobs>", "exec"), ns)
    return ns["parse_jobs"]


def _tags(text: str) -> tuple:
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "TAGS5" for t in n.targets))
    return ast.literal_eval(node.value)


def test_training_vendor_is_byte_identical_to_the_manifest(mk):
    manifest = json.loads((KIT / "vendor" / "vendor_manifest.json").read_text(encoding="utf-8"))["sha256"]
    sources = mk.vendor_sources()
    assert sorted(sources) == sorted(manifest)
    for name, path in sources.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest[name], name


def test_relief_vendor_is_shared_with_inference(mk):
    """Каналы рельефа обучения и инференса считает один и тот же код (inference/cnx/vendor)."""
    sources = mk.vendor_sources()
    for name in ("expds/lidar.py", "expds/accel.py", "relief_ext.py", "expds_core.abi3.so"):
        assert sources[name] == ROOT / "inference" / "cnx" / "vendor" / name


def test_family_folds_do_not_leak():
    folds = json.loads((KIT / "folds_li_59.json").read_text(encoding="utf-8"))
    fam, clu = folds["family"], folds["cluster"]
    assert len(fam) == 59 and set(fam) == set(clu)
    fold_of: dict[str, int] = {}
    for f in folds["folds"]:
        for family in f["test_families"]:
            assert family not in fold_of, f"семья {family} в двух фолдах"
            fold_of[family] = f["fold"]
    assert sorted({f["fold"] for f in folds["folds"]}) == [0, 1, 2, 3, 4]
    assert set(fam.values()) <= set(fold_of)
    for region, family in fam.items():         # кластер соседних участков целиком в одном фолде
        same = [r for r, c in clu.items() if c == clu[region]]
        assert len({fold_of[fam[r]] for r in same}) == 1, region


def test_c5_kernel_embeds_vendor_folds_and_s3_free_ground_truth(mk):
    text = mk.build("c5-in:0")
    compile(text, "kernel_c5.py", "exec")
    assert "# @@EMBED@@" not in text
    assert "CNX_JOBS_DEFAULT = 'c5-in:0'\n" in text and "CNX_HEAD_DEFAULT = 'centerhead'\n" in text
    files = _vendor_files(text)
    manifest = json.loads(files["vendor_manifest.json"])["sha256"]
    assert sorted(set(files) - {"vendor_manifest.json"}) == sorted(manifest)
    for name, data in files.items():
        if name != "vendor_manifest.json":
            assert hashlib.sha256(data).hexdigest() == manifest[name], name
    assert _folds(text) == json.loads((KIT / "folds_li_59.json").read_text(encoding="utf-8"))
    assert 'os.environ.setdefault("CNX_WORKERS", "0")' in text           # правка 1, как в ядрах 05.11
    assert "def _load_gt_file(path):" in text and "if CLI is None:" in text   # правки 3–5
    assert _parse_jobs(text)("c5-in:0") == [dict(variant="c5-in", ch=5, init="in", fold=0)]
    assert _tags(text) == ("lrm_s", "lrm_l", "slope", "svf_r", "opp_r")
    with pytest.raises(AssertionError):
        _parse_jobs(text)("c12-in:0")                                     # c12 только в своём ядре


def test_c12_kernel_reads_twelve_channels(mk):
    text = mk.build("c12-in:3", prep_dataset="owner/koz3-cnx-li-prep12")
    compile(text, "kernel_c12.py", "exec")
    assert _parse_jobs(text)("c12-in:3") == [dict(variant="c12-in", ch=12, init="in", fold=3)]
    assert _tags(text) == mk.C12_TAGS and len(mk.C12_TAGS) == 12
    assert mk.HEIGHT_NEW in text and mk.FINDPREP_NEW in text
    assert sorted(_vendor_files(text)) == sorted(_vendor_files(mk.build("c5-in:0")))


@pytest.mark.parametrize("kwargs", [
    dict(jobs="c12-in:0,c5-in:0", prep_dataset="o/x"),      # разные входы в одном ядре
    dict(jobs="c12-in:0"),                                   # c12 без каналов prep12
    dict(jobs="c5-in:0", prep_dataset="o/x"),                # каналы c12 для c5
    dict(jobs="c5-in:5"),                                    # фолдов пять: 0–4
    dict(jobs="c5-in:0", regions="001_x"),                   # подмножество участков — только smoke
    dict(jobs="c12-in:0", prep_dataset="o/x", mode="prep"),  # c12 строит prep12.py
])
def test_builder_refuses_inconsistent_requests(mk, kwargs):
    with pytest.raises(SystemExit):
        mk.build(**kwargs)


def test_channel_tags_agree_between_training_and_inference(mk):
    prep12 = _load("cnx_prep12", KIT / "prep12.py")
    ds12 = _load("cnx_make_dataset12", KIT / "make_dataset12.py")
    cnx_prep = _load("cnx_prep_infer", ROOT / "inference" / "cnx" / "cnx_prep.py")
    relief_ext = _load("cnx_relief_ext", ROOT / "inference" / "cnx" / "vendor" / "relief_ext.py")
    assert mk.C12_TAGS == prep12.TAGS12 == ds12.TAGS12 == cnx_prep.TAGS12
    assert mk.C12_TAGS == relief_ext.BASE_TAGS + relief_ext.EXTRA_TAGS
    assert mk.C12_TAGS[:5] == cnx_prep.TAGS5


def test_dataset12_manifest_vendor_is_what_the_kernel_checks(mk):
    """manifest.json входов c12 несёт sha вендора каналов в форме RELIEF_SHA ядра — иначе ядро их не примет."""
    ds12 = _load("cnx_make_dataset12_v", KIT / "make_dataset12.py")
    vend, full = ds12.relief_sha()
    assert vend == {k: v for k, v in full.items() if k.startswith("expds") or k == "relief_ext.py"}
    assert set(vend) == {"expds/__init__.py", "expds/accel.py", "expds/candidates.py", "expds/crs.py", "expds/lidar.py",
                         "expds_core.abi3.so", "relief_ext.py"}
