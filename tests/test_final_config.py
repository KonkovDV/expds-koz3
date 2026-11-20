"""Финальная версия решения — конфигурация третьей попытки финала (19.11.2026, 22:29 МСК).

`configs/default.toml` заморожен побайтно: тот же файл лежит в сданном архиве как `configs/default.toml`,
а его sha256 печатает протокол каждого прогона («конфиг SHA256 …»). Любая правка конфига меняет архив
решения, поэтому тест падает, пока не обновлены FROZEN_SHA256, ARCHIVE_SHA256 и docs/README.md.
"""
from __future__ import annotations

import hashlib
import importlib.util
import tomllib

import pytest

from conftest import ROOT

#: sha256 конфига в сданном архиве (configs/default.toml архива).
FROZEN_SHA256 = "3e00af39e2fe44baa51502fca4ac12ff6470508351fde22575dfb4da873f353b"
#: sha256 сданного архива: scripts/build_submission.py без аргументов, веса по models/MANIFEST.sha256.
ARCHIVE_SHA256 = "171bed3024aa6b55252c0f6b70116952e815a806bb29271d0f26e9737f8b7252"
ARCHIVE_MEMBERS = 178

CFG_PATH = ROOT / "configs" / "default.toml"
DISKS = {"radius_m": {"arkhitektury": 0.05, "object_poly": 0.05, "fortifikatsii": 0.05}, "anchor": "rp"}


def _cfg() -> dict:
    return tomllib.loads(CFG_PATH.read_text(encoding="utf-8"))


def _builder():
    spec = importlib.util.spec_from_file_location("build_submission", ROOT / "scripts" / "build_submission.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _manifest() -> dict[str, str]:
    rows = (ROOT / "models" / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines()
    return {path: sha for sha, path in (row.split() for row in rows if row.strip())}


def test_config_is_byte_identical_to_the_submitted_one():
    assert hashlib.sha256(CFG_PATH.read_bytes()).hexdigest() == FROZEN_SHA256


def test_repository_keeps_a_single_configuration():
    assert sorted(p.name for p in (ROOT / "configs").glob("*.toml")) == ["default.toml"]


def test_lidar_kurgans_are_the_ensemble_of_ten_convnext_networks():
    cnx = _cfg()["li_cnx"]
    assert cnx["enabled"] is True and cnx["decode"] == "head" and cnx["replace_li_kurgans"] is True
    assert cnx["weights"] == [f"models/cnx/weights_cnx_c5-in_f{k}.pt" for k in range(5)]
    assert cnx["weights_b"] == [f"models/cnx/weights_cnx_c12-in_f{k}.pt" for k in range(5)]
    assert cnx["threshold_group"] == "li_cnxin"


def test_final_post_processing_and_gray_maggr_threshold():
    cfg = _cfg()
    assert cfg["post"]["micro_disks"] == DISKS
    assert cfg["post"]["micro_circles"]["radius_m"] == {"kurgany_povrezhdennye": 0.5, "kurgany_tselye": 0.5}
    assert cfg["thresholds_by_markup"]["maggr_pix_grhi"] == {"finds_points": 0.22}


def test_inference_precision_is_the_one_thresholds_were_selected_in():
    assert _cfg()["infer"] == {"amp": False, "tf32": False, "deterministic": True}


def test_manifest_lists_exactly_the_weights_the_config_uses():
    cfg = _cfg()
    manifest = set(_manifest())
    used = set(_builder().model_paths(cfg))
    pix_dir = cfg["geophys"]["pixgbm"]["model_dir"].rstrip("/") + "/"
    assert used <= manifest, sorted(used - manifest)
    unused = sorted(p for p in manifest if p not in used and not p.startswith(pix_dir))
    assert not unused, unused


def test_default_build_reproduces_the_submitted_archive(tmp_path):
    """Сборка по умолчанию даёт сданный архив до байта (если веса, vendor/ и колесо RVT на месте)."""
    missing = [p for p in _manifest() if not (ROOT / p).is_file()]
    if missing or not (ROOT / "vendor" / "lleaves").is_dir() or not any((ROOT / "inference" / "cnx" / "wheels").glob("*.whl")):
        pytest.skip("веса, vendor/ и колесо RVT в git не хранятся (models/README.md)")
    builder = _builder()
    files = builder.collect(_cfg(), CFG_PATH.resolve())
    out = tmp_path / "submission.zip"
    try:
        sha = builder.write_zip(files, out)
    finally:
        out.unlink(missing_ok=True)       # 1,4 ГБ не оставляем в каталоге pytest
    assert len(files) == ARCHIVE_MEMBERS
    assert sha == ARCHIVE_SHA256
