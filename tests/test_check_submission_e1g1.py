"""04.11: чистая геофизика E1_G1 ([geophys.pixgbm] impl=clean, chain=e1g1, expds/pix_e1g1.py).

Проверка сабмита писалась под прежнюю реализацию: требовала expds/geo_pixgbm.py и models/pix_clean/finds.txt и валила
готовую чистую сборку (`submission_pixgbm_clean_e1g1_p4_lleaves.zip`, sha256 ed51cae4…) двумя ложными FAIL.
Теперь для e1g1 проверяются её собственные модули и шесть моделей ens6 + ORG3, а EN-модуль не нужен.
"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEMBERS = ("n40r8", "n40r6", "n40r10", "aug", "gw1d", "gw3d")
E1G1 = {"enabled": True, "impl": "clean", "chain": "e1g1", "model_dir": "models/pix_clean", "classes": ["finds_points"],
        "maps": ["maggr"], "e1g1": {"backend": "lleaves", "lleaves_target": "x86-64-v2"}}


def _cs():
    spec = importlib.util.spec_from_file_location("check_submission_e1g1", ROOT / "scripts" / "check_submission.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.RESULTS.clear()
    return m


def _names(drop=()):
    n = {"expds/pix_e1g1.py", "expds/pix_clean.py", "expds/pix_org_et.py",
         "models/pix_clean/org/org_etgr_map.txt", "models/pix_clean/org/org_etgr_prof.txt"}
    for m in MEMBERS:
        n |= {f"models/pix_clean/ens6/{m}/finds.txt", f"models/pix_clean/ens6/{m}/meta.json",
              f"models/pix_clean/ens6/{m}/finds.lleaves.x86-64-v2.o"}
    return n - set(drop)


def test_members_match_module_constant():
    cs = _cs()
    assert cs.E1G1_MEMBERS == MEMBERS


def test_e1g1_complete_pack_passes_without_en_module():
    cs = _cs()
    owned = cs.check_pixgbm_static(E1G1, _names())
    cs.check_pixgbm_module(E1G1, _names())
    assert not [r for r in cs.RESULTS if r[0] in ("FAIL", "WARN")], cs.RESULTS
    assert any(r[0] == "PASS" and "e1g1" in r[2] for r in cs.RESULTS)
    assert "models/pix_clean/ens6/gw3d/finds.txt" in owned and "models/pix_clean/org/org_etgr_map.txt" in owned


def test_e1g1_missing_member_or_module_fails():
    cs = _cs()
    cs.check_pixgbm_static(E1G1, _names(drop={"models/pix_clean/ens6/aug/finds.txt"}))
    assert any(r[0] == "FAIL" and "ens6/aug/finds.txt" in r[2] for r in cs.RESULTS)
    cs.RESULTS.clear()
    cs.check_pixgbm_static(E1G1, _names(drop={"expds/pix_e1g1.py"}))
    assert any(r[0] == "FAIL" and "pix_e1g1.py" in r[2] for r in cs.RESULTS)


def test_e1g1_missing_lleaves_only_warns():
    cs = _cs()
    cs.check_pixgbm_static(E1G1, _names(drop={f"models/pix_clean/ens6/{m}/finds.lleaves.x86-64-v2.o" for m in MEMBERS}))
    assert not [r for r in cs.RESULTS if r[0] == "FAIL"]
    assert any(r[0] == "WARN" and "lleaves" in r[2] for r in cs.RESULTS)


def test_probe_loads_ens6_models():
    cs = _cs()
    md, need, extra = cs.pixgbm_needed(E1G1)
    assert md == "models/pix_clean" and need == [f"models/pix_clean/ens6/{m}/finds.txt" for m in MEMBERS] and extra == []


def test_en_line_still_requires_its_module():
    cs = _cs()
    cs.check_pixgbm_module({"enabled": True}, {"inference/main.py"})
    assert any(r[0] == "FAIL" and "pix_own.py" in r[2] for r in cs.RESULTS)
