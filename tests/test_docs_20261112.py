"""Живые документы не спорят с configs/default.toml и не являются копиями друг друга."""
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PHRASE = "не собирает архив 0,43274"


def _text(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def _cfg() -> dict:
    return tomllib.loads((ROOT / "configs/default.toml").read_text(encoding="utf-8"))


def test_default_toml_is_the_build_the_docs_describe():
    cfg = _cfg()
    assert cfg["li_cnx"]["enabled"] is False
    assert cfg["hillfort"]["enabled"] is True
    assert cfg["rerank_arch"]["enabled"] is False
    assert cfg["geophys"]["pixgbm"]["enabled"] is False
    assert cfg["sp_dfine"]["enabled"] is False
    assert cfg["dfine_aeor"]["enabled"] is False
    assert cfg["gor_gate"]["enabled"] is False
    assert cfg["verify"]["enabled"] is True
    assert cfg["speed"]["enabled"] is False
    assert cfg["post"]["micro_circles"]["radius_m"]["kurgany_povrezhdennye"] == 0.5
    paths = [item["path"] for item in cfg["models"]["sp_or"]]
    assert paths == [
        "models/spor_gor_r18.pt",
        "models/spor_focal_r18.pt",
        "models/surf_arch_r18.pt",
    ]
    assert cfg["models"]["or"]["path"] == "models/surf_arch_r18.pt"
    assert cfg["verify"]["weights"] == "models/sp_crop.pt"


def test_living_docs_name_the_same_two_builds():
    for name in (
        "README.md",
        "docs/README.md",
        "docs/README_FINAL.md",
        "docs/ARCHITECTURE.md",
        "docs/DOCS.md",
    ):
        assert PHRASE in _text(name), name
    models = _text("models/README.md")
    assert "surf_arch_r18.pt" in models
    assert "в конфиге не стоит" in models
    final = _text("docs/README_FINAL.md")
    assert "--config" not in final
    assert "pantum/n7q:20261024" in final


def test_one_live_plan_and_it_is_linked_from_the_entry_points():
    plan = "docs/plans/AI_PLAN_20261112.md"
    body = _text(plan)
    assert "0,41024" in body and "D1" in body and "T1" in body
    for name in ("README.md", "docs/DOCS.md", "docs/AI_MASTER_PROMPT_061.md", "docs/plans/PLAN_TO_061_20261105.md"):
        assert "AI_PLAN_20261112.md" in _text(name), name
    assert "VERIFY_2026-11-12.md" in _text("README.md")


def test_declaration_is_not_a_second_copy_of_the_journal():
    summary = _text("docs/ai_declaration.md")
    journal = _text("docs/AI_DECLARATION.md")
    assert summary != journal
    assert "AI_DECLARATION.md" in summary
    assert "ai_declaration.md" in journal
    assert "only_known_classes" not in summary
    assert "_ФИО_" in summary
