"""bot-8: область сопоставления по умолчанию — global везде (метрика, scoreboard, env.sh), site только явно."""
import os
import re
from pathlib import Path

from expds import metric as M

ROOT = Path(__file__).resolve().parents[1]


def test_default_scope_is_global(monkeypatch):
    monkeypatch.delenv("EXPDS_MATCH_SCOPE", raising=False)
    assert M.match_scope() == "global"
    assert M.match_scope(None) == "global"
    monkeypatch.setenv("EXPDS_MATCH_SCOPE", "")
    assert M.match_scope() == "global"


def test_site_is_explicit_and_loud(monkeypatch, capsys):
    monkeypatch.setenv("EXPDS_MATCH_SCOPE", "site")
    monkeypatch.setattr(M, "_SCOPE_WARNED", [])
    assert M.match_scope() == "site"
    assert "scope=site" in capsys.readouterr().err


def test_no_code_path_sets_site_scope():
    """Ни один скрипт/модуль (кроме тестов) не включает site сам: ни scope="site", ни EXPDS_MATCH_SCOPE=site."""
    bad = []
    for p in list(ROOT.glob("expds/**/*.py")) + list(ROOT.glob("scripts/**/*.py")) + list(ROOT.glob("inference/**/*.py")) \
            + list(ROOT.glob("**/*.sh")) + list(ROOT.glob("**/*.bat")) + list(ROOT.glob("**/*.ps1")):
        t = p.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"""[(,]\s*scope\s*=\s*["']site["']|EXPDS_MATCH_SCOPE["']?\]?\s*=\s*["']?site|setdefault\(\s*["']EXPDS_MATCH_SCOPE""", t):
            bad.append(str(p.relative_to(ROOT)))
    assert not bad, bad
