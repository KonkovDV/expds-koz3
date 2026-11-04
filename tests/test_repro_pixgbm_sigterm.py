"""Linux fork repro: old PIXGBM pool hangs on SIGTERM; eb268d6 and v3 do not."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "repro_pixgbm_sigterm.py"


@pytest.mark.skipif(sys.platform == "win32", reason="fork is Linux-only")
@pytest.mark.skipif(
    any(subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[1]), "cat-file", "-e", f"{rev}:expds/geo_pixgbm.py"],
                       capture_output=True).returncode for rev in ("67284ab", "8594842", "af97af0", "eb268d6", "36af6b7", "78af57f")),
    reason="old PIXGBM revisions are not in this clone (expds/geo_pixgbm.py is held out of main)")
def test_sigterm_hangs_on_old_pool_and_not_on_fix():
    r = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    rows = json.loads(r.stdout)
    by = {row["commit"]: row for row in rows}
    assert by["67284ab"]["result"] == "hang"
    assert by["67284ab"]["handler_ran"] is True
    assert by["8594842"]["result"] == "hang"
    assert by["af97af0"]["result"] == "no_fork_pool"
    assert by["eb268d6"]["result"] == "clean"
    assert by["eb268d6"]["handler_ran"] is False
    assert by["36af6b7"]["result"] == "clean"
    assert by["78af57f"]["result"] == "clean"
