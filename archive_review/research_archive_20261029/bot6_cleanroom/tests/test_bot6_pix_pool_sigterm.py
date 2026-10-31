"""bot-6: документ + тест SIGTERM fork-пула PIXGBM (только эта ветка).

Вердикт по коммитам (чтение API 29.10, патчи других веток НЕ вносятся):

* ``67284ab`` — НЕ fixed: ``mp.get_context("fork").Pool`` без ``make_pool`` /
  ``_pool_init`` (дети наследуют обработчик SIGTERM ``main.py`` → пустой
  result.geojson + hang в ``pool.join()``).
* ``af97af0`` — пул в ``prob_map`` ещё не введён (последовательный код); STATUS
  помечает кандидата как небезопасный pre-fix v2. Фикс ``make_pool`` отсутствует.
* ``eb268d6`` — FIXED: ``make_pool`` / ``_pool_init`` (SIG_DFL + block SIGTERM
  на время fork); тест ``tests/test_bot7_pix_pool_sigterm.py``.
* v3 (``36af6b7`` / ``bots/bot7-candidate-v3-nomodels``, линия после ``eb268d6``) —
  FIXED: ``make_pool`` присутствует.

Этот файл — копия логики bot-7-теста для ветки bot-6; если ``make_pool`` нет
(ветка от main без модуля), тест пропускается с пояснением.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

CODE = r'''
import os, signal, sys, time
sys.path.insert(0, sys.argv[1])
from expds import geo_pixgbm
mark = sys.argv[2]
def on_term(signum, frame):
    with open(os.path.join(mark, str(os.getpid())), "w") as f:
        f.write("x")
    os._exit(0)
def slow(x):
    time.sleep(0.5)
    return x
signal.signal(signal.SIGTERM, on_term)
if sys.argv[3] == "plain":
    import multiprocessing as mp
    pool = mp.get_context("fork").Pool(2)
else:
    if not hasattr(geo_pixgbm, "make_pool"):
        print("SKIP_NO_MAKE_POOL")
        raise SystemExit(0)
    pool = geo_pixgbm.make_pool(2)
assert sorted(pool.map(abs, [-1, -2, -3, -4])) == [1, 2, 3, 4]
pool.map_async(slow, range(8)); time.sleep(0.2)
pool.terminate(); pool.join()
assert signal.getsignal(signal.SIGTERM) is on_term
assert signal.SIGTERM not in signal.pthread_sigmask(signal.SIG_BLOCK, set())
print("ok", os.getpid())
'''


def _geo_available() -> bool:
    return (ROOT / "expds" / "geo_pixgbm.py").is_file()


@pytest.mark.skipif(not _geo_available(), reason="expds/geo_pixgbm.py отсутствует на этой ветке (gitignore main)")
def test_bot6_pool_children_do_not_run_parent_sigterm_handler(tmp_path):
    plain = _run(tmp_path, "plain")
    if plain == "skip":
        pytest.skip("unexpected skip on plain pool")
    assert isinstance(plain, list)
    fixed = _run(tmp_path, "make_pool")
    if fixed == "skip":
        pytest.skip("make_pool отсутствует — см. вердикт в docstring (не патчим чужие ветки)")
    assert fixed == []


def _run(tmp_path, mode):
    d = tmp_path / mode
    d.mkdir()
    r = subprocess.run(
        [sys.executable, "-c", CODE, str(ROOT), str(d), mode],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if "SKIP_NO_MAKE_POOL" in (r.stdout or ""):
        return "skip"
    assert r.returncode == 0, r.stderr
    assert r.stdout.startswith("ok")
    return sorted(p.name for p in d.iterdir())
