import pytest
"""bot-7: fork-пул PIXGBM не наследует обработчик SIGTERM main.py (watchdog.arm_signal).

Без этого pool.terminate() шлёт детям SIGTERM, ребёнок исполняет обработчик родителя (пишет пустой
result.geojson) и может повиснуть; родитель висит в pool.join() (work5/check_8594842_217.log)."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CODE = r'''
import os, signal, sys, time
sys.path.insert(0, sys.argv[1])
geo_pixgbm = pytest.importorskip("expds.geo_pixgbm", reason="expds/geo_pixgbm.py not in main (EN-similarity hold, docs/MERGE_2026-10-29.md)")
mark = sys.argv[2]
def on_term(signum, frame):
    with open(os.path.join(mark, str(os.getpid())), "w") as f:
        f.write("x")
    os._exit(0)
def slow(x):
    time.sleep(0.5)
    return x
signal.signal(signal.SIGTERM, on_term)
if sys.argv[3] == "plain":   # контроль: обычный fork-пул — дети исполняют обработчик родителя
    import multiprocessing as mp
    pool = mp.get_context("fork").Pool(2)
else:
    pool = geo_pixgbm.make_pool(2)
assert sorted(pool.map(abs, [-1, -2, -3, -4])) == [1, 2, 3, 4]
pool.map_async(slow, range(8)); time.sleep(0.2)            # дети заняты в момент terminate, как в prob_map
pool.terminate(); pool.join()
assert signal.getsignal(signal.SIGTERM) is on_term          # обработчик родителя на месте
assert signal.SIGTERM not in signal.pthread_sigmask(signal.SIG_BLOCK, set())   # маска восстановлена
print("ok", os.getpid())
'''


def _run(tmp_path, mode):
    d = tmp_path / mode
    d.mkdir()
    r = subprocess.run([sys.executable, "-c", CODE, str(ROOT), str(d), mode],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert r.stdout.startswith("ok")
    return sorted(p.name for p in d.iterdir())


def test_pool_children_do_not_run_parent_sigterm_handler(tmp_path):
    assert _run(tmp_path, "plain")                             # контроль: проблема воспроизводится
    assert _run(tmp_path, "make_pool") == []                   # ни один ребёнок не вызвал обработчик родителя
