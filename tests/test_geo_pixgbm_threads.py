"""PIXGBM: потоки и процессы по доступным ядрам (sched_getaffinity), env-приоритет, process_t0 без psutil."""
import builtins
import os
import sys

import pytest

from expds import geo_pixgbm as gp


@pytest.fixture
def clean_env(monkeypatch):
    for k in ("EXPDS_PIXGBM_THREADS", "EXPDS_THREADS"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_threads_follow_affinity(clean_env):
    clean_env.setattr(os, "sched_getaffinity", lambda pid: {0, 1}, raising=False)
    clean_env.setattr(os, "cpu_count", lambda: 64)
    assert gp.avail_cores() == 2
    assert gp.pix_threads() == 2
    assert gp.pix_procs({}, gp.pix_threads()) == 2
    assert gp.pix_procs({"procs": 0}, 8) == 2          # и при явных потоках пул не шире доступных ядер
    assert gp.pix_procs({"procs": 3}, 2) == 3          # явное procs в конфиге — как задано


def test_threads_cap_and_fallback(clean_env):
    clean_env.setattr(os, "sched_getaffinity", lambda pid: set(range(32)), raising=False)
    assert gp.pix_threads() == 8                       # потолок 8
    assert gp.pix_procs(None, 8) == 4                  # пул признаков ≤ 4
    clean_env.delattr(os, "sched_getaffinity", raising=False)
    clean_env.setattr(os, "cpu_count", lambda: 3)
    assert gp.avail_cores() == 3                       # нет sched_getaffinity → os.cpu_count()

    def boom(pid):
        raise OSError("нет")
    clean_env.setattr(os, "sched_getaffinity", boom, raising=False)
    clean_env.setattr(os, "cpu_count", lambda: None)
    assert gp.avail_cores() == 1


def test_env_overrides(clean_env):
    clean_env.setattr(os, "sched_getaffinity", lambda pid: {0, 1}, raising=False)
    clean_env.setenv("EXPDS_THREADS", "5")
    assert gp.pix_threads() == 5
    clean_env.setenv("EXPDS_PIXGBM_THREADS", "3")
    assert gp.pix_threads() == 3                       # EXPDS_PIXGBM_THREADS важнее EXPDS_THREADS
    clean_env.setenv("EXPDS_PIXGBM_THREADS", "x")
    assert gp.pix_threads() == 5                       # не число → следующий ключ


def test_rate_uses_procs():
    gp.PHASE["rate"].clear()
    base = gp.PRIOR_S_PER_MPX["lightgbm"][("maggr", "full")]
    assert gp._rate("maggr", "full", 8) == pytest.approx(base / 4)
    assert gp._rate("maggr", "full", 8, "lightgbm", 2) == pytest.approx(base / 2)


def test_process_t0_without_psutil(monkeypatch):
    real = builtins.__import__

    def no_psutil(name, *a, **k):
        if name == "psutil":
            raise ImportError("нет psutil")
        return real(name, *a, **k)
    monkeypatch.delitem(sys.modules, "psutil", raising=False)
    monkeypatch.setattr(builtins, "__import__", no_psutil)
    assert gp.process_t0() == gp._T_IMPORT


def test_model_dir_relative_to_root(tmp_path, monkeypatch):
    from pathlib import Path
    root = Path(gp.__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)                        # платформа может запускать не из корня архива
    assert gp.resolve_model_dir("configs") == root / "configs"
    assert gp.resolve_model_dir("нет/такого") == Path("нет/такого")
    (tmp_path / "configs").mkdir()
    assert gp.resolve_model_dir("configs") == Path("configs")   # есть от cwd — как раньше
    assert gp.resolve_model_dir(str(tmp_path)) == tmp_path
