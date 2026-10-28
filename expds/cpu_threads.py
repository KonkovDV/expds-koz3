"""Число физических ядер и потоки torch для CPU-запасных путей (каналы рельефа без GPU).

Каналы рельефа на CPU (torch: oneDNN/SLEEF) дают побайтно одинаковый результат при любом числе
потоков (проверено 1 и 2 потока на 016 и 105, sha совпадает), поэтому запасной путь без GPU
можно отдавать всем физическим ядрам: на 105_Вщиж 9,4 → 6,0 с при 1 → 2 потоках.
inference/main.py ставит OMP_NUM_THREADS=8 для всего процесса; здесь поднимаем только на время
вызова. EXPDS_CPU_THREADS=N — задать явно.
"""
from __future__ import annotations

import contextlib
import glob
import math
import os


def _quota() -> int | None:
    """Ограничение CPU контейнера (cgroup v2 cpu.max / v1 cfs_quota), округлённое вверх."""
    try:
        q, p = open("/sys/fs/cgroup/cpu.max").read().split()[:2]
        if q != "max":
            return max(1, math.ceil(int(q) / int(p)))
    except (OSError, ValueError):
        pass
    try:
        q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
        p = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
        if q > 0 and p > 0:
            return max(1, math.ceil(q / p))
    except (OSError, ValueError):
        pass
    return None


def physical_cores() -> int:
    """Физические ядра среди доступных процессу (affinity), не больше квоты cgroup."""
    env = os.environ.get("EXPDS_CPU_THREADS")
    if env and env.isdigit() and int(env) > 0:
        return int(env)
    try:
        allowed = os.sched_getaffinity(0)
    except (AttributeError, OSError):
        allowed = set(range(os.cpu_count() or 1))
    cores = set()
    for cpu in allowed:
        base = f"/sys/devices/system/cpu/cpu{cpu}/topology/"
        try:
            pkg = open(base + "physical_package_id").read().strip()
            core = open(base + "core_id").read().strip()
        except OSError:
            cores.add(("cpu", cpu))
            continue
        cores.add((pkg, core))
    n = len(cores) or len(allowed) or 1
    q = _quota()
    return max(1, min(n, q) if q else n)


@contextlib.contextmanager
def torch_all_cores():
    """with torch_all_cores(): … — torch.set_num_threads(physical_cores()) на время блока."""
    import torch

    old = torch.get_num_threads()
    torch.set_num_threads(physical_cores())
    try:
        yield
    finally:
        torch.set_num_threads(old)


# для отладки: python -m expds.cpu_threads
if __name__ == "__main__":
    print(physical_cores(), "physical cores; logical", len(glob.glob("/sys/devices/system/cpu/cpu[0-9]*")))
