"""Item (6) (spec §5): DTMs of all LAS files of a run built in worker processes while the main process does relief + GPU.

ground.make_dtm is pure numpy/scipy and deterministic, so a worker returns exactly the arrays the sequential call would
(byte-equal). Scheduling: longest-processing-time first (point count) under a RAM budget; a job larger than the whole
budget runs alone. With the largest file at ~58 % of all points the DTM stage cannot drop below DTM(largest): wall ≈
max(DTM_largest, ΣDTM / N) and the main process overlaps relief/GPU of the smaller files with it.

Memory model (per job peak RSS, bytes): BYTES_PER_CELL · cells + BYTES_PER_POINT · min(points, 5 M) + BYTES_WORKER.
Box fit (fresh process, one DTM build, bench/cpu_a_speedup/pool_calib.txt): 016 2.8 Mcell/0.27 Mpt → 266 MB,
005 2.5/2.7 → 450, 040 4.7/2.1 → 494, 001 6.4/7.4 → 678, 022 13.5/8.9 → 1120 MB; the model gives 280/469/~510/910/1364
(conservative everywhere). The returned dtm+valid (5 B/cell) stay counted until the consumer takes them.

Workers are plain `python -m expds.lidar_a.dtm_workers` subprocesses (not multiprocessing): spawn/forkserver re-import the
caller's __main__ in every worker (torch and the models for run.py: +1 s and +0.3 GB each on the box), fork is unsafe
once the consumer holds CUDA / OpenMP pools. Arrays come back as .npy files in a tmpfs directory (/dev/shm when present).
"""
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

BYTES_PER_CELL = 64
BYTES_PER_POINT = 84            # one 5 M-point laspy chunk + x/y/z/ix/iy/k temporaries
CHUNK_POINTS = 5_000_000        # ground.STREAM_POINTS
BYTES_WORKER = 80 << 20
RESULT_BYTES_PER_CELL = 5
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS")


def plan(path):
    """→ (points, cells) from the LAS header (same grid as ground.make_dtm)."""
    from . import ground
    mins, maxs, n = ground.header_box(path)
    g = ground.grid_of(mins, maxs)
    return int(n), int(g.rows * g.cols)


def est_bytes(points, cells):
    return BYTES_PER_CELL * cells + BYTES_PER_POINT * min(points, CHUNK_POINTS) + BYTES_WORKER


def _worker_main(outdir):
    """Worker loop: stdin 'i\tpath' → ground.make_dtm → outdir/i_d.npy, i_v.npy; stdout one JSON line per job."""
    from . import ground
    for line in sys.stdin:
        i, path = line.rstrip("\n").split("\t", 1)
        t0, c0 = time.perf_counter(), time.process_time()
        try:
            d, v, origin = ground.make_dtm(path)
            np.save(os.path.join(outdir, f"{i}_d.npy"), d); np.save(os.path.join(outdir, f"{i}_v.npy"), v)
            del d, v
            msg = {"i": int(i), "origin": list(origin), "wall": time.perf_counter() - t0, "cpu": time.process_time() - c0, "pid": os.getpid()}
        except Exception as e:  # noqa: BLE001 — reported to the consumer
            msg = {"i": int(i), "error": f"{type(e).__name__}: {e}"}
        sys.stdout.write(json.dumps(msg) + "\n"); sys.stdout.flush()


class DtmPool:
    """Iterate (index, path, (dtm, valid, origin), stats) as DTMs complete; workers keep building while the consumer runs.

    workers: process count (platform: e.g. 6 of 24 cores, the rest for relief/torch); ram_budget: bytes for all DTM jobs
    plus finished-but-unconsumed results (platform: 96 GB − main process − safety; box: ≤ 2 GB in total).
    ordered=True yields in the input order (identical side effects to the sequential loop), False in completion order.
    """

    def __init__(self, paths, workers=2, ram_budget=1 << 30, worker_threads=1, ordered=True, tmpdir=None):
        self.paths = [str(p) for p in paths]
        self.workers, self.budget, self.ordered = max(1, min(int(workers), len(self.paths) or 1)), int(ram_budget), ordered
        self.sizes = [plan(p) for p in self.paths]
        self.order = sorted(range(len(self.paths)), key=lambda i: -self.sizes[i][0])     # LPT by point count
        self.worker_threads, self.tmpdir = worker_threads, tmpdir
        self._q = queue.Queue(); self._lock = threading.Condition(); self._used = 0
        self.max_used = 0

    def _release(self, b):
        with self._lock:
            self._used -= b; self._lock.notify_all()

    def _serve(self, proc, todo):
        """One thread per worker process: take the next job that fits the budget, wait for its result."""
        while True:
            with self._lock:
                while True:
                    if not todo:
                        return
                    i = todo[0]; need = est_bytes(*self.sizes[i])
                    if self._running == 0 or self._used + need <= self.budget:
                        break           # with nothing running a job always starts (progress; over-budget runs alone)
                    self._lock.wait(0.1)
                todo.pop(0); self._used += need; self._running += 1; self.max_used = max(self.max_used, self._used)
            proc.stdin.write(f"{i}\t{self.paths[i]}\n"); proc.stdin.flush()
            line = proc.stdout.readline()
            msg = json.loads(line) if line else {"i": i, "error": "worker died"}
            keep = RESULT_BYTES_PER_CELL * self.sizes[i][1]
            with self._lock:
                self._running -= 1
            self._release(need - keep)
            self._q.put((i, msg, keep))

    def __iter__(self):
        own = self.tmpdir is None
        base = "/dev/shm" if own and os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else None
        outdir = tempfile.mkdtemp(prefix="lidar_a_dtm_", dir=base) if own else self.tmpdir
        env = dict(os.environ, **{k: str(self.worker_threads) for k in THREAD_VARS})
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        procs = [subprocess.Popen([sys.executable, "-m", "expds.lidar_a.dtm_workers", outdir], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True, env=env, cwd=root) for _ in range(self.workers)]
        self._running = 0
        todo = list(self.order)
        ths = [threading.Thread(target=self._serve, args=(p, todo), daemon=True) for p in procs]
        try:
            for t in ths:
                t.start()
            ready, nxt = {}, 0
            for _ in range(len(self.paths)):
                while True:
                    if self.ordered and nxt in ready:
                        i = nxt; break
                    item = self._q.get()
                    ready[item[0]] = item
                    if not self.ordered:
                        i = item[0]; break
                _, msg, keep = ready.pop(i)
                nxt += 1
                if "error" in msg:
                    raise RuntimeError(f"dtm worker failed on {self.paths[i]}: {msg['error']}")
                fd, fv = (os.path.join(outdir, f"{i}_{k}.npy") for k in "dv")
                d, v = np.load(fd), np.load(fv)
                os.unlink(fd); os.unlink(fv)
                self._release(keep)    # the consumer owns the arrays now (counted by the caller's own reserve)
                yield i, self.paths[i], (d, v, tuple(msg["origin"])), msg
        finally:
            with self._lock:
                todo.clear(); self._lock.notify_all()
            for p in procs:
                try:
                    p.stdin.close()
                except OSError:
                    pass
            for p in procs:
                try:
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    p.kill()
            if own:
                shutil.rmtree(outdir, ignore_errors=True)


if __name__ == "__main__":
    _worker_main(sys.argv[1])
