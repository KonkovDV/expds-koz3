#!/usr/bin/env python3
"""Show the PIXGBM fork-pool SIGTERM hang, and that eb268d6 / v3 do not hang.

Old pool (67284ab, and 8594842 which failed on region 217): children inherit the
parent SIGTERM handler. pool.terminate() sends SIGTERM. If that handler returns
without exiting — watchdog.arm_signal does this when the write lock is already
held — pool.join() waits forever.

eb268d6 and v3 (36af6b7, dd58b4c, 78af57f; make_pool text is identical) install
SIG_DFL in the child and block SIGTERM in the parent only for the duration of
fork. terminate() then kills the children.

af97af0 has no fork pool in prob_map, so this hang cannot happen there. It is
still an older v2 candidate and must not be submitted.

Linux only (multiprocessing fork). Each case is a subprocess with a timeout.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# v3 line: 36af6b7 is the v3 tree named in STATUS; 78af57f is the later
# nomodels tip. make_pool/_pool_init hashes match eb268d6.
CASES = (
    ("67284ab", "plain", 6.0),
    ("8594842", "plain", 6.0),
    ("af97af0", "no_pool", 3.0),
    ("eb268d6", "make_pool", 8.0),
    ("36af6b7", "make_pool", 8.0),
    ("78af57f", "make_pool", 8.0),
)


def _funcs(rev: str) -> str:
    src = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{rev}:expds/geo_pixgbm.py"], text=True)
    chunks = []
    for name in ("_pool_init", "make_pool"):
        m = re.search(rf"^def {name}\(.*?(?=^def |\Z)", src, re.M | re.S)
        if not m:
            raise SystemExit(f"{rev}: no {name}")
        chunks.append(m.group(0))
    return "\n".join(chunks)


def _has_fork_pool(rev: str) -> bool:
    src = subprocess.check_output(["git", "-C", str(ROOT), "show", f"{rev}:expds/geo_pixgbm.py"], text=True)
    return 'get_context("fork").Pool' in src


CHILD = r'''
import os, signal, sys, time, threading
MODE = sys.argv[1]
MARK = sys.argv[2]
FUNCS = sys.argv[3]
if MODE == "no_pool":
    print("no_pool", flush=True)
    raise SystemExit(0)
ns = {}
exec(compile(open(FUNCS, encoding="utf-8").read(), FUNCS, "exec"), ns)
lock = threading.Lock()
lock.acquire()
def on_term(signum, frame):
    with open(MARK, "a", encoding="utf-8") as fh:
        fh.write("handler %s\n" % os.getpid())
    # Same branch as watchdog.arm_signal when _write_lock is already held.
    if not lock.acquire(blocking=False):
        return
    os._exit(0)
def slow(x):
    time.sleep(30)
    return x
signal.signal(signal.SIGTERM, on_term)
if MODE == "plain":
    import multiprocessing as mp
    pool = mp.get_context("fork").Pool(2)
else:
    pool = ns["make_pool"](2)
pool.map_async(slow, range(4))
time.sleep(0.3)
t0 = time.time()
pool.terminate()
pool.join()
dt = time.time() - t0
with open(MARK, "a", encoding="utf-8") as fh:
    fh.write("joined %.4f\n" % dt)
print("joined %.4f" % dt, flush=True)
'''


def run_case(rev: str, mode: str, timeout: float) -> dict:
    if mode == "plain" and not _has_fork_pool(rev):
        mode = "no_pool"
    rec = {"commit": rev, "mode": mode, "timeout_s": timeout}
    if mode == "no_pool":
        rec["fork_pool"] = _has_fork_pool(rev)
        rec["result"] = "no_fork_pool" if not rec["fork_pool"] else "unexpected_pool"
        rec["hung"] = False
        return rec
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        mark = td_path / "mark.txt"
        funcs = td_path / "funcs.py"
        if mode == "make_pool":
            funcs.write_text(_funcs(rev), encoding="utf-8")
        else:
            funcs.write_text("# plain\n", encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, "-c", textwrap.dedent(CHILD), mode, str(mark), str(funcs)],
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        hung = False
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            hung = True
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5)
        stdout = proc.stdout.read() if proc.stdout else ""
        stderr = proc.stderr.read() if proc.stderr else ""
        rec.update(
            hung=hung,
            returncode=proc.returncode,
            stdout=stdout[-500:],
            stderr=stderr[-500:],
            mark=mark.read_text(encoding="utf-8") if mark.is_file() else "",
        )
        rec["handler_ran"] = "handler " in rec["mark"]
        rec["joined"] = (not hung) and rec["returncode"] == 0 and "joined" in stdout
        if mode == "plain":
            rec["result"] = "hang" if hung and rec["handler_ran"] else "unexpected"
        else:
            rec["result"] = "clean" if rec["joined"] and not rec["handler_ran"] else "unexpected"
        return rec


def main() -> None:
    if sys.platform == "win32":
        print("fork repro is Linux-only", file=sys.stderr)
        sys.exit(2)
    rows = [run_case(*case) for case in CASES]
    text = json.dumps(rows, ensure_ascii=False, indent=2)
    print(text)
    out = os.environ.get("PIXGBM_SIGTERM_OUT")
    if out:
        Path(out).write_text(text + "\n", encoding="utf-8")
    bad = [r for r in rows if r["result"] in ("unexpected", "unexpected_pool")]
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
