#!/usr/bin/env python3
"""coord watcher v2 (заменяет gpu_watcher.py и cpu_watcher.sh).
Каждые 5 мин: статусы ядер под арендой слотов → renew (RUNNING) / release + pull (COMPLETE|ERROR|CANCEL);
свободный слот → первый готовый элемент QUEUE.json["dispatch"] → lease.sh acquire → kaggle push (или box-команда).
Каждые 15 мин: check_conflicts.sh --all --log. Аренды только через lease.sh (flock). Токены не печатаются.
Элемент dispatch: {id, owner, accel: gpu|tpu|cpu|box, account: <user>|any-cpu, kernel_dir | cmd, ttl_h,
  status: ready|hold|pushed|running|done|failed, after:[ids], pull_to?, kernel?}
Запуск: nohup nice -n 10 python3 orch/watcher.py >> orch/logs/watcher.out 2>&1 &   (один экземпляр, pidfile)"""
from __future__ import annotations
import datetime as dt, fcntl, json, os, pathlib, re, shutil, subprocess, sys, tempfile, time

O = pathlib.Path(os.environ.get("KOZ3_ORCH", "/workspace/koz3/orch"))
ROOT = pathlib.Path("/workspace/koz3")
PK, LEASE = str(O / "pool_kaggle.sh"), str(O / "lease.sh")
MSK = dt.timezone(dt.timedelta(hours=3))
ACC = {"acc1": "elenakonkova", "acc2": "dmitrijkv"}
# fix 30.10 13:10: + acc1/cpu и 2-я CPU-сессия на acc1/acc2 (Kaggle допускает параллельные CPU; проверено gorheads RUNNING рядом с kurgan)
SLOTS = ["acc1/gpu-1", "acc1/gpu-2", "acc1/tpu", "acc2/gpu-1", "acc2/gpu-2", "acc2/tpu", "acc1/cpu", "acc1/cpu-2", "acc2/cpu", "acc2/cpu-2",
         "box/q1", "box/q2", "box/cpu"]
# fix 30.10 13:40 (A-exec, orch-audit): CPU-пул грузим из kaggle_pool json (только username, токены не читаем в лог) → <user>/cpu + <user>/cpu-2
POOL_JSON = ROOT / "local_ctx/ctx/kaggle_pool_4x2_t4.json"
_POOL_FALLBACK = ["usertest1234", "aaaa0000004", "aaaadg4gg4g4g"]
def _pool_users():
    try: return [n["username"] for n in json.load(open(POOL_JSON))["nodes"] if n.get("username") and n.get("accel", "cpu") in ("cpu", "", None)] or _POOL_FALLBACK
    except Exception: return _POOL_FALLBACK
for _u in _pool_users():
    SLOTS += [f"{_u}/cpu", f"{_u}/cpu-2"]
TERMINAL = ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED", "CANCELLED", "CANCEL_REQUESTED")
POLL, CONFLICT_EVERY = int(os.environ.get("WATCH_POLL", 300)), 900
DRY = "--dry" in sys.argv
procs: dict[str, subprocess.Popen] = {}


def now(): return dt.datetime.now(MSK)
def log(m):
    with open(O / "COORD_LOG.md", "a", encoding="utf-8") as f: f.write(f"- {now():%d.%m %H:%M} MSK | watcher | {m}\n")
    print(f"{now():%H:%M} {m}", flush=True)
def run(cmd, t=120):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=t); return r.returncode, (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired: return 124, "timeout"
def account(slot): a = slot.split("/")[0]; return ACC.get(a, a)
def accel(slot): s = slot.split("/")[1]; return "box" if slot.startswith("box/") else s.split("-")[0]


class Queue:
    """QUEUE.json под flock (.queue.lock): читаем/меняем dispatch атомарно."""
    def __enter__(self):
        self.fd = open(O / ".queue.lock", "w"); fcntl.flock(self.fd, fcntl.LOCK_EX)
        self.d = json.load(open(O / "QUEUE.json", encoding="utf-8")); self.d.setdefault("dispatch", []); return self.d
    def __exit__(self, *a):
        tmp = O / ".QUEUE.tmp"; json.dump(self.d, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1); os.replace(tmp, O / "QUEUE.json")
        fcntl.flock(self.fd, fcntl.LOCK_UN); self.fd.close()


def leases():
    d = json.load(open(O / "LEASES.json", encoding="utf-8")); t = time.time()
    return {x["resource"]: x for x in d.get("leases", []) if x["expires"] > t}, d.get("slots", [])


def kstatus(ref):
    rc, o = run([PK, ref.split("/")[0], "kernels", "status", ref], 90)
    m = re.search(r'status "?(?:KernelWorkerStatus\.)?(\w+)', o)
    return m.group(1).upper() if m else "UNKNOWN"


def cooldown(d, slot):
    until = d.get("cooldown", {}).get(slot)
    if not until: return False
    try: u = dt.datetime.strptime(until, "%d.%m.%Y %H:%M").replace(tzinfo=MSK)
    except ValueError: return False
    return now() < u


def watch_running():
    act, _ = leases()
    for res, L in act.items():
        if not res.startswith("slot:"): continue
        slot = res[5:]
        if slot.startswith("box/"):
            # fix 30.10: трогаем только аренды, выданные самим watcher-ом (note == id элемента dispatch);
            # ручные аренды ботов на box/* снимаются только по expiry или владельцем.
            it = next((x for x in json.load(open(O / "QUEUE.json"))["dispatch"] if x.get("slot") == slot and x["status"] == "running"), None)
            if not it or L.get("note") != it.get("id"):
                procs.pop(slot, None) if (procs.get(slot) is not None and procs[slot].poll() is not None) else None
                continue
            p = procs.get(slot)
            if p is not None:
                if p.poll() is not None: finish(slot, L, "COMPLETE" if p.returncode == 0 else f"EXIT{p.returncode}")
            else:  # после рестарта watcher-а: смотрим pid из dispatch
                it = next((x for x in json.load(open(O / "QUEUE.json"))["dispatch"] if x.get("slot") == slot and x["status"] == "running"), None)
                alive = False
                if it and it.get("pid"):
                    try: os.kill(it["pid"], 0); alive = True
                    except ProcessLookupError: pass
                if not alive: finish(slot, L, "GONE")
            continue
        k = L.get("kernel")
        if not k: continue
        st = kstatus(k)
        if st in TERMINAL:
            finish(slot, L, st)
        elif st in ("RUNNING", "QUEUED") and L["expires"] - time.time() < 3600:
            run([LEASE, "renew", res, L["owner"], "2"]); run([LEASE, "renew", f"kernel:{k}", L["owner"], "2"])


def finish(slot, L, st):
    k = L.get("kernel")
    with Queue() as d:
        it = next((x for x in d["dispatch"] if x.get("kernel") == k and x["status"] in ("pushed", "running")), None) if k else \
             next((x for x in d["dispatch"] if x.get("slot") == slot and x["status"] == "running"), None)
        if it:
            it["status"] = "done" if st == "COMPLETE" else "failed"; it["finished"] = f"{now():%d.%m %H:%M} {st}"
            if it.get("pull_to") and k and not DRY:
                dst = ROOT / it["pull_to"]; dst.mkdir(parents=True, exist_ok=True)
                rc, o = run([PK, k.split("/")[0], "kernels", "output", k, "-p", str(dst), "-q"], 1800)
                it["pulled"] = f"rc={rc}"
    if not DRY:
        run([LEASE, "release", f"slot:{slot}", "coord", "--force"])
        if k: run([LEASE, "release", f"kernel:{k}", "coord", "--force"])
    log(f"{slot}: {k or 'box'} → {st}; аренда снята" + (f"; вывод → {it['pull_to']}" if it and it.get('pull_to') else ""))


def fill_free():
    act, slots_legacy = leases()
    legacy_busy = {s["slot"] for s in slots_legacy if s.get("kernel")}
    with Queue() as d:
        done = {x["id"] for x in d["dispatch"] if x["status"] == "done"}
        for slot in SLOTS:
            if f"slot:{slot}" in act or (ac := accel(slot)) is None or slot in legacy_busy or cooldown(d, slot): continue
            acc, ac = account(slot), accel(slot)
            for it in d["dispatch"]:
                if it["status"] != "ready" or it["accel"] != ac or not set(it.get("after", [])) <= done: continue
                if ac != "box" and it["account"] not in (acc, "any-cpu" if ac == "cpu" else None): continue
                ok = push(slot, it, acc, d)
                if ok: break
                # не вышло — пробуем следующий элемент


def push(slot, it, acc, d):
    ttl = str(it.get("ttl_h", 10))
    if it["accel"] == "box":
        if DRY: log(f"DRY {slot} ← {it['id']} (box)"); return True
        rc, o = run([LEASE, "acquire", f"slot:{slot}", it["owner"], ttl, "--note", it["id"], "--force"])
        if rc: return False
        (O / "logs").mkdir(exist_ok=True)
        procs[slot] = subprocess.Popen(["nice", "-n", "10", "bash", "-lc", it["cmd"]], cwd=str(ROOT),
                                       stdout=open(O / "logs" / f"{it['id']}.log", "a"), stderr=subprocess.STDOUT)
        it.update(status="running", slot=slot, started=f"{now():%d.%m %H:%M}", pid=procs[slot].pid)
        log(f"{slot} ← {it['id']} (box, pid {procs[slot].pid})"); return True
    src = ROOT / it["kernel_dir"]; meta = json.load(open(src / "kernel-metadata.json"))
    if (it["accel"] == "gpu") != bool(meta.get("enable_gpu")) or (it["accel"] == "tpu") != bool(meta.get("enable_tpu")):
        it["status"] = "hold"; it["why"] = "kernel-metadata ускоритель ≠ accel"; log(f"{it['id']}: hold (ускоритель в metadata)"); return False
    slug = meta["id"].split("/")[1]; ref = f"{acc}/{slug}"
    # guard 30.10 13:40: 2 CPU-слота на аккаунт → тот же slug нельзя пушить поверх живого ядра (push = новая версия = отмена старой)
    if f"kernel:{ref}" in leases()[0] or kstatus(ref) in ("RUNNING", "QUEUED"):
        log(f"{slot}: skip {it['id']} — {ref} уже занят/RUNNING (второй CPU-слот)"); return False
    if DRY: log(f"DRY {slot} ← {ref} ({it['id']})"); return True
    rc, o = run([LEASE, "acquire", f"slot:{slot}", it["owner"], ttl, "--kernel", ref, "--note", it["id"]])
    if rc: log(f"{slot}: acquire отказ: {o[-160:]}"); return False
    run([LEASE, "acquire", f"kernel:{ref}", it["owner"], ttl])
    path = src
    if meta["id"] != ref:  # any-cpu: копия с id другого аккаунта
        path = pathlib.Path(tempfile.mkdtemp(prefix="kpush_")); shutil.copytree(src, path, dirs_exist_ok=True)
        meta["id"] = ref; json.dump(meta, open(path / "kernel-metadata.json", "w"))
    rc, o = run([PK, acc, "kernels", "push", "-p", str(path)], 600)
    last = (o.splitlines() or [""])[-1][:200]
    if "successfully" in o or "Kernel version" in o:
        it.update(status="pushed", kernel=ref, slot=slot, started=f"{now():%d.%m %H:%M}")
        log(f"{slot} ← {ref} ({it['id']}): {last}"); return True
    run([LEASE, "release", f"slot:{slot}", it["owner"]]); run([LEASE, "release", f"kernel:{ref}", it["owner"]])
    if re.search(r"quota|exceed|limit", o, re.I):
        nxt = next(r for r in ("31.10.2026 03:00", "07.11.2026 03:00", "14.11.2026 03:00")
                   if dt.datetime.strptime(r, "%d.%m.%Y %H:%M").replace(tzinfo=MSK) > now())
        d.setdefault("cooldown", {})[slot] = nxt
        it.setdefault("errors", []).append(f"{now():%d.%m %H:%M} quota → {slot} пауза до {nxt}")
    else:
        it.setdefault("errors", []).append(f"{now():%d.%m %H:%M} {last}")
        if len(it["errors"]) >= 3: it["status"] = "failed"
    log(f"{slot}: push {ref} не удался: {last}"); return False


def main():
    pid = O / ".watcher.pid"
    if pid.exists() and not DRY:
        try: os.kill(int(pid.read_text()), 0); sys.exit("watcher уже запущен")
        except (ProcessLookupError, ValueError): pass
    if not DRY: pid.write_text(str(os.getpid()))
    log(f"старт v2 (poll {POLL}s, conflicts 15 мин{', DRY' if DRY else ''})")
    last_c = 0
    while True:
        try:
            watch_running(); fill_free()
            if time.time() - last_c >= CONFLICT_EVERY:
                rc, o = run([str(O / "check_conflicts.sh"), "--all", "--log"], 600); last_c = time.time()
                print(o.splitlines()[-1] if o else "", flush=True)
        except Exception as e:  # watcher не должен падать
            log(f"ошибка цикла: {type(e).__name__}: {str(e)[:200]}")
        if DRY: break
        time.sleep(POLL)


if __name__ == "__main__":
    main()
