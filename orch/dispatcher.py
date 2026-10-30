#!/usr/bin/env python3
"""KOZ3 orch dispatcher: board.yaml → Kaggle. Идемпотентный, с dry-run, лок-файлом и логом.

Цикл: опрос статусов → обновление state.json → освобождение аренд → оценка квоты →
пуш следующей задачи с наибольшим priority, если слот свободен и (остаток − est_hours) ≥ safety_margin.
Никогда не останавливает и не перезапускает RUNNING/QUEUED ядра. Токены не печатает.

  python orch/dispatcher.py --dry-run      # ничего не пушит
  python orch/dispatcher.py                # live
"""
from __future__ import annotations
import argparse, datetime as dt, fcntl, hashlib, json, logging, os, re, shutil, subprocess, sys, tempfile
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
BOARD = ROOT / "orch" / "board.yaml"
STATE = Path(os.environ.get("ORCH_STATE", ROOT / "orch" / "state.json"))
KAGGLE = os.environ.get("KAGGLE_BIN", os.path.expanduser("~/.local/bin/kaggle"))
LIVE = {"RUNNING", "QUEUED"}
log = logging.getLogger("orch")


def token_env(var: str) -> dict:
    v = (os.environ.get(var) or "").strip()
    if not v:
        raise RuntimeError(f"нет {var}")
    e = {k: x for k, x in os.environ.items() if not k.startswith("KAGGLE_JSON")}
    if v.startswith("{"):
        j = json.loads(v)
        e.update(KAGGLE_USERNAME=j["username"], KAGGLE_KEY=j["key"])
    else:
        e["KAGGLE_API_TOKEN"] = v
    return e


def kg(acc: dict, *args, t=120) -> tuple[int, str]:
    try:
        r = subprocess.run([KAGGLE, *args], env=token_env(acc["token_env"]), capture_output=True, text=True, timeout=t)
        return r.returncode, (r.stdout + r.stderr)
    except subprocess.TimeoutExpired:
        return 124, "timeout"


def kstatus(acc, slug) -> str:
    rc, out = kg(acc, "kernels", "status", f"{acc['user']}/{slug}", t=60)
    m = re.search(r'status "(?:KernelWorkerStatus\.)?([A-Z_]+)"', out)
    if m:
        return m.group(1)
    return "ABSENT" if ("404" in out or "not found" in out.lower()) else "UNKNOWN"


def week_start(now):
    return (now - dt.timedelta(days=(now.weekday() - 5) % 7)).replace(hour=0, minute=0, second=0, microsecond=0)


def recent_kernels(acc, since):
    """[(slug, lastRunUTC)] своих ядер с запуском после since."""
    rc, out = kg(acc, "kernels", "list", "--mine", "--page-size", "100", "--sort-by", "dateRun", "--csv")
    res = []
    for line in out.strip().splitlines()[1:]:
        r = line.split(",")
        if len(r) < 4 or not r[0].startswith(acc["user"] + "/"):
            continue
        try:
            lr = dt.datetime.fromisoformat(r[3].split(".")[0])
        except ValueError:
            continue
        if lr >= since:
            res.append((r[0].split("/", 1)[1], lr))
    return res


def accel_of(acc, slug, cache: Path) -> str:
    d = cache / acc["user"] / slug
    mf = d / "kernel-metadata.json"
    if not mf.exists():
        d.mkdir(parents=True, exist_ok=True)
        kg(acc, "kernels", "pull", f"{acc['user']}/{slug}", "-p", str(d), "-m", t=60)
    try:
        m = json.loads(mf.read_text())
    except Exception:
        return "CPU"
    ms = str(m.get("machine_shape") or "").lower()
    if m.get("enable_tpu") or "tpu" in ms:
        return "TPU"
    if m.get("enable_gpu") or any(x in ms for x in ("gpu", "t4", "p100")):
        return "GPU"
    return "CPU"


def runtime_h(acc, slug, status, lastrun, now, cache: Path, st_cache: dict) -> float:
    if status == "RUNNING":
        return (now - lastrun).total_seconds() / 3600
    if status not in ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED"):
        return 0.0
    key = f"{acc['user']}/{slug}@{lastrun.isoformat()}"
    if key in st_cache:
        return st_cache[key]
    d = cache / acc["user"] / slug
    d.mkdir(parents=True, exist_ok=True)
    lf = d / f"{slug}.log"
    if lf.exists():
        lf.unlink()
    kg(acc, "kernels", "output", f"{acc['user']}/{slug}", "-p", str(d), "--file-pattern", r".*\.log$", t=180)
    h = 0.0
    try:
        ts = re.findall(r'"time":\s*([0-9.]+)', lf.read_text(errors="ignore"))
        h = float(ts[-1]) / 3600 if ts else 0.0
    except Exception:
        pass
    st_cache[key] = h
    return h


def dir_hash(d: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(d.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(d).as_posix().encode()); h.update(p.read_bytes())
    return h.hexdigest()[:16]


def preflight(acc, meta) -> str | None:
    for ds in meta.get("dataset_sources") or []:
        rc, out = kg(acc, "datasets", "files", ds, "--page-size", "1", t=60)
        if "403" in out or "don't have access" in out or "404" in out:
            return f"нет доступа к датасету {ds}"
    for ks in meta.get("kernel_sources") or []:
        if kstatus(acc, ks.split("/", 1)[1]) == "ABSENT" and not ks.startswith(acc["user"] + "/"):
            return f"нет доступа к ядру-источнику {ks}"
    return None


def push(acc, task, dry) -> tuple[bool, str]:
    src = ROOT / task["dir"]
    meta = json.loads((src / "kernel-metadata.json").read_text())
    meta["id"] = f"{acc['user']}/{task['kernel']}"
    meta["title"] = task["kernel"]
    why = preflight(acc, meta)
    if why:
        return False, why
    if dry:
        return True, "dry-run: пуш пропущен"
    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / task["kernel"]
        shutil.copytree(src, dst)
        (dst / "kernel-metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
        rc, out = kg(acc, "kernels", "push", "-p", str(dst), t=300)
    ok = rc == 0 and "error" not in out.lower()
    return ok, out.strip().splitlines()[-1][:300] if out.strip() else f"rc={rc}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-pushes", type=int, default=4)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    lockf = open(STATE.with_suffix(".lock"), "w")
    try:
        fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log.info("другой экземпляр работает — выход"); return 0

    board = yaml.safe_load(BOARD.read_text())
    state = json.loads(STATE.read_text()) if STATE.exists() else {"tasks": {}, "runtime_cache": {}, "log": []}
    pol, q = board["policy"], board["quota"]
    now = dt.datetime.utcnow()
    wk = week_start(now)
    cache = STATE.parent / "cache"
    accs = board["accounts"]

    # 1) квота и занятость слотов (по ВСЕМ ядрам аккаунта, не только с доски)
    usage = {}
    for name, acc in accs.items():
        if not os.environ.get(acc["token_env"]):
            log.warning("%s: нет токена %s — аккаунт пропущен", name, acc["token_env"]); continue
        u = {"GPU": 0.0, "TPU": 0.0, "live": {"GPU": [], "TPU": []}}
        for slug, lr in recent_kernels(acc, wk - dt.timedelta(days=2)):
            st = kstatus(acc, slug)
            ac = accel_of(acc, slug, cache)
            if ac == "CPU":
                continue
            if st in LIVE:
                u["live"][ac].append(slug)
            if lr >= wk:
                u[ac] += runtime_h(acc, slug, st, lr, now, cache, state["runtime_cache"])
        usage[name] = u
        log.info("%s: GPU %.1f/%s ч live=%s | TPU %.1f/%s ч live=%s [ОЦЕНКА]", acc["user"], u["GPU"], q["GPU"]["hours"],
                 u["live"]["GPU"], u["TPU"], q["TPU"]["hours"], u["live"]["TPU"])

    # 2) статусы задач доски
    tasks = {t["id"]: t for t in board["tasks"]}
    for t in board["tasks"]:
        s = state["tasks"].setdefault(t["id"], {"status": "pending", "retries": 0})
        if t["resource"] == "cpu-box":
            s.setdefault("status", "pending"); continue
        res = s.get("pushed_to") or t["resource"]
        acc_name = res.split("@")[1]
        if acc_name not in usage:
            continue
        ks = kstatus(accs[acc_name], t["kernel"])
        prev = s["status"]
        if ks in LIVE:
            s["status"] = ks.lower()
        elif ks == "COMPLETE" and (t.get("external") or prev in ("running", "queued", "pushed")):
            s["status"] = "complete"
        elif ks in ("ERROR", "CANCEL_ACKNOWLEDGED") and prev in ("running", "queued", "pushed"):
            s["retries"] += 1
            s["status"] = "failed" if s["retries"] > pol["max_retries"] else "retry"
        elif ks == "COMPLETE" and prev == "pending" and t.get("external") is None:
            pass  # старая версия ядра — не считаем выполнением новой
        s["kaggle"] = ks
        if prev != s["status"]:
            log.info("%s: %s → %s (kaggle %s)", t["id"], prev, s["status"], ks)

    def done(i):
        return state["tasks"].get(i, {}).get("status") == "complete"

    # 3) диспетчеризация
    pushes = 0
    cand = sorted((t for t in board["tasks"] if not t.get("external") and t["resource"] != "cpu-box"
                   and state["tasks"][t["id"]]["status"] in ("pending", "retry")), key=lambda t: -t["priority"])
    plan = []
    for t in cand:
        s = state["tasks"][t["id"]]
        if not all(done(d) for d in t.get("depends_on") or []):
            plan.append(f"{t['id']}: ждёт {t.get('depends_on')}"); continue
        if s.get("pushed_hash") == dir_hash(ROOT / t["dir"]) and s["status"] != "retry":
            continue
        reason = []
        for res in [t["resource"], *(t.get("alt_resources") or [])]:
            kind, acc_name = res.split("@"); K = kind.upper()
            if acc_name not in usage:
                reason.append(f"{res}: нет токена"); continue
            u = usage[acc_name]
            if len(u["live"][K]) >= q[K]["concurrent"]:
                reason.append(f"{res}: слоты заняты {u['live'][K]}"); continue
            left = q[K]["hours"] - u[K]
            if left - t.get("est_hours", 1) < q["safety_margin_h"]:
                reason.append(f"{res}: квота ≈{left:.1f} ч"); continue
            if pushes >= a.max_pushes:
                reason.append("лимит пушей за цикл"); break
            acc = accs[acc_name]
            if kstatus(acc, t["kernel"]) in LIVE:
                reason.append(f"{res}: уже идёт"); continue
            ok, msg = push(acc, t, a.dry_run)
            log.info("%s → %s: %s %s", t["id"], res, "OK" if ok else "SKIP", msg)
            if not ok:
                reason.append(f"{res}: {msg}"); continue
            if not a.dry_run:
                s.update(status="pushed", pushed_to=res, pushed_hash=dir_hash(ROOT / t["dir"]),
                         pushed_at=now.isoformat() + "Z")
                u["live"][K].append(t["kernel"]); u[K] += 0  # время учтётся при следующем опросе
            pushes += 1
            reason = [f"→ {res}" + (" (dry-run)" if a.dry_run else "")]
            break
        plan.append(f"{t['id']}: " + "; ".join(reason))
    for p in plan:
        log.info("план: %s", p)
    state["last_run"] = now.isoformat() + "Z"
    state["last_plan"] = plan
    state["usage"] = {k: {"GPU_h": round(v["GPU"], 2), "TPU_h": round(v["TPU"], 2), "live": v["live"]} for k, v in usage.items()}
    if not a.dry_run:
        STATE.write_text(json.dumps(state, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
