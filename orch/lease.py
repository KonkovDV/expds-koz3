#!/usr/bin/env python3
"""Аренды KOZ3 (вызывать через orch/lease.sh — он держит flock на orch/.leases.lock).

Ресурсы (строка вида <тип>:<id>):
  slot:<acc>/<gpu-1|gpu-2|tpu|cpu>   slot:acc1/gpu-1, slot:usertest1234/cpu, slot:box/cpu
  kernel:<owner>/<slug>               kernel:elenakonkova/koz3-w1-dfine-ae-final
  dataset:<owner>/<slug>              dataset:usertest1234/koz3-w3-kurgan-cpu-data
  branch:<name>                       branch:bots/w1-dfine-ae
Владелец: bot-W1-aeor | bot-W2-lidar | bot-W3-bigobj | bot-W4-gate | bot-W5-rerank | coord (бот 1).
Хранилище: orch/LEASES.json, раздел "leases" (+ зеркалирование holder/kernel/until в "slots" для slot:*).
Время — epoch и строка MSK; TTL в часах. Истёкшая аренда = свободна для acquire, но check_conflicts её покажет.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import sys
import tempfile

ORCH = os.environ.get("KOZ3_ORCH", "/workspace/koz3/orch")
PATH = os.path.join(ORCH, "LEASES.json")
MSK = dt.timezone(dt.timedelta(hours=3))
OWNERS = re.compile(r"^(coord|bot-W[1-9][0-9]?(-[a-z0-9]+)?)$")
RES = re.compile(r"^(slot|kernel|dataset|branch):\S+$")
PREFIX = {"bot-W1": "koz3-w1-", "bot-W2": "koz3-w2-", "bot-W3": "koz3-w3-", "bot-W4": "koz3-w4-", "bot-W5": "koz3-w5-"}


def now() -> float:
    return dt.datetime.now(MSK).timestamp()


def msk(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts, MSK).strftime("%d.%m %H:%M MSK")


def load() -> dict:
    with open(PATH, encoding="utf-8") as f:
        return json.load(f)


def save(d: dict) -> None:
    fd, tmp = tempfile.mkstemp(dir=ORCH, prefix=".LEASES.", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)
        f.write("\n")
    os.replace(tmp, PATH)


def log(msg: str) -> None:
    with open(os.path.join(ORCH, "COORD_LOG.md"), "a", encoding="utf-8") as f:
        f.write(f"- {msk(now())} | lease | {msg}\n")


def active(d: dict):
    t = now()
    return [x for x in d.setdefault("leases", []) if x.get("expires", 0) > t]


def find(d: dict, res: str):
    return [x for x in d.setdefault("leases", []) if x["resource"] == res]


def owner_base(owner: str) -> str:
    m = re.match(r"^(bot-W\d+)", owner)
    return m.group(1) if m else owner


def check_name(res: str, owner: str) -> str | None:
    """Префикс koz3-w<N>- для ядер/датасетов и bots/w<N>- для веток своего потока."""
    base = owner_base(owner)
    if base == "coord":
        return None
    kind, rid = res.split(":", 1)
    n = base.split("W")[1]
    if kind in ("kernel", "dataset"):
        slug = rid.split("/")[-1]
        if not slug.startswith(PREFIX.get(base, f"koz3-w{n}-")):
            return f"{kind} {rid}: у {owner} префикс должен быть {PREFIX.get(base)}"
    if kind == "branch" and not rid.startswith(f"bots/w{n}-"):
        return f"branch {rid}: у {owner} ветка должна быть bots/w{n}-*"
    return None


def sync_slot(d: dict, lease: dict | None, slot: str, released_by: str | None = None) -> None:
    for s in d.get("slots", []):
        if s.get("slot") == slot:
            if lease is None:
                s.update({"holder": None, "kernel": None, "state": "свободен",
                          "last": f"снято {released_by} {msk(now())}"})
                s.pop("until", None); s.pop("since", None)
            else:
                s.update({"holder": lease["owner"], "kernel": lease.get("kernel"), "since": msk(lease["since"]),
                          "until": msk(lease["expires"]), "purpose": lease.get("note", s.get("purpose"))})
                s.pop("state", None)
            return
    if lease is not None:
        d.setdefault("slots", []).append({"slot": slot, "holder": lease["owner"], "kernel": lease.get("kernel"),
                                          "since": msk(lease["since"]), "until": msk(lease["expires"])})


def cmd_acquire(res, owner, ttl_h, note="", kernel=None, force=False):
    if not RES.match(res):
        sys.exit(f"bad resource {res!r}")
    if not OWNERS.match(owner):
        sys.exit(f"bad owner {owner!r}")
    err = check_name(res, owner)
    if err and not force:
        sys.exit("ОТКАЗ: " + err)
    d = load()
    t = now()
    cur = [x for x in find(d, res) if x["expires"] > t]
    if cur and cur[0]["owner"] != owner:
        x = cur[0]
        sys.exit(f"ЗАНЯТО: {res} у {x['owner']} до {msk(x['expires'])} ({x.get('note','')})")
    d["leases"] = [x for x in d["leases"] if x["resource"] != res]
    lease = {"resource": res, "owner": owner, "since": cur[0]["since"] if cur else t,
             "expires": t + float(ttl_h) * 3600, "note": note, "host": os.uname().nodename, "pid": os.getppid()}
    if kernel:
        lease["kernel"] = kernel
    d["leases"].append(lease)
    if res.startswith("slot:"):
        sync_slot(d, lease, res[5:])
    save(d)
    log(f"acquire {res} → {owner} до {msk(lease['expires'])} {note}")
    print(f"OK {res} → {owner} до {msk(lease['expires'])}")


def cmd_renew(res, owner, ttl_h):
    d = load()
    cur = find(d, res)
    if not cur or cur[0]["owner"] != owner:
        sys.exit(f"НЕТ АРЕНДЫ {res} у {owner}")
    cur[0]["expires"] = now() + float(ttl_h) * 3600
    if res.startswith("slot:"):
        sync_slot(d, cur[0], res[5:])
    save(d)
    print(f"OK renew {res} до {msk(cur[0]['expires'])}")


def cmd_set_kernel(res, owner, kernel):
    """P2: привязать ядро к уже взятой аренде слота (без смены since/expires)."""
    d = load(); cur = [x for x in find(d, res) if x["expires"] > now()]
    if not cur or (cur[0]["owner"] != owner and owner != "coord"):
        sys.exit(f"НЕТ АРЕНДЫ {res} у {owner}")
    cur[0]["kernel"] = kernel
    if res.startswith("slot:"):
        sync_slot(d, cur[0], res[5:])
    save(d); log(f"set-kernel {res} → {kernel} by {owner}"); print(f"OK set-kernel {res} → {kernel}")


def cmd_release(res, owner, force=False):
    d = load()
    cur = find(d, res)
    if not cur:
        sys.exit(f"нет аренды {res}")
    if cur[0]["owner"] != owner and not (force and owner == "coord"):
        sys.exit(f"ОТКАЗ: {res} у {cur[0]['owner']}, снимать может владелец или coord --force")
    d["leases"] = [x for x in d["leases"] if x["resource"] != res]
    if res.startswith("slot:"):
        sync_slot(d, None, res[5:], owner)
    save(d)
    log(f"release {res} ({cur[0]['owner']}) by {owner}")
    print(f"OK release {res}")


def cmd_list(all_=False):
    d = load()
    t = now()
    rows = d.get("leases", []) if all_ else active(d)
    for x in sorted(rows, key=lambda x: x["resource"]):
        flag = "" if x["expires"] > t else "  [ИСТЕКЛА]"
        print(f"{x['resource']:<55} {x['owner']:<16} до {msk(x['expires'])}{flag}  {x.get('note','')}")
    if not rows:
        print("(аренд нет)")


def cmd_who(res):
    d = load()
    cur = [x for x in find(d, res) if x["expires"] > now()]
    print(cur[0]["owner"] if cur else "FREE")
    sys.exit(0 if cur else 3)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("acquire"); a.add_argument("resource"); a.add_argument("owner"); a.add_argument("ttl_h", type=float)
    a.add_argument("--note", default=""); a.add_argument("--kernel"); a.add_argument("--force", action="store_true")
    r = sp.add_parser("renew"); r.add_argument("resource"); r.add_argument("owner"); r.add_argument("ttl_h", type=float)
    e = sp.add_parser("release"); e.add_argument("resource"); e.add_argument("owner"); e.add_argument("--force", action="store_true")
    sk = sp.add_parser("set-kernel"); sk.add_argument("resource"); sk.add_argument("owner"); sk.add_argument("kernel")
    li = sp.add_parser("list"); li.add_argument("--all", action="store_true")
    w = sp.add_parser("who"); w.add_argument("resource")
    g = ap.parse_args()
    if g.cmd == "acquire":
        cmd_acquire(g.resource, g.owner, g.ttl_h, g.note, g.kernel, g.force)
    elif g.cmd == "renew":
        cmd_renew(g.resource, g.owner, g.ttl_h)
    elif g.cmd == "release":
        cmd_release(g.resource, g.owner, g.force)
    elif g.cmd == "set-kernel":
        cmd_set_kernel(g.resource, g.owner, g.kernel)
    elif g.cmd == "list":
        cmd_list(g.all)
    else:
        cmd_who(g.resource)
