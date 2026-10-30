#!/usr/bin/env python3
"""Детектор конфликтов KOZ3. Запуск: orch/check_conflicts.sh [--kaggle] [--git] [--all] [--log]
ERROR → код выхода 1. Токены не печатаются (Kaggle — только через pool_kaggle.sh)."""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import subprocess
import sys

ORCH = os.environ.get("KOZ3_ORCH", "/workspace/koz3/orch")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "/workspace/koz3/repo"
WT = "/workspace/koz3/wt"
MSK = dt.timezone(dt.timedelta(hours=3))
ACCS = ["elenakonkova", "dmitrijkv", "usertest1234", "aaaa0000004", "aaaadg4gg4g4g"]
# допустимые авторы коммитов по владельцу ветки (регэксп, без учёта регистра)
AUTHORS = {"coord": r"bot.?1\b|coord|konkov|w0", "bot-W1": r"w1|bot.?2\b", "bot-W2": r"w2|bot.?3\b",
           "bot-W3": r"w3|bot.?4\b", "bot-W4": r"w4|bot.?5\b", "bot-W5": r"w5"}
issues: list[tuple[str, str]] = []


def add(level, msg):
    issues.append((level, msg))


def now():
    return dt.datetime.now(MSK).timestamp()


def msk(ts):
    return dt.datetime.fromtimestamp(ts, MSK).strftime("%d.%m %H:%M")


def base(owner):
    m = re.match(r"^(bot-W\d+)", owner or "")
    return m.group(1) if m else (owner or "")


def sh(cmd, cwd=None, timeout=90):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


def check_leases(d):
    t = now()
    leases = d.get("leases", [])
    seen = {}
    for x in leases:
        r = x["resource"]
        if r in seen:
            add("ERROR", f"двойная запись аренды {r}: {seen[r]} и {x['owner']}")
        seen[r] = x["owner"]
        if x["expires"] <= t:
            add("WARN", f"аренда истекла: {r} ({x['owner']}, {msk(x['expires'])}) — renew или release")
        if r.startswith(("kernel:", "dataset:")):
            slug = r.split("/")[-1]
            b = base(x["owner"])
            if b.startswith("bot-W") and not slug.startswith(f"koz3-w{b[5:]}-"):
                add("INFO", f"{r}: legacy-префикс (нужен koz3-w{b[5:]}-*), новые ядра только с префиксом")
    # slot → kernel: одно ядро не может держать два слота; holder в slots[] == владелец аренды
    act = {x["resource"]: x for x in leases if x["expires"] > t}
    kern_slots = {}
    for s in d.get("slots", []):
        k, h, slot = s.get("kernel"), s.get("holder"), s.get("slot")
        if k:
            if k in kern_slots:
                add("ERROR", f"ядро {k} записано в двух слотах: {kern_slots[k]} и {slot}")
            kern_slots[k] = slot
            acc = slot.split("/")[0]
            accname = {"acc1": "elenakonkova", "acc2": "dmitrijkv"}.get(acc, acc)
            if not k.startswith(accname + "/"):
                add("ERROR", f"слот {slot} ({accname}) содержит ядро другого аккаунта {k}")
        L = act.get(f"slot:{slot}")
        if h and k and not L:
            add("WARN", f"слот {slot} занят {h} ({k}) без аренды lease.sh (старый watcher?)")
        elif L and h and base(L["owner"]) != base(h):
            add("ERROR", f"слот {slot}: slots[].holder={h}, аренда у {L['owner']}")
    return act, kern_slots


def check_kaggle(act, kern_slots):
    leased = {r.split(":", 1)[1] for r in act if r.startswith("kernel:")} | set(kern_slots)
    cutoff = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None) - dt.timedelta(hours=36)
    busy = {}
    for acc in ACCS:
        rc, out, err = sh([os.path.join(HERE, "pool_kaggle.sh"), acc, "kernels", "list", "--mine", "--page-size", "20",
                           "--sort-by", "dateRun", "--csv"])
        if rc:
            add("WARN", f"kaggle list {acc}: rc={rc} (пропуск)")
            continue
        for line in out.splitlines()[1:]:
            parts = line.split(",")
            if len(parts) < 4 or not parts[0].split("/")[-1].startswith("koz3"):
                continue
            try:
                lr = dt.datetime.fromisoformat(parts[-2].strip())
            except ValueError:
                continue
            if lr < cutoff:
                continue
            ref = parts[0]
            rc, o, _ = sh([os.path.join(HERE, "pool_kaggle.sh"), acc, "kernels", "status", ref], timeout=60)
            st = (re.search(r"status \"?(?:KernelWorkerStatus\.)?(\w+)", o) or [None, "?"])[1].upper()
            if st in ("RUNNING", "QUEUED"):
                busy.setdefault(acc, []).append(ref)
                if ref not in leased:
                    add("ERROR", f"ядро {ref} {st} без аренды")
    for acc, refs in busy.items():
        print(f"  kaggle {acc}: активны {refs}")


def check_git(act):
    # 1) общий клон только main и чистый
    rc, br, _ = sh(["git", "-C", REPO, "rev-parse", "--abbrev-ref", "HEAD"])
    if br.strip() != "main":
        add("ERROR", f"общий клон {REPO} на ветке {br.strip()!r}, должен быть main (только чтение)")
    rc, st, _ = sh(["git", "-C", REPO, "status", "--porcelain", "--untracked-files=no"])
    if st.strip():
        add("ERROR", f"общий клон {REPO} имеет изменения в отслеживаемых файлах: {len(st.splitlines())} шт")
    # 2) worktrees: путь /workspace/koz3/wt/<owner>, своя ветка, одна ветка — один worktree
    rc, wl, _ = sh(["git", "-C", REPO, "worktree", "list", "--porcelain"])
    cur, used = {}, {}
    for line in wl.splitlines() + [""]:
        if line.startswith("worktree "):
            cur = {"path": line[9:]}
        elif line.startswith("branch "):
            cur["branch"] = line[7:].replace("refs/heads/", "")
        elif line == "detached":
            cur["branch"] = "(detached)"
        elif not line and cur:
            p, b = cur["path"], cur.get("branch", "?")
            if p != REPO:
                if not p.startswith(WT + "/"):
                    add("ERROR", f"worktree вне {WT}: {p} [{b}]")
                own = os.path.basename(p)
                if b in used:
                    add("ERROR", f"ветка {b} в двух worktree: {used[b]} и {p}")
                used[b] = p
                if b == "main" or b == "(detached)":
                    add("WARN", f"worktree {p} на {b} — работать только в своей ветке")
                m = re.match(r"w(\d+)$", own)
                if m and not re.match(rf"(bots/)?w{m.group(1)}-", b):
                    add("ERROR", f"worktree {p} (владелец w{m.group(1)}) на чужой ветке {b}")
            cur = {}
    # 3) лишние клоны koz3 вне worktree-схемы
    for root in ("/workspace/koz3",):
        for dname in sorted(os.listdir(root)):
            for cand in (os.path.join(root, dname, "repo"),):
                if os.path.isdir(os.path.join(cand, ".git")) and cand != REPO:
                    add("WARN", f"отдельный клон {cand} — перейти на git worktree {WT}/<owner>")
    # 4) ветки bots/*: авторы за 7 дней ↔ владелец (по аренде branch: или по имени bots/w<N>-)
    sh(["git", "-C", REPO, "fetch", "-q", "--prune", "origin"], timeout=120)
    rc, refs, _ = sh(["git", "-C", REPO, "for-each-ref", "--format=%(refname:short)", "refs/remotes/origin/bots/"])
    owners = {r.split(":", 1)[1]: x["owner"] for r, x in act.items() if r.startswith("branch:")}
    for ref in refs.split():
        b = ref.replace("origin/", "", 1)
        o = owners.get(b)
        if not o:
            m = re.match(r"bots/w(\d+)-", b)
            o = f"bot-W{m.group(1)}" if m and m.group(1) != "0" else ("coord" if m else None)
        if not o:
            continue  # старые bots/botN-* — см. BRANCHES.md
        rc, log, _ = sh(["git", "-C", REPO, "log", ref, "--since=7.days", "--format=%h|%an|%ae", "--not", f"--exclude={ref}", "--remotes=origin"])
        pat = AUTHORS.get(base(o), re.escape(o))
        bad = [l for l in log.splitlines() if not re.search(pat, l.split("|", 1)[1], re.I)]
        if bad:
            add("WARN", f"ветка {b} (владелец {o}): {len(bad)} коммит(ов) чужих авторов, напр. {bad[0]}")


def main():
    args = set(sys.argv[1:])
    if "--all" in args:
        args |= {"--kaggle", "--git"}
    d = json.load(open(os.path.join(ORCH, "LEASES.json"), encoding="utf-8"))
    print(f"check_conflicts {dt.datetime.now(MSK):%d.%m %H:%M} MSK")
    act, ks = check_leases(d)
    if "--git" in args:
        check_git(act)
    if "--kaggle" in args:
        check_kaggle(act, ks)
    order = {"ERROR": 0, "WARN": 1, "INFO": 2}
    for lv, m in sorted(issues, key=lambda x: order[x[0]]):
        print(f"[{lv}] {m}")
    ne = sum(1 for lv, _ in issues if lv == "ERROR")
    nw = sum(1 for lv, _ in issues if lv == "WARN")
    print(f"итог: ERROR={ne} WARN={nw}")
    if "--log" in args and (ne or nw):
        with open(os.path.join(ORCH, "COORD_LOG.md"), "a", encoding="utf-8") as f:
            f.write(f"- {dt.datetime.now(MSK):%d.%m %H:%M} MSK | check_conflicts | ERROR={ne} WARN={nw}: "
                    + "; ".join(m for lv, m in issues if lv == "ERROR")[:600] + "\n")
    sys.exit(1 if ne else 0)


if __name__ == "__main__":
    main()
