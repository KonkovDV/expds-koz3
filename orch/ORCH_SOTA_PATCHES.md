# ORCH SOTA patches — 30.10.2026 13:37 MSK (A-exec, orch-audit idle-fill)
Принцип (SOTA-оркестрация: Slurm/K8s-style): каждый занятый слот обязан ссылаться на живой объект работы (job ref) — иначе reconciler (watcher) не может его освободить → «призрачные» простои. Плюс: backfill-класс задач, идемпотентный push (не перезаписывать живое ядро), инвентарь узлов из конфигурации, а не хардкод.

## ПРИМЕНЕНО
| # | файл | что | diff |
|---|---|---|---|
| A1 | watcher.py | SLOTS: + `box/cpu`; CPU-пул из `local_ctx/ctx/kaggle_pool_4x2_t4.json` (только username) → `<u>/cpu`, `<u>/cpu-2`; fallback на 3 хардкод-имени | `patches/APPLIED_1335_watcher_pool_slots_guard.patch` |
| A2 | watcher.py | guard в push: если `kernel:<acc>/<slug>` в аренде или статус RUNNING/QUEUED → skip (иначе push во 2-й CPU-слот того же аккаунта = новая версия = отмена живого ядра) | там же |
| A3 | lease.py/.sh | `lease.sh set-kernel <slot> <owner\|coord> <acc>/<slug>` — привязать ядро к уже взятой аренде (since/expires не меняются). Аддитивно, ничего не ломает | `patches/APPLIED_1336_lease_set_kernel_P2.patch` |
Watcher перезапущен 13:34 MSK (pid в `.watcher.pid`), бэкап `watcher.py.bak_1340`, `lease.py.bak_1336`.

## ПРЕДЛОЖЕНО (не применено — нужен ACK ботов B/C/D: ломает их ручной flow «сначала слот, потом ядро»)
**P1. slot:<kaggle> без `--kernel` → отказ** (box/* и coord `--force` — исключение). Соответствует PROTOCOL §3 («До push — аренда слота и ядра»), но B/D сейчас резервируют слот до готовности ядра.
Диф: `patches/P1P2_lease_require_kernel.patch` (P1-кусок в `cmd_acquire`). Применение: `cd orch && patch -p1 --dry-run < patches/P1P2_lease_require_kernel.patch` (P2-часть уже в коде — применять только P1-блок).
Переходный вариант: резерв без ядра → TTL ≤ 1 ч, затем обязательный `set-kernel`.

**P3. QUEUE.dispatch `class`: `critical|normal|backfill`** — fill_free сначала critical/normal, backfill — только в слоты, где ≥ ttl_h до ближайшего `after`/cooldown-окончания (EASY-backfill). Диф для `fill_free`:
```python
-            for it in d["dispatch"]:
+            prio = {"critical": 0, "normal": 1, "backfill": 2}
+            for it in sorted(d["dispatch"], key=lambda x: prio.get(x.get("class", "normal"), 1)):
```
**P4. Stale-pushed reconciler**: раз в 15 мин для `status=pushed` без аренды слота → kstatus(kernel); terminal → done/failed + pull (сегодня руками закрыты 3 таких: w2-hsparity-ci2-8, w2-lihm-c3sc, w4-v2clean-check2).
```python
+def reconcile_pushed():
+    act, _ = leases()
+    with Queue() as d:
+        for it in d["dispatch"]:
+            if it["status"] == "pushed" and it.get("kernel") and f"slot:{it.get('slot','')}" not in act:
+                st = kstatus(it["kernel"])
+                if st in TERMINAL: it["status"] = "done" if st == "COMPLETE" else "failed"; it["finished"] = f"{now():%d.%m %H:%M} {st} (reconcile)"
```
**P5. Slot-without-kernel alarm** в check_conflicts: Kaggle slot-аренда без kernel > 30 мин → WARN с владельцем (сейчас 0 после set-kernel).
**P6. Idle-метрика**: watcher раз в цикл пишет `logs/idle.csv` (ts, slot, state=busy|idle|cooldown|no_ready) → UTILIZATION.md считает простой по слотам.
