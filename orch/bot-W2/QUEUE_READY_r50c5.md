# QUEUE-ready draft — P-W2-9 r50c5 (для A @ ~02:55 MSK 31.10)
Готовил B 30.10 13:37. **Не enqueue-ил сам** (QUEUE правит только A). Push только после 03:05 + lease A.
Старый dispatch `w2-r50c5` устарел (один id, account=dmitrijkv, kernel_dir без elenakonkova/). Заменить на два id ниже.

## (a) w2-r50c5-f01 — elena first
```bash
# вариант: set старого id → или add новый и hold старого
orch/enqueue.py add \
  --id w2-r50c5-f01 \
  --owner bot-W2-lidar \
  --accel gpu \
  --account elenakonkova \
  --kernel-dir w2/kernels/elenakonkova/koz3-w2-cnx-r50c5-f01 \
  --pull-to w2/oof/r50c5/f01 \
  --ttl 10 \
  --hold \
  --note "P-W2-9 f0+f1 ResNet50-c5; ready_after 31.10 03:05; B push после A lease+GO; early_stop → gate f234"
# затем @≥03:05: enqueue.py set w2-r50c5-f01 status=ready
# (старый w2-r50c5: set status=done note="superseded by w2-r50c5-f01/f234")
```
Payload:
```json
{
  "id": "w2-r50c5-f01",
  "owner": "bot-W2-lidar",
  "accel": "gpu",
  "account": "elenakonkova",
  "ttl_h": 10,
  "status": "hold",
  "after": [],
  "kernel_dir": "w2/kernels/elenakonkova/koz3-w2-cnx-r50c5-f01",
  "pull_to": "w2/oof/r50c5/f01",
  "kernel": "elenakonkova/koz3-w2-cnx-r50c5-f01",
  "ready_after": "31.10.2026 03:05",
  "note": "P-W2-9 f0+f1; B: lease slot:acc1/gpu-* + push; early_stop_r50c5.py; FAIL→cancel f234"
}
```
Kernel meta: gpu T4, secret_hits=0, dataset elenakonkova/koz3-s3-cred. Checklist: orch/bot-W2/R50C5_GO_CHECKLIST.md

## (b) w2-r50c5-f234 — dmitrij only after f01 PASS + A lease
```bash
orch/enqueue.py add \
  --id w2-r50c5-f234 \
  --owner bot-W2-lidar \
  --accel gpu \
  --account dmitrijkv \
  --kernel-dir w2/kernels/dmitrijkv/koz3-w2-cnx-r50c5-f234 \
  --pull-to w2/oof/r50c5/f234 \
  --ttl 12 \
  --hold \
  --after w2-r50c5-f01 \
  --note "P-W2-9 f2+f3+f4; HOLD до early_stop PASS на f01 + отдельный lease A; B сам не пушит"
```
Payload:
```json
{
  "id": "w2-r50c5-f234",
  "owner": "bot-W2-lidar",
  "accel": "gpu",
  "account": "dmitrijkv",
  "ttl_h": 12,
  "status": "hold",
  "after": ["w2-r50c5-f01"],
  "kernel_dir": "w2/kernels/dmitrijkv/koz3-w2-cnx-r50c5-f234",
  "pull_to": "w2/oof/r50c5/f234",
  "kernel": "dmitrijkv/koz3-w2-cnx-r50c5-f234",
  "note": "P-W2-9 f2–f4; push ONLY after A lease + early_stop PASS on f01; never self-push"
}
```
Kernel meta: gpu T4, secret_hits=0, dataset dmitrijkv/koz3-s3-cred, kernel_sources dmitrijkv/koz3-cnx-li-prep.

## B runbook (после A GO ≥03:05)
1. A: set f01 ready + lease GPU slot → B acquire kernel+slot → push f01
2. Pull OOF → `venv/bin/python w2_tools/early_stop_r50c5.py <out_f01>`
3. FAIL → cancel f234 (A set done/cancel), release; PASS → ждать A lease f234 → push
4. hmcp (P-W2-11) — только после r50c5 + lease A; ядро уже на диске

## Files on disk
- `w2/kernels/elenakonkova/koz3-w2-cnx-r50c5-f01/`
- `w2/kernels/dmitrijkv/koz3-w2-cnx-r50c5-f234/`
- `expds-koz3/w2_tools/early_stop_r50c5.py`
- Mirror JSON: `orch/bot-W2/QUEUE_READY_r50c5.json`

## Elena night 31.10 OVERRIDE
ONE GPU: `aaaa0000004/koz3-w2-cnx-r50c5-f01` (`w2/kernels/aaaa0000004/…`). s3-cred stripped. Wait A slot+cred clean before push. Stop after first honest slice if damaged kurgans don’t rise. No LB. f01-elena/f234-dmitrij hold.
