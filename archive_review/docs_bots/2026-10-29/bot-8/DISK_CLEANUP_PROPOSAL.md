# Disk cleanup proposal (bot-8, 2026-10-28 23:40 MSK) — NOTHING DELETED
/ : 128G, used 120G, free 7.7G (target ≥12G).

| path | size | newest mtime (MSK) | owner | in use / refs | class |
|---|---|---|---|---|---|
| /home/box/.cache/uv | ~2.0G (hardlinks; cache total 1.4G du) | 23:32 | shared | uv venvs may hardlink into it; offline reinstall needs network | A |
| /home/box/.cache/pip | 1.2G | — | shared | none | A |
| /home/box/.cache/torch | 45M | — | shared | hub weights; offline builds may need them | A (keep) |
| /tmp/t9.tif,t15.tif,t19.tif | 1.07G | 20:33 | unknown | no proc, no refs, <24h | D (likely C tomorrow) |
| /tmp/fsrc,/tmp/fout,/tmp/pdest | 0.67G | 19:32 | unknown | no proc, no refs, <24h | D |
| /tmp/b7 | 1.6G | 23:26 | prob. bot-7/bot-8 (repo copy, not a git worktree) | written 10 min ago | D |
| /tmp/geo,/tmp/lid,/tmp/wt_s4 | 0.82G each | 23:26 | unknown (repo copies, no .git) | written 10 min ago | D |
| /tmp/pix | 0.83G | 23:03 | prob. bot-3 (pixgbm) | active today | D |
| /tmp/dfine_venv | 1.6G | 21:25 | bot-9/bot-6 (DFINE.md, REPORT.md) | referenced | D |
| /workspace/kaggle_out | 6.1G | 22:19 | bot-9 (+refs bot-2,10) | referenced; **no byte-identical dups ≥1MB** (size+sha256 incl. kgl_out) | D |
| /workspace/rerank | 3.0G | 20:36 | bots 3/4/5 | **PID 418495 running** from rerank/venv | D |
| /workspace/kgl_out | 437M | 21:37 | bot-9 | referenced | D |
| /workspace/simbuild | 1.6G | 23:36 | unknown | being written now | D |
| /workspace/expds-koz3 + gh_sync_wt | 0.84G+0.84G | 23:30 | registered worktree pair | active | D |
| /workspace/xfer_pc, xfer_venv | 436M+267M | 23:06 | unknown | xfer_venv **PIDs 202729, 84057 running** | D |

No (B) and no (C) candidates: every item was modified within the last 24h.

## Commands (class A only, safe now)
```
uv cache clean      # or: uv cache prune  (keeps entries used by venvs)
pip cache purge
```
## Totals
- A: ~2.6–3.2 GB (pip 1.2G + uv ~1.4–2.0G) → free ≈10.5–11G, short of 12G.
- B: 0. C: 0.
- D: ~22 GB; biggest wins need owner confirmation: /tmp t*.tif+fsrc/fout/pdest (1.7G, orphan-looking), /tmp repo copies b7/geo/lid/wt_s4/pix (~4.9G), kaggle_out gpu-s1..s5 (6.1G, bot-9).
Candidate commands after owner OK:
```
rm -f /tmp/t9.tif /tmp/t15.tif /tmp/t19.tif; rm -rf /tmp/fsrc /tmp/fout /tmp/pdest
rm -rf /tmp/b7 /tmp/geo /tmp/lid /tmp/wt_s4 /tmp/pix   # not git worktrees, plain rm
```
