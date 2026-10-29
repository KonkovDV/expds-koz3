# C:\Sloi\TRASH: moves done 2026-10-29 09:44 MSK (Move-Item only, nothing deleted)

| From | To | GB |
|---|---|---|
| C:\Sloi\oof59 | C:\Sloi\TRASH\oof59 | 6.96 |
| C:\Sloi\models_59 | C:\Sloi\TRASH\models_59 | 4.39 |
| C:\Sloi\work59 | C:\Sloi\TRASH\work59 | 2.63 |
| C:\Sloi\lidump | C:\Sloi\TRASH\lidump | 2.89 (the fold weights li.pt/done.json were already copied to pc_jobs\lihalf\work) |
| C:\Sloi\s5 | C:\Sloi\TRASH\s5 | 1.81 |
| **Total** | | **18.68** |

Checks before moving: no running process command line, scheduled task, user/process PATH entry, or .ps1/.cmd under pc_jobs, sim or expds3c referenced oof59, models_59, work59, lidump or s5.

C: free before: 4.42 GB. After: 4.42 GB. **This is unchanged, as expected: a move inside the same C: volume frees nothing.**
To actually reclaim the 18.7 GB, the user must delete C:\Sloi\TRASH (or move it to F:, which has 16.8 GB free, so it would not fit whole).

Kept:
- expds-koz3_v0.2.0 (11.55 GB). It is **in use**: its .venv\Scripts\python.exe runs the li_half job (done) and the v3-vs-v2 e2e job (running now).
  Its dist (2.73) and models (2.07) are possible later candidates. I did not move them because I am not sure nothing references them.
- expds3c, pc_jobs, wt_lihalf, wt_bots, sim, files, kur: per the instructions and/or touched 10-29.
