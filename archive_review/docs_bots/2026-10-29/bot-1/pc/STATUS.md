# bot-1 PC status, 2026-10-29 09:46 MSK
DONE  li_half OOF dump (bot-2 patch, db332db tree, nn_oof --predict-only, 5 folds): 09:21 to about 09:33 MSK, 54 npz / 167 MB.
      PC: C:\Sloi\pc_jobs\lihalf\{li_half_dump,lihalf.log}. Box: /workspace/koz3_shared/bot-2/li_half_dump (sha256 manifest OK).
      Box eval started 09:40: work/lihalf/eval/li_half_run.sh > work/lihalf/eval/run.log (cwd bot-2).
RUN   v3on (D-FINE R) vs v2 eb268d6 e2e on 101+103 Gnezdovo (honest for f0): C:\Sloi\pc_jobs\v3e2e\pc_e2e_v3_pc.ps1
      logs: v3e2e\run.log, out\time.txt, out\main_{v2,v3}.{log,err}, out\e2e.log. Started 09:41:45 MSK.
NEXT  v3on vs v2 full mock_val in the Docker sim (sim_eb.ps1 style) after the e2e run (F: single reader).
DONE  v3e2e 09:44 (scoring rerun 09:58): v3 == v2 on all slices, see v3on/result/E2E_RESULT.md.
BLOCK 09:59 PC local-exec refused (EPERM / outside allowed root). PC idle until access restored.
BLOCK 10:21-10:30 After reboot: 1 PC Shell call OK, then spawn EPERM again; local-exec root excludes C:\Sloi. Task A (F:\expds3 TRASH) and Task B (v3e2e2 + main-vs-v2) not started. Fixed trash script v2 on box. Needs owner to re-grant local-exec with C:\ (or C:\Sloi + F:\) as allowed root.
BLOCK 10:35-10:40 Access back at 10:35: 3 PC calls OK (mkdir C:\Sloi\pc_jobs\expds3_trash; copied fixed script from %USERPROFILE% -> C:\Sloi\pc_jobs\expds3_trash\f_expds3_trash.ps1, sha256 0338e868… = box). Dry-run call (4th) -> spawn EPERM; never started (scan.txt not checkable). 3 retries over ~2.5 min all EPERM. CopyFromBox/Read only allowed under C:\Users\<user>; C:\Sloi refused.
NOTE Task B: use models/dfine_f0 (+ fold thresholds) per bot-9 §7; final (bdb579ca…, box /workspace/kaggle_out/koz3-dfine-spor-final/dfine_final, 125 MB) = reserve only, and in-sample for 101/103 (trained on all SpOr).
