# v3on (D-FINE R) vs v2 eb268d6, PC e2e on 101+103 Gnezdovo — bot-1, 29.10 10:00 MSK
PC: C:\Sloi\pc_jobs\v3e2e\out (e2e.json, e2e.log, time.txt, main_v{2,3}.log, dfine_lines_v3.txt).
main.py: v2 97.4 s, WS 1972 MB, VRAM 2953 MiB; v3 67.4 s, WS 1999 MB, VRAM 3107 MiB. main_*.err empty (exit code printed blank = PS Start-Process quirk, not a failure).
Original script died at step 2 (expds.gt INFO on stderr + ErrorActionPreference Stop); gt+score rerun by bot-1 via cmd redirection, exit 0.
Metric (built-in defaults, config.py absent):
ALL/ALL gt450 n248: v2 0.31251 = v3 0.31251 (delta 0.0)
ALL/ae 0.09556 = ; ALL/li 0.33698 = ; ALL/sp_or 0.24291 =
101/ALL 0.34640 = ; 101/ae 0.24291 ; 101/li 0.33698 ; 101/sp_or 0.34800
103/ALL 0.00699 = ; 103/ae 0.0 ; 103/sp_or 0.09524
=> v3 output identical to v2 on every slice: D-FINE R apparently did not change any prediction (not firing / flags not on in v3on zip?). Check dfine_lines_v3.txt on PC.
PC access from box broke at ~09:59 (spawn EPERM, path outside allowed local-exec root) — could not copy files.
