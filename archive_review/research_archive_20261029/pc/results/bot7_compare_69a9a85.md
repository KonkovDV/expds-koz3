# compare_candidates @69a9a85 (PC, ENV=docker:pantum/n7q:20261024, DONE 19:50 MSK)
baseline v2 docker ALL=0.312509 li=0.336983 sp_or=0.242915 ae=0.095556; rule: no layer drops >0.01

| candidate | env | ALL | li | sp_or | ae | dALL | rule |
|---|---|---:|---:|---:|---:|---:|---|
| v2 | docker | 0.312509 | 0.336983 | 0.242915 | 0.095556 | 0 | PASS |
| v3 | docker | 0.287985 | 0.340909 | 0.000000 | 0.095556 | -0.024524 | FAIL(sp_or) |
| hotfix_B | host | 0.312873 | 0.336983 | 0.245064 | 0.095556 | +0.000364 | PASS |
| v4_off | host | 0.312873 | 0.336983 | 0.245064 | 0.095556 | +0.000364 | PASS |
| bot4_off | host | 0.312873 | 0.336983 | 0.245064 | 0.095556 | +0.000364 | PASS |
| bot4_on | host | 0.296584 | 0.321598 | 0.245064 | 0.095556 | -0.015925 | FAIL(li) |
Source: F:\Sloi_pc_jobs\bot7_compare_candidates_69a9a85\compare\compare.md
