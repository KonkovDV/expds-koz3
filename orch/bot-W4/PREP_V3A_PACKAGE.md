# ⛔ НЕ ДЛЯ LB (30.10 14:27 MSK)
Elena LB 0.41238→0.37838: весь спад — геофизика (PIXGBM cut / TerFl off). v3a_min **не** следующий LB. См. `orch/LB_DROP_20261030.md`. Путь: cleanroom PIXGBM + TerFl 0.58017; gate geo P/R ≥ v2 (~.769/.656) vs bot7cand_v2/geo5, не vs A.

# PREP: пакет v2clean + ORG3 + M_rp для Елены (C, 30.10 13:32 MSK) — DONE
Цель LB#1. Не сабмитить ботам — только Елена.

## Готово
| | |
|---|---|
| Zip | `/workspace/koz3/w4geo/dist/koz3_v3a_min_4da2eb9_2f2d236a_wb042594f.zip` |
| sha256 | `132c22ebc6ff421982cb09c542f2fa48ea6acabce6729ce044e9e95dabbfc2ab` |
| README | `/workspace/koz3/w4geo/dist/README_v3a_min.md` |
| Commit | `bots/w4-pix-clean` @ `4da2eb9` (M_rp on v2_clean; default.toml comment only) |
| DOC-1 | PASS (`check_archive_clean.py`) |
| check4 base | `out_v2clean_check4` PASS FAIL0 WARN2 (545ef32a skeleton) |

## Состав
1. База v2clean: `org_etgr=true`, TerFl pix off, impl=clean, weights org_etgr_*.
2. M_rp: `[post.micro_circles] anchor="rp"` + r=0.05 for gor/arch/obj/fort; kurgans 0.5 m.
3. НЕ включено: xlayer, WS-G1, τ levers, obj>30m, dedup (FAIL).

## Offline Δ
ORG3 +0.023191 (18-site geo) / M_rp +0.00437 P0.989 (honest-10 OOF A). ORG3∉LB 0.4124.

## Осталось
- Рекомендуется check_submission нового zip на CPU (aaaa) перед LB — дельта только M_rp.
- WS-G1 retry только после vitb TPU COMPLETE.
