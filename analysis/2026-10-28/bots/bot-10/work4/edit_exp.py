import re
p='/workspace/koz3_shared/EXPERIMENTS.md'
L=open(p,encoding='utf-8').read().split('\n')
# locate by content (file may have been edited by others)
i_p5=next(i for i,s in enumerate(L) if s.startswith('- **П5.**'))
add_rule=['- **П6 (bot-10, 28.10 21:40 МСК). Одинаковый режим unseen.** База и вариант считаются при одном `unseen` (`drop` или `keep`, ветка `bot10-unseen-keep`). Сравнивать keep с drop нельзя: одно это даёт +0,0186 (object_poly etgr/terfl Гнёздово), см. `bot-10/REPORT3.md`.',
 '- **Вторичная проверка (не условие приёма): устойчивость к site-scope.** Главное табло — глобальное (`bot-8/CORRECTNESS.md`). Δ того же изменения при `EXPDS_MATCH_SCOPE=site` (патч `bot-8/patch/bot8-site-scope-worktree.diff`) не должна быть отрицательной; если отрицательна или заметно меньше глобальной — изменение подозрительно (выигрыш от межучасткового сопоставления). Прошлые глобальные строки не пересматриваются.']
L[i_p5+1:i_p5+1]=add_rule
i_cmd=next(i for i,s in enumerate(L) if s.startswith('Команда bot-10 для абляций'))
cmd=L.pop(i_cmd)
if L[i_cmd-1]=='': L.pop(i_cmd-1)
# fix rerank_arch row
i_r=next(i for i,s in enumerate(L) if '| rerank_arch (' in s)
L[i_r]=L[i_r].replace('LOFO +0,0086 (пороги прибл.) / 5 из 5 (+0,0052…+0,0101)','LOFO +0,0086 (пороги прибл., НЕ строгий П3); строгий перемер bot-10 — строка ниже').replace('П1–П4 ✓ → кандидат; перемерить пакетом с crop verify','П3 здесь не строгий (пороги LOFO приближённые); строгий П3 ✓ подтверждён bot-10 (см. строку rerank_arch strict)')
# find last journal row
last=max(i for i,s in enumerate(L) if s.startswith('| 28.10') or s.startswith('| 27.10'))
B='bot-10/work4/'
new=[
'| 28.10 | bot-10 | sota_stack6 separate, unseen **drop** (одинаковый режим) | геофизика multipole/ws, отдельные группы | 0,314615 | +0,006703 [0,0000; +0,0197], P 0,948 (структурно: геофизика в 3 семьях) | — | LOFO +0,006703 / 5 из 5 (мин. +0,001586) | — | `'+B+'stack6_global.json`, `'+B+'stack6_fam.py` | П1–П4 ✓; site +0,006699 ✓ |',
'| 28.10 | bot-10 | sota_stack6 separate, unseen **keep** vs A keep 0,333237 | то же, как в сдаче (сдача выдаёт всё) | 0,333237 | +0,015057, P 0,948; heavy 0, new34 +0,021888 | object_poly +0,009837, finds +0,00522 | LOFO = k10 / 5 из 5 | — | `'+B+'stack6_global.json` | **принять условно** (П1–П4 ✓; P структурный, константы подобраны на тех же 3 семьях) → проверка LB; site +0,014454 ✓ |',
'| 28.10 | bot-10 | sota_stack6 unified (keep vs keep) | unified group | 0,333237 | +0,015057 (= separate) | — | — | — | `'+B+'stack6_global.json` | заявленные +0,0337 = артефакт unseen (drop: unified +0,033687 из-за смены режима, 0,01863 — keep-эффект); unified в заявленном виде **отклонить** |',
'| 28.10 | bot-10 | ae_fort_mask, перепроверка global/site, keep=drop | маска фортификаций ae | 0,314615 | +0,005240 [−0,00058; +0,00915], P 0,926; site +0,005260 P 0,923 | fortifikatsii +0,00524; new34 +0,010977 | LOFO +0,006843 / 5 из 5 | — | `'+B+'aefort_global.json`, `'+B+'aefort_site.json` | подтверждено, site ✓ (Δ<0,006 → в пакете) |',
'| 28.10 | bot-10 | rerank_arch strict (nested LOFam, пороги inner-5 внутри фолда) | реранкер arkhitektury | 0,314615 (LOFO-база 0,315067) | +0,009536 [+0,00063; +0,01535], P 0,957; без Елизаветинского +0,006792; site +0,009590 P 0,980 | arkhitektury +0,009536; heavy 0, new34 0 | LOFO +0,007869 / 5 из 5 (+0,00241…+0,01051); site LOFO +0,008152, семена 5/5 | — | `'+B+'rerank/nested_lofo5.py`, `'+B+'rerank/res_*_nested5_micro*.json`, `'+B+'rerank/*.log` | П1–П4 ✓ (строго). Оговорки: поправка на отбор (~7 вариантов) → ≈+0,0035, P≈0,74; без rg_n/rg_pct +0,004407 P 0,859; без всех rg_* −0,003054 → проверка LB |',
'| 28.10 | bot-10 (по bot-7) | bot7-bundle 42ada8d: kurgany 0,5 м + arch micro 0,25 м | геометрия выдачи; пороги перефитены внутри фолдов (nested, честно) | 0,314615 | bundle +0,002447 [+0,00049; +0,00500], P 0,995; kur +0,001371 P 0,953; archM +0,001076 P 0,867 | kur: povr+tselye +0,001371, new34 +0,000967; arch +0,001076 | bundle LOFO +0,001596 / 5 из 5; kur LOFO **+0,000064**; archM LOFO +0,001531 | — | `'+B+'bot7_global.json`, `bot-7/REPORT2.md` | цифры bot-7 воспроизведены точно; П5: только в пакете. archM в site-scope +0,00023 (P 0,63) → выигрыш micro почти весь межучастковый, **micro отклонить** (согласно bot-5); kur — в пакет, слабый LOFO |',
'| 28.10 | bot-10 | ПАКЕТ kur 0,5 м + archM + rerank_arch (без crop: стека crop на коробке нет) | состав пакета, классы аддитивны | 0,314615 | +0,011955 [+0,00369; +0,01705], P 0,991; R+kur +0,010907; R+archM +0,010585; R +0,009536 | heavy +0,001371, new34 +0,000967 | LOFO +0,008285 / 5 из 5 (+0,00408…+0,01241) | — | `'+B+'compose.py`, `'+B+'package.json/.log` | П1–П4 ✓. Leave-one-out: −kur −0,00137, −archM −0,00105, −R −0,00951. Взаимодействие R×micro — bot-5 (`bot-5/analysis/joint/summary.json`). Crop × kur не аддитивны → совместный замер на ПК |',
'| 28.10 | bot-3 | `bot3-geophys` (ws+pole, симуляция) | постобработка object_poly | — | только симуляция: F1 0,748→0,784 LOFO | ожид. ALL ≤ +0,0013…+0,0026 | — | — | `bot-3/REPORT.md`, `bot-3/geophys.diff` | **PENDING-on-PC**; перекрывается со stack6 multipole, не аддитивно |',
'| 28.10 | bot-9 | s4 crop-verify курганов (Kaggle, AUC) | классификатор кропов, AUC r18 fam10 0,9316 | — | только AUC, балла нет | — | — | — | `bot-9/s4_eval.json`, `bot-9/REPORT2.md` §3 | **PENDING-on-PC** (`bot-9/PC_RESULTS.md` на 21:40 нет); AUC ≠ Δ балла, П4 по курганам обязателен |',
]
L[last+1:last+1]=new
# command line after table
j=last+1+len(new)
L[j:j]=['',cmd]
open(p,'w',encoding='utf-8').write('\n'.join(L))
print('ok')
