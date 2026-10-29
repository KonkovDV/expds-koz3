p='/workspace/koz3_shared/EXPERIMENTS.md'; s=open(p,encoding='utf-8').read()
old1="⚠ Поверх stack6 sep keep −0,001587 → v2 мерить S+stack6 на ПК в порядке pix → stack6 |"
new1="⚠ Поверх stack6 sep keep −0,001587 → v2 мерить S+stack6 на ПК в порядке pix → stack6. **29.10: −0,001587 — артефакт порядка; в порядке сдачи Ss6−S +0,001304, Ss6−A +0,013938 (bot-10/REPORT9_v2 §A)** |"
old2="перемерить пакет S+stack6 (pix → stack6) до сдачи |"
new2="перемерить пакет S+stack6 (pix → stack6) до сдачи → **сделано 29.10: Ss6−A keep +0,013938, P 1,0 (REPORT9_v2 §A)** |"
assert s.count(old1)==1 and s.count(old2)==1
s=s.replace(old1,new1).replace(old2,new2)
s=s.rstrip('\n')+'\n\n'+open('/workspace/koz3_shared/bot-10/work9/exp_block.md',encoding='utf-8').read()
open(p,'w',encoding='utf-8').write(s)
