"""Пункт 2 (вложенный LOFO): модели на ОДНОЙ семье = M_{−F,−G} для пары отложенных (F, G).
Признаки — pixgbm/feats (те же, что у S), тот же fit() из scripts/pixgbm.py, ROUNDS, seed 0."""
import sys, json, time
sys.path.insert(0, '/workspace/koz3_shared/bot-3/audit/wt/scripts')
import pixgbm as P
from pathlib import Path
parts = P.load_feats('/workspace/koz3_shared/bot-3/pixgbm/feats')
out = Path('/workspace/koz3_shared/bot-3/audit/calib/models_single'); info = {}
for fam in sorted({p['family'] for p in parts}):
    tr = [p for p in parts if p['family'] == fam]
    (out / fam).mkdir(parents=True, exist_ok=True)
    for target, name in (('yp', 'poly'), ('yf', 'finds')):
        sub = [p for p in tr if (p[target] >= 0).any()]
        if not sub or not any((p[target] == 1).any() for p in sub):
            print(fam, name, 'нет положительных — модели нет', flush=True); continue
        t0 = time.time(); b, inf = P.fit(sub, target, P.G.ROUNDS)
        b.save_model(str(out / fam / f'{name}.txt')); inf.update(sec=round(time.time() - t0, 1), maxrss_mb=round(P.rss_mb()))
        info[f'{fam}/{name}'] = inf; print(fam, name, json.dumps(inf, ensure_ascii=False), flush=True)
(out / 'train_info.json').write_text(json.dumps(info, ensure_ascii=False, indent=1))
