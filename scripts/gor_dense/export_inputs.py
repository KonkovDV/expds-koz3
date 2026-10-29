# bot-5. ТОЛЬКО ЧЕСТНЫЙ ЗАМЕР. Экспорт входов пробы (dense_probe.py) с бокса: эталонные полигоны городищ ОБУЧАЮЩИХ
# участков (UTM участка, WKT, флаг разметки sp_or), раскладки фолдов honest-10 / LOFO / seed0-4 (списки участков) и SCN.
# Нужно, чтобы на Kaggle проба училась на тех же метках и фолдах, что и оценка на боксе. Env: OUT.
import os, json
from pathlib import Path
src = Path('/workspace/koz3_shared/bot-5/work/hillfort/gor_scene.py').read_text(encoding='utf-8').split('VARS = [')[0]
exec(compile(src, 'gor_scene_head', 'exec'))
out = Path(os.environ.get('OUT', '/workspace/koz3_shared/bot-5/analysis/gor_dense')); out.mkdir(parents=True, exist_ok=True)
hf = {}
for j in range(len(GG)):
    hf.setdefault(str(greg[j]), []).append([shapely.to_wkt(GG[j], rounding_precision=2), int(gm[j] == 'sp_or')])
J = dict(hf=hf, scn=list(SCN), layouts={k: [sorted(f) for f in v] for k, v in layouts.items()})
json.dump(J, open(out / 'gor_dense_inputs.json', 'w'), ensure_ascii=False)
log('written', out / 'gor_dense_inputs.json', 'regions with hillforts', len(hf), 'layouts', {k: len(v) for k, v in layouts.items()})
