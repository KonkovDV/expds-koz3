"""Потоковая разрезка большого стека GeoJSON по семьям (≤ ~300 МБ ОЗУ).
Каждая строка выхода: <индекс в файле>\t<исходный текст объекта, байт-в-байт>.
python split_stackA.py STACK OUTDIR"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, "/workspace/koz3_shared/bot-8/repo")
from expds.naming import site_family
src, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
dec = json.JSONDecoder()
t0 = time.time()
fh = open(src, "r", encoding="utf-8")
buf = fh.read(1 << 20)
pos = buf.index('"features"'); pos = buf.index("[", pos) + 1
files, counts, n, maxx = {}, {}, 0, {}
def more():
    global buf, pos
    chunk = fh.read(8 << 20)
    buf = buf[pos:] + chunk; pos = 0
    return bool(chunk)
while True:
    while True:
        while pos < len(buf) and buf[pos] in " \n\r\t,":
            pos += 1
        if pos >= len(buf) - 1 and more():
            continue
        break
    if pos < len(buf) and buf[pos] == "]":
        break
    try:
        obj, end = dec.raw_decode(buf, pos)
    except json.JSONDecodeError:
        if not more():
            raise
        continue
    if end >= len(buf) and more():   # объект мог кончиться на границе куска — перечитать
        continue
    text = buf[pos:end]; pos = end
    reg = (obj.get("properties") or {}).get("region_name") or ""
    fam = site_family(str(reg)) if reg else "_none"
    f = files.get(fam)
    if f is None:
        f = files[fam] = open(out / f"{fam}.tsv", "w", encoding="utf-8")
    f.write(f"{n}\t{text}\n"); counts[fam] = counts.get(fam, 0) + 1; n += 1
    if len(buf) - pos < (1 << 20) and more():
        pass
for f in files.values(): f.close()
(out / "_counts.json").write_text(json.dumps({"total": n, "families": counts, "seconds": round(time.time() - t0, 1)},
                                             ensure_ascii=False, indent=1))
print(n, len(counts), round(time.time() - t0, 1))
