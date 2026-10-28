# Потоковое извлечение кругов курганов из большого стека (без загрузки 1.1 ГБ в память).
import json, sys
src, dst = sys.argv[1], sys.argv[2]
KEEP = {"kurgany_povrezhdennye", "kurgany_tselye"}
dec = json.JSONDecoder()
buf = ""; n = k = 0; started = False
out = open(dst, "w"); out.write('{"type": "FeatureCollection", "features": [')
first = True
with open(src, "r") as f:
    while True:
        chunk = f.read(8 << 20)
        buf += chunk
        if not started:
            i = buf.find('"features": [')
            if i < 0: continue
            buf = buf[i + 13:]; started = True
        pos = 0
        while True:
            while pos < len(buf) and buf[pos] in " ,\n\r\t": pos += 1
            if pos >= len(buf) or buf[pos] == "]": break
            try:
                obj, end = dec.raw_decode(buf, pos)
            except json.JSONDecodeError:
                break
            n += 1
            if obj.get("properties", {}).get("class_name") in KEEP:
                out.write(("" if first else ", ") + json.dumps(obj)); first = False; k += 1
            pos = end
        buf = buf[pos:]
        if not chunk: break
out.write("]}"); out.close()
print("всего", n, "курганов", k)
