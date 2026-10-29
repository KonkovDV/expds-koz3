"""Вырезка 4 участков из стека /workspace/rerank/stack_arch.geojson (порядок строк = индекс table.pkl)."""
import ijson, pickle, sys, time
REG = {"008_Герите", "014_Лешки", "031_Новотитаровская", "116_Каменные_выкладки_6"}
t = time.time(); out = {r: [] for r in REG}; n = 0
with open("/workspace/rerank/stack_arch.geojson", "rb") as fh:
    for i, f in enumerate(ijson.items(fh, "features.item", use_float=True)):
        r = f["properties"].get("region_name")
        if r in REG:
            out[r].append((i, f))
        n = i + 1
print("rows", n, {r: len(v) for r, v in out.items()}, round(time.time() - t, 1), "s")
pickle.dump(out, open("equiv/regions4.pkl", "wb"), protocol=5)
