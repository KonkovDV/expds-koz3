"""Метаданные чекпойнтов без torch: data.pkl из zip, тензоры заглушены (форма только у первого слоя)."""
import zipfile, pickle, sys, json, collections
class Stub:
    def __init__(self, *a, **k): self.a = a
    def __setstate__(self, s): pass
def rebuild(storage, off, size, stride, *a): return ("T", tuple(size))
class U(pickle.Unpickler):
    def find_class(self, m, n):
        if n == "_rebuild_tensor_v2": return rebuild
        if m == "collections" and n == "OrderedDict": return collections.OrderedDict
        return Stub
    def persistent_load(self, pid): return None
out = {}
for p in sys.argv[1:]:
    z = zipfile.ZipFile(p); nm = [n for n in z.namelist() if n.endswith("data.pkl")][0]
    d = U(z.open(nm)).load()
    sd = d.get("state_dict", {}); first = next(((k, v[1]) for k, v in sd.items() if isinstance(v, tuple)), None)
    last = [(k, v[1]) for k, v in sd.items() if isinstance(v, tuple)][-1]
    m = {k: v for k, v in d.items() if k != "state_dict" and not isinstance(v, Stub)}
    m["first_layer"] = first; m["last_layer"] = last
    out[p.split("/")[-1]] = m
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
