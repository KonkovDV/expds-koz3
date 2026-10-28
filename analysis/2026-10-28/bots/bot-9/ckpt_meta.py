"""Читает метаданные чекпойнта torch.save без torch: тензоры заменяются заглушками."""
import json, pickle, sys, zipfile
class Stub:
    def __init__(self,*a,**k): self.a=a
    def __setstate__(self,s): pass
class U(pickle.Unpickler):
    def find_class(self, mod, name):
        if mod.startswith("torch") or mod.startswith("collections") and name=="OrderedDict":
            if name=="OrderedDict":
                import collections; return collections.OrderedDict
            return Stub
        return super().find_class(mod, name)
    def persistent_load(self, pid): return None
for p in sys.argv[1:]:
    z=zipfile.ZipFile(p); n=[x for x in z.namelist() if x.endswith("data.pkl")][0]
    d=U(z.open(n)).load()
    meta={k:v for k,v in d.items() if k!="state_dict"}
    sd=d["state_dict"]; meta["n_tensors"]=len(sd); meta["has_n_averaged"]=any("n_averaged" in k for k in sd)
    print(p, json.dumps(meta, ensure_ascii=False, default=str))
