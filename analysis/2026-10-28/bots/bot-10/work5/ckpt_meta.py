"""Читает не-тензорные поля чекпойнта torch без torch (zip+pickle, тензоры заглушены)."""
import sys, zipfile, pickle, json
class Stub:
    def __init__(s,*a,**k): pass
    def __setstate__(s,st): pass
    def __repr__(s): return "<T>"
class U(pickle.Unpickler):
    def find_class(s, mod, name):
        if mod.startswith("torch") or mod.startswith("numpy") or mod=="collections" and name=="OrderedDict":
            if name=="OrderedDict":
                import collections; return collections.OrderedDict
            return lambda *a,**k: "<T>"
        return super().find_class(mod,name)
    def persistent_load(s,pid): return "<S>"
def short(o,d=0):
    if isinstance(o,dict):
        if len(o)>40 and all(v=="<T>" for v in list(o.values())[:5]): return f"<state_dict {len(o)} keys>"
        return {str(k):short(v,d+1) for k,v in o.items()}
    if isinstance(o,(list,tuple)): return [short(x,d+1) for x in o[:30]]
    return o if isinstance(o,(int,float,str,bool,type(None))) else repr(o)[:80]
for p in sys.argv[1:]:
    z=zipfile.ZipFile(p); n=[x for x in z.namelist() if x.endswith("data.pkl")][0]
    o=U(z.open(n)).load()
    print("==",p); print(json.dumps(short(o),ensure_ascii=False,indent=1,default=str)[:4000])
