import pickle,zipfile,sys
class Dummy:
    def __init__(s,*a,**k): pass
    def __setstate__(s,st): pass
class U(pickle.Unpickler):
    def find_class(s,mod,name):
        if mod.startswith('torch') : return lambda *a,**k: None
        if mod=='collections' and name=='OrderedDict':
            import collections; return collections.OrderedDict
        return super().find_class(mod,name)
    def persistent_load(s,pid): return None
for p in sys.argv[1:]:
    z=zipfile.ZipFile(p)
    n=[x for x in z.namelist() if x.endswith('data.pkl')][0]
    d=U(z.open(n)).load()
    print(p,{k:v for k,v in d.items() if k!='state_dict'})
