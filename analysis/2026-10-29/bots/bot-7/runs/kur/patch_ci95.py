import sys; f=sys.argv[1]; s=open(f).read()
fn='''def boot95(a,b,n=2000,seed=0):
    # копия M.paired_bootstrap_counts (те же семья/seed -> те же выборки), добавлены перцентили 2.5/97.5
    regs={r for cs in (a,b) for by in cs.values() for r in by}; fams={}
    for r in sorted(regs): fams.setdefault(site_family(r),[]).append(r)
    names=sorted(fams); rng=np.random.default_rng(seed); v=[]
    for _ in range(n):
        pick=np.bincount(rng.integers(0,len(names),len(names)),minlength=len(names))
        w={r:float(k) for f,k in zip(names,pick) if k for r in fams[f]}
        v.append(M.score_counts(a,None,w)[0]-M.score_counts(b,None,w)[0])
    v=np.asarray(v); return {'p2_5':float(np.percentile(v,2.5)),'p97_5':float(np.percentile(v,97.5))}
def criteria('''
s=s.replace("def criteria(",fn,1)
s=s.replace("    pb=M.paired_bootstrap_counts(a,b,site_family,n=2000,seed=0)\n","    pb=M.paired_bootstrap_counts(a,b,site_family,n=2000,seed=0); pb.update(boot95(a,b))\n")
s=s.replace("CI=[%.5f,%.5f] P1","CI90=[%.5f,%.5f] CI95=[%.5f,%.5f] P1").replace("r['boot']['p5'],r['boot']['p95'],","r['boot']['p5'],r['boot']['p95'],r['boot']['p2_5'],r['boot']['p97_5'],")
open(f,'w').write(s)
