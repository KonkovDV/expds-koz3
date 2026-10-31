import csv, math
from collections import Counter, defaultdict
D=list(csv.DictReader(open('diag.csv')))
B=lambda v: v=='True'
F=lambda r,k: float(r[k]) if r.get(k) not in (None,'','nan') else math.nan
def ring(r,soft=False,pre='morph_'):
    rp,rg,hm=F(r,pre+'r_peak'),F(r,pre+'ring'),F(r,pre+'hmax')
    if any(math.isnan(x) for x in (rp,rg,hm)): return None
    if F(r,pre+'valid')==0: return None
    return rp>=1.5 and rg>=(0.05 if soft else 0.15) and hm>=0.10
def pit(r,pre='morph_'):
    h0,hm=F(r,pre+'h0'),F(r,pre+'hmax')
    if math.isnan(h0) or math.isnan(hm) or F(r,pre+'valid')==0: return None
    return h0<0 and hm>=0.10
def share(rows,fn):
    v=[fn(r) for r in rows]; v=[x for x in v if x is not None]
    return f"{sum(v)}/{len(v)}" + (f" ({100*sum(v)/len(v):.1f}%)" if v else "")
for cls in ('kurgany_povrezhdennye','kurgany_tselye'):
    R=[r for r in D if r['cls']==cls]; n=len(R)
    c=lambda k: sum(B(r[k]) for r in R)
    print(cls,'n',n,'acc_li',c('acc_li'),'acc_all',c('acc_all'),'acc_cross(no same)',sum(B(r['acc_cross']) and not B(r['acc_all']) for r in R),
          'li_anyconf',c('li_anyconf'),'li_anyconf_same',c('li_anyconf_same'))
    miss=[r for r in R if not B(r['acc_li'])]; found=[r for r in R if B(r['acc_li'])]
    missall=[r for r in R if not B(r['acc_all'])]
    print(' missed_li',len(miss),'missed_all',len(missall),' missed but li circle exists (any conf)',sum(B(r['li_anyconf']) for r in miss),
          ' missed no li circle',sum(not B(r['li_anyconf']) for r in miss))
    for name,S in (('missed_li',miss),('found_li',found)):
        print('  ',name,'ring',share(S,ring),'soft',share(S,lambda r:ring(r,True)),'pit',share(S,pit))
    # per region
    if cls=='kurgany_povrezhdennye':
        pr=defaultdict(lambda:[0,0,0]); 
        for r in R: pr[r['region']][0]+=1; pr[r['region']][1]+=B(r['acc_li']); pr[r['region']][2]+=B(r['acc_all'])
        print('  per region n/acc_li/acc_all:',{k[:3]:v for k,v in sorted(pr.items(),key=lambda x:-x[1][0]) if v[0]>=5})
# DEM at GT
G=list(csv.DictReader(open('gtdem.csv')))
key=lambda r:(r['region'],r['cls'],round(float(r['gx']),2),round(float(r['gy']),2))
dk={key(r):r for r in D}
print('DEM rows',len(G),'matched to diag',sum(key(r) in dk for r in G))
GG=[r for r in G if not r['region'].startswith('041') and F(r,'morph_valid')!=0 and not math.isnan(F(r,'morph_hmax'))]
print('DEM valid (без 041)',len(GG))
for cls in ('kurgany_povrezhdennye','kurgany_tselye'):
    R=[r for r in GG if r['cls']==cls and key(r) in dk]
    miss=[r for r in R if not B(dk[key(r)]['acc_li'])]; found=[r for r in R if B(dk[key(r)]['acc_li'])]
    nolic=[r for r in miss if not B(dk[key(r)]['li_anyconf'])]
    for name,S in (('all',R),('missed_li',miss),('missed_no_li_circle',nolic),('found_li',found)):
        print(' DEM',cls[8:12],name,len(S),'ring',share(S,ring),'soft',share(S,lambda r:ring(r,True)),'pit',share(S,pit))
    for reg in sorted({r['region'] for r in R}):
        S=[r for r in R if r['region']==reg]; M_=[r for r in S if r in miss]
        print('   ',reg[:3],'n',len(S),'missed',len(M_),'ring all',share(S,ring),'ring missed',share(M_,ring))
