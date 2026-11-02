import numpy as np,struct,glob,json,sys
res={}
for f in sorted(glob.glob('/workspace/koz3/**/*.las',recursive=True)):
    with open(f,'rb') as h: hd=h.read(375)
    ver=hd[24:26]; off=struct.unpack('<I',hd[96:100])[0]; fmt=hd[104]&0x3f; rl=struct.unpack('<H',hd[105:107])[0]
    n=struct.unpack('<I',hd[107:111])[0]
    if ver[1]>=4 and n==0: n=struct.unpack('<Q',hd[247:255])[0]
    m=np.memmap(f,dtype=np.uint8,mode='r',offset=off,shape=(n,rl))
    c=(m[:,15]&0x1f) if fmt<6 else m[:,16]
    cnt=np.zeros(256,np.int64)
    for s in range(0,n,20_000_000): cnt+=np.bincount(np.asarray(c[s:s+20_000_000]),minlength=256)
    res[f]={'ver':f'{ver[0]}.{ver[1]}','fmt':fmt,'n':int(n),'classes':{int(k):int(v) for k,v in enumerate(cnt) if v},'frac_ground':float(cnt[2]/max(n,1))}
    print(f,res[f],flush=True)
json.dump(res,open('/workspace/koz3/out/w0s-dem-noground/class_counts.json','w'),ensure_ascii=False,indent=1)
print('DONE',sum(1 for v in res.values() if v['frac_ground']==0),'of',len(res),'without class 2')
