import json,subprocess,time,os,sys
from pathlib import Path
exec(open('plan.py').read().split("print(len(objs)")[0])
t=[o for o in tif if not dest(o[0]).exists() and 100e6<o[1]<600e6][::-1]
W=[int(x) for x in sys.argv[1:]]
k=0
for w in W:
    sel=t[k:k+w*2]; k+=w*2
    tmp=Path('/workspace/data/s3_tmp/bench'); tmp.mkdir(parents=True,exist_ok=True)
    cmds='\n'.join(f'cp --concurrency 8 "s3://train-expds-3/{r}" "{tmp}/{i}_{Path(r).name}"' for i,(r,s) in enumerate(sel))
    open('/tmp/run.txt','w').write(cmds+'\n')
    t0=time.time(); subprocess.run(f'. ~/.koz3s3/env; ~/.local/bin/s5cmd --endpoint-url $S3EP --numworkers {w} run /tmp/run.txt',shell=True,check=True,executable='/bin/bash')
    dt=time.time()-t0; b=sum(s for _,s in sel); print(w,'files',len(sel),round(b/1e9,2),'GB',round(dt,1),'s',round(b/1e6/dt,1),'MB/s',flush=True)
    subprocess.run(['rm','-rf',str(tmp)])
