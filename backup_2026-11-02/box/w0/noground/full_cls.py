# w0s-dem-noground: full-file class histogram (stream, no disk) for every _La_ LAS on S3; creds from env, never printed
import os,struct,json,boto3,numpy as np,sys
from botocore.config import Config
c=boto3.client('s3',endpoint_url=os.environ['S3_ENDPOINT'],aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],config=Config(signature_version='s3v4'))
B=os.environ['S3_BUCKET']; O='/workspace/koz3/w0/noground/full_cls.json'
out=json.load(open(O)) if os.path.exists(O) else {}
prio=['037_','101_','102_','187_']
keys=[l.rstrip('\n').split('\t') for l in open('/workspace/koz3/s3_keys.tsv',encoding='utf-8')]
keys=[k for k in keys if k[0].lower().endswith('.las') and '_La_' in k[0]]
keys.sort(key=lambda k:(not any(k[0].startswith(p) for p in prio),k[0]))
for k,s in keys:
  site=k.split('/')[0]
  if site in out: continue
  body=c.get_object(Bucket=B,Key=k)['Body']; h=body.read(375)
  off=struct.unpack('<I',h[96:100])[0]; fmt=h[104]&0x3f; rl=struct.unpack('<H',h[105:107])[0]
  body.read(off-375); hist=np.zeros(256,np.int64); buf=b''
  while True:
    b=body.read(64<<20)
    if not b: break
    buf+=b; n=len(buf)//rl
    if n:
      rec=np.frombuffer(buf[:n*rl],np.uint8).reshape(n,rl)
      cls=rec[:,16] if fmt>=6 else rec[:,15]&0x1f
      hist+=np.bincount(cls,minlength=256); buf=buf[n*rl:]
  tot=int(hist.sum()); out[site]={'fmt':int(fmt),'n':tot,'cls':{int(i):int(v) for i,v in enumerate(hist) if v},'has_ground':bool(hist[2]>0)}
  print(site,out[site],flush=True); json.dump(out,open(O,'w'),ensure_ascii=False,indent=1)
print('DONE no_ground=',[s for s,v in out.items() if not v['has_ground']])
