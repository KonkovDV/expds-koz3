# Range-read first ~3MB of every _La_ LAS on S3 and report classification histogram (creds from env, never printed)
import os,io,struct,json,boto3,numpy as np
from botocore.config import Config
c=boto3.client('s3',endpoint_url=os.environ['S3_ENDPOINT'],aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],config=Config(signature_version='s3v4'))
B=os.environ['S3_BUCKET']; out={}
keys=[l.rstrip('\n').split('\t') for l in open('/workspace/koz3/s3_keys.tsv',encoding='utf-8')]
for k,s in keys:
  if not (k.lower().endswith('.las') and '_La_' in k): continue
  b=c.get_object(Bucket=B,Key=k,Range='bytes=0-3000000')['Body'].read()
  ver=b[24:26]; off=struct.unpack('<I',b[96:100])[0]; fmt=b[104]&0x3f; rl=struct.unpack('<H',b[105:107])[0]
  n=(len(b)-off)//rl; rec=np.frombuffer(b[off:off+n*rl],np.uint8).reshape(n,rl)
  if fmt>=6: cls=rec[:,16]; rn=rec[:,14]&0x0f; nr=rec[:,14]>>4
  else: cls=rec[:,15]&0x1f; rn=rec[:,14]&7; nr=(rec[:,14]>>3)&7
  u,cn=np.unique(cls,return_counts=True)
  out[k.split('/')[0]]={'size_mb':round(int(s)/1e6),'fmt':int(fmt),'n_sample':int(n),'cls':{int(a):round(float(x)/n,3) for a,x in zip(u,cn)},'last_return_frac':round(float((rn==nr).mean()),3)}
  print(k.split('/')[0],out[k.split('/')[0]],flush=True)
json.dump(out,open('/workspace/koz3/w2/las/cls_probe.json','w'),ensure_ascii=False,indent=1)
