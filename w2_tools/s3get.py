# usage: s3get.py <dst_dir> <prefix>...   (creds from env, never printed)
import os,sys,boto3
from botocore.config import Config
c=boto3.client('s3',endpoint_url=os.environ['S3_ENDPOINT'],aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],config=Config(signature_version='s3v4'))
B=os.environ['S3_BUCKET']; dst=sys.argv[1]
keys=[l.split('\t') for l in open('/workspace/koz3/s3_keys.tsv',encoding='utf-8')]
for p in sys.argv[2:]:
  for k,s in [(k,int(s)) for k,s in keys if k.startswith(p) and int(s)>0]:
    if k.endswith(('_Li_c.tif','_Li_ch.tif')): continue
    o=os.path.join(dst,k); os.makedirs(os.path.dirname(o),exist_ok=True)
    if os.path.exists(o) and os.path.getsize(o)==s: continue
    c.download_file(B,k,o); print('ok',k,s,flush=True)
