"""Замер скорости S3 малыми range-GET (данные не пишутся на диск). Ключи — из окружения, не печатаются."""
import os, sys, time, json, concurrent.futures as cf
import boto3
from botocore.config import Config
s3 = boto3.client("s3", endpoint_url=os.environ["S3EP"], config=Config(max_pool_connections=16, retries={"max_attempts": 5}))
B = "train-expds-3"
key = sys.argv[1]; chunk = int(sys.argv[2]) << 20; par = int(sys.argv[3]); n = int(sys.argv[4])
h = s3.head_object(Bucket=B, Key=key); size = h["ContentLength"]
def get(i):
    a = (i * chunk) % max(1, size - chunk)
    t = time.time(); b = s3.get_object(Bucket=B, Key=key, Range=f"bytes={a}-{a+chunk-1}")["Body"].read(); return len(b), time.time() - t
t0 = time.time()
with cf.ThreadPoolExecutor(par) as ex: res = list(ex.map(get, range(n)))
dt = time.time() - t0; tot = sum(r[0] for r in res)
print(json.dumps({"key_size": size, "chunk_MB": chunk >> 20, "parallel": par, "requests": n, "MB": round(tot / 1e6, 1),
                  "sec": round(dt, 2), "MB_s": round(tot / 1e6 / dt, 1), "head_ok": True, "time_msk": time.strftime("%H:%M:%S")}, ensure_ascii=False))
