import os, sys, boto3, pathlib
env = dict(l.strip().split("=",1) for l in open("/workspace/koz3/local_ctx/s3_train.env") if "=" in l and not l.startswith("#"))
s3 = boto3.client("s3", endpoint_url=env["S3_ENDPOINT"], aws_access_key_id=env["AWS_ACCESS_KEY_ID"], aws_secret_access_key=env["AWS_SECRET_ACCESS_KEY"])
out = pathlib.Path("/workspace/koz3/w4geo/train")
keys = []
for line in open("/workspace/koz3/s3_keys.tsv", encoding="utf-8"):
    k, sz = line.rstrip("\n").split("\t")
    try: reg = int(k[:3])
    except ValueError: continue
    if not (200 <= reg <= 217) or k.endswith("/"): continue
    low = k.lower()
    if low.endswith(".las") or "_li_" in low or "_la_" in low: continue
    keys.append((k, int(sz)))
print(len(keys), sum(s for _,s in keys)/2**30, "GiB", flush=True)
for k, sz in keys:
    p = out / k
    if p.exists() and p.stat().st_size == sz: continue
    p.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(env["S3_BUCKET"], k, str(p) + ".part"); os.replace(str(p)+".part", p)
    print("ok", k, sz, flush=True)
print("ALLDONE", flush=True)
