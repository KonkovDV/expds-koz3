import os, pathlib, boto3
env = dict(l.strip().split("=",1) for l in open("/workspace/koz3/local_ctx/s3_train.env") if "=" in l and not l.startswith("#"))
ep = env.get("S3_ENDPOINT") or env.get("AWS_ENDPOINT_URL")
s3 = boto3.client("s3", endpoint_url=ep, aws_access_key_id=env["AWS_ACCESS_KEY_ID"], aws_secret_access_key=env["AWS_SECRET_ACCESS_KEY"])
out = pathlib.Path("/workspace/koz3/w4geo/train")
keys=[]
for line in open("/workspace/koz3/s3_keys_geo_200_217.tsv", encoding="utf-8"):
    k, sz = line.rstrip("\n").split("\t")
    keys.append((k, int(sz)))
print(len(keys), sum(s for _,s in keys)/2**30, "GiB", flush=True)
done=skip=0
for k, sz in keys:
    p = out / k
    if p.exists() and p.stat().st_size == sz:
        skip+=1; continue
    p.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(env["S3_BUCKET"], k, str(p) + ".part"); os.replace(str(p)+".part", p)
    done+=1
    if done % 25 == 0: print("ok", done, k, sz, flush=True)
print("ALLDONE", done, "skipped", skip, flush=True)
