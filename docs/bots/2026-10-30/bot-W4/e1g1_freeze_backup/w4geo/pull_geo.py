#!/usr/bin/env python3
"""Pull geo sites 200-217 (MagGr/EtGr/TerFl + разметка + UTM.json/extra) from S3. Never log secrets."""
import os, sys, time, re
from pathlib import Path
import boto3
from botocore.config import Config
ROOT = Path("/workspace/koz3/w4geo/raw")
SKIP = ("_Li_", "_Ae_", "_Or_", "_SpOr_", "_La_")
def main():
    c = boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL") or os.environ["S3_ENDPOINT"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"], aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        config=Config(retries={"max_attempts": 8, "mode": "adaptive"}))
    b = os.environ.get("S3_BUCKET") or "train-expds-3"
    t0=time.time(); n=0; tot=0
    for p in c.get_paginator("list_objects_v2").paginate(Bucket=b, Prefix="2", Delimiter="/"):
        for cp in p.get("CommonPrefixes") or []:
            site = cp["Prefix"]
            m = re.match(r"(\d{3})_", site)
            if not m or not (200 <= int(m.group(1)) <= 217): continue
            print("SITE", site, flush=True)
            for pg in c.get_paginator("list_objects_v2").paginate(Bucket=b, Prefix=site):
                for o in pg.get("Contents") or []:
                    k, sz = o["Key"], o["Size"]
                    if k.endswith("/") or any(s in k for s in SKIP): continue
                    d = ROOT / k; d.parent.mkdir(parents=True, exist_ok=True)
                    if not (d.exists() and d.stat().st_size == sz):
                        c.download_file(b, k, str(d))
                    n += 1; tot += sz
    print(f"DONE files={n} MB={tot/2**20:.1f} sec={time.time()-t0:.1f}", flush=True)
    (ROOT/"PULL_DONE").write_text(f"files={n} bytes={tot}\n")
if __name__ == "__main__": sys.exit(main())
