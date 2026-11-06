# Driver: for every site with a LAS in the organizers' bucket: download LAS + UTM.json, build 12-ch input, delete LAS.
import os, sys, json, subprocess, concurrent.futures as cf
from pathlib import Path
sys.path.append("/root/.local/lib/python3.13/site-packages")
import boto3
from botocore.config import Config
env = dict(l.strip().split("=", 1) for l in Path.home().joinpath(".secrets/s3_train.env").read_text().splitlines() if "=" in l)
s3 = boto3.client("s3", endpoint_url=env["S3_ENDPOINT"], aws_access_key_id=env["S3_ACCESS_KEY_ID"],
                  aws_secret_access_key=env["S3_SECRET_ACCESS_KEY"], region_name="ru-7",
                  config=Config(s3={"addressing_style": "path"}, retries={"max_attempts": 8}))
B = env["S3_BUCKET"]; OUT = Path("/workspace/c12/relief12"); TMP = Path("/workspace/c12/tmp"); OUT.mkdir(exist_ok=True, parents=True)
keys = [l.rstrip("\n").split("\t", 2) for l in open(os.environ.get("S3_KEYS", sys.argv[1] if len(sys.argv) > 1 else "s3_keys.tsv"), encoding="utf-8")]
las = sorted((k, int(s)) for s, _, k in keys if k.lower().endswith(".las"))
def one(item):
    k, size = item; site = k.split("/")[0]
    if (OUT / f"{site}.npy").exists(): return site, "skip"
    d = TMP / site; d.mkdir(parents=True, exist_ok=True)
    lp = d / Path(k).name; up = d / "UTM.json"
    s3.download_file(B, k, str(lp)); s3.download_file(B, f"{site}/UTM.json", str(up))
    crs = json.loads(up.read_text()).get("crs") or ""
    r = subprocess.run(["/workspace/koz3/venv312/bin/python", "/workspace/c12/prep12.py", str(lp), crs, str(OUT / site)],
                       capture_output=True, text=True, timeout=7200)
    lp.unlink(missing_ok=True)
    return site, (r.stdout.strip().splitlines() or ["?"])[-1] if r.returncode == 0 else f"FAIL rc={r.returncode} {r.stderr[-300:]}"
print(f"{len(las)} LAS, {sum(s for _, s in las) / 1e9:.1f} GB", flush=True)
with cf.ThreadPoolExecutor(2) as ex:
    for site, res in ex.map(one, sorted(las, key=lambda x: -x[1])):
        print(site, res, flush=True)
print("DONE", flush=True)
