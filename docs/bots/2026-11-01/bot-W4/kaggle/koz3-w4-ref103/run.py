# W4: опорный участок 103 (SpOr + UTM.json) из S3 train-expds-3 для пробы весов D-FINE в check_submission. Ключи не печатаются.
import glob, hashlib, json, os, subprocess, sys
from pathlib import Path
try:
    import boto3
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "boto3"], check=True); import boto3
from botocore.config import Config
def cred():
    kid, sec = os.environ.get("KOZ3_S3_ID"), os.environ.get("KOZ3_S3_KEY")
    if kid and sec: return kid, sec
    for p in sorted(glob.glob("/kaggle/input/**/cred.json", recursive=True)):
        try:
            d = json.load(open(p)); kid = d.get("id") or d.get("aws_access_key_id"); sec = d.get("key") or d.get("aws_secret_access_key")
            if kid and sec: return kid, sec
        except Exception as e: print("cred.json unreadable", type(e).__name__)
    raise SystemExit("no S3 creds")
kid, sec = cred()
c = boto3.client("s3", endpoint_url="https://s3.ru-7.storage.selcloud.ru", aws_access_key_id=kid, aws_secret_access_key=sec,
                 region_name="ru-7", config=Config(signature_version="s3v4"))
R = "103_Гнёздово_Ольшанская"
keys = [R + "/UTM.json", R + "/04_Гнёздово_Ольшанская_SpOr_спутник/01_Гнёздово_Ольшанская_SpOr_спутник.tif"]
out = Path("/kaggle/working/ref103"); man = {}
for k in keys:
    d = out / k; d.parent.mkdir(parents=True, exist_ok=True)
    try:
        c.download_file("train-expds-3", k, str(d))
        man[k] = {"size": d.stat().st_size, "sha256": hashlib.sha256(d.read_bytes()).hexdigest()}
    except Exception as e:
        man[k] = {"err": type(e).__name__}
    print(k, man[k], flush=True)
(out / "MANIFEST.json").write_text(json.dumps(man, ensure_ascii=False, indent=1))
