# @@S3@@
CLI = s3_client()
DATA = TMP / "data"; DATA.mkdir(parents=True, exist_ok=True)
def df_tmp():
    st = os.statvfs(str(TMP)); free_gb = st.f_bavail * st.f_frsize / 2**30
    log("df", TMP, f"{free_gb:.1f} ГБ свободно"); SUMMARY["tmp_free_gb"] = round(free_gb, 2); dump(); return free_gb
df_tmp()
# список регионов
regions = []
for pref in list_prefix(CLI, "", "/"):
    name = pref.rstrip("/")
    if re.match(r"^\d{3}_", name): regions.append(name)
log("S3 регионов", len(regions))
SMOKE_REGIONS = [x for x in os.environ.get("B9_REGIONS", "185_Изборск,015_Демидовка,034_Усть_река").split(",") if x]
want = set(SMOKE_REGIONS) if MODE == "smoke" else None

def download_region(reg):
    dest = DATA / reg
    if dest.is_symlink(): dest.unlink()
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for k, s in list_prefix(CLI, reg + "/"):
        name = k.rsplit("/", 1)[-1]
        if name == "UTM.json" or ("_SpOr_" in name and name.lower().endswith((".tif", ".tiff"))):
            # fetch пишет в root/key — нам нужен dest/name
            from boto3.s3.transfer import TransferConfig
            dst = dest / name
            if dst.exists() and dst.stat().st_size == s:
                n += 1; continue
            tmp = dst.with_name(dst.name + ".part")
            for a in range(5):
                try:
                    CLI.download_file(BUCKET, k, str(tmp), Config=TransferConfig(max_concurrency=4))
                    tmp.rename(dst); assert dst.stat().st_size == s; n += 1; break
                except Exception as e:
                    log("повтор", a + 1, k, type(e).__name__); time.sleep(5 * (a + 1))
            else:
                raise RuntimeError("не скачан " + k)
    return n

downloaded = []
for reg in regions:
    if want is not None and reg not in want: continue
    try:
        n = download_region(reg)
        if n: downloaded.append(reg); log("скачан", reg, "файлов", n)
    except Exception as e:
        log("пропуск", reg, repr(e))
SUMMARY["regions_downloaded"] = len(downloaded); dump()
log("скачано участков", len(downloaded))
if len(downloaded) < (1 if MODE == "smoke" else 10):
    raise SystemExit(f"мало участков SpOr: {len(downloaded)}")

# GT
gthits = sorted(glob.glob("/kaggle/input/**/gt_train_v2.geojson.gz", recursive=True))
gthits += sorted(glob.glob("/kaggle/input/**/gt_train_v2.geojson", recursive=True))
if not gthits: raise SystemExit("нет gt_train_v2 — dataset dmitrijkv/koz3-gt-train-v2")
gt_src = Path(gthits[0]); GT = TMP / "gt.geojson"
if str(gt_src).endswith(".gz"):
    GT.write_bytes(gzip.decompress(gt_src.read_bytes()))
else:
    shutil.copy(gt_src, GT)
got = hashlib.sha256(GT.read_bytes()).hexdigest()
assert got == GT_SHA256, f"GT sha {got} != {GT_SHA256}"
log("GT OK", got[:16], GT.stat().st_size)
df_tmp()
