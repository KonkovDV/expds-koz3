# Сборка ядер convnext_lidar из src/convnext_main.py (код + вендор + фолды встраиваются: Kaggle загружает только code_file).
# Ядра: prep (CPU, каналы рельефа), smoke (GPU, 10 ит.), 10 full (GPU, по 2 задания «вариант:фолд» на 2×T4).
import base64, csv, hashlib, io, json, pathlib, shutil, tarfile
H = pathlib.Path(__file__).resolve().parent
BOT9 = H.parent.parent
REPO = BOT9 / "repo"
OWNER = "dmitrijkv"
DOCKER = "gcr.io/kaggle-private-byod/python@sha256:9a6bcfda6d03b2cd9aa347ee8c1bae62043328ed0f4afe9bda014e4f8bbf9435"
PREP_SLUG = "koz3-cnx-li-prep"
import os
HEAD_DEFAULT = os.environ.get("CNX_HEAD_DEFAULT", "centerhead")    # голова по умолчанию во всех ядрах; env CNX_HEAD в ядре переопределяет
SMOKE_REGIONS = "191_Осечки_1.3км,016_Хога"          # 4,4 МБ + 8,1 МБ LAS (bot-9/s3/listing.txt); фолды 3 и 4
FULL = {                                              # slug-хвост → задания (вариант:фолд); 1 задание на GPU
    "c3in-f01": "c3-in:0,c3-in:1", "c3in-f23": "c3-in:2,c3-in:3",
    "c3sc-f01": "c3-sc:0,c3-sc:1", "c3sc-f23": "c3-sc:2,c3-sc:3",
    "c5in-f01": "c5-in:0,c5-in:1", "c5in-f23": "c5-in:2,c5-in:3",
    "c5sc-f01": "c5-sc:0,c5-sc:1", "c5sc-f23": "c5-sc:2,c5-sc:3",
    "c3sc-f4": "c3-sc:4", "c5-f4": "c5-in:4,c5-sc:4",
}
# c3-f4 (c3-in:4 + c3-sc:4) отработал 29.10: c3-in:4 OK (OOF в /workspace/kaggle_out/koz3-cnx-li-c3-f4), c3-sc:4 упал на fork/filelock
# → c3-sc:4 перенесён в отдельное ядро c3sc-f4; ядро koz3-cnx-li-c3-f4 больше НЕ собирается (не перезапускать c3-in:4).
DONE = {"c3-f4": "c3-in:4"}
ONLY = {s.strip() for s in os.environ.get("CNX_BUILD_ONLY", "").split(",") if s.strip()}   # пусто = все ядра; иначе только эти slug

# ── фолды: семьи из index.csv li_tin_59_t128, кластеры — folds_family_k5 (обе карты проверяются в ядре) ─────────
L = json.load(open(BOT9 / "configs/folds_li_tin_59_t128.json", encoding="utf-8"))
fam = {}
with open(L["source"], encoding="utf-8") as fh:
    for row in csv.DictReader(fh): fam[row["region"]] = row["family"]
K = json.load(open(BOT9 / "kaggle/nolidar_verifier/folds_family_k5.json", encoding="utf-8"))
clu = {r: K["family_of"][r] for r in fam if r in K["family_of"]}
folds = {"family": fam, "cluster": clu, "source": [str(BOT9 / "configs/folds_li_tin_59_t128.json"), L["source"],
         str(BOT9 / "kaggle/nolidar_verifier/folds_family_k5.json")],
         "folds": [{"fold": f["fold"], "test_families": f["test_families"]} for f in L["folds"]]}
assert len(fam) == 59, len(fam)

# ── вендор: копии (не правки) expds + Rust .so + relief_ext бота 2 + compute_metrics ───────────────────────────
V = H / "src/vendor"; shutil.rmtree(V, ignore_errors=True); (V / "expds").mkdir(parents=True)
for n in ("accel.py", "lidar.py", "candidates.py", "crs.py"): shutil.copy2(REPO / "expds" / n, V / "expds" / n)
(V / "expds/__init__.py").write_text('"""Минимальный expds для convnext_lidar: accel, lidar, candidates, crs (копии из repo/expds)."""\n', encoding="utf-8")
shutil.copy2(REPO / "inference/lib/expds_core.abi3.so", V / "expds_core.abi3.so")
shutil.copy2(BOT9.parent / "bot-2/relief/relief_ext.py", V / "relief_ext.py")
shutil.copy2(REPO / "materials/compute_metrics.py", V / "compute_metrics.py")
shutil.copy2(H / "src/cnx_head.py", V / "cnx_head.py")                 # сменная голова (HEAD_INTERFACE.md)
CH_REPO = BOT9.parent / "bot-2/repo"; CH_SHA = os.environ.get("CNX_CENTERHEAD_SHA", "788202a")   # голова bot-2: снимок по коммиту
import subprocess
CH_FULL = subprocess.run(["git", "-C", str(CH_REPO), "rev-parse", CH_SHA + "^{commit}"], check=True, capture_output=True, text=True).stdout.strip()
_t = subprocess.run(["git", "-C", str(CH_REPO), "archive", "--format=tar", CH_FULL, "centerhead"], check=True, capture_output=True).stdout
with tarfile.open(fileobj=io.BytesIO(_t)) as tf:                               # git archive (только чтение репо bot-2), без tests/
    for m in tf.getmembers():
        if m.isfile() and "/tests/" not in m.name and not m.name.endswith(".pyc"):
            d = V / m.name; d.parent.mkdir(parents=True, exist_ok=True); d.write_bytes(tf.extractfile(m).read())
files = sorted(p.relative_to(V).as_posix() for p in V.rglob("*") if p.is_file() and p.name != "vendor_manifest.json")
sha = {n: hashlib.sha256((V / n).read_bytes()).hexdigest() for n in files}
man = {"sha256": sha, "source": {"expds/*": str(REPO / "expds"), "expds_core.abi3.so": str(REPO / "inference/lib"),
       "relief_ext.py": str(BOT9.parent / "bot-2/relief/relief_ext.py"), "compute_metrics.py": str(REPO / "materials/compute_metrics.py"),
       "cnx_head.py": str(H / "src/cnx_head.py"), "centerhead/*": f"git {CH_REPO} {CH_FULL}:centerhead"}, "centerhead_commit": CH_FULL}
(V / "vendor_manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=9) as tf:
    for n in files + ["vendor_manifest.json"]:
        ti = tf.gettarinfo(str(V / n), arcname=n); ti.mtime = 0; ti.uid = ti.gid = 0; ti.uname = ti.gname = ""
        with open(V / n, "rb") as fh: tf.addfile(ti, fh)
B64 = base64.b64encode(buf.getvalue()).decode()

# ── GPU-бенч bot-8 (только smoke): ЧИСТАЯ ветка bot8-cpu-a-clean (worktree bot-8/wt-cpua-clean, только чтение) ─────────
# expds/lidar_a + expds/{__init__,cpu_threads}.py — git archive закоммиченного состояния ветки (не рабочего дерева);
# bench/bench_acc_gpu.py — файл bot-8/bench (вне git), фиксируется sha256. wt-cpua / bot8-cpu-a-speedup НЕ используются.
B8 = BOT9.parent / "bot-8"; B8_WT = B8 / "wt-cpua-clean"; B8_BRANCH = "bot8-cpu-a-clean"
B8_SHA = subprocess.run(["git", "-C", str(B8_WT), "rev-parse", os.environ.get("CNX_BENCH_COMMIT", B8_BRANCH) + "^{commit}"],
                        check=True, capture_output=True, text=True).stdout.strip()
_on = subprocess.run(["git", "-C", str(B8_WT), "branch", "--contains", B8_SHA, "--format=%(refname:short)"], check=True, capture_output=True, text=True).stdout.split()
assert B8_BRANCH in _on, ("коммит бенча не в чистой ветке", B8_SHA, _on)
B8_FILES = ["expds/__init__.py", "expds/cpu_threads.py"] + [f"expds/lidar_a/{n}.py" for n in ("__init__", "ground", "terrain", "tiling", "runner", "dtm_workers")]
_t = subprocess.run(["git", "-C", str(B8_WT), "archive", "--format=tar", B8_SHA, *B8_FILES], check=True, capture_output=True).stdout
bench = {}
with tarfile.open(fileobj=io.BytesIO(_t)) as tf:
    for m in tf.getmembers():
        if m.isfile(): bench["repo/" + m.name] = tf.extractfile(m).read()
assert sorted(bench) == sorted("repo/" + n for n in B8_FILES), sorted(bench)
bench["bench/bench_acc_gpu.py"] = (B8 / "bench/bench_acc_gpu.py").read_bytes()
import re as _re
for n, b in bench.items():                               # лицензионный барьер: ни строки кода EhimenNathan в бандле
    assert not _re.search(rb"(?i)ehimen|nathan", b), ("бенч: найден след EhimenNathan", n)
bman = {"branch": B8_BRANCH, "commit": B8_SHA, "worktree": str(B8_WT), "from": "git archive (закоммиченное состояние)",
        "bench_acc_gpu.py": str(B8 / "bench/bench_acc_gpu.py") + " (вне git, sha256 ниже)", "doc": str(B8 / "bench/acc_gpu_T4.md"),
        "sha256": {n: hashlib.sha256(b).hexdigest() for n, b in sorted(bench.items())}}
bbuf = io.BytesIO()
with tarfile.open(fileobj=bbuf, mode="w:gz", compresslevel=9) as tf:
    for n, b in sorted(bench.items()) + [("bench_manifest.json", json.dumps(bman, ensure_ascii=False, indent=1).encode())]:
        ti = tarfile.TarInfo(n); ti.size = len(b); ti.mtime = 0; ti.mode = 0o644; tf.addfile(ti, io.BytesIO(b))
BENCH_B64 = base64.b64encode(bbuf.getvalue()).decode()

src = (H / "src/convnext_main.py").read_text(encoding="utf-8")
assert src.count("# @@EMBED@@") == 1
def emit(name, mode, jobs, regions, gpu, ksrc, extra=""):
    emb = (f"{extra}CNX_MODE_DEFAULT = {mode!r}\nCNX_HEAD_DEFAULT = {HEAD_DEFAULT!r}\nCNX_JOBS_DEFAULT = {jobs!r}\nCNX_REGIONS_DEFAULT = {regions!r}\n"
           f"FOLDS = __import__('json').loads(r'''{json.dumps(folds, ensure_ascii=False)}''')\nVENDOR_B64 = {B64!r}")
    if ONLY and name not in ONLY: return None
    d = H / name; d.mkdir(exist_ok=True)
    code = src.replace("# @@EMBED@@", emb); (d / f"{name}.py").write_text(code, encoding="utf-8")
    md = {"id": f"{OWNER}/{name}", "title": name, "code_file": f"{name}.py", "language": "python", "kernel_type": "script",
          "is_private": True, "enable_gpu": gpu, "enable_tpu": False, "enable_internet": True,
          "dataset_sources": [f"{OWNER}/koz3-s3-cred"], "competition_sources": [], "kernel_sources": ksrc, "model_sources": [],
          "docker_image": DOCKER}
    if gpu: md["machine_shape"] = "NvidiaTeslaT4"
    json.dump(md, open(d / "kernel-metadata.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return d, len(code.encode())
out = [emit(PREP_SLUG, "prep", "", "", False, [])]
out.append(emit("koz3-cnx-li-smoke", "smoke", "c5-in:4,c3-sc:3", SMOKE_REGIONS, True, [], f"BENCH_B64 = {BENCH_B64!r}\n"))
for tail, jobs in FULL.items(): out.append(emit(f"koz3-cnx-li-{tail}", "full", jobs, "", True, [f"{OWNER}/{PREP_SLUG}"]))
# все 20 пар (вариант, фолд) покрыты ровно один раз
pairs = [p for j in [*FULL.values(), *DONE.values()] for p in j.split(",")]
assert len(pairs) == len(set(pairs)) == 20, pairs
for d, n in filter(None, out): print(d.name, n, "байт")
print("vendor sha256:", json.dumps(sha, indent=0))
print("bench bot-8:", B8_BRANCH, B8_SHA, json.dumps(bman["sha256"], indent=0))
