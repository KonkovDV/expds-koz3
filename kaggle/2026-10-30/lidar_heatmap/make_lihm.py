#!/usr/bin/env python3
"""P-W2-7: собрать ядро elenakonkova/koz3-w2-lihm-c3sc-f01234 из проверенного ядра coord s3f4 (c3-sc, BUILD_RELIEF).
Изменения только флагами: CNX_HEAD=w2hm_head:make_head (голова встраивается в /tmp/w2hm), CNX_TGROUP=li_hm,
jobs c3-sc:0..4, seed 0, deadline 8,5 ч. Значения по умолчанию в исходнике ядра не меняются (CNX_HEAD_DEFAULT, li_cnx)."""
import base64, json, pathlib, sys
HERE = pathlib.Path(__file__).resolve().parent
SRC = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/koz3/orch/coordk/s3f4/koz3-w2-cnx-c3sc-s3f4.py")
SLUG = "koz3-w2-lihm-c3sc-f01234"; ACC = "elenakonkova"
OUT = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else f"/workspace/koz3/w2/kernels/{ACC}/{SLUG}")
s = SRC.read_text()
def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:80], s.count(a)); s = s.replace(a, b)
rep("CNX_JOBS_DEFAULT = 'c3-sc:4'", "CNX_JOBS_DEFAULT = 'c3-sc:0,c3-sc:1,c3-sc:2,c3-sc:3,c3-sc:4'")
rep('os.environ.setdefault("CNX_SEED", "3"); os.environ.setdefault("CNX_DEADLINE_H", "3.0"); os.environ.setdefault("CNX_BUILD_RELIEF", "1")  # bot1 filler: seed3 f4 on acc1 gpu-2',
    '# P-W2-7 li_hm (PREREG_W2_cnx.md): голова QFL-теплокарты шага 1, своя группа порогов li_hm; 5 фолдов в 2 полосах T4×2\n'
    'os.environ.setdefault("CNX_SEED", "0"); os.environ.setdefault("CNX_DEADLINE_H", "8.5"); os.environ.setdefault("CNX_BUILD_RELIEF", "1")\n'
    'os.environ.setdefault("CNX_HEAD", "w2hm_head:make_head"); os.environ.setdefault("CNX_TGROUP", "li_hm")')
head_b64 = base64.b64encode((HERE / "w2hm_head.py").read_bytes()).decode()
rep("import cnx_head as HD\n",
    "import base64 as _b64\nos.makedirs(\"/tmp/w2hm\", exist_ok=True)\n"
    f"open(\"/tmp/w2hm/w2hm_head.py\", \"wb\").write(_b64.b64decode(\"{head_b64}\"))  # P-W2-7, флаг CNX_HEAD\n"
    "sys.path.insert(0, \"/tmp/w2hm\")\nimport cnx_head as HD\n")
rep('threshold_group="li_cnx", original_crs', 'threshold_group=os.environ.get("CNX_TGROUP", "li_cnx"), original_crs')
OUT.mkdir(parents=True, exist_ok=True)
(OUT / f"{SLUG}.py").write_text(s)
meta = json.loads((SRC.parent / "kernel-metadata.json").read_text())
meta.update(id=f"{ACC}/{SLUG}", title=SLUG, code_file=f"{SLUG}.py", kernel_sources=[], dataset_sources=[f"{ACC}/koz3-s3-cred"])
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
print(OUT, len(s))
