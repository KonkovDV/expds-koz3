import json, pathlib
H = pathlib.Path(__file__).parent
F = json.load(open(H / "folds_family_k5.json", encoding="utf-8"))
folds = {"source": F["source"], "family_of": F["family_of"],
         "folds": [{k: f[k] for k in ("fold", "test_families", "val_family", "test_regions")} for f in F["folds"]]}
src = (H / "src/verifier_main.py").read_text(encoding="utf-8")
head, rest = src.split("import json, time, csv", 1)
emb = ("FOLDS = __import__('json').loads(r'''" + json.dumps(folds, ensure_ascii=False) + "''')\n"
       "TABLE_B64 = '" + (H / "table_embed.b64").read_text().strip() + "'\n")
code = head + emb + "import json, time, csv" + rest
(H / "kernel/koz3-nolidar-verifier.py").write_text(code, encoding="utf-8")
md = {"id": "dmitrijkv/koz3-nolidar-verifier", "title": "koz3-nolidar-verifier", "code_file": "koz3-nolidar-verifier.py",
      "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
      "enable_internet": True, "machine_shape": "NvidiaTeslaT4", "dataset_sources": ["dmitrijkv/koz3-s3-cred"], "competition_sources": [],
      "kernel_sources": [], "model_sources": [],
      "docker_image": "gcr.io/kaggle-private-byod/python@sha256:9a6bcfda6d03b2cd9aa347ee8c1bae62043328ed0f4afe9bda014e4f8bbf9435"}
json.dump(md, open(H / "kernel/kernel-metadata.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("kernel bytes", len(code.encode()))
