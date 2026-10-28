"""Собирает самодостаточные скрипты ядер: mn_lib + main + встроенные фолды (без ключей)."""
import json, pathlib
H = pathlib.Path(__file__).parent
F = json.load(open(H / "folds_matanerdy_k5.json", encoding="utf-8"))
folds = {"source": F["source"], "family_of": F["family_of"],
         "folds": [{k: f[k] for k in ("fold", "test_families", "val_family", "test_regions")} for f in F["folds"]]}
lib = (H / "src/mn_lib.py").read_text(encoding="utf-8")
emb = "FOLDS = json.loads(r'''" + json.dumps(folds, ensure_ascii=False) + "''')\n"
for kind, main, slug, title, gpu, src in [
    ("prep", "prep_main.py", "koz3-mn-prep", "koz3 mn prep", False, []),
    ("train", "train_main.py", "koz3-mn-deeplab", "koz3 mn deeplab", True, ["dmitrijkv/koz3-mn-prep"])]:
    d = H / kind; d.mkdir(exist_ok=True)
    code = lib + "\n" + emb + "\n" + (H / "src" / main).read_text(encoding="utf-8")
    (d / f"{slug}.py").write_text(code, encoding="utf-8")
    md = {"id": f"dmitrijkv/{slug}", "title": title, "code_file": f"{slug}.py", "language": "python",
          "kernel_type": "script", "is_private": True, "enable_gpu": gpu, "enable_tpu": False,
          "enable_internet": True, "dataset_sources": [], "competition_sources": [], "kernel_sources": src}
    if gpu: md["machine_shape"] = "NvidiaTeslaT4"
    json.dump(md, open(d / "kernel-metadata.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(d / f"{slug}.py", len(code))
