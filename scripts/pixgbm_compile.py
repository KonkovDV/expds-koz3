#!/usr/bin/env python3
"""Сборка бэкендов predict PIXGBM рядом с моделями. Нужны только на машине сборки:
lleaves (MIT) + llvmlite 0.44 (BSD-2 / Apache-2.0 + LLVM exception) + gcc; treelite/tl2cgen (Apache-2.0) — для --treelite.

    python scripts/pixgbm_compile.py models/pixgbm/all                 # → poly.lleaves.so/.json, finds.lleaves.so/.json
    python scripts/pixgbm_compile.py models/pixgbm/all --treelite      # + poly.tl2cgen.so, finds.tl2cgen.so

lleaves: target_cpu по умолчанию "x86-64" (базовый набор, без AVX-512 — check_submission требует «без zmm»,
и i9-14900KS AVX-512 не имеет). Объект lleaves (reloc=pic) линкуется gcc -shared в .so; в рантайме он грузится
ctypes (expds/geo_pixgbm.SoForest) — ни lleaves, ни llvmlite в образе не нужны. В .json — sha256 модели:
при несовпадении рантайм откатывается на lightgbm.
"""
import argparse, hashlib, json, subprocess, sys, tempfile, time
from pathlib import Path


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--names", nargs="*", default=["poly", "finds"])
    ap.add_argument("--target-cpu", default="x86-64")
    ap.add_argument("--treelite", action="store_true")
    a = ap.parse_args()
    md = Path(a.model_dir)
    for k in a.names:
        txt = md / f"{k}.txt"
        if not txt.exists():
            continue
        import lleaves, llvmlite
        t = time.perf_counter()
        with tempfile.TemporaryDirectory() as td:
            obj = Path(td) / f"{k}.o"
            m = lleaves.Model(model_file=str(txt))
            m.compile(cache=str(obj), target_cpu=a.target_cpu, target_cpu_features="")
            so = md / f"{k}.lleaves.so"
            subprocess.run(["gcc", "-shared", "-o", str(so), str(obj)], check=True)
        zmm = subprocess.run(["objdump", "-d", str(so)], capture_output=True, text=True).stdout.count("zmm")
        meta = dict(sha256=sha256(txt), num_feature=m.num_feature(), target_cpu=a.target_cpu,
                    lleaves=getattr(lleaves, "__version__", "?"), llvmlite=llvmlite.__version__,
                    compile_s=round(time.perf_counter() - t, 1), so_bytes=so.stat().st_size, zmm=zmm)
        (md / f"{k}.lleaves.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        print(k, "lleaves", meta, flush=True)
        if a.treelite:
            import treelite, tl2cgen
            t = time.perf_counter()
            lib = md / f"{k}.tl2cgen.so"
            tl2cgen.export_lib(treelite.frontend.load_lightgbm_model(str(txt)), toolchain="gcc", libpath=str(lib),
                               params={"parallel_comp": 8}, nthread=2, options=["-O2"])
            print(k, "tl2cgen", dict(compile_s=round(time.perf_counter() - t, 1), so_bytes=lib.stat().st_size))


if __name__ == "__main__":
    sys.exit(main())
