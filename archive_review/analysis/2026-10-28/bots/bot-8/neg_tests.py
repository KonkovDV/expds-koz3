#!/usr/bin/env python3
"""bot-8: негативные проверки check_submission.py — каждый испорченный архив обязан дать FAIL (код ≠ 0).

python neg_tests.py BASE.zip [--python venv/bin/python]  → tests_out/neg/<случай>.zip, .log, сводка summary.tsv
"""
import argparse, os, subprocess, sys, time, zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "tests_out" / "neg"

NAN_PATCH = '''

_b8_orig_fc = to_feature_collection
def to_feature_collection(feats):  # bot-8 neg-test: NaN в первой вершине
    fc = _b8_orig_fc(feats)
    if fc["features"]:
        ring = fc["features"][0]["geometry"]["coordinates"][0]
        ring[1] = [float("nan"), ring[1][1]]
    return fc
'''


def mutate(base: zipfile.ZipFile, dst: Path, edit) -> None:
    with zipfile.ZipFile(dst, "w") as z:
        extra = []
        for info in base.infolist():
            data = base.read(info)
            res = edit(info.filename, data)
            if res is None:
                continue
            for name, d in (res if isinstance(res, list) else [(info.filename, res)]):
                zi = zipfile.ZipInfo(name, info.date_time)
                zi.compress_type, zi.external_attr = info.compress_type, info.external_attr
                z.writestr(zi, d)


def sub(name, old, new, count=1):
    def f(n, d):
        if n != name:
            return d
        s = d.decode()
        assert old in s, (name, old)
        return s.replace(old, new, count).encode()
    return f


CASES = {
    # имя: (правка, доп. аргументы, нужен смоук)
    "n1_missing_weight": (lambda n, d: None if n == "models/li_r18_hard2.pt" else d, ["--no-smoke"]),
    "n2_las_inside": (lambda n, d: [(n, d), ("data/region_9/x.las", b"LASF")] if n == "metadata.json" else d, ["--no-smoke"]),
    "n3_bad_toml": (sub("configs/default.toml", "[geophys]\n", "[geophys\n"), ["--no-smoke"]),
    "n4_bad_class": (sub("configs/default.toml", "arkhitektury = 1.01", "arhitektury = 1.01"), ["--no-smoke"]),
    "n5_maps_typo": (sub("configs/default.toml", "[geophys.maggr.finds_points]\nbackground_m",
                         "[geophys.maggr]\nmaps = \"grey\"\n\n[geophys.maggr.finds_points]\nbackground_m"), ["--no-smoke"]),
    "n6_nested_root": (lambda n, d: [("koz3/" + n, d)], ["--no-smoke"]),
    "n7_net_import": (sub("expds/pipeline.py", "from __future__ import annotations\n",
                          "from __future__ import annotations\nimport urllib.request\n"), ["--no-smoke"]),
    "n8_crash_empty": (lambda n, d: d + b"\n\ndef run(*a, **k):\n    raise RuntimeError('bot-8 neg-test')\n"
                       if n == "expds/pipeline.py" else d, []),
    "n9_timeout": (sub("expds/pipeline.py", "from __future__ import annotations\n",
                       "from __future__ import annotations\nimport time as _t; _t.sleep(120)\n"), ["--timeout", "20"]),
    "n10_nan": (lambda n, d: d + NAN_PATCH.encode() if n == "expds/writer.py" else d, []),
    "n11_rss_limit": (lambda n, d: d, ["--max-rss-mb", "300"]),
    "n12_bad_metadata": (sub("metadata.json", "inference/main.py", "inference/run.py"), ["--no-smoke"]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base", type=Path)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    base = zipfile.ZipFile(a.base)
    rows = []
    for name, (edit, extra) in CASES.items():
        if a.only and name not in a.only:
            continue
        zp = OUT / f"{name}.zip"
        mutate(base, zp, edit)
        t = time.monotonic()
        r = subprocess.run([sys.executable, str(HERE / "check_submission.py"), str(zp), "--python", a.python,
                            "--workdir", str(HERE / "tmp_check"), *extra], capture_output=True, text=True)
        dt = time.monotonic() - t
        (OUT / f"{name}.log").write_text(r.stdout + r.stderr)
        fails = [ln for ln in r.stdout.splitlines() if ln.startswith("[FAIL]")]
        ok = r.returncode != 0 and "ИТОГ: FAIL" in r.stdout
        rows.append((name, r.returncode, "ok" if ok else "ПРОПУЩЕН", f"{dt:.1f}", fails[0][:160] if fails else "-"))
        print("\t".join(map(str, rows[-1])), flush=True)
        zp.unlink()  # копии по 540 МиБ не храним
    (OUT / "summary.tsv").write_text("case\texit\tverdict\tsec\tfirst_fail\n" + "\n".join("\t".join(map(str, r)) for r in rows) + "\n")
    sys.exit(0 if all(r[2] == "ok" for r in rows) else 1)


if __name__ == "__main__":
    main()
