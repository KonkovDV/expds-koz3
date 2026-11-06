#!/usr/bin/env python3
"""Архив-кандидат «v2 + лидар c5-in» после PASS гейта (PREREG c5-in, «Если PASS»; поправка 1 — режим head).

База — v2 (`eb268d6`, LB 0,41238): дерево `main` (HEAD) + артефакты v2 (`scripts/v2_release_artifacts.py`) + конфиг
`eb268d6:configs/default.toml`. 05.11 это побайтно равно финалу v2 на 5 участках S3 (`docs/AUDIT_2026-11-05.md` §1.1).
К конфигу дописывается секция `[li_cnx]`: decode = "head", веса пяти фолдов (среднее логитов), группа порогов li_cnxin
с медианой порогов 10 честных фолдов из отчёта гейта; вариант R — `replace_li_kurgans = true`.

    python make_candidate.py --variant K|R --gate gate_cnxin_ctx.json --weights W0.pt ... W4.pt --tree DIR --out ZIP
        [--base-config FR1h/configs/default.toml --base-sha256 HEX]

`--base-config` — другой базовый конфиг вместо `eb268d6:configs/default.toml`, например конфиг из архива FR1h
(`submission_fr1h_on_eb268d6.zip`: v2 + `[post.frame_prior]` + 4 порога курганов спутника). Код FR1h в `main` есть;
сборку с ним сверять с архивом FR1h на участках SpOr без LAS — там фаза cnx не работает и ответ обязан совпасть побайтно.

Сборка — `scripts/build_submission.py --allow-en` (линия EN по решению владельца 04.11). Ключи и токены не нужны.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
V2 = "eb268d6"
GROUP = "li_cnxin"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cnx_section(variant: str, weights: list[str], taus: dict, weights_b: list[str] | None = None) -> str:
    lines = ["", "# 05.11 кандидат v2 + c5-in (kaggle/2026-11-05/li_cnx_in/make_candidate.py): курганы ConvNeXt-T ImageNet,",
             f"# декодирование как в OOF ядра (decode = \"head\", паритет PASS), вариант {variant} гейта c5-in.",
             "[li_cnx]", "enabled = true",
             "weights = [" + ", ".join(json.dumps(w) for w in weights) + "]",
             *(["# ансамбль (PREREG li_ens_in): второй набор весов, логиты 10 сетей усредняются",
                "weights_b = [" + ", ".join(json.dumps(w) for w in weights_b) + "]"] if weights_b else []),
             'decode = "head"', f'threshold_group = "{GROUP}"',
             'classes = ["kurgany_povrezhdennye", "kurgany_tselye"]',
             f"replace_li_kurgans = {'true' if variant == 'R' else 'false'}", "",
             f"[thresholds_by_markup.{GROUP}]",
             "# медиана порогов 10 честных фолдов гейта (gate_cnxin_ctx.json → taus_group.median)"]
    lines += [f"{c} = {float(v):.6f}" for c, v in sorted(taus.items())]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--variant", required=True, choices=("K", "R"))
    ap.add_argument("--gate", type=Path, required=True)
    ap.add_argument("--weights", type=Path, nargs=5, required=True, help="weights_cnx_c5-in_f0.pt … _f4.pt")
    ap.add_argument("--weights-b", type=Path, nargs=5, default=None,
                    help="ансамбль: второй набор, например weights_cnx_c12-in_f0.pt … _f4.pt (гейт — gate_e512in_ctx.json)")
    ap.add_argument("--tree", type=Path, required=True, help="пустой каталог сборки (вне репозитория)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base-config", type=Path, default=None, help="базовый конфиг вместо eb268d6 (например FR1h)")
    ap.add_argument("--sp-yolo", type=Path, default=None, help="TorchScript YOLO спутника (expds/sp_yolo.py)")
    ap.add_argument("--sp-yolo-thr", type=float, default=0.10, help="порог группы sp_yolo (по OOF)")
    ap.add_argument("--sp-yolo-replace", action="store_true", help="вариант R: YOLO заменяет спутник участка")
    ap.add_argument("--sp-yolo-classes", default="kurgany_povrezhdennye,kurgany_tselye,gorodishcha,arkhitektury")
    ap.add_argument("--wheels-dir", type=Path, default=Path("/workspace/whl"),
                    help="колёса inference/cnx/wheels (в git их нет, .gitignore); sha256 сверяется с inference/cnx/MANIFEST.json")
    ap.add_argument("--base-sha256", default="", help="ожидаемый sha256 файла --base-config (обязателен вместе с ним)")
    a = ap.parse_args()
    if a.base_config is not None and (not a.base_sha256 or sha256(a.base_config) != a.base_sha256):
        sys.exit(f"--base-config: sha256 {sha256(a.base_config) if a.base_config.is_file() else 'нет файла'} ≠ "
                 f"--base-sha256 {a.base_sha256 or '(не задан)'}")
    gate = json.loads(a.gate.read_text(encoding="utf-8"))
    name = {"K": "K_add_kurgans", "R": "R_replace_li_kurgans"}[a.variant]
    v = gate["variants"][name]
    if not v["gate"]["PASS"]:
        sys.exit(f"вариант {a.variant} не прошёл гейт — кандидат не собирается")
    taus = v["taus_group"]["median"]
    if set(taus) != {"kurgany_povrezhdennye", "kurgany_tselye"}:
        sys.exit(f"нет порогов обоих классов курганов: {taus}")
    if a.tree.exists() and any(a.tree.iterdir()):
        sys.exit(f"{a.tree} не пуст")
    a.tree.mkdir(parents=True, exist_ok=True)
    arch = subprocess.run(["git", "-C", str(REPO), "archive", "HEAD"], capture_output=True, check=True).stdout
    subprocess.run(["tar", "-x", "-C", str(a.tree)], input=arch, check=True)
    env = dict(os.environ, EXPDS_V2_DEST=str(a.tree))
    subprocess.run([sys.executable, str(REPO / "scripts" / "v2_release_artifacts.py")], env=env, check=True)
    cnx_man = json.loads((a.tree / "inference" / "cnx" / "MANIFEST.json").read_text(encoding="utf-8"))
    for rel, want in (cnx_man.get("sha256") or {}).items():          # {путь: sha256}, в т.ч. wheels/*.whl
        dst = a.tree / "inference" / "cnx" / rel
        if not rel.startswith("wheels/") or not isinstance(want, str) or dst.is_file():
            continue
        src = a.wheels_dir / Path(rel).name
        if not src.is_file() or sha256(src) != want:
            sys.exit(f"нет колеса {rel} с sha256 {want[:16]}… в {a.wheels_dir} (скачать с PyPI и сверить)")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"колесо {rel} ← {src} (sha256 {want[:16]}… сходится)", flush=True)
    lost = [r for r in (cnx_man.get("sha256") or {}) if r.startswith("wheels/") and not (a.tree / "inference" / "cnx" / r).is_file()]
    if lost:
        sys.exit(f"после копирования нет колёс: {lost}")
    cfg = (a.base_config.read_text(encoding="utf-8") if a.base_config is not None else
           subprocess.run(["git", "-C", str(REPO), "show", f"{V2}:configs/default.toml"], capture_output=True,
                          check=True).stdout.decode("utf-8"))
    if "[li_cnx]" in cfg:
        sys.exit("в базовом конфиге уже есть [li_cnx]")
    rels = []
    man_p = a.tree / "inference" / "cnx" / "WEIGHTS_MANIFEST.json"
    man = json.loads(man_p.read_text(encoding="utf-8"))
    sets = [(f, w) for f, w in enumerate(a.weights)] + [(f, w) for f, w in enumerate(a.weights_b or [])]
    rels_b = []
    for i, (f, w) in enumerate(sets):
        if not w.name.endswith(f"_f{f}.pt"):
            sys.exit(f"{w}: ожидается файл фолда {f} (…_f{f}.pt)")
        rel = f"models/cnx/{w.name}"
        dst = a.tree / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(w, dst)
        except OSError:
            shutil.copy2(w, dst)
        man["weights"] = [e for e in man["weights"] if e.get("path") != rel] + [
            {"path": rel, "sha256": sha256(dst), "size": dst.stat().st_size, "source": str(w),
             "status": f"{w.name.split('weights_cnx_')[-1].rsplit('_f', 1)[0]} fold {f}, ImageNet init; "
                       f"гейт {Path(a.gate).name} вариант {a.variant}"}]
        (rels if i < len(a.weights) else rels_b).append(rel)
    man_p.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    extra = ""
    if a.sp_yolo is not None:
        rel_y = f"models/sp_yolo/{a.sp_yolo.name}"
        (a.tree / rel_y).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(a.sp_yolo, a.tree / rel_y)
        cls_y = [c for c in a.sp_yolo_classes.split(",") if c]
        extra = ("\n# 05.11 YOLO спутника (docs/plans/OPTICS_DETECTOR_PLAN_20261105.md): фолд 4 OOF — YOLO 0,248 против стека 0,146\n"
                 "[sp_yolo]\nenabled = true\n" + f"weights = {json.dumps(rel_y)}\n" +
                 f"replace_sp = {'true' if a.sp_yolo_replace else 'false'}\n" +
                 "classes = [" + ", ".join(json.dumps(c) for c in cls_y) + "]\n\n[thresholds_by_markup.sp_yolo]\n" +
                 "".join(f"{c} = {a.sp_yolo_thr:.6f}\n" for c in cls_y))
    (a.tree / "configs" / "default.toml").write_text(cfg + cnx_section(a.variant, rels, taus, rels_b) + extra, encoding="utf-8")
    r = subprocess.run([sys.executable, "scripts/build_submission.py", "--allow-en", "--out", str(a.out.resolve())],
                       cwd=str(a.tree), capture_output=True, text=True)
    sys.stdout.write(r.stdout[-3000:])
    if r.returncode:
        sys.exit(r.stderr[-2000:])
    print(json.dumps({"variant": a.variant, "taus": taus, "base_config": str(a.base_config or f"{V2}:configs/default.toml"),
                      "zip": str(a.out), "sha256": sha256(a.out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
