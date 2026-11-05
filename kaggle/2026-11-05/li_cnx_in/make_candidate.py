#!/usr/bin/env python3
"""Архив-кандидат «v2 + лидар c5-in» после PASS гейта (PREREG c5-in, «Если PASS»; поправка 1 — режим head).

База — v2 (`eb268d6`, LB 0,41238): дерево `main` (HEAD) + артефакты v2 (`scripts/v2_release_artifacts.py`) + конфиг
`eb268d6:configs/default.toml`. 05.11 это побайтно равно финалу v2 на 5 участках S3 (`docs/AUDIT_2026-11-05.md` §1.1).
К конфигу дописывается секция `[li_cnx]`: decode = "head", веса пяти фолдов (среднее логитов), группа порогов li_cnxin
с медианой порогов 10 честных фолдов из отчёта гейта; вариант R — `replace_li_kurgans = true`.

    python make_candidate.py --variant K|R --gate gate_cnxin_ctx.json --weights W0.pt ... W4.pt --tree DIR --out ZIP

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


def cnx_section(variant: str, weights: list[str], taus: dict) -> str:
    lines = ["", "# 05.11 кандидат v2 + c5-in (kaggle/2026-11-05/li_cnx_in/make_candidate.py): курганы ConvNeXt-T ImageNet,",
             f"# декодирование как в OOF ядра (decode = \"head\", паритет PASS), вариант {variant} гейта c5-in.",
             "[li_cnx]", "enabled = true",
             "weights = [" + ", ".join(json.dumps(w) for w in weights) + "]",
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
    ap.add_argument("--tree", type=Path, required=True, help="пустой каталог сборки (вне репозитория)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
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
    cfg = subprocess.run(["git", "-C", str(REPO), "show", f"{V2}:configs/default.toml"], capture_output=True,
                         check=True).stdout.decode("utf-8")
    rels = []
    man_p = a.tree / "inference" / "cnx" / "WEIGHTS_MANIFEST.json"
    man = json.loads(man_p.read_text(encoding="utf-8"))
    for f, w in enumerate(a.weights):
        rel = f"models/cnx/weights_cnx_c5-in_f{f}.pt"
        dst = a.tree / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(w, dst)
        except OSError:
            shutil.copy2(w, dst)
        man["weights"] = [e for e in man["weights"] if e.get("path") != rel] + [
            {"path": rel, "sha256": sha256(dst), "size": dst.stat().st_size, "source": str(w),
             "status": f"c5-in fold {f} (dmitrijkv/koz3-li-cnx-in-*), ImageNet init; гейт c5-in вариант {a.variant}"}]
        rels.append(rel)
    man_p.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    (a.tree / "configs" / "default.toml").write_text(cfg + cnx_section(a.variant, rels, taus), encoding="utf-8")
    r = subprocess.run([sys.executable, "scripts/build_submission.py", "--allow-en", "--out", str(a.out.resolve())],
                       cwd=str(a.tree), capture_output=True, text=True)
    sys.stdout.write(r.stdout[-3000:])
    if r.returncode:
        sys.exit(r.stderr[-2000:])
    print(json.dumps({"variant": a.variant, "taus": taus, "zip": str(a.out), "sha256": sha256(a.out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
