#!/usr/bin/env python3
"""Build /workspace/kaggle/push/geo-gate from bots/bot7-geo-gate tip.

Inlines kaggle/common/koz3_s3.py into koz3-bot7-geo-gate.py between
BEGIN/END markers. Does NOT push to Kaggle.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_HELPER = ROOT / "kaggle" / "common" / "koz3_s3.py"
SRC_KERNEL_TMPL = ROOT / "kaggle" / "geo-gate" / "koz3-bot7-geo-gate.py"
PKG = Path("/workspace/kaggle/push/geo-gate")
BEGIN = "# --- BEGIN koz3_s3 (inlined from kaggle/common/koz3_s3.py; do not edit by hand) ---"
END = "# --- END koz3_s3 ---"


def tip_sha() -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()


def inline_helper(kernel_text: str) -> str:
    helper = SRC_HELPER.read_text(encoding="utf-8")
    # __future__ only legal at file start — drop it when inlining mid-file
    lines = []
    for ln in helper.splitlines(True):
        if ln.startswith("from __future__"):
            continue
        lines.append(ln)
    helper = "".join(lines)
    block = BEGIN + "\n" + helper + ("\n" if not helper.endswith("\n") else "") + END + "\n"
    if BEGIN in kernel_text and END in kernel_text:
        pre, _, rest = kernel_text.partition(BEGIN)
        _, _, post = rest.partition(END)
        # post still starts after END line content — drop first line remnant
        if post.startswith("\n"):
            post = post[1:]
        elif post.lstrip().startswith("#"):
            # END may leave trailing on same structure
            pass
        # re-find clean: remove from BEGIN through END inclusive
        import re

        return re.sub(
            re.escape(BEGIN) + r".*?" + re.escape(END) + r"\n?",
            block,
            kernel_text,
            count=1,
            flags=re.S,
        )
    # insert after imports marker
    needle = "# <<INLINE_KOZ3_S3>>\n"
    if needle not in kernel_text:
        raise SystemExit("kernel template missing INLINE marker or BEGIN/END")
    return kernel_text.replace(needle, block)


def main() -> None:
    PKG.mkdir(parents=True, exist_ok=True)
    tmpl = SRC_KERNEL_TMPL.read_text(encoding="utf-8")
    (PKG / "koz3-bot7-geo-gate.py").write_text(inline_helper(tmpl), encoding="utf-8")
    meta = {
        "id": "dmitrijkv/koz3-bot7-geo-gate",
        "title": "koz3-bot7-geo-gate",
        "code_file": "koz3-bot7-geo-gate.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": ["cpu", "honest-10", "geo-gate", "bot7", "s3"],
        "dataset_sources": [
            "dmitrijkv/koz3-gt-train-v2",
            "dmitrijkv/koz3-xfer-s000",
            "dmitrijkv/koz3-s3-cred",
        ],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
    }
    (PKG / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    # copy notes / helper mirror for offline inspect
    notes = PKG / "notes"
    notes.mkdir(exist_ok=True)
    src_notes = ROOT / "notes" / "lb_ceiling_types.json"
    if src_notes.exists():
        shutil.copy2(src_notes, notes / "lb_ceiling_types.json")
    common = PKG / "common"
    common.mkdir(exist_ok=True)
    shutil.copy2(SRC_HELPER, common / "koz3_s3.py")
    sha = tip_sha()
    readme = ROOT / "kaggle" / "geo-gate" / "README.md"
    if readme.exists():
        shutil.copy2(readme, PKG / "README.md")
    (PKG / "TIP.txt").write_text(sha + "\n", encoding="utf-8")
    print("built", PKG, "tip", sha[:12])


if __name__ == "__main__":
    main()
