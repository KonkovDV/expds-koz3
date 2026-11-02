"""Применяет DOC-1 к scripts/build_submission.py и configs/default.toml текущего checkout (идемпотентно)."""
import re, sys
from pathlib import Path
R = Path(sys.argv[1])
b = R / "scripts" / "build_submission.py"; s = b.read_text(encoding="utf-8")
if "check_archive_clean" not in s:
    s = s.replace('FLAG_MODULES = (', '#: W4 DOC-1: файлы, производные от EN, и старые модели PIXGBM в архив не идут (scripts/check_archive_clean.py)\n'
                  'EN_EXCLUDE = {"geo_pixgbm.py", "geo_finds_gbm.py"}\nFLAG_MODULES = (', 1)
    old = "if not (p.parent == ROOT / \"expds\" and (p.name in TRAIN_ONLY or p.name in off)))"
    assert old in s, "expds glob not found"
    s = s.replace(old, "if p.name not in EN_EXCLUDE\n                    and not (p.parent == ROOT / \"expds\" and (p.name in TRAIN_ONLY or p.name in off)))", 1)
    old2 = "    size = out.stat().st_size\n    if size > LIMIT:"
    assert s.count(old2) == 1, "size check not found"
    s = s.replace(old2, "    _doc1_check(out)\n" + old2, 1)
    old3 = "def collect() -> list[Path]:"
    s = s.replace(old3, '''def _doc1_check(out: Path) -> None:
    """W4 DOC-1: архив без EN-файлов и без включённого legacy PIXGBM, иначе архив удаляется и сборка падает."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from check_archive_clean import check
    bad = check(out)
    if bad:
        out.unlink(missing_ok=True)
        sys.exit("DOC-1: архив не чистый:\\n" + "\\n".join(bad))


''' + old3, 1)
    # --from-zip тоже проверяем
    old4 = "        digest = variant_from_zip(Path(args.from_zip), out, args.only_class or None, changes)\n"
    if old4 in s:
        s = s.replace(old4, old4 + "        _doc1_check(out)\n", 1)
    b.write_text(s, encoding="utf-8")
c = R / "configs" / "default.toml"; t = c.read_text(encoding="utf-8")
i = t.index("[geophys.pixgbm]"); j = t.find("\n[", i + 1); j = len(t) if j < 0 else j
sec = t[i:j]
new, n = re.subn(r"(?m)^enabled\s*=\s*true\b", '# W4 DOC-1: выключено явно — legacy PIXGBM (по мотивам EN) не используется и в архив не идёт\n# (scripts/check_archive_clean.py); clean-room перепись — только impl = "clean" за флагом.\nenabled = false', sec, count=1)
if n == 0 and not re.search(r"(?m)^enabled\s*=\s*false", sec):
    new = sec.replace("[geophys.pixgbm]\n", '[geophys.pixgbm]\nenabled = false   # W4 DOC-1: выключено явно\n', 1)
c.write_text(t[:i] + new + t[j:], encoding="utf-8")
print("patched", R)
