"""DOC-1 правки тестов, завязанных на включённый legacy PIXGBM (идемпотентно, с отчётом что применилось)."""
import sys
from pathlib import Path
R = Path(sys.argv[1]); p = R / "tests" / "test_bot7_candidate_v2.py"
if not p.exists():
    print("no test_bot7_candidate_v2"); sys.exit()
s = p.read_text(encoding="utf-8"); done = []
def rep(old, new, tag):
    global s
    if old in s and new not in s:
        s = s.replace(old, new, 1); done.append(tag)
rep('''    return pipeline.load_config(ROOT / "configs" / "default.toml")
''', '''    # W4 DOC-1: в default.toml legacy PIXGBM выключен; механику варианта S проверяем с флагом, включённым в памяти
    c = pipeline.load_config(ROOT / "configs" / "default.toml")
    return dict(c, geophys=dict(c["geophys"], pixgbm=dict(c["geophys"]["pixgbm"], enabled=True)))
''', "fixture")
rep('''def test_flags_independent_and_as_documented(cfg):
    pix = cfg["geophys"]["pixgbm"]
''', '''def test_flags_independent_and_as_documented(cfg):
    assert pipeline.load_config(ROOT / "configs" / "default.toml")["geophys"]["pixgbm"]["enabled"] is False   # DOC-1
    pix = cfg["geophys"]["pixgbm"]
''', "documented")
rep('''    md = ROOT / cfg["geophys"]["pixgbm"]["model_dir"]
    for name, h in SHA.items():
''', '''    md = ROOT / cfg["geophys"]["pixgbm"]["model_dir"]
    if not all((md / name).is_file() for name in SHA):
        pytest.skip("ветка -nomodels: весов legacy PIXGBM нет (W4 DOC-1: в архив они и не идут)")
    for name, h in SHA.items():
''', "shipped")
old = '''def test_build_submission_collects_pix_models():'''
if old in s:
    i = s.index(old); j = s.index("\n\n\n", i)
    s = s[:i] + '''def test_build_submission_excludes_legacy_pix():
    """W4 DOC-1 (было: веса legacy PIXGBM в архиве): [geophys.pixgbm] выключен, в архив не идут
    ни веса models/pixgbm/*, ни модули expds/geo_pixgbm.py / geo_finds_gbm.py."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("bs", ROOT / "scripts" / "build_submission.py")
    bs = importlib.util.module_from_spec(spec); spec.loader.exec_module(bs)
    rel = {p.relative_to(ROOT).as_posix() for p in bs.collect()}
    assert not any(r.startswith("models/pixgbm/") for r in rel)
    assert not any(Path(r).name in {"geo_pixgbm.py", "geo_finds_gbm.py"} for r in rel)''' + s[j:]
    done.append("collect")
p.write_text(s, encoding="utf-8"); print("tests patched:", done)
