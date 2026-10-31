"""Сравнение check_submission.py: эталон (expds-koz3 f932945, analysis/.../bot-8) vs bot-8 02:01 (под PIXGBM).
Только чтение чужих файлов: модули грузятся importlib, архивы — в памяти."""
import importlib.util, io, zipfile, sys, tempfile, pathlib, textwrap, re
OLD = '/workspace/expds-koz3/analysis/2026-10-28/bots/bot-8/check_submission.py'
NEW = '/workspace/koz3_shared/bot-8/check_submission.py'
CFG = pathlib.Path('/workspace/koz3_shared/bot-4/work/review/pixgbm428/configs/default.toml').read_text(encoding='utf-8')
def load(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
O, N = load(OLD, 'cs_old'), load(NEW, 'cs_new')
def mkzip(cfg, extra):
    b = io.BytesIO(); z = zipfile.ZipFile(b, 'w')
    z.writestr('configs/default.toml', cfg)
    for n in re.findall(r'"(models/[^"]+\.(?:pt|txt|json))"', cfg): z.writestr(n, b'x')  # все прочие веса «есть»
    for n in extra: z.writestr(n, b'x')
    z.close(); return zipfile.ZipFile(io.BytesIO(b.getvalue()))
def run(mod, z):
    mod.RESULTS.clear()
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()): mod.check_config_static(z)
    return [(s, m[:110]) for s, c, m in mod.RESULTS if s in ('FAIL',) or (s == 'PASS' and 'путей' in m)]
on = CFG.replace('enabled = false\nmodel_dir = "models/pixgbm/all"', 'enabled = true\nmodel_dir = "models/pixgbm/all"')
assert on != CFG
cases = [
 ('T1 PIXGBM выкл., папки моделей нет (как в архиве 428257f)', CFG, []),
 ('T2 PIXGBM вкл., папки нет', on, []),
 ('T3 PIXGBM вкл., в папке только README (нет poly.txt/finds.txt)', on, ['models/pixgbm/all/README.md']),
 ('T4 PIXGBM вкл., есть poly.txt, нет finds.txt', on, ['models/pixgbm/all/poly.txt']),
 ('T5 PIXGBM вкл., poly.txt+finds.txt', on, ['models/pixgbm/all/poly.txt', 'models/pixgbm/all/finds.txt']),
 ('T6 PIXGBM вкл., похожая папка models/pixgbm/all_old/ (префикс без «/»)', on, ['models/pixgbm/all_old/poly.txt']),
 ('T7 контроль: порог неизвестного класса', CFG.replace('[thresholds]', '[thresholds]\nkurgany_x = 0.3', 1), []),
]
for name, cfg, extra in cases:
    z = mkzip(cfg, extra); print(name); print('   OLD:', run(O, z)); print('   NEW:', run(N, z))
# ленивые импорты: классификация _module_level_imports
d = pathlib.Path(tempfile.mkdtemp()); (d / 'expds').mkdir(); (d / 'inference').mkdir()
(d / 'inference/main.py').write_text('import expds.a\n'); (d / 'expds/__init__.py').write_text('')
(d / 'expds/a.py').write_text(textwrap.dedent('''
    def f():
        import psutil            # БЕЗ try/except: упадёт в рантайме, если пакета нет
        return psutil.Process()
    try:
        import tl2cgen           # уровень модуля, с фолбэком
    except ImportError:
        tl2cgen = None
'''))
print('T8 модульные импорты (NEW._module_level_imports):', N._module_level_imports(d, {'__main__', 'expds', 'expds.a'}))
print('   => psutil (импорт в функции без фолбэка) попадёт в WARN, tl2cgen (модуль, с фолбэком) — в FAIL')
