# bot-5. PREREG bot-10/PREREG_gor_dense_bot5.md, поправка (1): ДО прогона записать генератор сидов 5–9, хэши списков фолдов
# и sha всех входов. Пишет DOUT/gor_dense_inputs.json (hf, scn — как в analysis/gor_dense; layouts — только seed5..seed9)
# и DOUT/prerun.json. Раскладки: gor_scene.py rand_folds(s) = bot-10/repo/scripts/honest_bootstrap.py:random_family_folds
# (np.random.default_rng(s).permutation(n_семей) % 10 по sorted(семьи site_family)).
import os, json, hashlib, subprocess
from pathlib import Path
src = Path('/workspace/koz3_shared/bot-5/work/hillfort/gor_scene.py').read_text(encoding='utf-8').split('VARS = [')[0]
exec(compile(src, 'gor_scene_head', 'exec'))
B = Path('/workspace/koz3_shared/bot-5'); D0 = B / 'analysis/gor_dense'; DOUT = Path(os.environ.get('DOUT', str(B / 'analysis/gor_dense_s59')))
J0 = json.load(open(D0 / 'gor_dense_inputs.json', encoding='utf-8'))
NEWL = {f'seed{s}': [sorted(f) for f in rand_folds(s)] for s in range(5, 10)}
for s in range(5):   # проверка: генератор воспроизводит seed0-4 из прежних входов
    assert [sorted(f) for f in rand_folds(s)] == J0['layouts'][f'seed{s}'], f'seed{s} mismatch'
J = dict(hf=J0['hf'], scn=J0['scn'], layouts=NEWL)
assert list(SCN) == J0['scn']
json.dump(J, open(DOUT / 'gor_dense_inputs.json', 'w'), ensure_ascii=False)
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()
def lh(L): return hashlib.sha256(json.dumps(L, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
for f in ('dense.npz', 'files.json'):
    if not (DOUT / f).exists(): os.symlink(D0 / f, DOUT / f)
ins = [D0 / 'dense.npz', D0 / 'files.json', D0 / 'gor_dense_inputs.json', DOUT / 'gor_dense_inputs.json', D0 / 'peaks.json',
       Path('/workspace/rerank/work/table.pkl'), Path('/workspace/rerank/work/gt_wkb.pkl'), Path('/workspace/rerank/gt_train_v2.geojson'),
       Path('/workspace/data/s3_listing.jsonl'), B / 'repo/notes/regions_las_new.txt', B / 'repo/notes/las_boxes_59.json',
       B / 'repo/materials/compute_metrics.py', B / 'work/hillfort/gor_scene.py', B / 'repo/scripts/gor_patch/gor_patch_eval.py',
       B / 'repo/scripts/gor_dense/dense_probe.py', B / 'repo/scripts/gor_dense/peaks_ckpt.py', B / 'repo/scripts/gor_dense/gor_dense_eval.py',
       B / 'repo/scripts/gor_dense/gor_dense_diag.py', Path('/workspace/koz3_shared/bot-10/work10/rw/big_scenes.json'),
       Path('/workspace/koz3_shared/bot-10/PREREG_gor_dense_bot5.md')]
gp = Path(os.environ.get('OUT', str(B / 'analysis/gor_patch')))
ins += sorted(p for p in gp.glob('*') if p.is_file() and p.suffix in ('.npz', '.json'))
git = lambda *a: subprocess.run(['git', '-C', str(B / 'repo'), *a], capture_output=True, text=True).stdout.strip()
P = dict(prereg='/workspace/koz3_shared/bot-10/PREREG_gor_dense_bot5.md', written_before_run=True,
         seed_generator='gor_scene.py:rand_folds(seed,k=10) = honest_bootstrap.random_family_folds: names=sorted(site_family families of all regions); lab=np.random.default_rng(seed).permutation(len(names)) % 10; seeds 5,6,7,8,9; check: reproduces seed0-4 of analysis/gor_dense/gor_dense_inputs.json exactly',
         n_families=len({site_family(r) for r in regions}), n_regions=len(regions),
         fold_list_sha256={**{k: lh(v) for k, v in NEWL.items()}, **{k: lh(v) for k, v in J0['layouts'].items()}},
         fold_sizes={k: [len(f) for f in v] for k, v in NEWL.items()},
         inputs_sha256={str(p): sha(p) for p in ins},
         code=dict(head=git('rev-parse', 'HEAD'), dense_probe_last_commit=git('log', '-1', '--format=%H', '--', 'scripts/gor_dense/dense_probe.py'),
                   probe_vs_4a39427_diff=git('diff', '--stat', '4a39427', 'HEAD', '--', 'scripts/gor_dense/dense_probe.py', 'scripts/gor_dense/peaks_ckpt.py') or 'none',
                   dirty=git('status', '--short', '--', 'scripts')),
         probe_cmd='DOUT=analysis/gor_dense_s59 VARIANTS=pu,pu_edge,all LAYOUTS=seed5,...,seed9 RESUME=1 dense_probe.py (ckpt per fold in DOUT/ckpt)',
         s_grid='dense_probe.py CP_GRID [150,250,400,700,inf], chosen per outer fold by inner OOF top-1 hit (nested)',
         D_grid_FN='prereg {inf,30,100,300,1000 m}; inf == drop every point in a scene that has a stack gor row (= only scenes without stack hillfort); chosen per outer fold by train F1, ties -> larger D')
json.dump(P, open(DOUT / 'prerun.json', 'w'), ensure_ascii=False, indent=1)
log('prerun written', DOUT / 'prerun.json', P['fold_list_sha256'], P['code'])
