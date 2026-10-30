"""W2: генератор ready-to-push ядер c5-sc (seed0) из проверенного исходника s1f23 (fork-lock патч).
Меняются только CNX_JOBS_DEFAULT, setdefault-строка (seed/deadline/relief) и метаданные. Ничего не пушит."""
import json, re, sys, hashlib
from pathlib import Path
SRC = Path('/workspace/koz3/w2/src/koz3-cnx-li-c3sc-s1f23')
base = (SRC / 'koz3-cnx-li-c3sc-s1f23.py').read_text(encoding='utf-8')
meta0 = json.loads((SRC / 'kernel-metadata.json').read_text(encoding='utf-8'))
OUT = Path('/workspace/koz3/w2/kernels')
PLAN = {'f01': ('c5-sc:0,c5-sc:1', '4.0'), 'f23': ('c5-sc:2,c5-sc:3', '5.5'), 'f4': ('c5-sc:4', '3.5')}
for acc in ('elenakonkova', 'dmitrijkv'):
    for tag, (jobs, dl) in PLAN.items():
        kid = f'koz3-w2-cnx-c5sc-{tag}'
        s = base
        s, n1 = re.subn(r"^CNX_JOBS_DEFAULT = '.*'$", f"CNX_JOBS_DEFAULT = '{jobs}'", s, flags=re.M)
        relief = '1' if acc == 'elenakonkova' else '0'   # acc2 берёт каналы из kernel_sources koz3-cnx-li-prep
        line = (f'os.environ.setdefault("CNX_SEED", "0"); os.environ.setdefault("CNX_DEADLINE_H", "{dl}"); '
                f'os.environ.setdefault("CNX_BUILD_RELIEF", "{relief}")  # W2 [li_cnx]: c5-sc seed0 {tag}; PREREG_W2_cnx P-W2-1')
        s, n2 = re.subn(r'^os\.environ\.setdefault\("CNX_SEED".*$', lambda m: line, s, flags=re.M)
        assert n1 == 1 and n2 == 1, (acc, tag, n1, n2)
        d = OUT / acc / kid; d.mkdir(parents=True, exist_ok=True)
        (d / f'{kid}.py').write_text(s, encoding='utf-8')
        m = dict(meta0); m.pop('id_no', None)
        m.update(id=f'{acc}/{kid}', title=kid, code_file=f'{kid}.py', dataset_sources=[f'{acc}/koz3-s3-cred'],
                 kernel_sources=(['dmitrijkv/koz3-cnx-li-prep'] if acc == 'dmitrijkv' else []), is_private=True,
                 enable_gpu=True, enable_tpu=False, machine_shape='NvidiaTeslaT4')
        (d / 'kernel-metadata.json').write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding='utf-8')
        print(acc, kid, jobs, 'dl', dl, 'relief', relief, hashlib.sha256(s.encode()).hexdigest()[:12])
