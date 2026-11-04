"""W2 P-W2-9: генератор ядер r50c5 из подготовленного koz3-w2-cnx-r50c5-f01 (dmitrijkv, T5). Ничего не пушит.
Меняются только CNX_JOBS_DEFAULT, setdefault-строка (seed/deadline/relief) и метаданные."""
import json, re, hashlib
from pathlib import Path
K = Path('/workspace/koz3/w2/kernels'); SRC = K / 'koz3-w2-cnx-r50c5-f01'
base = (SRC / 'koz3-w2-cnx-r50c5-f01.py').read_text(encoding='utf-8')
meta0 = json.loads((SRC / 'kernel-metadata.json').read_text(encoding='utf-8'))
PLAN = [('elenakonkova', 'f01', 'c5-sc:0,c5-sc:1', '6.0', '1'),          # acc1: свой relief (нет доступа к dmitrijkv prep)
        ('dmitrijkv', 'f234', 'c5-sc:2,c5-sc:3,c5-sc:4', '11.3', '0')]   # acc2: каналы из kernel_sources koz3-cnx-li-prep
for acc, tag, jobs, dl, relief in PLAN:
    kid = f'koz3-w2-cnx-r50c5-{tag}'
    s, n1 = re.subn(r"^CNX_JOBS_DEFAULT = '.*'$", f"CNX_JOBS_DEFAULT = '{jobs}'", base, flags=re.M)
    line = (f'os.environ.setdefault("CNX_SEED", "0"); os.environ.setdefault("CNX_DEADLINE_H", "{dl}"); '
            f'os.environ.setdefault("CNX_BUILD_RELIEF", "{relief}")  # W2 P-W2-9 r50c5 {tag}: ResNet-50 U-Net, c5, scratch, seed0')
    s, n2 = re.subn(r'^os\.environ\.setdefault\("CNX_SEED".*$', lambda m: line, s, flags=re.M)
    assert n1 == 1 and n2 == 1, (acc, tag, n1, n2)
    d = K / acc / kid; d.mkdir(parents=True, exist_ok=True)
    (d / f'{kid}.py').write_text(s, encoding='utf-8')
    m = dict(meta0)
    m.update(id=f'{acc}/{kid}', title=kid, code_file=f'{kid}.py', dataset_sources=[f'{acc}/koz3-s3-cred'],
             kernel_sources=(['dmitrijkv/koz3-cnx-li-prep'] if acc == 'dmitrijkv' else []), is_private=True,
             enable_gpu=True, enable_tpu=False, machine_shape='NvidiaTeslaT4')
    (d / 'kernel-metadata.json').write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding='utf-8')
    print(acc, kid, jobs, 'dl', dl, 'relief', relief, hashlib.sha256(s.encode()).hexdigest()[:12])
