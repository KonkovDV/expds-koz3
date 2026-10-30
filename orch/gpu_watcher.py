"""coord watcher: push W1 D-FINE kernels to elenakonkova GPU slots when W2 s1 kernels finish. Logs to orch/COORD_LOG.md."""
import json, subprocess, time, datetime, pathlib
O = pathlib.Path('/workspace/koz3/orch')
PK = str(O / 'pool_kaggle.sh')
SLOTS = {'acc1/gpu-1': 'elenakonkova/koz3-cnx-li-c3sc-s1f4', 'acc1/gpu-2': 'elenakonkova/koz3-cnx-li-c3sc-s1f23'}
QUEUE = [('elenakonkova/koz3-w1-dfine-ae-final', O / 'bot-W1/kernels/koz3-w1-dfine-ae-final', 'W1 D-FINE ae final+5 фолдов OOF (noflip/flipnms/flipwbf), ~5.5–6 ч'),
         ('elenakonkova/koz3-w1-dfine-or-final', O / 'bot-W1/kernels/koz3-w1-dfine-or-final', 'W1 D-FINE or final (P_OR_FIX) + 3 фолда OOF, ~6–9 ч')]
now = lambda: datetime.datetime.now().strftime('%d.%m %H:%M MSK')
def log(m):
    with open(O / 'COORD_LOG.md', 'a') as f: f.write(f'- {now()} | gpu_watcher | {m}\n')
def st(k):
    r = subprocess.run([PK, k.split('/')[0], 'kernels', 'status', k], capture_output=True, text=True)
    return (r.stdout + r.stderr).strip().split('"')[-2] if '"' in r.stdout else 'UNKNOWN'
log('старт; очередь: ' + ', '.join(q[0] for q in QUEUE))
while QUEUE:
    L = json.load(open(O / 'LEASES.json'))
    for s in L['slots']:
        if s['slot'] not in SLOTS or not QUEUE: continue
        if s.get('kernel') != SLOTS[s['slot']]: continue          # slot already re-leased by someone/us
        state = st(SLOTS[s['slot']])
        if state.split('.')[-1] in ('COMPLETE', 'ERROR', 'CANCEL_ACKNOWLEDGED', 'CANCELLED'):
            kid, path, purpose = QUEUE[0]
            old = dict(s)
            s.clear(); s.update({'slot': old['slot'], 'account': 'elenakonkova', 'holder': 'bot-W1-aeor', 'kernel': kid,
                                 'since': now(), 'until': '+12 ч', 'purpose': purpose, 'pushed_by': 'coord gpu_watcher',
                                 'prev': f"{old.get('kernel')} {state}"})
            json.dump(L, open(O / 'LEASES.json', 'w'), ensure_ascii=False, indent=1)   # lease BEFORE push
            r = subprocess.run([PK, 'elenakonkova', 'kernels', 'push', '-p', str(path)], capture_output=True, text=True)
            out = (r.stdout + r.stderr).strip().splitlines()[-1:] or ['']
            log(f"{old['slot']}: {old.get('kernel')} → {state}; аренда W1; push {kid}: {out[0][:200]}")
            if 'successfully' in out[0]: QUEUE.pop(0)
            else:
                s['state'] = 'push failed, retry'; s['kernel'] = SLOTS[old['slot']]   # keep watching
                json.dump(L, open(O / 'LEASES.json', 'w'), ensure_ascii=False, indent=1)
            break
    time.sleep(300)
log('очередь пуста — выход')
