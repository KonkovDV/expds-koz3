#!/bin/bash
# Autopilot ensemble (PREREG li_ens_in): after the c12 autopilot ends → push koz3-li-ens-oof (T4) → OOF → gate e512-in
# vs the current lidar line (c12-in if it won its rule, else c5-in) → commit JSON → Drive. Tokens never printed.
set -u
K=/workspace/koz3; REPO=/home/user/expds-koz3; PY=$K/w4venv/bin/python; KG=$HOME/.local/bin/kaggle
OOF=$K/w2/oof/ens; GJ=$REPO/kaggle/2026-11-05/li_ens_in/gate_e512in_ctx.json; G12=$REPO/kaggle/2026-11-05/li_c12_in/gate_c12in_ctx.json
echo "$(date -u +%H:%M) start"
while ! grep -qE "^[0-9:]+ end|СТОП" /workspace/autopilot_c12.log; do sleep 120; done
grep -q "СТОП" /workspace/autopilot_c12.log && { echo "$(date -u +%H:%M) СТОП: автопилот c12 остановился — ансамбль не считается"; exit 1; }
for i in $(seq 1 30); do st=$($KG datasets status dmitrijkv/koz3-ens-code 2>&1 | tail -1); echo "$st" | grep -q ready && break; sleep 60; done
echo "$(date -u +%H:%M) code ds: $st; push: $($KG kernels push -p /workspace/ens/build/koz3-li-ens-oof 2>&1 | tail -1)"
sleep 120
while :; do s=$($KG kernels status dmitrijkv/koz3-li-ens-oof 2>&1 | tail -1); echo "$s" | grep -qE "COMPLETE|ERROR|CANCEL" && break; sleep 180; done
echo "$(date -u +%H:%M) ens kernel: $s"
mkdir -p $OOF/koz3-li-ens-oof
for t in 1 2 3 4 5; do (cd $OOF/koz3-li-ens-oof && $KG kernels output dmitrijkv/koz3-li-ens-oof -p . >/dev/null 2>&1) && break; sleep 120; done
echo "$s" | grep -q COMPLETE || { echo "СТОП: ядро ансамбля не COMPLETE"; tail -c 1500 $OOF/koz3-li-ens-oof/*.log; exit 1; }
CMP=c5-in; CMPD=$K/w2/oof/cnxin
if [ -s $G12 ] && $PY -c "
import json,sys; d=json.load(open('$G12')); v=d['variants']['R_replace_li_kurgans']; c=d.get('cmp',{}).get('R_replace_li_kurgans',{})
sys.exit(0 if v['gate']['PASS'] and c.get('main_replaces_cmp') else 1)"; then CMP=c12-in; CMPD=$K/w2/oof/c12in; fi
echo "$(date -u +%H:%M) текущая линия лидара: $CMP"
cd $REPO/kaggle/2026-11-05/li_cnx_in
CNXIN_VARIANT=e512-in CNXIN_OOF_DIR=$OOF nice -n 10 $PY gate_cnxin_ctx.py check > /workspace/ens/gate_check.json 2>&1 \
  || { echo "СТОП: check OOF ансамбля"; tail -3 /workspace/ens/gate_check.json; exit 1; }
CNXIN_VARIANT=e512-in CNXIN_OOF_DIR=$OOF CNXIN_CMP_VARIANT=$CMP CNXIN_CMP_DIR=$CMPD nice -n 10 $PY gate_cnxin_ctx.py 2000 $GJ > /workspace/ens/gate.log 2>&1
tail -5 /workspace/ens/gate.log
if [ -s $GJ ]; then
  MSG=$($PY -c "
import json; d=json.load(open('$GJ')); v=d.get('variants',{}); c=d.get('cmp',{})
f=lambda x:('%+.4f P %.3f'%(x['delta'],x['P'] or 0)) if x else 'n/a'
s=lambda n: v[n]['scores']['all'] if n in v else {}
g=lambda n: c[n]['scores']['all'] if n in c else {}
print('Gate ensemble e512-in: K %s, R %s, PASS_any %s; E-%s K %s, R %s (autopilot)'%(f(s('K_add_kurgans')),f(s('R_replace_li_kurgans')),d.get('PASS_any'),c.get('vs'),f(g('K_add_kurgans')),f(g('R_replace_li_kurgans'))))")
  cd $REPO && git add $GJ && git -c user.name=Claude -c user.email=noreply@anthropic.com commit -q -m "$MSG

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0186JU8ZVBBQysUJfZ9g5uMM" && for i in 1 2 3 4; do git push -q origin claude/stoic-turing-mahw6w && break; sleep $((2**i)); done
  echo "$(date -u +%H:%M) $MSG"
fi
cd /workspace/ens && tar -cf koz3-li-ens-oof.tar -C $OOF koz3-li-ens-oof && python3 /workspace/gauth/gd_upload.py 13_ens_2026-11-05 koz3-li-ens-oof.tar $GJ
echo "$(date -u +%H:%M) end"
