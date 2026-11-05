#!/bin/bash
# Autopilot c12 (05.11): first of a/b COMPLETE → push c (f4); all three → OOF → gate c12-in (+ paired c12 − c5) → commit JSON;
# kernel outputs → Drive «2»/12_c12in_2026-11-05. Any ERROR → stop for a manual look. Tokens from ~/.kaggle, never printed.
set -u
K=/workspace/koz3; REPO=/home/user/expds-koz3; PY=$K/w4venv/bin/python; KG=$HOME/.local/bin/kaggle
OOF=$K/w2/oof/c12in; GJ=$REPO/kaggle/2026-11-05/li_c12_in/gate_c12in_ctx.json
pushretry() { while :; do o=$($KG kernels push -p "$1" 2>&1 | tail -1); echo "$(date -u +%H:%M) push $(basename $1): $o"; echo "$o" | grep -q "successfully pushed" && break; echo "$o" | grep -qi "quota" || break; sleep 1800; done; }
st() { $KG kernels status dmitrijkv/$1 2>&1 | tail -1 | sed 's/.*status "//; s/"$//'; }
term() { echo "$1" | grep -qE "COMPLETE|ERROR|CANCEL"; }
bad() { echo "$1" | grep -qE "ERROR|CANCEL"; }
echo "$(date -u +%H:%M) start"
while :; do sa=$(st koz3-li-c12-in-a); sb=$(st koz3-li-c12-in-b)
  if bad "$sa" || bad "$sb"; then echo "$(date -u +%H:%M) СТОП: a=$sa b=$sb"; exit 1; fi
  if term "$sa" || term "$sb"; then break; fi; sleep 300; done
echo "$(date -u +%H:%M) a=$sa b=$sb"
if term "$(st koz3-li-c12-in-c)" || st koz3-li-c12-in-c | grep -qE "RUNNING|QUEUED"; then echo "$(date -u +%H:%M) c уже есть: $(st koz3-li-c12-in-c) (push c пропущен)"; else
python3 $REPO/kaggle/2026-11-05/li_cnx_in/make_kernel.py --slug koz3-li-c12-in-c --jobs c12-in:4 \
  --prep-dataset dmitrijkv/koz3-cnx-li-prep12 --out /workspace/c12/BUILD >/dev/null \
  && pushretry /workspace/c12/BUILD/koz3-li-c12-in-c; fi
sleep 120
while :; do sa=$(st koz3-li-c12-in-a); sb=$(st koz3-li-c12-in-b); sc=$(st koz3-li-c12-in-c)
  if bad "$sa" || bad "$sb" || bad "$sc"; then echo "$(date -u +%H:%M) СТОП: a=$sa b=$sb c=$sc"; exit 1; fi
  if term "$sa" && term "$sb" && term "$sc"; then break; fi; sleep 300; done
echo "$(date -u +%H:%M) all COMPLETE"
for k in a b c; do d=$OOF/koz3-li-c12-in-$k; mkdir -p $d
  for t in 1 2 3 4 5; do (cd $d && $KG kernels output dmitrijkv/koz3-li-c12-in-$k -p . >/dev/null 2>&1) && break; sleep 120; done
  echo "$(date -u +%H:%M) output $k: $(ls $d | grep -c oof_cnx_c12-in_f) oof"; done
cd $REPO/kaggle/2026-11-05/li_cnx_in
CNXIN_VARIANT=c12-in CNXIN_OOF_DIR=$OOF nice -n 10 $PY gate_cnxin_ctx.py check > /workspace/c12/gate_check.json 2>&1 \
  || { echo "СТОП: check OOF"; tail -3 /workspace/c12/gate_check.json; exit 1; }
CNXIN_VARIANT=c12-in CNXIN_OOF_DIR=$OOF CNXIN_CMP_VARIANT=c5-in CNXIN_CMP_DIR=$K/w2/oof/cnxin \
  nice -n 10 $PY gate_cnxin_ctx.py 2000 $GJ > /workspace/c12/gate.log 2>&1
tail -4 /workspace/c12/gate.log
if [ -s $GJ ]; then
  MSG=$($PY -c "
import json; d=json.load(open('$GJ')); v=d.get('variants',{}); c=d.get('cmp',{})
f=lambda x:('%+.4f P %.3f'%(x['delta'],x['P'] or 0)) if x else 'n/a'
s=lambda n: v[n]['scores']['all'] if n in v else {}
g=lambda n: c[n]['scores']['all'] if n in c else {}
print('Gate c12-in: K %s, R %s, PASS_any %s; c12-c5 K %s, R %s (autopilot)'%(f(s('K_add_kurgans')),f(s('R_replace_li_kurgans')),d.get('PASS_any'),f(g('K_add_kurgans')),f(g('R_replace_li_kurgans'))))")
  cd $REPO && git add $GJ && git -c user.name=Claude -c user.email=noreply@anthropic.com commit -q -m "$MSG

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0186JU8ZVBBQysUJfZ9g5uMM" && for i in 1 2 3 4; do git push -q origin claude/stoic-turing-mahw6w && break; sleep $((2**i)); done
  echo "$(date -u +%H:%M) $MSG"
fi
cd /workspace/c12 && for k in a b c; do tar -cf koz3-li-c12-in-$k.tar -C $OOF koz3-li-c12-in-$k; done
sha256sum koz3-li-c12-in-?.tar > SHA256SUMS_c12 && python3 /workspace/gauth/gd_upload.py 12_c12in_2026-11-05 \
  koz3-li-c12-in-a.tar koz3-li-c12-in-b.tar koz3-li-c12-in-c.tar SHA256SUMS_c12 $GJ
echo "$(date -u +%H:%M) end"
