#!/usr/bin/env bash
source /workspace/koz3_shared/bot-8/env.sh
L=/workspace/koz3_shared/bot-8/work/rel7
for n in ${NAMES:-wt-base-main wt-base-b7 wt-rel wt-rel7 wt-rel7c}; do
  cd /workspace/koz3_shared/bot-8/$n
  echo "== $n $(git rev-parse --short HEAD) $(git status --short | wc -l) dirty"
  nice -n 10 python -m pytest -q -p no:cacheprovider --basetemp=/workspace/koz3_shared/bot-8/pytest_tmp/$n tests > $L/tests_$n.log 2>&1
  echo "exit $?" >> $L/tests_$n.log
  tail -3 $L/tests_$n.log
done
# регрессионные тесты bot-8 на чистом bot7-bundle (без исправлений): должны падать
cd /workspace/koz3_shared/bot-8/wt-base-b7
cp ../wt-rel7/tests/test_bot8_faults.py ../wt-rel7/tests/test_bot8_window_deadline.py tests/
nice -n 10 python -m pytest -q -p no:cacheprovider --basetemp=/workspace/koz3_shared/bot-8/pytest_tmp/regr-b7 tests/test_bot8_faults.py tests/test_bot8_window_deadline.py > $L/regr_on_bot7_bundle.log 2>&1
echo "exit $?" >> $L/regr_on_bot7_bundle.log
rm tests/test_bot8_faults.py tests/test_bot8_window_deadline.py
git status --short
tail -15 $L/regr_on_bot7_bundle.log
