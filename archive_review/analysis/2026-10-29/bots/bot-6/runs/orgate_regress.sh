#!/bin/bash
export TMPDIR=/workspace/koz3_shared/bot-6/work/pytest_tmp OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 MALLOC_ARENA_MAX=2 RAYON_NUM_THREADS=2
PY=/workspace/koz3_shared/bot-7/venv/bin/python   # bot-8 venv нет после рестора бокса 29.10 ~13:11
T="tests/test_bot6_orgate.py tests/test_dfine_aeor.py tests/test_post_writer_geo.py tests/test_fusion.py tests/test_bot6_sp_fusion.py tests/test_bot6_twins_only.py tests/test_bot7_rerank_arch.py tests/test_bot7_v3_flags.py tests/test_bot7_limits.py tests/test_bot8_faults.py tests/test_bot8_window_deadline.py tests/test_claude_kurgan_priors.py tests/test_claude_frame_time.py tests/test_optional_ideas.py tests/test_platform_robustness.py tests/test_bot8_check_submission.py tests/test_bot7_dfine_import.py"
(cd /workspace/koz3_shared/bot-6/repo && timeout 1500 prlimit --as=2000000000 nice -n 10 taskset -c 0,1 $PY -m pytest -q -p no:cacheprovider -rf $T > /workspace/koz3_shared/bot-6/work/orgate_regress3.log 2>&1)
cd /workspace/koz3_shared/bot-6/work/orgate_identity && rm -rf run2 && prlimit --as=2000000000 nice -n 10 $PY check_identity.py $PWD/run2 > run2.log 2>&1
echo DONE
