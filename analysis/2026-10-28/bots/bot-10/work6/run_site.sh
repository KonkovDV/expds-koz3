cd /workspace/koz3_shared/bot-10/work6
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 EXPDS_METRIC=/workspace/koz3_shared/bot-10/repo/materials/compute_metrics.py
ulimit -v 3000000
nice -n 10 /workspace/rerank/venv/bin/python twins_eval.py site eval_site.pkl > eval_site.log 2>&1
cd ..; nice -n 10 /workspace/rerank/venv/bin/python work6/stats_twins.py work6/eval_site.pkl work6/stats_site.json 2>&1 | grep -v config.py > work6/stats_site.log
