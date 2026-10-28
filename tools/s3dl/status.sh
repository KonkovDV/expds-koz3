#!/bin/bash
# KOZ3 S3 download status
pgrep -f "python s3_dl.py" >/dev/null && echo "RUNNING (pid $(pgrep -of 'python s3_dl.py'))" || echo "NOT RUNNING"
grep STAT /workspace/data/s3_dl.log | tail -1
grep -E "PAUSE|died|FAILED|FATAL|ERROR|ALL DONE" /workspace/data/s3_dl.log | tail -3
echo "manifest rows: $(($(wc -l < /workspace/data/s3_manifest.tsv)-1)); kept raw: $(grep -c 'kept raw' /workspace/data/s3_manifest.tsv)"
df -h / | tail -1
