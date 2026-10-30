#!/bin/bash
P="nice -n 19 /workspace/koz3/w4venv/bin/python -m pytest -q --continue-on-collection-errors --tb=no -rfE -p no:cacheprovider"
cd /workspace/koz3/w4clean && $P > /workspace/koz3/w4geo/pytest_after.txt 2>&1; echo DONE >> /workspace/koz3/w4geo/pytest_after.txt
cd /workspace/koz3/w4doc1/v4 && $P > ../pytest_v4_after.txt 2>&1; echo DONE >> ../pytest_v4_after.txt
