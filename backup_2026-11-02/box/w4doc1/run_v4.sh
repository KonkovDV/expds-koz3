cd /workspace/koz3/w4doc1/v4
P="nice -n 19 /workspace/koz3/w4venv/bin/python -m pytest -q --continue-on-collection-errors --tb=no -rfE -p no:cacheprovider"
git status --short > ../v4_state_before.txt
$P > ../pytest_v4_before.txt 2>&1
git stash pop -q
git status --short > ../v4_state_after.txt
$P > ../pytest_v4_after.txt 2>&1
echo DONE >> ../pytest_v4_after.txt
