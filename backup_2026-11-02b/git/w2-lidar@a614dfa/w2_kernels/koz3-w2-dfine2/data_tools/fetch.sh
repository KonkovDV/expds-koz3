#!/bin/bash
# private copy of elenakonkova/koz3-dino-gor-raw rasters for dmitrijkv (flat names <region>__<file>); no credentials in output
# usage: fetch.sh a|b
PART=${1:?a|b}
cd /workspace/koz3/w2/dfine2_data; mkdir -p files_$PART
PK=/workspace/koz3/orch/pool_kaggle.sh
while IFS=, read -r f sz; do
  reg=$(echo "$f" | cut -d/ -f2); base=$(basename "$f"); dst="files_$PART/${reg}__${base}"
  [ -f "$dst" ] && [ "$(stat -c %s "$dst")" = "$sz" ] && continue
  rm -rf tmpdl_$PART; mkdir tmpdl_$PART
  for a in 1 2 3; do $PK elenakonkova datasets download elenakonkova/koz3-dino-gor-raw -f "$f" -p tmpdl_$PART -q >/dev/null 2>&1 && break; sleep 5; done
  got=$(find tmpdl_$PART -type f | head -1)
  if [[ "$got" == *.zip ]]; then (cd tmpdl_$PART && unzip -q -o "$(basename "$got")" && rm -f "$(basename "$got")"); got=$(find tmpdl_$PART -type f | head -1); fi
  if [ -n "$got" ]; then mv "$got" "$dst"; echo "ok $dst $(stat -c %s "$dst")/$sz"; else echo "FAIL $f"; fi
done < rawlist_$PART.csv
rm -rf tmpdl_$PART; echo DONE
