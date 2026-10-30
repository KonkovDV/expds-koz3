#!/bin/bash
# Собрать каталог датасета (без загрузки). Код — только чистые файлы из bots/w4-pix-clean.
set -e; H=$(dirname "$0"); R=/workspace/koz3/w4clean; A=${1:-usertest1234}
rm -rf $H/dataset; mkdir -p $H/dataset/code/scripts
ln -s /workspace/koz3/w4geo/cache_clean $H/dataset/cache_clean
cp -r $R/expds $H/dataset/code/; cp $R/scripts/pix_clean_train.py $H/dataset/code/scripts/
find $H/dataset/code -name __pycache__ -prune -exec rm -rf {} +
rm -f $H/dataset/code/expds/geo_pixgbm.py $H/dataset/code/expds/geo_finds_gbm.py
printf '{"title":"koz3-w4-pixclean-cache","id":"%s/koz3-w4-pixclean-cache","licenses":[{"name":"other"}]}\n' $A > $H/dataset/dataset-metadata.json
echo "ok: $H/dataset"
