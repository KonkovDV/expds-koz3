#!/bin/bash
# запуск с лимитом RAM 2 ГБ (ulimit -v) и печатью пикового RSS
ulimit -v 2500000
OMP_NUM_THREADS=2 RAYON_NUM_THREADS=2 EXPDS_THREADS=2 nice -n 10 "$@"
