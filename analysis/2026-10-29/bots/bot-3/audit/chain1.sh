#!/bin/bash
A=/workspace/koz3_shared/bot-3/audit
$A/pipe_variant.sh SAT PIXGBM_SAT_CORE_M2=60 PIXGBM_SAT_FLAG=1
$A/pipe_variant.sh SATPICK PIXGBM_SAT_CORE_M2=60 PIXGBM_SAT_FLAG=1 PIXGBM_PICK=hp_energy
$A/pipe_variant.sh PICK PIXGBM_PICK=hp_energy
