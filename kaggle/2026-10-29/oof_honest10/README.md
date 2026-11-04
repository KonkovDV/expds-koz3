# koz3-bot8-oof-honest10 (GPU-5)

OOF-инференс моделей `[models]` на трейне КОЗ №3 для калибровки порогов по слою (honest-10).

## Запуск (исполнитель с токеном Kaggle)

1. Прикрепить private dataset со snapshot репо (нужны `scripts/nn_oof.py`, `scripts/oof_kaggle_jobs.py`, `expds/`, `configs/default.toml`, `models/*.pt`) **или** выставить `OOF_REPO` / `OOF_MODELS_ROOT` после S3 sync.
2. Данные: attach `train-expds-3` **или** `OOF_S3_SYNC=1` + `dmitrijkv/koz3-s3-cred`.
3. Push kernel `dmitrijkv/koz3-bot8-oof-honest10`, GPU T4.
4. Забрать `/kaggle/working/oof_honest10/{STATUS.txt,SHA256SUMS,oof/*.geojson,jobs/*/done.json,logs/}`.

## Env

| Var | Default | Meaning |
|-----|---------|---------|
| OOF_MODE | weights | `weights` (nn_oof --weights) или `fold` (--predict-only --resume) |
| OOF_JOBS | all | comma ids: spor_gor,spor_focal,… |
| OOF_DEADLINE_H | 11 | stop & leave RUNNING for resume |
| OOF_SMOKE | | `1` = CPU synthetic, no S3/GPU |
| OOF_S3_SYNC | | `1` = download train-expds-3 from S3 |
| OOF_TTA | | pass --tta to nn_oof |

## Оценка GPU

~8–12 ч T4 на все 10 jobs в режиме weights (зависит от числа участков с растрами/LAS). Рестарт идемпотентен.
