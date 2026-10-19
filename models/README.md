# models/

Сюда кладутся обученные модели:

- `li_r18.pt`, `ae_r18.pt`, … — U-Net из `python -m expds.train`;
- `kurgan_lgbm.txt` — оценщик CPU-кандидатов из `python -m expds.candidates train`.

Подключение — в `configs/default.toml`, секции `[models]` и `[candidates]`. В архив попадают только модели, указанные в конфиге (`scripts/build_submission.py`). В git модели не коммитятся: они производные конкурсных данных.
