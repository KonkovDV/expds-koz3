# Посылка v4 — где лежит и как собрать (проверено 03.11.2026)

Архив: `koz3_v4c1f_07795d2_51882efd_w6dfd2e07.zip`, 722 705 579 Б,
sha256 `f0def9822cf185aee0829b72c36e4def94c17d03044d16adefc59c9547202cef`.
Состав: v3_on `b447649` + `scene_drel` + C1F-all, конфиг `configs/v4_c1f.toml` @ `bots/w4-v3on` 07795d2 (теперь в `main`).
Honest-10 0,361641 (Δ+0,0162 к v3_on, P 1,0). Kaggle-паритет v4chk PASS.

## Где

**Одним файлом (04.11, без склейки):** Google Drive «2» → `KOZ3_backup_2026-11-04/00_SUBMIT_v4_f0def982/koz3_v4c1f_07795d2_51882efd_w6dfd2e07.zip` — https://drive.google.com/file/d/12J_BRXkOipEv4JZtaenvYOgP3TRR1fiG/view (папка: https://drive.google.com/drive/folders/1NrqHOD57oNTVfL_Ufe7LYLVEASPQlo_H). Перед выкладкой 04.11: sha256 `f0def982…` совпал, CRC всех 3904 файлов zip в порядке, в корне `metadata.json` (`pantum/n7q:20261024`, `python inference/main.py`); md5 файла на Drive совпал с проверенным. Код сабмита в `main` с проверки пересборки (1bee81de) не менялся — v4 остаётся самой свежей версией: все гейты 03–04.11 поверх неё провалились.

Частями (исходная выкладка 02.11):

Google Drive → KOZ3-final → `v4c1f_f0def982`:
https://drive.google.com/drive/folders/1h8AAWDV6xJkgxN7FwZXeLMYaAIEUx5Ia

В папке 8 частей `koz3_v4c1f_f0def982.zip.part00…part07`, **каждая загружена дважды**. 03.11 все 16 файлов скачаны и сверены с `SHA256SUMS_parts.txt`: обе копии каждой части одинаковы, брать можно любую. Склеенный архив совпал с `SHA256SUMS` (`f0def982…`: OK). Смоук на синтетике (CPU): код 0, 63 с, 82 объекта, фаза `sp_dfine` отработала, `V3_DEGRADED` нет.

| часть | Drive id (проверенная копия) |
|---|---|
| part00 | `1oNk9eX5sWivVIPK4iZ2-zt0YSsrYqPpI` |
| part01 | `1LnPtOaGX_p-OSqFxWvxBG3OBI0l66pBQ` |
| part02 | `15KAbhloPfv95lPgt2mScN3MSxJXEFB17` |
| part03 | `1170qFIdD8IPDhVyKNosLYU1Bl9ML4UHZ` |
| part04 | `1B2NHOxH52XOCY1QCLaXHyt2BbXZofkx-` |
| part05 | `1ymLoP6G5YUfMKLe56BiqfURlB7fdu0nC` |
| part06 | `1k01WrsFPZByD5QyS6WZuDPZOWrgUlLCd` |
| part07 | `1DTBnVRPCjX1H0_NciiwhjHAMwqIzVIC4` |

## Сборка

Linux / macOS / WSL:

```bash
cat koz3_v4c1f_f0def982.zip.part0* > koz3_v4c1f_07795d2_51882efd_w6dfd2e07.zip && sha256sum -c SHA256SUMS
```

Windows (cmd):

```
copy /b koz3_v4c1f_f0def982.zip.part00+koz3_v4c1f_f0def982.zip.part01+koz3_v4c1f_f0def982.zip.part02+koz3_v4c1f_f0def982.zip.part03+koz3_v4c1f_f0def982.zip.part04+koz3_v4c1f_f0def982.zip.part05+koz3_v4c1f_f0def982.zip.part06+koz3_v4c1f_f0def982.zip.part07 koz3_v4c1f_07795d2_51882efd_w6dfd2e07.zip
certutil -hashfile koz3_v4c1f_07795d2_51882efd_w6dfd2e07.zip SHA256
```

Хэш должен быть `f0def9822cf185aee0829b72c36e4def94c17d03044d16adefc59c9547202cef`.

## Посылка

Загружает Елена. Порядок: v4 → если LB ниже 0,41238, то v3_on (`1406c01a…`, папка `v3_on_b447649_1406c01a`) → v2clean (`545ef32a…`). После посылки — строка в `docs/LB_LOG.md` с sha256 архива.

## Пересборка из `main` даёт тот же выход (проверено 04.11)

Архив v4 распакован, `expds/`, `inference/main.py` и `inference/lib/expds_core.abi3.so` заменены на версии из `main` (`1bee81de`, после сведения всех веток и исправлений аудита), `configs/default.toml` — на `configs/v4_c1f.toml` из `main`; веса, D-FINE и вендор — из архива. Прогон `inference/main.py` на синтетике из `tests/test_golden_synthetic.py` (лидар, аэрофото, магниторазведка, спутник, ортофото; CPU) у оригинального архива и у гибрида: **sha256 `c3a5294e…` у обоих, 82 объекта, тот же состав** (в т. ч. курган `sp_or` от D-FINE). Конфиг архива совпадает с `configs/v4_c1f.toml` без учёта комментариев.

Вывод: сведение веток 03.11 и правки аудита выход финала не меняют; v4 собирается из `main` командой `scripts/build_submission_git.py --config configs/v4_c1f.toml` при наличии весов D-FINE и вендора (их нет в git — берутся из архива на Drive).
