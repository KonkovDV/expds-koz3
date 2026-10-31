Бот 7 — патч-кит v3fix для ПК бота 1 (29.10.2026, МСК)
=====================================================
Что это: submission_bot7cand_v3on_0876090.zip = v3on_bc5aa15 + фикс географической CRS в UTM.json
(ветка bot7-candidate-v3fix @0876090: catalog.read_utm_json -> crs.project_region_crs, защита в sp_dfine.utm_for).
Пороги/конфиг те же, что в v3on_bc5aa15 (configs/default.toml побайтно тот же). Отличие от v3on_bc5aa15 —
ровно 3 члена: expds/catalog.py, expds/crs.py, expds/sp_dfine.py (CHG), добавленных/удалённых нет; членов 3534.
Эталонный zip на боксе: bot-7/work6/dist/submission_bot7cand_v3on_0876090.zip, 669 299 212 Б,
sha256 a9e62802a70fe2480e4860276e6fc075f2f0cc9abef01d962fb37e41f22c1fdc; check_submission --no-smoke PASS.
На ПК sha zip будет ДРУГИМ (ZipArchive пишет иначе) — это норма, сверка идёт по членам (манифест).

1. Что скопировать на ПК в C:\Sloi\pc_jobs\v3e2e\kit\
   patch_v3fix_a9e62802.zip
   manifest_v3fix_a9e62802.txt
   manifest_v3on_341a2394.txt   (тот же, что уже лежит в ките от v3on; sha256 77a0dd82…)
   apply_patch_v3fix.ps1
   pc_e2e_v3.ps1                (отличается от прежнего ТОЛЬКО дефолтом -ZipV3 = v3on_0876090; можно заменить)
   README.txt
   НЕ копировать SHA256SUMS этого кита поверх C:\Sloi\pc_jobs\v3e2e\kit\SHA256SUMS: там суммы zip v2, tools_v3e2e.tar
   и zip ПК, их читает pc_e2e_v3.ps1. SHA256SUMS кита — только для проверки файлов при передаче.
   Нужен уже лежащий на ПК C:\Sloi\pc_jobs\v3e2e\kit\submission_bot7cand_v3on_bc5aa15.zip (собран apply_patch_v3on.ps1).

2. Сборка zip (только C:, PowerShell 5.1):
   powershell -ExecutionPolicy Bypass -File C:\Sloi\pc_jobs\v3e2e\kit\apply_patch_v3fix.ps1
   (если C:\…\submission_bot7cand_v3on_0876090.zip уже есть — добавить -Force)
   Что делает: проверяет sha256 патча и обоих манифестов; входной zip — ПО ЧЛЕНАМ по manifest_v3on_341a2394.txt
   (имена, число 3534, размеры, CRC если доступен, sha256 всех членов; ~1–3 мин; -QuickInput — sha256 только у 3
   заменяемых членов); применяет патч; проверяет результат по manifest_v3fix_a9e62802.txt (все 3534 члена);
   0-байтные члены — константа e3b0c442…b855, остальные — MemoryStream.ToArray() (баг PS 5.1 учтён);
   пишет .sha256 рядом и строку в kit\SHA256SUMS (для pc_e2e_v3.ps1). Нужно ≥1,5 ГБ на C:.
   Ожидаемо: «OK: входной zip = v3on_bc5aa15 по членам (3534 …)», «записано членов: 3, удалено: 0»,
   «OK: все 3534 членов совпадают с новым манифестом», «ГОТОВО».
   Пути по умолчанию — C:\Sloi\pc_jobs\v3e2e\kit (параметры -Kit, -OldZip, -OldManifest, -PatchZip, -Manifest,
   -OutZip, -SumsFile).

3. e2e на 101/103 (как run4; в C:\Sloi\pc_jobs\v3e2e\regions должны быть только 101 и 103 — запускать ДО того,
   как класть туда Герите):
   powershell -ExecutionPolicy Bypass -File C:\Sloi\pc_jobs\v3e2e\kit\pc_e2e_v3.ps1 -Mode docker -ZipV3 C:\Sloi\pc_jobs\v3e2e\kit\submission_bot7cand_v3on_0876090.zip -Out C:\Sloi\pc_jobs\v3e2e\out_v3fix
   Ожидаемо: у 101/103 UTM.json = EPSG:32636 (проекционная) -> фикс ничего не меняет: проба D-FINE ok,
   «sp_dfine n_det=482 total» (как run4), res_v3 = run4 res_v3 (13 объектов) с точностью до fp16-шума
   (возможен ±1 пограничный объект, score ≈ порогу), res_v2 = run4 побайтно, метрика как в run4.
   Строк «[crs] … географическая» в логе 101/103 быть НЕ должно.

4. Смоук на 008_Герите (v2 для сравнения не нужен).
   Данные на боксе: /workspace/data/train-expds-3/008_Герите  (всего 5 478 984 564 Б, 3 файла):
     UTM.json                                            59 Б  sha256 95c20b0e0cf903541adc4649aea31081142c74c7119bc67481134b3b5961532f
                                                         ({"utm":"WGS84","crs":"urn:ogc:def:crs:EPSG::4326"})
     05_Герите_Or_ортофото/01_Герите_Or_ортофото.tif     5 478 863 788 Б  sha256 5797bc449c0438225b665a52f567dfcee37aa28a7965529cb91e802d4535dead
     06_Герите_разметка/Герите_Or_архитектуры.geojson    120 717 Б  sha256 e2f06c59d52af65c12cfaadcdcde86bcc509a0631b906aa60ac7810c67cba1eb
   На ПК оригинал, по заметкам бота 3 (bot-3/repo/notes/gt_stats_summary.md), лежит на F:\expds3\train-expds-3\008_Герите
   (не проверено). С F: НЕ читать: нужна копия на C:\Sloi\pc_jobs\v3e2e\regions\008_Герите (с бокса или копией
   владельцем), сверить sha256 трёх файлов. ВНИМАНИЕ: нужно ≥6 ГБ свободного на C: (по bot-1/pc/SLOI_DISK.md на
   29.10 09:39 было 0,5–4,7 ГБ) — сначала проверить (Get-PSDrive C).Free.
   Запуск (docker, как на площадке; zip_v3 уже распакован шагом 3):
   New-Item -ItemType Directory -Force C:\Sloi\pc_jobs\v3e2e\out_008 | Out-Null
   $a = @('run','--rm','--gpus','all','--network','none','--read-only','--user','12345:12345','--memory','96g','--pids-limit','8192','--shm-size','16g','-v','C:\Sloi\pc_jobs\v3e2e\out_v3fix\zip_v3:/app:ro','-v','C:\Sloi\pc_jobs\v3e2e\regions\008_Герите:/data/input/008_Герите:ro','-v','C:\Sloi\pc_jobs\v3e2e\out_008:/data/output','-w','/app','pantum/n7q:20261024','python','inference/main.py','/data/input','/data/output/result.geojson')
   Start-Process -FilePath docker -ArgumentList $a -Wait -NoNewWindow -RedirectStandardOutput C:\Sloi\pc_jobs\v3e2e\out_008\main.log -RedirectStandardError C:\Sloi\pc_jobs\v3e2e\out_008\main.err
   Select-String -Path C:\Sloi\pc_jobs\v3e2e\out_008\main.log,C:\Sloi\pc_jobs\v3e2e\out_008\main.err -Pattern '\[crs\]|sp_dfine|UNAVAILABLE' | % Line
   Критерии (смоук прошёл): есть WARNING «[crs] 008_Герите: CRS участка EPSG:4326 географическая — беру UTM по центру
   участка EPSG:32638 …»; строка «sp_dfine: 01_Герите_Or_ортофото.tif WxH px …» НЕ 1x1 (ожидается 2070x1883, как в
   репро на боксе); «sp_dfine n_det=N total» с N>0; нет строк *_UNAVAILABLE; result.geojson не пустой.

Файлы кита и sha256 — SHA256SUMS. Служебное (только бокс): tools/ (mkpatch_v3fix.py, emulate_apply.py, repack.py),
src/ (шаблон ps1), build/ (логи сборки, check_submission, эмуляции и pwsh-теста).
