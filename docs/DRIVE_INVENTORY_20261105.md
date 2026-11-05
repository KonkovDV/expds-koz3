# Опись Google Drive «2» на 05.11.2026 (вечер)

Корень: `KOZ3_backup_2026-11-04` — https://drive.google.com/drive/folders/1O_9svjsxWv0rvEunry6QGSCUHejzayqZ.
Загрузки — `/workspace/gauth/gd_upload.py`: Drive v3 resumable, сверка md5 по каждому файлу.

| Папка | Что лежит |
|---|---|
| `00_SUBMIT_v2_c5in_R_98e0ab74` | **лучшая посылка LB 0,43274**: `v2_c5in_R.zip` (sha256 `98e0ab74…`), SHA256SUMS, лог `check_submission` |
| `00_SUBMIT_FINAL_v2_EN_167478e3` | v2, LB 0,41238 (пересборка из `eb268d6`) |
| `00_SUBMIT_FR1h_fe8d60b9` | FR1h, LB 0,41112 — не берём |
| `00_SUBMIT_NEXT_e1g1_ed51cae4`, `00_SUBMIT_v4_f0def982` | E1_G1 (0,38988), v4 (0,34418) — история |
| `11_c5in_2026-11-05` | выходы ядер c5-in a/b/c (веса 5 фолдов, OOF, карты), `gate_cnxin_ctx.json` |
| `12_c12in_2026-11-05` | создаётся автопилотом c12: выходы ядер c12-in a/b/c и гейт |
| `13_ens_2026-11-05` | создаётся автопилотом ансамбля: OOF ансамбля и гейт |
| `14_competitors_public_2026-11-05` | 11 публичных репозиториев конкурентов (`INDEX.md`), `comp_gh_arhipelag.tar.gz` (GigaFlex), `rtcore_v2_solution_code.tar.gz` (код из образа `strekalov/rt-core:v2`, без весов), `rt-core_v2.tar` (образ целиком, загрузил владелец) |
| `15_session_2026-11-05` | датасет YOLO спутника (910 окон), датасет рельефа c12 (59 LAS, 12 каналов, 3,9 ГБ), выход smoke c12, git-бандл репозитория (`--all`), логи автопилотов, SHA256SUMS |
| `01_gate_base` … `10_github_repo` | бокс гейта, OOF, D-FINE, лидар, зеркала Kaggle и GitHub (до 05.11) |

Доступ к папке — «все со ссылкой — редактор». После сдачи понизить до «читатель».
