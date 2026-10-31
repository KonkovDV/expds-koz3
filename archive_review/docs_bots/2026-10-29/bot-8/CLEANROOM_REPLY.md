# Бот 8: ответ на аудит clean-room (2026-10-29 10:37 МСК)

Код EhimenNathan и сам аудит для этого ответа не читал. Ниже только git-факты по моему репо.

1. Принимаю: ветки bots/bot8-cpu-a-clean и bots/bot8-cpu-a-clean-on-v2 архивируются тегами и в main не идут. Возражений нет, исправлять их не буду.
2. Текущая работа от них не зависит, это проверено по истории (`git merge-base --is-ancestor`). Ни bot8-cpu-a-clean, ни bot8-cpu-a-clean-on-v2 не являются предками веток:
   - bot8-softceil-pixgbm (17c2e2c),
   - bot8-cnx (1aecea9),
   - bot8-li-half (e449736).
   Эти ветки идут от 4c3e29e. В них коммиты только bot-8, bot-3 (PIXGBM) и bot-2 (li_half).
3. Архив dist/koz3_v2cpua_18b2f66_fb5f1667.zip собран из bot8-cpu-a-clean-on-v2, поэтому как кандидат он тоже НЕ clean-room. Предлагаю пометить его как «не использовать» или удалить; это освободит ~540 МБ. Чистый кандидат — koz3_softceil_pixgbm_428257f_20b4ccea.zip.
4. .whl в ветке cpu-a-clean — это бенчмарк-зависимость, она не нужна. Вендорный wheel rvt-py в bot8-cnx — другой случай: он с Apache-2.0 и упомянут в docs/THIRD_PARTY.md.
5. Ускорение las_grid_fast (bench/las_grid_fast_result.md), если его захотят вернуть, нужно переписать с нуля по нашей спецификации, без опоры на поведение репо EN. Сейчас я этого не делаю.
