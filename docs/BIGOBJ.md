# bot-2 bigobj (задача 16): крупные объекты > 5000 м² — флаги, по умолчанию ВЫКЛ.

Аудит и замер: bot-2/REPORT2.md §16, пред-декларация: bot-2/EXPERIMENTS.md «Задача 16».
Патч применим к 58ab645 (`git apply --check` на индексе 58ab645 — ок); затронуты только
expds/post.py, expds/watchdog.py, expds/pipeline.py (output_floors). Без ключей поведение не меняется.

F1 — своя группа порога у крупных масок (`expds/post.py: big_area_groups, big_group_thresholds`,
`watchdog.finish` до apply_thresholds, `pipeline.output_floors` — ранний пол не выше порога _big):

    [post.big_groups]
    min_area_m2 = 5000
    classes = ["gorodishcha", "fortifikatsii"]
    # groups = []           # пусто — все группы
    [post.big_groups.thresholds]       # пороги <группа>_big; без них — порог родителя (выдача та же)
    # sp_or_big = { gorodishcha = ... }

F2 — круг-ядро у крупных масок городищ (существующий `[post.core]` + новый ключ `min_area_m2`):

    [post.core.gorodishcha]
    frac = 0.3
    min_r_m = 8
    min_area_m2 = 5000

Скрипты замера (офлайн, стек A bot-10/work11/fam0): scripts/bigobj/{bigobj_recall,var_curves,tab,bigobj_paired}.py.
