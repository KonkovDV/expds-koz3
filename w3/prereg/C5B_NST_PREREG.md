# C5B complement (NST) PREREG — W3, 30.10 16:00 MSK (до замера). ОДИН вариант: NST.
База B = A+M_rp (MRP: диск r = 0.05 м в representative_point для gor/arch/obj/fort). Стек split_arch, w3_gate honest-10, 10 семейных фолдов.
Кандидаты для каждого класса c ∈ {gorodishcha, arkhitektury, fortifikatsii, object_poly}, все поверх MRP:
- m0 = MRP (conf как есть);
- m1 = MRP + C5B conf;
- m2 = MRP + среднее перцентильных рангов (conf, C5B conf) внутри класса. Ранги без меток; вне скоупа C5B ранг тот же.
Вложенный выбор: для внешнего фолда k и класса c выбирается m* = argmax F1_c по внутреннему honest на train-фолдах.
- Внутренний honest: для каждого j ≠ k порог настраивается на фолдах ≠ j,k и проверяется на j; счётчики суммируются по j.
- При равенстве выбирается m0.
- Порог для фолда k настраивается на всех train-фолдах ≠ k (стандартно).
Классы курганы/finds идут из B без изменений.
Гейт NST vs B: Δ ≥ +0.003, парный семейный бутстрап P ≥ 0.9, 34new и optical не ниже B.
ORG3 (справочно, после главного гейта): C19etonly_orget_s0.geojson (финал W4 N0) заменяет object_poly·etgr в 212–217. Порог etgr·object_poly фиксирован 0 (как --fix etgr:object_poly=0). Сравнение (B+ORG3 → NST+ORG3) той же процедурой.
Флаг: bigobj_c5b (по умолчанию false). Clean-room, без EN-кода.
