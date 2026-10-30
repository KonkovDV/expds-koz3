# C5B joint PREREG — W3, 30.10 15:45 MSK (до замера)
Проверка C5B на перекрытие с уже принятыми M_rp и ORG3 EtGr. Тот же w3_gate honest-10 на split_arch, A = 0.314615.
Компоненты в терминах split_arch (замены строк):
- MRP: все строки gorodishcha/arkhitektury/object_poly/fortifikatsii → диск r = 0.05 м в representative_point(P). Это как W0 post.micro_circles anchor=rp и W5 M_rp (пары w0_disk_pairs 'rp'); пороги заново подбирает гейт.
- ORG: строки object_poly·etgr в 212–217 (их ровно столько же, сколько в A18: 43/3/37/153/132/35) → object_poly·etgr из w4geo/eval/C19etonly_orget_s0.geojson (финал W4 N0, PASS +0.023191).
- C5B: variants/C5/C5B.* (conf := p_LR при площади ≥ 2000 м², non-li). Строки, которые ORG заменил, C5B не трогает. В MRP сначала применяется C5B (conf), затем диск.
Варианты против A0: MRP, ORG, C5B, S0 = MRP+ORG, S1 = MRP+ORG+C5B.
Решение (заранее):
1. Аддитивность: Δ(S1) сравнивается с ΔMRP+ΔORG+ΔC5B; сообщается доля сохранённого прироста ΔS1−ΔS0 к ΔC5B.
2. Главный критерий: парный бутстрап семей S1 vs S0. PASS, если ΔS1−ΔS0 ≥ +0.003, P ≥ 0.9, а на 34new и optical/no-lidar S1 не ниже S0.
3. Если ΔS1−ΔS0 ∈ [0; 0.003): C5B не передаётся в финал (отмечается как перекрытие).
C5B только за флагом (по умолчанию off), clean-room, без EN-кода.
