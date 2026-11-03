# ROSTER KOZ3 — кто есть кто (30.10.2026 13:05 MSK, минимальная версия; правит coord/A)
| Агент | Роль | Потоки | Worktree @ SHA | Ветка-аренда |
|---|---|---|---|---|
| A (этот бот, coord) | оркестрация, STATE/QUEUE/LEASES, аренды, сводка Елене | W0/coord | wt/orch @ 46c23b3 (bots/orch-protocol) | branch:bots/orch-protocol до 13.11 |
| B (агент 2, id 09c5d647…) | лидар | W2 | wt/w2 = /workspace/koz3/expds-koz3 @ 8c37e94 (bots/w2-lidar) | branch:bots/w2-lidar до 30.10 15:05 (под закрытый P-W2-10 → продлить под P-W2-9) |
| C (агент 3, id c77110f3…) | gate + реранкер | W4, W5 | wt/w4 @ 7eb271c (bots/w4-gate); wt/w5 @ c4e18a9 (bots/w5-rerank) | branch:bots/w4-gate, bots/w5-rerank до 12.11 |
| D (агент 4, id f92e0225…) | аэро/орто + крупные/курганы | W1, W3 | wt/w1 @ 1b457f3 (bots/w1-aeor); wt/w3 @ b259371 (bots/w3-kurgan-seeds) | до 01.11 12:57 |
Правила: PROTOCOL.md v2; LB сдаёт только Елена; main не пушить; GPU push — только с арендой от A.
