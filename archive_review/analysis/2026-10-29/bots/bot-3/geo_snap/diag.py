"""Шаг 1 (bot-3, geo_snap): GAP.md §5 на честном выходе S. finds: FN S (honest-10, maски по фолдам) и ближайший
кандидат S (любой / оставленный) — корзины ≤2, 2–3, 3–5, >5 м, вектор смещения (кандидат − эталон) по сторонам света;
смещение у TP. object_poly: FN с кандидатом рядом (центроид в буфере 5 м), не прошедшим геометрию; площади.
Режим масок: unseen keep, scope global и site. Потенциал: каждый FN с оставленным FP-кандидатом в 2–5 м → TP (tp+1, fp−1)."""
from common import *
out = {}
def f1(tp, fp, ng): return 2 * tp / (2 * tp + fp + (ng - tp)) if tp else 0.0
def ang_bucket(dx, dy):
    a = (np.degrees(np.arctan2(dx, dy)) + 360) % 360          # 0 = север, 90 = восток
    return ['N', 'E', 'S', 'W'][int(((a + 45) % 360) // 90)]
for scope in ['global', 'site']:
    R = out[scope] = {}
    # ── finds ──
    c = 'finds_points'; P = SV[c]; G = GTS[c]
    keep = kept_mask(P, c, 'keep', scope); kid = [i for i in range(len(P)) if keep[i]]
    m = greedy(P, c, kid); tpj = set(m.values()); fpi = [i for i in kid if i not in m]
    by_all, by_kfp = collections.defaultdict(list), collections.defaultdict(list)
    for i, p in enumerate(P):
        by_all[p[2]].append(i)
        if keep[i] and i not in m: by_kfp[p[2]].append(i)
    def nearest(reg_idx, r, g):
        ids = reg_idx.get(r, [])
        if not ids: return np.inf, None
        xy = np.array([[P[i][0].centroid.x, P[i][0].centroid.y] for i in ids]) if not hasattr(reg_idx, '_c') else None
        d = np.hypot(xy[:, 0] - g.x, xy[:, 1] - g.y); k = int(np.argmin(d)); return float(d[k]), ids[k]
    cache = {}
    def nn(kind, r, g):
        key = (kind, r)
        if key not in cache:
            ids = (by_all if kind == 'all' else by_kfp).get(r, [])
            cache[key] = (ids, np.array([[P[i][0].centroid.x, P[i][0].centroid.y] for i in ids]).reshape(-1, 2))
        ids, xy = cache[key]
        if not ids: return np.inf, None, 0, 0
        d = np.hypot(xy[:, 0] - g.x, xy[:, 1] - g.y); k = int(np.argmin(d))
        return float(d[k]), ids[k], float(xy[k, 0] - g.x), float(xy[k, 1] - g.y)
    fn = [j for j in range(len(G)) if j not in tpj]
    rec = {'n_gt': len(G), 'tp': len(tpj), 'fp': len(fpi), 'fn': len(fn),
           'f1': f1(len(tpj), len(fpi), len(G))}
    for kind in ['all', 'kept_fp']:
        bk = collections.Counter(); dirs = collections.Counter(); vec = []; fam = collections.defaultdict(collections.Counter)
        grp = collections.Counter(); used = set(); recov = 0
        for j in fn:
            g = G[j][0]; d, i, dx, dy = nn(kind, G[j][2], g)
            b = '≤2' if d <= 2 else '2–3' if d <= 3 else '3–5' if d <= 5 else '>5'
            bk[b] += 1; fam[site_family(G[j][2])][b] += 1
            if 2 < d <= 5:
                dirs[ang_bucket(dx, dy)] += 1; vec.append((dx, dy)); grp[P[i][3]] += 1
                if kind == 'kept_fp' and i not in used: used.add(i); recov += 1
        v = np.array(vec).reshape(-1, 2)
        rec[kind] = dict(buckets=dict(bk), by_family={k: dict(x) for k, x in fam.items()}, dir_2_5m=dict(dirs),
                         group_2_5m=dict(grp),
                         mean_dxdy_2_5m=v.mean(0).round(3).tolist() if len(v) else None,
                         median_dxdy_2_5m=np.median(v, 0).round(3).tolist() if len(v) else None)
        if kind == 'kept_fp':
            tp2, fp2 = len(tpj) + recov, len(fpi) - recov
            rec[kind]['unique_fp_2_5m'] = recov
            rec[kind]['oracle_snap_dF1'] = f1(tp2, fp2, len(G)) - rec['f1']
            rec[kind]['oracle_snap_dscore'] = 0.1 * rec[kind]['oracle_snap_dF1']
    # смещение у TP: кандидат − эталон
    tv = np.array([[P[i][0].centroid.x - G[j][0].x, P[i][0].centroid.y - G[j][0].y] for i, j in m.items()])
    tf = collections.defaultdict(list)
    for (i, j), v in zip(m.items(), tv): tf[(site_family(G[j][2]), P[i][3])].append(v)
    rec['tp_offset'] = {'all': dict(n=len(tv), mean=tv.mean(0).round(3).tolist(), median=np.median(tv, 0).round(3).tolist(),
                                    mean_abs_d=float(np.hypot(*tv.T).mean()))}
    for k, vv in tf.items():
        vv = np.array(vv); rec['tp_offset']['|'.join(k)] = dict(n=len(vv), mean=vv.mean(0).round(3).tolist(),
                                                                 median=np.median(vv, 0).round(3).tolist())
    R[c] = rec; print(scope, c, json.dumps(rec, ensure_ascii=False), flush=True)
    # ── object_poly ──
    c = 'object_poly'; P = SV[c]; G = GTS[c]; gg = [x[0] for x in G]
    keep = kept_mask(P, c, 'keep', scope); kid = [i for i in range(len(P)) if keep[i]]
    m = greedy(P, c, kid); tpj = set(m.values()); fpi = [i for i in kid if i not in m]
    pt = STRtree([p[0] for p in P]); cen = [p[0].centroid for p in P]
    fn = [j for j in range(len(G)) if j not in tpj]
    bk = collections.Counter(); a_fn, a_cand, a_tp, a_tpgt = [], [], [], []
    geomfail_kept = []; inside = 0; fam = collections.defaultdict(collections.Counter)
    for j in fn:
        near = [int(i) for i in pt.query(gg[j].buffer(5.0)) if P[int(i)][2] == G[j][2] and gg[j].buffer(5.0).contains(cen[int(i)])]
        if not near: bk['нет_в_5м'] += 1; fam[site_family(G[j][2])]['нет_в_5м'] += 1; continue
        okk = [i for i in near if calc._match_quality(P[i][0], gg[j])[0]]
        if okk: bk['рядом_сопоставим'] += 1; fam[site_family(G[j][2])]['рядом_сопоставим'] += 1; continue
        bk['рядом_не_проходит_геом'] += 1; fam[site_family(G[j][2])]['рядом_не_проходит_геом'] += 1
        a_fn.append(gg[j].area); ds = [cen[i].distance(gg[j].centroid) for i in near]; i0 = near[int(np.argmin(ds))]
        a_cand.append(P[i0][0].area)
        if gg[j].contains(cen[i0]): inside += 1
        kn = [i for i in near if keep[i] and i not in m]
        if kn: geomfail_kept.append(kn[int(np.argmin([cen[i].distance(gg[j].centroid) for i in kn]))])
    for i, j in m.items(): a_tp.append(P[i][0].area); a_tpgt.append(gg[j].area)
    q = lambda a: np.percentile(a, [10, 25, 50, 75, 90]).round(2).tolist() if len(a) else None
    u = len(set(geomfail_kept))
    rec = {'n_gt': len(G), 'tp': len(tpj), 'fp': len(fpi), 'fn': len(fn), 'f1': f1(len(tpj), len(fpi), len(G)),
           'buckets': dict(bk), 'by_family': {k: dict(x) for k, x in fam.items()},
           'geomfail_centroid_inside_gt': inside,
           'area_gt_fn_geomfail_p10_90': q(a_fn), 'area_nearest_cand_p10_90': q(a_cand),
           'area_tp_pred_p10_90': q(a_tp), 'area_tp_gt_p10_90': q(a_tpgt),
           'geomfail_with_kept_fp_unique': u,
           'oracle_geomfix_dscore': 0.1 * (f1(len(tpj) + u, len(fpi) - u, len(G)) - f1(len(tpj), len(fpi), len(G))),
           'groups_kept_fp_geomfail': dict(collections.Counter(P[i][3] for i in set(geomfail_kept)))}
    R[c] = rec; print(scope, c, json.dumps(rec, ensure_ascii=False), flush=True)
json.dump(out, open(f'{W}/diag_S.json', 'w'), ensure_ascii=False, indent=1, default=float)
