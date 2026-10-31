#!/usr/bin/env python3
"""Потоковый проход по oof_dfine_ae.geojson (3.3 ГБ, одна строка): без json.load целиком.
Делит поток по маркеру начала фичи, (1) пишет копию с ae_dfine→dfine_ae в threshold_group/group (байт-в-байт иначе),
(2) собирает компактный индекс (смещение, длина, участок, класс, conf, fold, bbox 3857) в npz."""
import json, sys, hashlib, collections, numpy as np
from array import array
SRC = '/workspace/kaggle_out/koz3-dfine-ae/oof_dfine_ae.geojson'
DST = '/workspace/koz3_shared/bot-9/for_bot10/oof_dfine_ae.geojson'
IDX = '/workspace/koz3_shared/bot-9/dfine_ae_oof/oof_index.npz'
TOK = b'{"type": "Feature", '
OLD, NEW = b'"ae_dfine"', b'"dfine_ae"'
CLS = ["kurgany_tselye", "kurgany_povrezhdennye", "gorodishcha", "fortifikatsii", "arkhitektury"]
regs, tifs = {}, {}
A = {k: array(t) for k, t in dict(off='q', ln='l', reg='h', cls='b', conf='f', fold='b', tif='h',
                                       x0='d', y0='d', x1='d', y1='d', nv='h').items()}
stat = collections.Counter(); props_vals = collections.defaultdict(collections.Counter)
h_src, h_dst = hashlib.sha256(), hashlib.sha256()
fo = open(DST + '.part', 'wb')
def handle(feat_bytes, off):
    # feat_bytes: TOK + тело + разделитель (', ' или ']}' в конце)
    n = feat_bytes.count(OLD); stat[f'n_old_per_feat={n}'] += 1
    out = feat_bytes.replace(OLD, NEW); fo.write(out); h_dst.update(out)
    body = feat_bytes.rstrip()
    if body.endswith(b']}'): body = body[:-2].rstrip()
    if body.endswith(b','): body = body[:-1]
    ft = json.loads(body); p = ft['properties']; g = ft['geometry']
    stat['geom_' + g['type']] += 1
    rings = g['coordinates'] if g['type'] == 'Polygon' else [r for poly in g['coordinates'] for r in poly]
    if not rings or not rings[0]:
        stat['geom_empty'] += 1; xy = np.full((1, 2), np.nan)
    else: xy = np.asarray(rings[0], dtype=np.float64)
    for k in ('crs', 'markup_type', 'group', 'threshold_group', 'original_crs', 'fold'): props_vals[k][str(p.get(k))] += 1
    A['off'].append(off); A['ln'].append(len(feat_bytes))
    A['reg'].append(regs.setdefault(p['region_name'], len(regs))); A['cls'].append(CLS.index(p['class_name']))
    A['conf'].append(float(p['confidence'])); A['fold'].append(int(p['fold'])); A['tif'].append(tifs.setdefault(p['tif'], len(tifs)))
    A['x0'].append(xy[:, 0].min()); A['y0'].append(xy[:, 1].min()); A['x1'].append(xy[:, 0].max()); A['y1'].append(xy[:, 1].max())
    A['nv'].append(len(xy))
with open(SRC, 'rb') as f:
    buf = b''; pos = 0; header_done = False; CH = 32 << 20
    while True:
        chunk = f.read(CH); h_src.update(chunk)
        buf += chunk; eof = not chunk
        if not header_done:
            i = buf.find(TOK)
            if i < 0 and not eof: continue
            hdr = buf[:i]; fo.write(hdr); h_dst.update(hdr); stat['header'] = hdr.decode(); buf = buf[i:]; pos = i; header_done = True
        # режем по маркеру; последний кусок оставляем в буфере до EOF
        start = 0
        while True:
            j = buf.find(TOK, start + len(TOK))
            if j < 0: break
            handle(buf[start:j], pos + start); start = j
        buf = buf[start:]; pos += start
        if eof:
            if buf: handle(buf, pos)
            break
fo.close()
import os; os.replace(DST + '.part', DST)
np.savez(IDX, **{k: np.frombuffer(v, dtype=v.typecode) if False else np.array(v) for k, v in A.items()},
         regions=np.array(list(regs)), tifs=np.array(list(tifs)))
conf = np.array(A['conf'])
res = dict(src=SRC, src_sha256=h_src.hexdigest(), dst=DST, dst_sha256=h_dst.hexdigest(), n_features=len(conf),
           stat=dict(stat), props=dict({k: dict(v) for k, v in props_vals.items()}),
           conf_min=float(conf.min()), conf_max=float(conf.max()), conf_median=float(np.median(conf)),
           conf_hist={f'{a}-{b}': int(((conf >= a) & (conf < b)).sum()) for a, b in
                      [(0, .05), (.05, .1), (.1, .2), (.2, .3), (.3, .5), (.5, .7), (.7, 1.01)]},
           x_range=[float(np.nanmin(A['x0'])), float(np.nanmax(A['x1']))], y_range=[float(np.nanmin(A['y0'])), float(np.nanmax(A['y1']))], nv=dict(collections.Counter(A['nv'])))
json.dump(res, open('/workspace/koz3_shared/bot-9/dfine_ae_oof/pass1.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in res.items()}, ensure_ascii=False)[:3000])
