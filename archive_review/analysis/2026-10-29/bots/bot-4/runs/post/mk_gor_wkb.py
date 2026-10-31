"""Разовый кэш: wkb только строк gorodishcha из /workspace/rerank/work/geom_wkb.pkl (строки gor одинаковы в table.pkl и в таблице пакета)."""
import pickle, numpy as np
T = pickle.load(open('/workspace/rerank/work/table.pkl', 'rb')); gidx = np.flatnonzero(T['cls'] == T['CL'].index('gorodishcha')); del T
G = pickle.load(open('/workspace/rerank/work/geom_wkb.pkl', 'rb'))
pickle.dump({'idx': gidx, 'wkb': G[gidx]}, open('/workspace/koz3_shared/bot-4/work/post/gor_wkb.pkl', 'wb')); print(len(gidx))
