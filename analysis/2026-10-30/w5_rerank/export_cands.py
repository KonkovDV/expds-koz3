"""table.pkl -> cands.csv for koz3-w5-dino-emb (cid = row in table)."""
from common import *
import numpy as np, shapely, csv
T = pickle.load(open(W/'work/table.pkl', 'rb')); g = shapely.from_wkb(pickle.load(open(W/'work/geom_wkb.pkl', 'rb')))
b = shapely.bounds(g)  # geoms are in UTM metres (prep), centres in T['cx'],T['cy']
with open(W/'work/cands.csv', 'w', newline='', encoding='utf-8') as f:
    wr = csv.writer(f); wr.writerow(['cid', 'region', 'markup_type', 'cx', 'cy', 'w', 'h'])
    for i in range(len(T['cls'])): wr.writerow([i, T['reg'][i], T['mk'][i], f"{T['cx'][i]:.2f}", f"{T['cy'][i]:.2f}", f"{b[i,2]-b[i,0]:.2f}", f"{b[i,3]-b[i,1]:.2f}"])
