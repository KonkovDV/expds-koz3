# Сравнение наших каналов c3/c5 (ours.npz) с 12-канальным стеком bot-8 (theirs.npz) на одном участке.
import json, numpy as np
o = np.load("ours.npz"); t = np.load("theirs.npz")
arr = o["arr"].astype(np.float32); st = t["stack"]; ins = t["inside"]; org = t["origin"]
dx = (float(o["left"]) - org[0]) / 0.5; dy = (org[1] - float(o["top"])) / 0.5
assert abs(dx - round(dx)) < 1e-6 and abs(dy - round(dy)) < 1e-6, (dx, dy)
dx, dy = int(round(dx)), int(round(dy)); H, W = arr.shape[1:]
st = st[:, dy:dy + H, dx:dx + W]; ins = ins[dy:dy + H, dx:dx + W]; assert st.shape[1:] == (H, W), st.shape
net = (arr - 128) / 127                                  # вход нашей сети: norm(x)=(x-128)/127
vo = arr[0] > 0; both = vo & ins
R = dict(offset_px=dict(dx=dx, dy=dy), shape=[H, W], valid_ours=int(vo.sum()), valid_b8=int(ins.sum()), valid_both=int(both.sum()),
         valid_xor=int((vo ^ ins).sum()))
def stat(a, b, m):
    d = np.abs(a[m] - b[m]); c = np.corrcoef(a[m], b[m])[0, 1]
    return dict(max_abs=round(float(d.max()), 4), mean_abs=round(float(d.mean()), 4), p99_abs=round(float(np.percentile(d, 99)), 4), pearson=round(float(c), 4))
# 1) DTM на 0.5 м: наш z (0.1→5x5 mean) vs avgpool 2x2 их DTM 0.25
import torch, torch.nn.functional as F
ah, aw = int(t["dtm_shape"][0]), int(t["dtm_shape"][1])
# их z на 0.5 м = восстановить нельзя из стека напрямую → пересчёт из dtm не сохранён; используем LRM-разности ниже, z сравним через ours.z vs (нет) — см. compare_dtm
B8 = ["LRM2", "LRM5", "LRM10", "LRM25", "slope", "hs315", "hs45", "lap", "pos_open", "neg_open", "svf", "valid"]
OURS = ["lrm_s", "lrm_l", "slope", "svf_r", "opp_r"]
# 2) вход сети как есть: каждая наша плоскость vs ближайший аналог bot-8 (оба на both; у bot-8 невалид=0, у нас -1.008)
pairs = [("lrm_s", "LRM5"), ("lrm_s", "LRM10"), ("lrm_l", "LRM25"), ("slope", "slope"), ("svf_r", "svf"), ("opp_r", "pos_open")]
R["net_input_as_is"] = {f"{a}~{b}": stat(net[OURS.index(a)], st[B8.index(b)], both) for a, b in pairs}
R["invalid_value"] = dict(ours=round(float(net[0][~vo].mean()), 4) if (~vo).any() else None, b8=float(st[0][~ins].max()) if (~ins).any() else None)
# 3) физические единицы (только где наш канал не в клипе)
S = json.load(open("ours.log"))["stats"] if False else None
phys = {}
lr_s = net[0] * 0.75; lr_l = net[1] * 2.0; m_s = both & (np.abs(net[0]) < 0.999); m_l = both & (np.abs(net[1]) < 0.999)
for b in ("LRM2", "LRM5", "LRM10"):
    phys[f"lrm_s[m]~{b}*0.5"] = stat(lr_s, st[B8.index(b)] * 0.5, m_s & (np.abs(st[B8.index(b)]) < 4))
phys["lrm_l[m]~LRM25*0.5"] = stat(lr_l, st[B8.index("LRM25")] * 0.5, m_l & (np.abs(st[B8.index("LRM25")]) < 4))
sd = (net[2] + 1) / 2 * 45; ms = both & (net[2] < 0.999) & (net[2] > -0.999)
phys["slope[deg]~slope*0.5[deg]"] = stat(sd, np.degrees(st[B8.index("slope")] * 0.5), ms)
R["physical_units"] = phys
json.dump(R, open("compare.json", "w"), indent=1); print(json.dumps(R, indent=1))
