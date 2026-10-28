# Контракт головы без S3 (обе головы): D8-согласованность целей, untta, decode идеальных карт, эквивалентность
# вспомогательных функций ядра функциям bot-2 (d8_objs, tile_origins, valid_to_stride). Запуск: python tests/test_head.py
import sys, json, numpy as np, torch
sys.path.insert(0, "src/vendor"); torch.set_num_threads(2); torch.manual_seed(0)
import cnx_head as HD
from centerhead.d8 import d8_objs, d8_vec, torch_untta_offsets
from centerhead.decode import tile_origins as ch_tiles
from centerhead.losses import valid_to_stride
CFG = dict(sigma_k=0.15, sigma_min_px=1.0, default_diam_m=8.0, min_score=0.05, peak_k=11, max_det=20000, hm_bias=-2.19)
def d8(a, k, fl): a = np.rot90(a, k, axes=(-2, -1)); return np.ascontiguousarray(a[..., ::-1] if fl else a)
D8 = [(k, f) for f in (False, True) for k in range(4)]
res = {"equiv": {}}
rng = np.random.default_rng(0); S = 128
objs = np.c_[rng.uniform(8, 120, (8, 2)), rng.uniform(8, 24, 8), rng.integers(0, 2, 8)]
res["equiv"]["d8_points==d8_objs"] = max(float(np.abs(np.c_[HD.d8_points(objs[:, :2], S, k, f), objs[:, 2:]] - d8_objs(objs, S, k, f)).max()) for k, f in D8)
res["equiv"]["tile_origins"] = all(HD.tile_origins(H, W, 512, 384, s) == ch_tiles(H, W, 512, 384, s) for H in (300, 512, 1001, 2999) for W in (513, 1777) for s in (1, 4))
va = rng.random((2, 37, 50)) > 0.7
res["equiv"]["block_any==valid_to_stride"] = bool(all((HD.block_any(va[i], 4) == (valid_to_stride(torch.from_numpy(np.pad(va, ((0, 0), (0, 3), (0, 2)))), 4)[i].numpy() > 0)).all() for i in range(2)))
for name in ("heat", "centerhead"):
    H = HD.get_head(name, CFG); HD.check_head(H); r = {"stride": H.stride, "circle_r_m": H.circle_r_m}; s = H.stride
    t0 = H.targets(objs, S); r["targets"] = {k: list(v.shape) for k, v in t0.items()}
    worst = 0.0; offerr = 0.0
    for k, fl in D8:      # цели по повёрнутым точкам vs повёрнутые цели (центр-пиксель floor ⇒ равенство для S/s целых)
        t1 = H.targets(d8_objs(objs, S, k, fl), S); worst = max(worst, float(np.abs(d8(t0["hm"], k, fl) - t1["hm"]).max()))
        if "off" in t1:
            m = t1["reg_mask"] > 0; ov = np.moveaxis(d8_vec(np.moveaxis(d8(t0["off"], k, fl), 0, -1), k, fl), -1, 0)
            offerr = max(offerr, float(np.abs(ov[:, m] - t1["off"][:, m]).max()))
    r["d8_hm_maxabs"] = round(worst, 5); r["d8_off_maxabs"] = round(offerr, 5) if "off" in t0 else None
    m = H.build(5, "tu-convnext_tiny", None); x = torch.randn(2, 5, S, S); o = m(x)
    r["out_shapes"] = {k: list(v.shape) for k, v in o.items() if v is not None}
    tb = {k: torch.from_numpy(np.stack([v, v])) for k, v in t0.items()}
    l, parts = H.loss(o, tb, torch.ones(2, S, S)); l.backward(); r["loss"] = round(float(l.detach()), 4); r["parts"] = {k: round(v, 4) for k, v in parts.items()}
    nog = sorted({n.rsplit(".", 2)[0] for n, p in m.named_parameters() if p.grad is None})   # без маскового лосса — ожидаемо: seg-голова и блоки декодера после точки съёма
    r["no_grad_modules(без mask_loss)"] = nog; r["grad_finite"] = all(torch.isfinite(p.grad).all().item() for p in m.parameters() if p.grad is not None)
    with torch.no_grad():   # untta на «модели»-тождестве: все 8 видов возвращаются к виду k=0
        hs = S // s; z = torch.randn(1, 5, hs, hs)
        f = lambda zz: {"hm": zz[:, :2], "wh": zz[:, 2:3], "off": torch.zeros(1, 2, hs, hs)} if name == "centerhead" else {"hm": zz[:, :2]}
        base = H.untta(f(z), 0, False); e = 0.0
        for k, fl in D8:
            zi = torch.rot90(z, k, (2, 3)); zi = zi.flip(3) if fl else zi; u = H.untta(f(zi), k, fl)
            e = max(e, max(float((u[q] - base[q]).abs().max()) for q in base))
        r["untta_scalar_maxabs"] = e
    mp = {kk: v for kk, v in t0.items() if kk in ("hm", "wh", "off")}
    hs = S // s; tt = H.decode(mp, 1000.0, 2000.0, 0.5 * s, np.ones((hs, hs), bool), np.zeros((hs, hs), bool))
    X0, Y0 = 1000 + objs[:, 0] * 0.5, 2000 - objs[:, 1] * 0.5
    r["decode_n"] = int(len(tt["X"])); r["decode_dist_max_m"] = round(max(float(np.min(np.hypot(tt["X"] - a, tt["Y"] - b))) for a, b in zip(X0, Y0)), 4)
    r["cols"] = sorted(tt); res[name] = r
print(json.dumps(res, ensure_ascii=False, indent=1))
