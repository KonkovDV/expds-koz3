# ── TPU: обучение без синхронизаций хоста в цикле (статические формы, bf16 autocast, лог через step closure) ──────
# Отличия от GPU-train_job (convnext_lidar/src/convnext_main.py) — только устройство/точность/оптимизатор/лог; данные
# (WinDS: кроп 512, copy-paste, D8, цели головы в numpy), лоссы bot-2, EMA, косинус lr, бюджет дедлайна — те же.
_ENC_TEST = os.environ.get("CNX_ENCODER_TEST")            # только бокс: маленький энкодер, чтобы XLA-CPU уложился в 2 ГБ
if _ENC_TEST and not ARGS.allow_cpu: raise SystemExit("CNX_ENCODER_TEST допустим только с --allow-cpu (бокс)")
if _ENC_TEST:
    def build_model(ch, init): return HEAD.build(ch, _ENC_TEST, None)
WARM_IT = 3                                               # ит. 1..3 — компиляция графа(ов) XLA; с/ит — по ит. > 3
def _sync():
    if XLA_ON:
        import torch_xla; torch_xla.sync()                # torch_xla 2.8: xm.mark_step устарел → torch_xla.sync
def _wait():
    if XLA_ON:
        import torch_xla.core.xla_model as xm; xm.wait_device_ops()
def _xla_metrics():
    if not XLA_ON: return {}
    import torch_xla.debug.metrics as met
    out = {}
    for k in ("CompileTime", "ExecuteTime", "TransferToDeviceTime", "TransferFromDeviceTime"):
        d = met.metric_data(k)
        if d: out[k] = dict(n=int(d[0]), total_s=round(d[1] / 1e9, 2))
    out["counters"] = {k: int(met.counter_value(k)) for k in met.counter_names() if k.startswith("aten::")}   # опы, упавшие на CPU
    return out
def _pad_bs(a, n):
    """Статическая форма батча инференса: дополнить нулями до n (лишние строки отбрасывает zip(bt, v))."""
    return a if len(a) == n else np.concatenate([a, np.zeros((n - len(a),) + a.shape[1:], a.dtype)])

class XAdamW:
    """AdamW как torch.optim.AdamW (betas 0.9/0.999, eps 1e-8, развязанный weight decay), но lr и номер шага — тензоры
    на устройстве: косинусный lr python-числом стал бы константой графа XLA → перекомпиляция на каждом шаге."""
    def __init__(self, params, wd, dev, b1=0.9, b2=0.999, eps=1e-8):
        self.p = [p for p in params if p.requires_grad]; self.wd, self.b1, self.b2, self.eps = wd, b1, b2, eps
        self.m = [torch.zeros_like(p) for p in self.p]; self.v = [torch.zeros_like(p) for p in self.p]
        self.t = torch.zeros((), dtype=torch.float32, device=dev)
    def zero_grad(self):
        for p in self.p: p.grad = None
    @torch.no_grad()
    def step(self, lr):
        self.t += 1; bc1 = 1 - torch.pow(self.b1, self.t); bc2 = 1 - torch.pow(self.b2, self.t)
        for p, m, v in zip(self.p, self.m, self.v):
            if p.grad is None: continue
            g = p.grad
            p.mul_(1 - lr * self.wd)
            m.mul_(self.b1).add_(g, alpha=1 - self.b1); v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            p.sub_((lr / bc1) * m / (v.sqrt() / bc2.sqrt() + self.eps))

def head_loss_nosync(out, tgt, valid):
    """= CenterHeadAdapter.loss (cnx_head.py) → centerhead.losses.center_loss (bot-2 @ 788202a) без float(): части — тензоры.
    Формула та же (focal_loss_logits + reg_l1_loss, valid max-pool до шага головы); проверка равенства — tests/test_tpu_parts.py."""
    if HEAD.name != "centerhead": raise SystemExit("TPU-ядро поддерживает только голову centerhead (heat синхронизирует хост в лоссе)")
    from centerhead.losses import focal_loss_logits, reg_l1_loss, valid_to_stride
    c = HEAD.cfg; v = valid_to_stride(valid, HEAD.stride) if HEAD.stride > 1 else valid.float()
    parts = {"hm": focal_loss_logits(out["hm"].float(), tgt["hm"], v, c["focal_alpha"], c["focal_beta"])}
    total = c["w_hm"] * parts["hm"]; rm = tgt["reg_mask"] * v.to(tgt["reg_mask"].dtype)
    for k in ("wh", "off"):
        if out.get(k) is not None and c[f"w_{k}"] > 0:
            parts[k] = reg_l1_loss(out[k].float(), tgt[k], rm); total = total + c[f"w_{k}"] * parts[k]
    return total, {f"loss_{k}": x.detach() for k, x in parts.items()}

def train_job(job, dev, lane_left_iters, log_j):
    torch.manual_seed(CFG["seed"] + job["fold"]); train, test = split(job["fold"], POOL)
    model = build_model(job["ch"], job["init"]).to(dev)    # без channels_last: раскладку на TPU выбирает компилятор XLA
    msd = model.state_dict(); ema = {k: v.detach().clone().float() for k, v in msd.items()}
    fk = [k for k, v in msd.items() if v.dtype.is_floating_point]; ok_ = [k for k in msd if k not in fk]
    e_l, m_l = [ema[k] for k in fk], [msd[k] for k in fk]
    opt = XAdamW(model.parameters(), CFG["wd"], dev)
    iters = CFG["iters"]; wu = min(CFG["warmup"], max(1, iters // 10))
    lr_at = lambda it, n: CFG["lr"] * (it + 1) / wu if it < wu else CFG["lr"] * 0.5 * (1 + math.cos(math.pi * (it - wu) / max(1, n - wu)))
    use_bf16 = CFG["amp"] and XLA_ON
    ds = WinDS(train, job["ch"], CFG["seed"] * 1000 + job["fold"] * 10 + job["ch"], 10 ** 8)
    nw = CFG["workers"]
    dl = torch.utils.data.DataLoader(ds, batch_size=CFG["bs"], shuffle=False, num_workers=nw, pin_memory=False,
                                     persistent_workers=False, prefetch_factor=4 if nw else None, drop_last=True,   # drop_last: форма батча постоянна
                                     worker_init_fn=lambda _: torch.set_num_threads(1))
    if XLA_ON:
        import torch_xla.distributed.parallel_loader as pl
        ploader = pl.ParallelLoader(dl, [dev], loader_prefetch_size=8, device_prefetch_size=4); it_src = ploader.per_device_loader(dev)
    else:
        ploader = None; it_src = ((x.to(dev), {k: v.to(dev) for k, v in t.items()}, m.to(dev), v_.to(dev)) for x, t, m, v_ in dl)
    log_j(f"train: {len(train)} участков, объектов {sum(len(TGT[r]['samp']) for r in train)}, банк copy-paste {len(ds.bank)}; test {test}; "
          f"bf16={use_bf16}, workers={nw}")
    hist, bad, it, s_it = [], [], 0, None; model.train(); tm = {"t0": time.time()}; t_wait = 0.0
    def _closure(it_, loss_, lh_, lm_, parts_, lr_, t_):       # на хосте, после исполнения графа шага it_ (без лишней синхронизации)
        L = float(loss_)
        if not math.isfinite(L): bad.append(it_); return
        pv = {k: round(float(v), 4) for k, v in parts_.items()}
        hist.append(dict(it=it_, loss=round(L, 4), head=round(float(lh_), 4), mask=round(float(lm_), 4), **pv, lr=round(lr_, 7), t=round(t_, 1)))
        log_j(f"ит {it_}/{iters} loss {L:.4f} (head {float(lh_):.4f} {pv} mask {float(lm_):.4f})")
    budget_it = min(100, iters) if iters > 100 else None
    tw = time.time()
    for x, tgt, m3, val in it_src:
        t_wait += time.time() - tw
        if it >= iters: break
        lr = lr_at(it, iters); dcy = min(CFG["ema"], (1 + it) / (10 + it))
        sc = torch.tensor([lr, dcy], dtype=torch.float32).to(dev)          # данные, не константы графа
        x = norm(x)
        with torch.autocast(device_type="xla" if XLA_ON else "cpu", dtype=torch.bfloat16, enabled=use_bf16):
            out = model(x)
        lh, parts = head_loss_nosync(out, tgt, val); lm = mask_loss(out["mask"], m3, val); loss = lh + lm
        opt.zero_grad(); loss.backward(); opt.step(sc[0])
        with torch.no_grad():
            d = sc[1]
            for e, mm in zip(e_l, m_l): e.mul_(d).add_(mm.float() * (1 - d))
            for k in ok_: ema[k].copy_(msd[k])
        it += 1
        if it % CFG["log_every"] == 0 or it == iters or it <= 2:
            if XLA_ON:
                import torch_xla.core.xla_model as xm
                xm.add_step_closure(_closure, args=(it, loss.detach(), lh.detach(), lm.detach(), parts, lr, time.time() - tm["t0"]))
            else: _closure(it, loss.detach(), lh.detach(), lm.detach(), parts, lr, time.time() - tm["t0"])
        _sync()
        if bad: raise RuntimeError(f"loss не конечен на ит. {bad[0]}")
        if it in (1, WARM_IT):
            _wait(); tm[f"it{it}"] = time.time() - tm["t0"]
            if it == 1: log_j(f"ит 1 (компиляция + шаг): {tm['it1']:.1f} с; XLA {json.dumps(_xla_metrics().get('CompileTime', {}))}")
            if it == WARM_IT: tm["c_warm"] = _xla_metrics().get("CompileTime", {}).get("n"); tm["w_warm"] = t_wait
        if budget_it and it == budget_it:            # бюджет полосы под дедлайн — по установившейся с/ит (без компиляции)
            _wait(); s_it = (time.time() - tm["t0"] - tm[f"it{WARM_IT}"]) / (it - WARM_IT)
            left = DEADLINE - time.time() - CFG["infer_reserve_h"] * 3600 * (1 + lane_left_iters // max(1, CFG["iters"]))
            need_s = s_it * ((iters - it) + lane_left_iters)
            if need_s > left:
                new = max(int(CFG["min_iter_frac"] * CFG["iters"]), int(iters * left / max(need_s, 1)))
                log_j(f"БЮДЖЕТ: {s_it:.3f} с/ит, нужно {need_s / 3600:.2f} ч > есть {left / 3600:.2f} ч → итераций {iters} → {new}")
                iters = new
            else: log_j(f"бюджет OK: {s_it:.3f} с/ит, прогноз обучения {s_it * iters / 3600:.2f} ч")
        tw = time.time()
    _wait(); t_train = time.time() - tm["t0"]
    if ploader is not None: ploader.close()
    if bad: raise RuntimeError(f"loss не конечен на ит. {bad[0]}")
    t_w = tm.get(f"it{WARM_IT}", 0.0); n_st = max(1, it - WARM_IT)
    s_steady = (t_train - t_w) / n_st if it > WARM_IT else t_train / max(1, it)
    if s_it is None: s_it = s_steady
    xmet = _xla_metrics(); c_end = xmet.get("CompileTime", {}).get("n")
    recomp = (c_end - tm["c_warm"]) if (c_end is not None and tm.get("c_warm") is not None) else None
    if recomp: log_j(f"ВНИМАНИЕ: {recomp} перекомпиляций после ит. {WARM_IT} (формы/константы графа меняются)")
    model.load_state_dict({k: v.to(model.state_dict()[k].dtype) for k, v in ema.items()}); model.eval()
    del dl
    return model, dict(iters_planned=CFG["iters"], iters_effective=it, s_per_it=round(s_it, 4), s_per_it_steady=round(s_steady, 4),
                       s_per_sample_steady=round(s_steady / CFG["bs"], 5), t_first_it_s=round(tm.get("it1", 0.0), 1),
                       t_warm_s=round(t_w, 1), data_wait_frac=round((t_wait - tm.get("w_warm", 0.0)) / max(1e-9, t_train - t_w), 3),
                       recompiles_after_warm=recomp, xla=xmet, t_train_s=round(t_train, 1),
                       bank=len(ds.bank), hist=hist, train_regions=train, test_regions=test, device=str(dev), chip=TPU_CHIP)

