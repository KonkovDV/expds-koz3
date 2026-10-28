# ── Ячейка 4. Честные фолды по семьям и обучение ───────────────────────────────────────
# site_family / family_folds — копия expds/naming.py:108 и expds/metric.py:538 (как scoreboard --honest 10).
def site_family(dirname):
    m = re.match(r"^(\d{3})_(.+)$", dirname)
    name = m.group(2) if m else dirname
    return re.split(r"[_\s]", name, maxsplit=1)[0]

def family_folds(regions, family_of, k=None):
    fams = {}
    for r in regions:
        fams.setdefault(family_of(r), set()).add(r)
    names = sorted(fams)
    if not k or k >= len(names):
        return [fams[f] for f in names]
    bins, sizes = [set() for _ in range(k)], [0] * k
    for f in sorted(names, key=lambda f: (-len(fams[f]), f)):
        i = int(np.argmin(sizes)); bins[i] |= fams[f]; sizes[i] += len(fams[f])
    return [b for b in bins if b]

family = np.array([site_family(r) for r in region])
if "family" in D.files and not np.array_equal(family, D["family"].astype(str)):
    print(f"ВНИМАНИЕ: поле family в npz расходится с site_family(region) у {int((family != D['family'].astype(str)).sum())} окон; беру site_family", flush=True)
FOLDS = family_folds(sorted(set(region)), site_family, CFG["lfo_folds"])
fold_of = {r: k for k, regs in enumerate(FOLDS) for r in regs}
fold = np.array([fold_of[r] for r in region])
for k in range(len(FOLDS)):
    te_f = set(family[fold == k]); tr_f = set(family[fold != k])
    if te_f & tr_f:
        raise RuntimeError(f"утечка: семьи {sorted(te_f & tr_f)} в обучении и проверке фолда {k}")
print(f"фолдов {len(FOLDS)}, семей {len(set(family))}; ни одна семья не делится между фолдами", flush=True)

def norm(xb):
    return (xb.float() / 255.0 - 0.5) / 0.25  # как crop_verify._tensor и кэш обучения

def augment(x, g):
    """x: (B,C,H,W) норм. Поворот на любой угол + отражение + сдвиг/масштаб (reflect) + усиление/сдвиг каналов."""
    B = x.shape[0]; A = CFG["aug"]; dev = x.device
    ang = torch.rand(B, device=dev, generator=g) * 2 * math.pi
    sc = 1 + (torch.rand(B, device=dev, generator=g) * 2 - 1) * A["scale"]
    flip = torch.where(torch.rand(B, device=dev, generator=g) < 0.5, -1.0, 1.0)
    sh = (torch.rand(B, 2, device=dev, generator=g) * 2 - 1) * (2 * A["shift_px"] / x.shape[-1])
    cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
    theta = torch.stack([torch.stack([cos * flip, -sin, sh[:, 0]], 1), torch.stack([sin * flip, cos, sh[:, 1]], 1)], 1)
    grid = torch.nn.functional.affine_grid(theta, list(x.shape), align_corners=False)
    x = torch.nn.functional.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    gain = 1 + (torch.rand(B, x.shape[1], 1, 1, device=dev, generator=g) * 2 - 1) * A["gain"]
    bias = (torch.rand(B, x.shape[1], 1, 1, device=dev, generator=g) * 2 - 1) * A["bias"]
    return x * gain + bias + torch.randn(x.shape, device=dev, generator=g) * A["noise"]

def tta_views(x):
    v = [torch.rot90(x, k, dims=(2, 3)) for k in range(4)]
    v += [t.flip(-1) for t in v]
    return v[:CFG["tta"]]

def fit(dev, Xd, tr, encoder, seed):
    torch.manual_seed(seed); g = torch.Generator(device=dev); g.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = make_model(encoder).to(dev)
    if CFG["require_gpu"] and not LOCAL_TEST:
        assert all(p.device.type == "cuda" for p in m.parameters()), "--require-gpu: модель не на GPU"
    opt = torch.optim.AdamW(m.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    steps_ep = max(1, len(tr) // CFG["batch"])
    if CFG.get("max_batches"):
        steps_ep = min(steps_ep, CFG["max_batches"])
    total = CFG["epochs"] * steps_ep
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, CFG["lr"], total_steps=total, pct_start=CFG["warmup_frac"])
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ytr = torch.from_numpy(y).to(dev).float()
    trg = torch.from_numpy(tr).to(dev)
    for ep in range(CFG["epochs"]):
        m.train()
        order = trg[torch.from_numpy(rng.permutation(len(tr))).to(dev)]
        for s in range(steps_ep):
            b = order[s * CFG["batch"]:(s + 1) * CFG["batch"]]
            if len(b) < 2:
                continue
            x = augment(norm(Xd[b]), g)
            if s == 0 and ep == 0 and CFG["require_gpu"] and not LOCAL_TEST:
                assert x.is_cuda, "--require-gpu: батч не на GPU"
            with torch.autocast(dev.type, dtype=torch.float16, enabled=use_amp):
                lo = torch.nn.functional.binary_cross_entropy_with_logits(m(x).float(), ytr[b])
            opt.zero_grad(set_to_none=True)
            scaler.scale(lo).backward(); scaler.step(opt); scaler.update(); sched.step()
    return m.eval()

@torch.no_grad()
def predict(m, dev, Xd, idx):
    out = np.zeros(len(idx)); use_amp = dev.type == "cuda"
    ig = torch.from_numpy(idx).to(dev)
    for s in range(0, len(idx), 512):
        x = norm(Xd[ig[s:s + 512]])
        with torch.autocast(dev.type, dtype=torch.float16, enabled=use_amp):
            p = sum(torch.sigmoid(m(v).float()) for v in tta_views(x)) / len(tta_views(x))
        out[s:s + 512] = p.cpu().numpy()
    return out

def auc(yy, pp):
    yy = np.asarray(yy); pp = np.asarray(pp)
    n1 = int((yy == 1).sum()); n0 = int((yy == 0).sum())
    if not n1 or not n0:
        return None
    o = np.argsort(pp, kind="mergesort"); r = np.empty(len(pp)); r[o] = np.arange(1, len(pp) + 1)
    for v in np.unique(pp):  # средние ранги при равенствах
        t = pp == v
        if t.sum() > 1: r[t] = r[t].mean()
    return float((r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
