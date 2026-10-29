# bot-1 (29.10, кандидат v4): вставка в ядро D-FINE Ae/Or. Своё: частичная разметка, TTA flip × масштабы, WBF внутри снимка.
# Встраивается build.py перед «Шаг 5»; использует torch, np, CLASSES, CFG, log из ядра.
PARTIAL = os.environ.get("DFINE_PARTIAL", globals().get("PARTIAL_DEFAULT", "1")) == "1"
TTA_SCALES = [float(s) for s in os.environ.get("DFINE_TTA_SCALES", globals().get("TTA_SCALES_DEFAULT", "1.0,1.5")).split(",") if s]
FUSE = os.environ.get("DFINE_FUSE", globals().get("FUSE_DEFAULT", "wbf"))   # wbf | nms
FUSE_IOU = float(os.environ.get("DFINE_FUSE_IOU", "0.55"))
MASK_LOGIT = -30.0   # sigmoid ≈ 1e-13: отрицательный пример отсутствующего в разметке участка класса не штрафуется


def region_class_mask(inv):
    """Классы, у которых на участке есть файл разметки этой модальности (размечены); остальные — «не размечено»."""
    have = {c for _, _, c in inv["marks"]}
    return [c in have for c in range(len(CLASSES))]


def _mask_like(t, m):
    """m (B, C) bool → форма t (B, …, C)."""
    return m.view(m.shape[0], *([1] * (t.dim() - 2)), m.shape[1])


def install_partial_labels(model_cls):
    """Обёртка loss_function: логиты неразмеченных на участке классов → MASK_LOGIT (через where — градиент 0).
    Метки этих классов в чипе отсутствуют по построению, так что позитивов они не теряют."""
    if getattr(model_cls, "_aeor_partial", False):
        return
    orig = model_cls.loss_function

    def prop(self):
        f = orig.fget(self) if isinstance(orig, property) else orig

        def g(logits, labels, *a, **k):
            ms = [l.pop("class_mask", None) for l in labels]
            if not PARTIAL or all(m is None for m in ms):
                return f(logits, labels, *a, **k)
            C = logits.shape[-1]
            m = torch.stack([torch.ones(C, dtype=torch.bool, device=logits.device) if x is None else x.to(logits.device).bool()
                             for x in ms])
            fill = lambda t: torch.where(_mask_like(t, m), t, torch.full_like(t, MASK_LOGIT)) if torch.is_tensor(t) and \
                t.dim() >= 2 and t.shape[0] == m.shape[0] and t.shape[-1] == C else t
            a = list(a)
            if len(a) >= 4:
                a[3] = fill(a[3])   # outputs_class (B, L, Q, C)
            for key in ("outputs_class", "enc_topk_logits"):
                if key in k:
                    k[key] = fill(k[key])
            return f(fill(logits), labels, *a, **k)
        return g

    model_cls.loss_function = property(prop)
    model_cls._aeor_partial = True


def _iou1(b, B):
    ix = np.clip(np.minimum(b[2], B[:, 2]) - np.maximum(b[0], B[:, 0]), 0, None)
    iy = np.clip(np.minimum(b[3], B[:, 3]) - np.maximum(b[1], B[:, 1]), 0, None)
    inter = ix * iy
    ua = (b[2] - b[0]) * (b[3] - b[1]) + (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]) - inter
    return inter / np.maximum(ua, 1e-9)


def wbf_np(B, L, C, n_views, iou=0.55):
    """Weighted boxes fusion по классу; уверенность = сумма / max(n_views, членов)."""
    oB, oS, oC = [], [], []
    for c in np.unique(C):
        idx = np.where(C == c)[0]; idx = idx[np.argsort(-L[idx], kind="stable")]
        fused, mem = [], []
        for i in idx:
            j = -1
            if fused:
                v = _iou1(B[i], np.asarray(fused)); k = int(np.argmax(v)); j = k if v[k] >= iou else -1
            if j < 0: fused.append(B[i].copy()); mem.append([i])
            else:
                mem[j].append(i); mm = np.asarray(mem[j]); w = L[mm]; fused[j] = (B[mm] * w[:, None]).sum(0) / max(w.sum(), 1e-12)
        for fb, mm in zip(fused, mem):
            oB.append(fb); oS.append(float(L[mm].sum() / max(n_views, len(mm)))); oC.append(int(c))
    if not oB: return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)
    return np.asarray(oB, np.float32), np.clip(np.asarray(oS, np.float32), 0, 1), np.asarray(oC, np.int64)


def tta_views():
    return [(s, f) for s in TTA_SCALES for f in (False, True)]


def run_view(model, px, scale, flip, dev):
    """Один вид TTA: масштаб = апсемплинг окна (кратно 32), flip — по горизонтали; боксы нормированы (cx, cy, w, h)."""
    import torch.nn.functional as F
    inp = px
    if scale != 1.0:
        n = max(32, int(round(px.shape[-1] * scale / 32)) * 32)
        inp = F.interpolate(px, size=(n, n), mode="bilinear", align_corners=False)
    if flip: inp = inp.flip(3)
    with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
        o = model(pixel_values=inp)
    prob = o.logits.float().sigmoid(); bx = o.pred_boxes.float()
    if flip: bx = bx.clone(); bx[..., 0] = 1 - bx[..., 0]
    return prob, bx


def fuse_region_views(per_view, nms_iou):
    """per_view: [(B, L, C) торч] по видам → NMS внутри вида (перекрытие окон) → WBF/NMS между видами (один снимок)."""
    from torchvision.ops import batched_nms
    vs = []
    for B, L, Cc in per_view:
        if not B: continue
        B, L, Cc = torch.cat(B), torch.cat(L), torch.cat(Cc); k = batched_nms(B, L, Cc, nms_iou)
        vs.append((B[k].numpy(), L[k].numpy(), Cc[k].numpy()))
    if not vs: return None
    B = np.concatenate([v[0] for v in vs]); L = np.concatenate([v[1] for v in vs]); Cc = np.concatenate([v[2] for v in vs])
    if FUSE == "wbf" and len(per_view) > 1:
        B, L, Cc = wbf_np(B.astype(np.float64), L.astype(np.float64), Cc.astype(np.int64), len(per_view), FUSE_IOU)
    else:
        k = batched_nms(torch.as_tensor(B), torch.as_tensor(L), torch.as_tensor(Cc), nms_iou).numpy(); B, L, Cc = B[k], L[k], Cc[k]
    return torch.as_tensor(B), torch.as_tensor(L), torch.as_tensor(Cc)
