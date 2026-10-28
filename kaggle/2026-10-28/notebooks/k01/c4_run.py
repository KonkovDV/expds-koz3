# ── Ячейка 5. 10 фолдов × модели, по одной модели на GPU параллельно ───────────────────
OOF = {}; LOG = []; ERR = []
def run_model(mi, spec):
    dev = DEVICES[spec["gpu"]]
    Xd = torch.from_numpy(X).to(dev)  # uint8 на GPU
    p = np.full(len(y), np.nan); t_all = time.time()
    for k in range(len(FOLDS)):
        te = np.flatnonzero(fold == k); tr = np.flatnonzero((fold != k) & lab)
        if len(te) == 0 or len(set(y[tr].tolist())) < 2:
            continue
        t0 = time.time()
        m = fit(dev, Xd, tr, spec["encoder"], spec["seed"] * 100 + k)
        p[te] = predict(m, dev, Xd, te)
        a = auc(y[te][lab[te]], p[te][lab[te]])
        rec = {"model": spec["name"], "fold": k, "train": int(len(tr)), "test": int(len(te)),
               "test_labeled": int(lab[te].sum()), "auc": a, "sec": round(time.time() - t0, 1), "device": str(dev)}
        LOG.append(rec); print(json.dumps(rec, ensure_ascii=False), flush=True)
        del m
        if dev.type == "cuda": torch.cuda.empty_cache()
    OOF[spec["name"]] = p
    print(f"{spec['name']}: все фолды за {time.time() - t_all:.0f} с", flush=True)

def _safe(mi, spec):
    try:
        run_model(mi, spec)
    except Exception as e:  # noqa: BLE001 — исключение потока иначе теряется
        import traceback; ERR.append(f"{spec['name']}: {e!r}\n{traceback.format_exc()}")

T0 = time.time()
th = [threading.Thread(target=_safe, args=(i, s)) for i, s in enumerate(CFG["models"])]
[t.start() for t in th]; [t.join() for t in th]
if ERR:
    raise RuntimeError("обучение упало:\n" + "\n".join(ERR))
print(f"OOF готово за {time.time() - T0:.0f} с", flush=True)
