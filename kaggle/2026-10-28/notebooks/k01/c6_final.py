# ── Ячейка 7. Финальные модели на всех размеченных (для теста), веса в /kaggle/working/weights ───────
if CFG["final_models"]:
    (WORK / "weights").mkdir(exist_ok=True)
    FERR = []
    def _final(spec):
        try:
            dev = DEVICES[spec["gpu"]]; Xd = torch.from_numpy(X).to(dev); t0 = time.time()
            m = fit(dev, Xd, np.flatnonzero(lab), spec["encoder"], spec["seed"] * 100 + 99)
            blob = {"state_dict": {k: (v.half() if v.is_floating_point() else v).cpu() for k, v in m.state_dict().items()},
                    "encoder": spec["encoder"], "dropout": CFG["dropout"], "norm": "(x/255-0.5)/0.25", "tile": int(X.shape[-1]),
                    "tta": CFG["tta"], "labeled": int(lab.sum()), "povr_share": float((y[lab] == 1).mean()),
                    "apply_thr": CFG["apply_thr"], "npz": str(NPZ), "label": "1 = kurgany_povrezhdennye, 0 = kurgany_tselye"}
            torch.save(blob, WORK / "weights" / f"{spec['name']}_all.pt")
            print(f"{spec['name']}: финальная модель {time.time() - t0:.0f} с", flush=True)
        except Exception as e:  # noqa: BLE001
            FERR.append(f"{spec['name']}: {e!r}")
    th = [threading.Thread(target=_final, args=(s,)) for s in CFG["models"]]
    [t.start() for t in th]; [t.join() for t in th]
    if FERR:
        raise RuntimeError("; ".join(FERR))
print("ГОТОВО за", round(time.time() - T0), "с", flush=True)
