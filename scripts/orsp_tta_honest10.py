#!/usr/bin/env python3
"""Заявка и прогон honest-10 для TTA/мультимасштаба U-Net спутника.

Четыре варианта спутника (sp_or, sp_focal2): baseline / tta / multiscale / tta+multiscale.
D-FINE в эти варианты не входит: выход D-FINE SpOr на mock v3on_0876090 сломан
(sp_or F1 0), фикс — ветка bots/bot2-e2e-v3-prep. Статус D-FINE: не оценено.

Без корня трейна и каталога work честного табло скрипт только печатает заявку
(код возврата 0 на --request, 3 если прогон запрошен, а данных нет). Ничего не скачивает.

    python scripts/orsp_tta_honest10.py --request

    python scripts/orsp_tta_honest10.py --root КОРЕНЬ_ТРЕЙНА --work КАТАЛОГ_OOF \\
        --out-dir work/orsp_tta --variant all
    # затем команды scoreboard, которые скрипт печатает, или --score
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Зум кадра U-Net, как Segmenter.ms_scales. 4 — отражения без поворота на 90°.
VARIANTS = {
    "baseline": {"enabled": False},
    "tta": {"enabled": True, "unet_tta": 4, "unet_scales": [1.0], "dfine_views": "off", "dfine_gsd": [1.0]},
    "multiscale": {"enabled": True, "unet_tta": 0, "unet_scales": [0.75, 1.0, 1.25],
                   "dfine_views": "off", "dfine_gsd": [1.0]},
    "tta_multiscale": {"enabled": True, "unet_tta": 4, "unet_scales": [0.75, 1.0, 1.25],
                       "dfine_views": "off", "dfine_gsd": [1.0]},
}

# Отложено до фикса bot2. В --variant all не входит.
DFINE_DEFERRED = {
    "dfine_tta_multiscale": {
        "enabled": True, "unet_tta": 0, "unet_scales": [1.0],
        "dfine_views": "d8", "dfine_gsd": [0.75, 1.0, 1.25], "dfine_fuse": "wbf",
    },
}

KUR = ("kurgany_povrezhdennye", "kurgany_tselye")
# ПК координатора, сообщение 29.10: GTX 1650 4 ГБ, Docker/WSL2.
PC_V2_S = 82.6
PC_V3_S = 143.0
# LB v2 eb268d6: инференс 3 мин 48 с. Отношение имеет смысл, только если это тот же объём, что 82,6 с.
PLATFORM_V2_S = 228.0
PLATFORM_BUDGET_S = 15 * 60


def request_text() -> str:
    ratio = PLATFORM_V2_S / PC_V2_S
    pc_cap = PLATFORM_BUDGET_S / ratio
    lines = [
        "Заявка координатору. На LB не сдавать. D-FINE не гонять: не оценено.",
        "",
        "Что: honest-10 и время TTA/мультимасштаба U-Net спутника (sp_or, sp_focal2).",
        "Ветка: bots/bot8-orsp-tta. Коммит: тот, что в PR (флаг [orsp_tta], по умолчанию false).",
        "Машина: ПК, GTX 1650 4 ГБ, Docker/WSL2. Не Kaggle.",
        "Веса: models/spor_gor_r18.pt, models/spor_focal_r18.pt, models/surf_arch_r18.pt,",
        "models/spor_focal_wide_r18.pt — как в configs/default.toml. Финальная модель",
        "Kaggle dmitrijkv/koz3-dfine-spor-final для этого прогона не нужна.",
        "",
        "Варианты (dfine_views=off, dfine_gsd=[1.0] — D-FINE не меняется):",
        "  baseline        [orsp_tta] enabled=false",
        "  tta             unet_tta=4 (отражения), unet_scales=[1.0]     множитель прямых проходов U-Net ×4",
        "  multiscale      unet_tta=0, unet_scales=[0.75, 1.0, 1.25]     ×3, маски усредняются",
        "  tta_multiscale  оба                                          ×12",
        "",
        "Команды (из корня репозитория, ветка bots/bot8-orsp-tta):",
        "  python scripts/orsp_tta_honest10.py --root КОРЕНЬ_ТРЕЙНА --work КАТАЛОГ_OOF \\",
        "      --out-dir work/orsp_tta --variant all --score",
        "КАТАЛОГ_OOF — тот же, что у scripts/scoreboard.py --honest 10 (gt_train_v2.geojson,",
        "fusion/li_hard2_fused.geojson, spor_*.geojson, mosaic_li). КОРЕНЬ_ТРЕЙНА — растры SpOr.",
        "",
        "Отдельно время на том же mock, где v2 = 82,6 с и v3 = 143,0 с (не на полном трейне):",
        "  четыре прогона inference/main.py с конфигом, где [orsp_tta] как в варианте выше.",
        "  В лог фазы «оптика» — секунды. Это и есть замер добавки.",
        "",
        "Экстраполяция на тестовый объём площадки (цель ≤ 15 мин = 900 с):",
        f"  Если mock {PC_V2_S} с — тот же объём, что LB v2 {PLATFORM_V2_S:.0f} с, то",
        f"  t_площадка ≈ t_mock * ({PLATFORM_V2_S:.0f}/{PC_V2_S}) ≈ t_mock * {ratio:.2f}.",
        f"  Тогда на mock нужно уложиться в {pc_cap:.0f} с (v3 сейчас {PC_V3_S} с, запас ~{pc_cap - PC_V3_S:.0f} с).",
        "  Добавка ≈ (множитель − 1) * время спутникового U-Net внутри этих 143 с, не всего прогона.",
        "  Если 82,6 с — меньший mock, чем прогон LB 3:48, коэффициент 2,76 не использовать:",
        "  прислать оба времени и объём (число участков / байты растров sp_or).",
        "  honest-10 по трейну в бюджет 15 мин не входит: 15 мин — это тест площадки.",
        "",
        "Критерий успеха:",
        "  1) В JSON каждого варианта есть honest-10 (все / 34 новых / без лидара), Δ к baseline,",
        "     и по слоям sp_or и or: precision = TP/(TP+0,5·FP), recall = TP/(TP+FN).",
        "     Ортофото (or) эта ветка не меняет — Δ по or около нуля, это проверка, что замена",
        "     спутниковых групп не задела or.",
        "  2) Включать флаг имеет смысл при Δ honest-10 ≥ +0,002 и t_площадка ≤ 900 с.",
        "     Иначе оставить enabled=false.",
        "  3) D-FINE не измерять, пока bots/bot2-e2e-v3-prep не починит CRS/порог. Статус: не оценено.",
        "",
        "Что вернуть: work/orsp_tta/summary.json, лог с секундами по вариантам,",
        "четыре строки honest-10 и таблицу P/R по sp_or и or, время mock e2e и оценку t_площадка.",
        "На LB не сдавать.",
    ]
    return "\n".join(lines)


def _group(feat: dict, model_key: str) -> str:
    if model_key == "sp_focal2":
        return "sp_focal2"
    cls = feat.get("class_name")
    if cls in KUR:
        return "sp_focal"
    if cls == "arkhitektury":
        return "sp_arch"
    return "sp_or"


def _overlay(cfg: dict, name: str) -> dict:
    own = dict(VARIANTS[name])
    out = dict(cfg)
    out["orsp_tta"] = own
    return out


def predict_variant(cfg: dict, train: Path, out_dir: Path) -> dict:
    """Спутниковые растры тем же _raster_with_model, что сдача. D-FINE не вызывается."""
    from expds.catalog import read_utm_json, scan_root
    from expds.pipeline import _cached_segmenters, _raster_with_model, _sp_floors

    out_dir.mkdir(parents=True, exist_ok=True)
    cache: dict = {}
    loaded = {key: bool(_cached_segmenters(cache, cfg, key, ROOT)) for key in ("sp_or", "sp_focal2")}
    if not loaded["sp_or"]:
        raise SystemExit("нет моделей [models].sp_or на диске — файлы не пишу, чтобы --replace не стёр стек")
    active = ["sp_or", "sp_focal", "sp_arch"] + (["sp_focal2"] if loaded["sp_focal2"] else [])
    buckets: dict[str, list] = {g: [] for g in active}
    t0 = time.perf_counter()
    n_reg = 0
    n_bytes = 0
    seen_bytes: set[str] = set()
    for reg in scan_root(train):
        crs = read_utm_json(getattr(reg, "utm_json", None))
        did = False
        for key in ("sp_or", "sp_focal2"):
            if key == "sp_focal2" and reg.las:
                continue
            paths = list((reg.rasters or {}).get("sp_or" if key == "sp_focal2" else key, []))
            if not paths:
                continue
            segs = _cached_segmenters(cache, cfg, key, ROOT)
            if not segs:
                continue  # sp_focal2 без файла весов: группу не подменяем
            floors = _sp_floors(cfg, "sp_focal2" if key == "sp_focal2" else "sp_or")
            for seg in segs:
                for path in paths:
                    key_b = str(path)
                    if key_b not in seen_bytes:
                        seen_bytes.add(key_b)
                        n_bytes += Path(path).stat().st_size
                    found = _raster_with_model(seg, Path(path), reg.name, "sp_or", min_conf=floors,
                                               fallback_crs=crs)
                    for feat in found:
                        if seg.keep is not None and feat["class_name"] not in seg.keep:
                            continue
                        feat["threshold_group"] = _group(feat, key)
                        buckets[feat["threshold_group"]].append(feat)
                    did = True
        if did:
            n_reg += 1
    elapsed = time.perf_counter() - t0
    files = {}
    for group, feats in buckets.items():
        path = out_dir / f"{group}.geojson"
        # writer округляет и ставит EPSG:3857 — тот же путь, что сдача
        from expds.writer import write_geojson

        write_geojson(feats, path)
        files[group] = str(path)
    return {"seconds": round(elapsed, 3), "regions": n_reg, "raster_bytes": n_bytes, "files": files,
            "n": {g: len(v) for g, v in buckets.items()}}


def score_commands(work: Path, train: Path, files: dict[str, str], out_json: Path) -> list[str]:
    repl = " ".join(
        f"--replace {g}={shlex.quote(files[g])}"
        for g in ("sp_or", "sp_focal", "sp_arch", "sp_focal2") if g in files)
    return [
        "python scripts/scoreboard.py "
        f"--work {work} --train {train} --metric materials/compute_metrics.py "
        f"--honest 10 {repl} --out {out_json}"
    ]


def layer_pr(pred_paths: list[Path], gt_path: Path, markup: str) -> dict:
    """P/R слоя до honest-порогов. И предсказания, и эталон отфильтрованы по markup_type.

    Это не цифра приёмки: порог ещё не подобран на других семьях.
    """
    from expds.metric import score

    feats = []
    for p in pred_paths:
        fc = json.loads(Path(p).read_text(encoding="utf-8"))
        feats += [f for f in fc.get("features", [])
                  if (f.get("properties") or {}).get("markup_type") == markup]
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    gt_feats = [f for f in gt.get("features", [])
                if (f.get("properties") or {}).get("markup_type") == markup]
    rep = score({"type": "FeatureCollection", "features": feats},
                {"type": "FeatureCollection", "features": gt_feats},
                str(ROOT / "materials" / "compute_metrics.py"))
    tp = fp = fn = 0
    for row in rep["per_class"].values():
        tp += row["tp"]
        fp += row["fp"]
        fn += row["fn"]
    precision = tp / (tp + 0.5 * fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    return {"markup": markup, "precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn,
            "final_score_in_layer": rep["final_score"]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--request", action="store_true", help="только текст заявки координатору")
    ap.add_argument("--root", type=Path, default=None, help="корень трейна с растрами SpOr")
    ap.add_argument("--work", type=Path, default=None, help="каталог OOF для scoreboard --honest 10")
    ap.add_argument("--out-dir", type=Path, default=Path("work/orsp_tta"))
    ap.add_argument("--variant", default="all", choices=["all", *VARIANTS])
    ap.add_argument("--score", action="store_true", help="после предсказаний вызвать scoreboard")
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "default.toml")
    args = ap.parse_args(argv)
    if args.request or args.root is None or args.work is None:
        print(request_text())
        if args.request or (args.root is None and args.work is None):
            return 0
        print("\nПрогон не начат: нужны --root и --work. Данные не скачиваются.", file=sys.stderr)
        return 3
    if not args.root.is_dir() or not args.work.is_dir():
        print(request_text())
        print(f"\nНет каталога root={args.root} или work={args.work}.", file=sys.stderr)
        return 3
    from expds.pipeline import load_config

    base = load_config(args.config)
    names = list(VARIANTS) if args.variant == "all" else [args.variant]
    summary = {"dfine": "не оценено", "variants": {}, "note": request_text()}
    for name in names:
        cfg = _overlay(base, name)
        dest = args.out_dir / name
        print(f"== {name} ==", flush=True)
        info = predict_variant(cfg, args.root, dest)
        info["commands"] = score_commands(args.work, args.root, info["files"], dest / "honest10.json")
        for cmd in info["commands"]:
            print(cmd, flush=True)
        if args.score:
            import subprocess
            rc = subprocess.call(info["commands"][0], shell=True, cwd=ROOT)
            info["score_rc"] = rc
        gt = args.work / "gt_train_v2.geojson"
        if gt.is_file():
            try:
                info["sp_or_pr_before_thresholds"] = layer_pr(
                    [Path(p) for p in info["files"].values()], gt, "sp_or")
            except Exception as exc:  # noqa: BLE001 — диагностика, прогон не роняем
                info["sp_or_pr_error"] = f"{type(exc).__name__}: {exc}"
        summary["variants"][name] = info
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"сводка: {args.out_dir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
