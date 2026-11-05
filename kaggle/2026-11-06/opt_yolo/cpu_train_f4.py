# CPU: YOLO26n, holdout fold 4 (train on folds 0-3), early signal before the Kaggle GPU quota resets.
import json, time
from pathlib import Path
from ultralytics import YOLO
t = time.time()
m = YOLO("yolo26n.pt")
m.train(data="data_f4.yaml", imgsz=640, epochs=30, batch=8, workers=2, device="cpu", project="/workspace/yolo_cpu/runs",
        name="f4_n640", exist_ok=True, plots=False, val=False, cos_lr=True, close_mosaic=5, fliplr=0.5, flipud=0.5,
        degrees=0.0, scale=0.25, mosaic=1.0, seed=0, patience=0)
best = Path("/workspace/yolo_cpu/runs/f4_n640/weights/last.pt")
val = [str(Path("/workspace/yolo_spor") / l.strip()[2:]) for l in open("/workspace/yolo_spor/val_f4.txt") if l.strip()]
mb = YOLO(str(best))
with open("/workspace/yolo_cpu/oof_f4.jsonl", "w") as fo:
    for i in range(0, len(val), 32):
        for r in mb.predict(source=val[i:i + 32], imgsz=640, conf=0.01, iou=0.6, device="cpu", verbose=False):
            b = r.boxes
            fo.write(json.dumps({"tile": Path(r.path).stem, "xyxy": b.xyxy.numpy().round(1).tolist(),
                                 "conf": b.conf.numpy().round(5).tolist(), "cls": b.cls.numpy().astype(int).tolist()}) + "\n")
print("CPU_TRAIN_DONE", round(time.time() - t), "s", flush=True)
