"""После OOF считает балл на прежних 49 участках, без новых 101 и 102.

Иначе общий балл нельзя сравнивать с 0,203: в файл попали два новых участка.
"""
import subprocess
import time
from pathlib import Path

PY = r"C:\Sloi\expds-koz3_v0.2.0\expds-koz3\.venv\Scripts\python.exe"
ROOT = r"C:\Sloi\expds-koz3_v0.2.0\expds-koz3"
W = Path(r"F:\expds3\work")
LOG = W / "hard3.log"
SWEEP = Path(r"C:\Users\Пользователь\AppData\Local\Temp\expds_materials\sweep_many.py")
FUSED = W / "fusion" / "li_hard3_fused.geojson"


def main() -> None:
    while "hard3 done" not in LOG.read_text(encoding="utf-8", errors="replace"):
        time.sleep(60)
    if not FUSED.is_file():
        raise SystemExit("нет файла предсказаний")
    r = subprocess.run(
        [PY, "-X", "utf8", str(SWEEP), str(W / "gt_train.geojson"),
         str(W / "regions_las49.txt"), "1", str(FUSED)],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    line = "sweep las49 " + r.stdout.strip()
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


if __name__ == "__main__":
    main()
