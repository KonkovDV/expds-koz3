# koz3-w4: чистая перепись PIXGBM — дообучение фолдовых моделей сидов 1, 2 (CPU).
# Датасет usertest1234/koz3-w4-pixclean-cache: cache_clean/*.npz (выборка пикселей 200–217, 548 МБ)
# + code/ (expds/, scripts/pix_clean_train.py из bots/w4-pix-clean). Растров и GT в датасете нет.
import glob, os, shutil, subprocess, sys
D = glob.glob("/kaggle/input/*koz3-w4-pixclean-cache*")[0]
shutil.copytree(f"{D}/code", "/kaggle/working/code", dirs_exist_ok=True)
os.chdir("/kaggle/working/code")
F = {"F1": "200", "F2": "201 202 203 204 205 206 207", "F3": "208 209 210 215",
     "F4": "211 212 216", "F5": "213 214 217"}
seeds = os.environ.get("SEEDS", "1 2").split()
for s in seeds:
    for k, ex in F.items():
        out = f"/kaggle/working/models/{k}_s{s}"
        print("==", k, s, flush=True)
        subprocess.run([sys.executable, "scripts/pix_clean_train.py", "fit", "--cache", f"{D}/cache_clean",
                        "--out", out, "--seed", s, "--threads", str(os.cpu_count() or 4),
                        "--exclude", *ex.split()], check=True)
shutil.rmtree("/kaggle/working/code")
print("ALLFIT")
