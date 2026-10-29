# ── общий блок (bot-9): пины, лог, распаковка исходников bot-5 с проверкой sha256 ──────────────────────────────
import os, sys, subprocess, importlib, json, time, hashlib, base64, gzip, glob, shutil
from pathlib import Path
T0 = time.time()
def log(*a): print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)
def deps(pins):
    need = []
    for mod, pin in pins.items():
        try:
            m = importlib.import_module(mod)
            if "==" in pin and getattr(m, "__version__", "") != pin.split("==")[1]: need.append(pin)
        except Exception: need.append(pin)
    if need and not os.environ.get("B9_DRY"):
        log("pip install:", need)
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--only-binary", ":all:", *need])
        if r.returncode: raise SystemExit(f"pip install не удался ({need}); нужен enable_internet=true")
    vers = {}
    for mod in ["numpy", "scipy", "torch", *pins]:
        try:
            out = subprocess.run([sys.executable, "-c", f"import {mod}; print({mod}.__version__)"], capture_output=True, text=True)
            vers[mod] = out.stdout.strip() or out.stderr.strip().splitlines()[-1]
        except Exception as e: vers[mod] = repr(e)
    log("версии:", json.dumps(vers))
    bad = {m: vers.get(m) for m, p in pins.items() if "==" in p and vers.get(m) != p.split("==")[1]}
    if bad and not os.environ.get("B9_DRY"): raise SystemExit(f"версии не совпали с пинами bot-5: {bad}")
    return vers
SRC_DIR = Path(os.environ.get("B9_SRC", "/tmp/b5repo"))
def unpack_sources():
    """Исходники bot-5 @COMMIT (git archive), встроены gzip+base64; sha256 сверяется с git-блобом при сборке."""
    for rel, (sha, b64) in SOURCES.items():
        data = gzip.decompress(base64.b64decode(b64))
        assert hashlib.sha256(data).hexdigest() == sha, f"sha256 не совпал: {rel}"
        p = SRC_DIR / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data)
    log(f"исходники bot-5 @{COMMIT}: {len(SOURCES)} файлов → {SRC_DIR}")
def run_step(name, args, env_extra, logfile, timeout_s):
    """Запуск скрипта bot-5 подпроцессом; stdout+stderr → консоль и logfile. Возврат: (код, секунды)."""
    env_extra = {k: str(v) for k, v in env_extra.items()}; env = dict(os.environ); env.update(env_extra); t = time.time()
    log(f"▶ {name}: {' '.join(map(str, args))} | env {json.dumps(env_extra, ensure_ascii=False)} | timeout {timeout_s:.0f} s")
    import threading
    with open(logfile, "w", encoding="utf-8") as fh:
        p = subprocess.Popen([sys.executable, "-u", *map(str, args)], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        killed = []; tm = threading.Timer(max(1.0, timeout_s), lambda: (killed.append(1), p.kill())); tm.start()
        for line in p.stdout:
            fh.write(line); fh.flush(); print("   ", line, end="", flush=True)
        rc = p.wait(); tm.cancel()
        if killed: rc = "timeout"
    dt = time.time() - t; log(f"■ {name}: rc={rc}, {dt:.0f} s"); return rc, dt
def sha256_file(p, bs=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bs), b""): h.update(b)
    return h.hexdigest()
