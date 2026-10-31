# Бот 9: проверка фикса fork/filelock (c3-f4 ERROR) на боксе (CPU). Берёт make_loader/_mp_ctx из src/convnext_main.py как есть.
# Аудит-хук filelock 4.2 (os.fork) падает, если ДРУГОЙ поток сейчас внутри своего fork (fork_owner_depths) — две полосы
# одновременно стартуют воркеры DataLoader. Здесь 2 потока-«полосы» по N раз строят загрузчик (fork, 2 воркера) + поток с filelock.
# GUARD=0: FORK_LOCK = пустой контекст (как было до фикса) → ожидаем «unsafe while filelock» (фолбэк workers=0 в make_loader);
# GUARD=1: настоящий FORK_LOCK → 0 фолбэков, все загрузчики с 2 воркерами.
import os, sys, json, time, threading, itertools, multiprocessing, pathlib, tempfile, types
import torch, filelock
H = pathlib.Path(__file__).resolve().parent.parent
src = (H / "src/convnext_main.py").read_text(encoding="utf-8")
blk = src[src.index("def _worker_init(_)"):src.index("def train_job(job, dev, lane_left_iters, log_j):")]
CFG = dict(workers=2, bs=4)
def log(*a): pass
GUARD = int(os.environ.get("GUARD", "1"))
import contextlib
FORK_LOCK = threading.RLock() if GUARD else contextlib.nullcontext()
exec(compile(blk, "convnext_main.py:make_loader", "exec"))
class DS(torch.utils.data.Dataset):
    def __len__(self): return 10 ** 6
    def __getitem__(self, i): return torch.tensor(i)
GUARD = int(os.environ.get("GUARD", "1")); N = int(os.environ.get("N", "15"))
dev = types.SimpleNamespace(type="cuda")         # путь воркеров как на GPU (pin_memory на CPU — только предупреждение)
stop = threading.Event(); lk = filelock.FileLock(os.path.join(tempfile.mkdtemp(), "x.lock"))
def hammer():
    while not stop.is_set():
        with FORK_LOCK:
            with lk: pass
        time.sleep(0)
ta = threading.Thread(target=hammer); ta.start()
res = []
def lane_(k):
  for i in range(N):
      msgs = []
      try:
          dl, it_, first, info = make_loader(DS(), dev, lambda *a: msgs.append(" ".join(map(str, a))))
          nb = sum(1 for _ in itertools.islice(it_, 3)); res.append(dict(ok=True, workers=info["loader_workers"], ctx=info["loader_ctx"],
                   fallback=info["loader_fallback_error"], lock_wait=info["loader_lock_wait_s"]))
          del it_, dl
      except Exception as e: res.append(dict(ok=False, err=repr(e)[:120]))
ls_ = [threading.Thread(target=lane_, args=(k,)) for k in range(2)]
for t in ls_: t.start()
for t in ls_: t.join()
stop.set(); ta.join()
fb = [r for r in res if r.get("fallback")]
print(json.dumps(dict(guard=GUARD, loaders=2 * N, ok=sum(r["ok"] for r in res), fallback_to_0=len(fb),
                      workers=sorted({r.get("workers") for r in res if r["ok"]}), ctx=sorted({str(r.get("ctx")) for r in res if r["ok"]}),
                      sample_fallback=(fb[0]["fallback"][:120] if fb else None)), ensure_ascii=False))
