# KOZ3: organizers' public S3 (s3://train-expds-3) -> box, lossless on-the-fly compression (.las->.laz, .tif->ZSTD GeoTIFF)
import os, sys, json, time, hashlib, shutil, subprocess, threading, queue, traceback
os.environ['GDAL_NUM_THREADS'] = '1'; os.environ['RAYON_NUM_THREADS'] = '1'; os.environ['GDAL_CACHEMAX'] = '128'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
DEST = Path('/workspace/data/train-expds-3'); DATA = DEST.parent
TMP = DATA / 's3_tmp'; LOGP = DATA / 's3_dl.log'; MANP = DATA / 's3_manifest.tsv'
LIST = DATA / 's3_listing.jsonl'; BUCKET = 's3://train-expds-3/'
MINFREE = float(os.environ.get('S3_MINFREE_GB', '12')) * 1e9
NPROC = int(os.environ.get('S3_NPROC', '3')); NDL = int(os.environ.get('S3_NDL', '2'))
BIG = 10e9; KEEP_RAW_BELOW = 1.3
S5 = os.path.expanduser('~/.local/bin/s5cmd')
LOG = open(LOGP, 'a', buffering=1); llock = threading.Lock()
def log(*a):
    with llock: LOG.write(time.strftime('%Y-%m-%d %H:%M:%S ') + ' '.join(map(str, a)) + '\n')
def free(): return shutil.disk_usage(DATA).free
def dest_of(rel): return DEST / (rel[:-4] + '.laz' if rel.lower().endswith('.las') else rel)

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()

def tif_meta(d):
    return dict(crs=d.crs.to_wkt() if d.crs else None, transform=list(d.transform), nodata=list(d.nodatavals), dtypes=list(d.dtypes), count=d.count,
                w=d.width, h=d.height, desc=list(d.descriptions), ci=[c.name for c in d.colorinterp], scales=list(d.scales), offsets=list(d.offsets),
                units=list(d.units), tags=d.tags(), btags=[d.tags(i) for i in range(1, d.count + 1)], novr=len(d.overviews(1)),
                masks=[[m.name for m in f] for f in d.mask_flag_enums])

def do_tif(raw, out):
    import numpy as np, rasterio, rasterio.shutil as rsh
    from rasterio.windows import Window
    with rasterio.open(raw) as d:
        kind = np.dtype(d.dtypes[0]).kind; ovr = len(d.overviews(1))
    o = dict(TILED='YES', BLOCKXSIZE=512, BLOCKYSIZE=512, BIGTIFF='IF_SAFER', NUM_THREADS='1', COMPRESS='ZSTD', ZSTD_LEVEL=int(os.environ.get('S3_ZLEVEL', '4')), PREDICTOR=3 if kind == 'f' else 2)
    if ovr: o['COPY_SRC_OVERVIEWS'] = 'YES'
    rsh.copy(str(raw), str(out), driver='GTiff', **o)
    with rasterio.open(raw) as a, rasterio.open(out) as b:
        ma, mb = tif_meta(a), tif_meta(b)
        diff = [k for k in ma if ma[k] != mb[k] and not (k == 'crs' and a.crs == b.crs)]
        if diff: return False, 'META DIFF ' + str({k: (ma[k], mb[k]) for k in diff})[:500], ma
        rows = max(1, int(48e6 // (a.width * a.count * np.dtype(a.dtypes[0]).itemsize)))
        rows = max(512, rows // 512 * 512) if a.height > 512 else a.height
        hh = hashlib.sha256()
        for r0 in range(0, a.height, rows):
            w = Window(0, r0, a.width, min(rows, a.height - r0))
            x = a.read(window=w).tobytes()
            if x != b.read(window=w).tobytes(): return False, f'pixel diff at row {r0}', ma
            hh.update(x)
    return True, f'tif {ma["w"]}x{ma["h"]}x{ma["count"]} {ma["dtypes"][0]} pixsha={hh.hexdigest()[:16]} full-compare-ok', ma

def do_las(raw, out):
    import laspy
    with laspy.open(raw) as r:
        with laspy.open(out, mode='w', header=r.header, do_compress=True, laz_backend=laspy.LazBackend.Lazrs) as w:
            for ch in r.chunk_iterator(5_000_000): w.write_points(ch)
            if r.header.evlrs: w.write_evlrs(r.header.evlrs)
    with laspy.open(raw) as r1, laspy.open(out) as r2:
        h1, h2 = r1.header, r2.header
        f = lambda h, skip: dict(version=str(h.version), pf=h.point_format.id, n=h.point_count, scales=list(map(float, h.scales)), offsets=list(map(float, h.offsets)),
                                 vlrs=[(v.user_id, v.record_id) for v in h.vlrs if not (skip and v.record_id == 22204)], evlrs=len(h.evlrs or []))
        m1, m2 = f(h1, False), f(h2, True)
        if m1 != m2: return False, f'header diff {m1} {m2}', m1
        it2 = r2.chunk_iterator(5_000_000); n = 0
        for c1 in r1.chunk_iterator(5_000_000):
            c2 = next(it2)
            if c1.array.tobytes() != c2.array.tobytes(): return False, f'points diff near {n}', m1
            n += len(c1)
        if n != h1.point_count: return False, f'count {n}', m1
    return True, f'las n={n} pf={m1["pf"]} full-point-compare-ok', m1

def work(rel, raw):
    os.nice(0)
    t = time.time(); raw = Path(raw); dst = dest_of(rel); size = raw.stat().st_size
    h = sha256_file(raw)
    out = raw.with_name(raw.stem + '.out' + ('.laz' if rel.lower().endswith('.las') else '.tif'))
    method = 'raw'; note = ''
    try:
        ok, note, _ = (do_las if rel.lower().endswith('.las') else do_tif)(raw, out)
        if ok:
            ratio = size / out.stat().st_size
            if rel.lower().endswith('.las') or ratio >= KEEP_RAW_BELOW: method = 'laz' if rel.lower().endswith('.las') else 'zstd'
            else: note += f' ratio {ratio:.2f}<{KEEP_RAW_BELOW} -> kept raw'
        else: note = 'VERIFY FAIL -> kept raw: ' + note
    except Exception as e:
        note = 'COMPRESS ERROR -> kept raw: ' + repr(e)[:300]
    if rel.lower().endswith('.las') and method == 'raw': dst = DEST / rel   # keep .las name if not compressed
    if dst.exists():
        raw.unlink(missing_ok=True); out.unlink(missing_ok=True); return dict(rel=rel, skipped='exists at finish', sec=time.time() - t)
    dst.parent.mkdir(parents=True, exist_ok=True)
    src = out if method != 'raw' else raw
    os.replace(src, dst)
    raw.unlink(missing_ok=True); out.unlink(missing_ok=True)
    return dict(rel=rel, size=size, sha=h, method=method, note=note, csize=dst.stat().st_size, dst=str(dst), sec=time.time() - t)

def init_worker(): pass  # main already runs under nice -n 10

def main():
  for npass in range(3):
      log('START s3_dl pid', os.getpid(), 'nproc', NPROC, 'ndl', NDL, 'pass', npass)
      shutil.rmtree(TMP, ignore_errors=True); TMP.mkdir(parents=True, exist_ok=True)
      done = set()
      if MANP.exists():
          for l in MANP.read_text(encoding='utf-8').splitlines()[1:]: done.add(l.split('\t')[0])
      else:
          MANP.write_text('key\tsize\tsha256_raw\tverification\tmethod\tcomp_size\tdest\n', encoding='utf-8')
      objs = []
      for l in open(LIST, encoding='utf-8'):
          d = json.loads(l)
          if d.get('type') != 'directory': objs.append((d['key'][len(BUCKET):], int(d['size'])))
      ext = lambda r: Path(r).suffix.lower()
      # PC (os.walk sorted) order reversed; rasters first, then small/annotation, then LAS, then >10GB orthophotos
      tif = [o for o in objs if ext(o[0]) in ('.tif', '.tiff')][::-1]; las = [o for o in objs if ext(o[0]) == '.las'][::-1]
      oth = [o for o in objs if ext(o[0]) not in ('.tif', '.tiff', '.las')]
      order = [o for o in tif if o[1] <= BIG] + oth + las + sorted([o for o in tif if o[1] > BIG], key=lambda o: o[1])
      todo = [o for o in order if o[0] not in done and not dest_of(o[0]).exists() and not (DEST / o[0]).exists()]
      log('plan', len(objs), 'objects; todo', len(todo), f'{sum(s for _, s in todo)/1e9:.1f} GB raw')
      if not todo: break
      sem = threading.Semaphore(NPROC)
      def run_work(rel, raw):
          with sem:
              for att in range(3):
                  p = subprocess.run([sys.executable, __file__, '--work', rel, raw], capture_output=True, text=True)
                  if p.returncode == 0 and p.stdout.strip(): return json.loads(p.stdout.strip().splitlines()[-1])
                  log('WORKER died rc', p.returncode, 'att', att, rel, p.stderr[-300:].replace('\n', ' | ')); time.sleep(60)
                  for x in Path(raw).parent.glob(Path(raw).stem + '.out.*'): x.unlink(missing_ok=True)
              Path(raw).unlink(missing_ok=True); raise RuntimeError('worker failed 3x, raw deleted, will retry next pass')
      class F:
          def __init__(s, rel, raw): s.r = None; s.e = None; s.rel = rel; s.raw = raw
          def result(s):
              if s.e: raise s.e
              return s.r
      def submit(rel, raw, cb):
          def go():
              f = F(rel, raw)
              try: f.r = run_work(rel, raw)
              except Exception as e: f.e = e
              cb(f)
          th = threading.Thread(target=go); th.start(); wths.append(th)
      wths = []
      lock = threading.Lock(); st = dict(reserve=0.0, dl_bytes=0, c_raw=0, c_comp=0, n=0, t0=time.time(), inflight=0)
      futs = []; mlock = threading.Lock()
      def on_done(fu, rel, size):
          with lock: st['reserve'] -= size * 1.6; st['inflight'] -= 1
          try:
              r = fu.result()
          except Exception:
              log('WORKER ERROR', rel, traceback.format_exc()); return
          if 'skipped' in r: log('SKIP', rel, r['skipped']); return
          with mlock, open(MANP, 'a', encoding='utf-8') as f:
              f.write('\t'.join(map(str, [rel, r['size'], r['sha'], r['note'], r['method'], r['csize'], r['dst']])) + '\n')
          with lock: st['c_raw'] += r['size']; st['c_comp'] += r['csize']; st['n'] += 1
          log(f'OK {r["method"]} {size/1e6:.0f}MB -> {r["csize"]/1e6:.0f}MB x{size/max(r["csize"],1):.2f} {r["sec"]:.0f}s {rel} | {r["note"][:120]}')
      q = queue.Queue()
      for i, o in enumerate(todo): q.put((i, o))
      def dl_thread():
          while True:
              try: i, (rel, size) = q.get_nowait()
              except queue.Empty: return
              if dest_of(rel).exists() or (DEST / rel).exists(): log('SKIP exists', rel); continue
              need = size * 1.6
              warned = False
              while True:
                  with lock:
                      ok = free() - st['reserve'] - need >= MINFREE and (st['inflight'] < NPROC * 3 or size > 2e9 and st['inflight'] == 0)
                      if ok: st['reserve'] += need; st['inflight'] += 1
                  if ok: break
                  if free() - st['reserve'] - need < MINFREE and not warned:
                      log(f'PAUSE LOW DISK: free {free()/1e9:.1f} GB, reserved {st["reserve"]/1e9:.1f}, need {need/1e9:.1f} for {rel}'); warned = True
                  time.sleep(15)
              if warned: log('RESUME', rel)
              raw = TMP / f'{i:05d}{Path(rel).suffix.lower()}'
              t = time.time()
              for att in range(5):
                  p = subprocess.run([S5, '--endpoint-url', os.environ['S3EP'], 'cp', '--concurrency', '16', BUCKET + rel, str(raw)], capture_output=True, text=True)
                  if p.returncode == 0 and raw.exists() and raw.stat().st_size == size: break
                  log('DL retry', att, rel, p.stderr[-300:]); raw.unlink(missing_ok=True); time.sleep(30 * (att + 1))
              else:
                  log('DL FAILED', rel); 
                  with lock: st['reserve'] -= need; st['inflight'] -= 1
                  continue
              dt = time.time() - t
              with lock: st['dl_bytes'] += size
              log(f'DL {size/1e6:.0f}MB {dt:.0f}s {size/1e6/max(dt,1e-3):.1f} MB/s {rel}')
              submit(rel, str(raw), lambda f, rel=rel, size=size: on_done(f, rel, size))
      def stats():
          while True:
              time.sleep(60)
              with lock: s = dict(st)
              el = time.time() - s['t0']
              log(f'STAT {el/60:.0f}min dl {s["dl_bytes"]/1e9:.2f} GB avg {s["dl_bytes"]/1e6/el:.1f} MB/s | done {s["n"]} files {s["c_raw"]/1e9:.2f}->{s["c_comp"]/1e9:.2f} GB '
                  f'x{s["c_raw"]/max(s["c_comp"],1):.2f} | inflight {s["inflight"]} free {free()/1e9:.1f} GB queue {q.qsize()}')
      threading.Thread(target=stats, daemon=True).start()
      ths = [threading.Thread(target=dl_thread) for _ in range(NDL)]
      for th in ths: th.start()
      for th in ths: th.join()
      for th in list(wths): th.join()
      log('ALL DONE', st['n'], 'files', f'{st["c_raw"]/1e9:.2f}->{st["c_comp"]/1e9:.2f} GB')
if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--work':
        try: open('/proc/self/oom_score_adj', 'w').write('800')   # under memory pressure kill this worker, not other bots' jobs
        except Exception: pass
        print(json.dumps(work(sys.argv[2], sys.argv[3]))); sys.exit(0)
    try: main()
    except Exception: log('FATAL', traceback.format_exc())
