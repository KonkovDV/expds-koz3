"""RSS-сторож: python watch.py LOG -- cmd...; убивает дерево при RSS > 2000 MB; пишет пик RSS и время."""
import os, sys, time, subprocess, signal
log = sys.argv[1]; cmd = sys.argv[sys.argv.index('--') + 1:]
def tree_rss(pid):
    tot = 0; kids = {pid}
    try:
        for p in os.listdir('/proc'):
            if p.isdigit():
                try:
                    st = open(f'/proc/{p}/stat').read().split(')')[-1].split()
                    if int(st[1]) in kids or int(p) == pid: kids.add(int(p))
                except Exception: pass
        for k in kids:
            try:
                for l in open(f'/proc/{k}/status'):
                    if l.startswith('VmRSS'): tot += int(l.split()[1])
            except Exception: pass
    except Exception: pass
    return tot / 1024
t0 = time.time(); peak = 0
with open(log, 'w') as f:
    pr = subprocess.Popen(['nice', '-n', '10'] + cmd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    while pr.poll() is None:
        r = tree_rss(pr.pid); peak = max(peak, r)
        if r > 2000:
            os.killpg(pr.pid, signal.SIGKILL); f.write(f'\nWATCHDOG KILL rss {r:.0f} MB\n'); break
        time.sleep(0.5)
    pr.wait()
    f.write(f'\nWATCH rc {pr.returncode} elapsed_s {time.time()-t0:.1f} peak_rss_mb {peak:.0f}\n')
