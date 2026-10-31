"""Run a command; kill its process tree if total RSS > limit MiB. usage: rss_guard.py LIMIT_MB cmd..."""
import os, signal, subprocess, sys, time
lim = float(sys.argv[1]); p = subprocess.Popen(sys.argv[2:], start_new_session=True)
peak = 0.0
def tree_rss(root):
    kids, tot = {}, 0
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                st = open(f"/proc/{d}/stat").read().rsplit(")", 1)[1].split()
                kids.setdefault(int(st[1]), []).append((int(d), int(st[21])))
            except (OSError, IndexError, ValueError):
                pass
    stack, seen = [root], set()
    try: tot += int(open(f"/proc/{root}/stat").read().rsplit(")", 1)[1].split()[21])
    except Exception: pass
    while stack:
        for c, r in kids.get(stack.pop(), []):
            if c not in seen: seen.add(c); tot += r; stack.append(c)
    return tot * os.sysconf("SC_PAGE_SIZE") / 2**20
while p.poll() is None:
    r = tree_rss(p.pid); peak = max(peak, r)
    if r > lim:
        sys.stderr.write(f"RSS guard: {r:.0f} MiB > {lim:.0f}, killing\n"); os.killpg(p.pid, signal.SIGKILL); p.wait(); sys.exit(3)
    time.sleep(0.2)
sys.stderr.write(f"RSS guard: peak {peak:.0f} MiB, exit {p.returncode}\n"); sys.exit(p.returncode)
