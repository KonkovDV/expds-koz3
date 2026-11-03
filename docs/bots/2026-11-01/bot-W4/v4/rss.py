import resource, subprocess, sys, time
t = time.time(); r = subprocess.run(sys.argv[1:]); dt = time.time() - t
ru = resource.getrusage(resource.RUSAGE_CHILDREN)
print(f"exit {r.returncode} wall_s {dt:.1f} peak_rss_mb {ru.ru_maxrss/1024:.0f} user_s {ru.ru_utime:.0f} sys_s {ru.ru_stime:.0f}", file=sys.stderr)
