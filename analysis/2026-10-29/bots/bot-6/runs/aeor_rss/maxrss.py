"""Замена /usr/bin/time -v (его нет на боксе, ставить системный пакет вне bot-6/ нельзя): запускает команду и
печатает ru_maxrss детей (то же поле ядра, что «Maximum resident set size» у GNU time) и время."""
import resource, subprocess, sys, time
t = time.monotonic()
rc = subprocess.call(sys.argv[1:])
ru = resource.getrusage(resource.RUSAGE_CHILDREN)
print(f"MAXRSS_KB={ru.ru_maxrss} elapsed_s={time.monotonic() - t:.1f} user_s={ru.ru_utime:.1f} sys_s={ru.ru_stime:.1f} rc={rc}", flush=True)
sys.exit(rc)
