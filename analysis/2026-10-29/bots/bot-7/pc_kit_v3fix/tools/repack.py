"""bot-7: перепаковка zip с другими метками времени, уровнем сжатия и порядком членов (эмуляция «чужого» zip на ПК)."""
import sys, zipfile
src, dst = zipfile.ZipFile(sys.argv[1]), sys.argv[2]
with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
    for i in reversed(src.infolist()):
        j = zipfile.ZipInfo(i.filename, date_time=(2026, 10, 29, 10, 0, 0)); j.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(j, src.read(i.filename))
