"""bot-7: патч v3fix для ПК (на основе work6/on/mkpatch_v3on.py) — члены нового zip, изменённые/добавленные
относительно старого (по CRC+размеру), список удалённых и манифест всех членов нового zip (sha256, crc32, size).
Старый zip на ПК побайтно другой (собран ZipArchive), поэтому опора — манифест старого zip по членам.
Usage: mkpatch_v3fix.py OLD.zip NEW.zip OUTDIR OLD_MANIFEST"""
import zipfile, hashlib, sys
from pathlib import Path
OLD, NEW, PC, OMAN = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
so, sm, sn = sha(OLD), sha(OMAN), sha(NEW); s8 = sn[:8]
zo, zn = zipfile.ZipFile(OLD), zipfile.ZipFile(NEW)
a = {i.filename: i for i in zo.infolist()}; b = {i.filename: i for i in zn.infolist()}
# старый zip должен совпадать со старым манифестом по членам
om = {}
for line in OMAN.read_text(encoding="utf-8").splitlines():
    if line and not line.startswith("#"):
        h, c, s, n = line.split("\t", 3); om[n] = (h, int(c, 16), int(s))
assert set(om) == set(a), "старый zip != старый манифест (имена)"
for n, i in a.items():
    assert (om[n][1], om[n][2]) == (i.CRC, i.file_size), n
    assert hashlib.sha256(zo.read(n)).hexdigest() == om[n][0], n
chg = [n for n in b if n in a and (a[n].CRC, a[n].file_size) != (b[n].CRC, b[n].file_size)]
add = [n for n in b if n not in a]; dele = [n for n in a if n not in b]
man = [f"# bot-7 manifest v3fix {NEW.name}", f"# new_zip_sha256 {sn}",
       f"# old_zip bot-7 sha256 {so} (на ПК другой — проверка по членам: {OMAN.name} sha256 {sm})",
       f"# members {len(b)} changed {len(chg)} added {len(add)} deleted {len(dele)}",
       "# sha256<TAB>crc32<TAB>size<TAB>name (все члены нового zip)"]
for n, i in b.items():
    man.append(f"{hashlib.sha256(zn.read(n)).hexdigest()}\t{i.CRC:08x}\t{i.file_size}\t{n}")
mp = PC / f"manifest_v3fix_{s8}.txt"; mp.write_text("\n".join(man) + "\n", encoding="utf-8")
pp = PC / f"patch_v3fix_{s8}.zip"
with zipfile.ZipFile(pp, "w", zipfile.ZIP_DEFLATED) as z:
    for n in chg + add:
        z.writestr(b[n], zn.read(n), compress_type=zipfile.ZIP_DEFLATED)
    z.writestr("_b7patch/DELETE.txt", "".join(x + "\n" for x in dele))
    z.writestr("_b7patch/CHANGED.txt", "".join(f"{'CHG' if x in a else 'ADD'}\t{b[x].CRC:08x}\t{x}\n" for x in chg + add))
print(s8, "chg", chg, "add", add, "del", dele)
print(pp, pp.stat().st_size, sha(pp)); print(mp, mp.stat().st_size, sha(mp))
