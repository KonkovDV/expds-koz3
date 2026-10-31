"""bot-7: патч v3on для ПК — члены нового zip, изменённые/добавленные относительно старого (по CRC+размеру),
и манифест всех членов нового zip (sha256, crc32, size). Usage: mkpatch_v3on.py OLD.zip NEW.zip OUTDIR [OLD_SHA_ON_PC]"""
import zipfile, hashlib, sys
from pathlib import Path
OLD, NEW, PC = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
so = sys.argv[4] if len(sys.argv) > 4 else sha(OLD); sn = sha(NEW); s8 = sn[:8]
zo, zn = zipfile.ZipFile(OLD), zipfile.ZipFile(NEW)
a = {i.filename: i for i in zo.infolist()}; b = {i.filename: i for i in zn.infolist()}
chg = [n for n in b if n in a and (a[n].CRC, a[n].file_size) != (b[n].CRC, b[n].file_size)]
add = [n for n in b if n not in a]; dele = [n for n in a if n not in b]
man = [f"# bot-7 manifest v3on {NEW.name}", f"# new_zip_sha256 {sn}", f"# old_zip (на ПК) sha256 {so}",
       f"# members {len(b)} changed {len(chg)} added {len(add)} deleted {len(dele)}",
       "# sha256<TAB>crc32<TAB>size<TAB>name (все члены нового zip)"]
for n, i in b.items():
    man.append(f"{hashlib.sha256(zn.read(n)).hexdigest()}\t{i.CRC:08x}\t{i.file_size}\t{n}")
mp = PC / f"manifest_v3on_{s8}.txt"; mp.write_text("\n".join(man) + "\n", encoding="utf-8")
pp = PC / f"patch_v3on_{s8}.zip"
with zipfile.ZipFile(pp, "w", zipfile.ZIP_DEFLATED) as z:
    for n in chg + add:
        z.writestr(b[n], zn.read(n), compress_type=zipfile.ZIP_DEFLATED)
    z.writestr("_b7patch/DELETE.txt", "".join(x + "\n" for x in dele))
    z.writestr("_b7patch/CHANGED.txt", "".join(f"{'CHG' if x in a else 'ADD'}\t{b[x].CRC:08x}\t{x}\n" for x in chg + add))
print(s8, "chg", chg, "add", len(add), "del", dele)
print(pp, pp.stat().st_size, sha(pp)); print(mp, mp.stat().st_size, sha(mp))
