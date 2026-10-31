"""bot-7: эмуляция apply_patch_v3fix.ps1 на питоне. Usage: emulate_apply.py OLD.zip OLD_MANIFEST PATCH.zip NEW_MANIFEST OUT.zip
Проверяет входной zip по членам (старый манифест), применяет патч, проверяет результат по новому манифесту."""
import sys, zipfile, hashlib
OLD, OMAN, PATCH, NMAN, OUT = sys.argv[1:6]
E = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
def man(p):
    d = {}
    for l in open(p, encoding="utf-8").read().splitlines():
        if l and not l.startswith("#"):
            h, c, s, n = l.split("\t", 3); d[n] = (h, c, int(s))
    return d
def check(zp, want):
    z = zipfile.ZipFile(zp); bad = 0; seen = set()
    for i in z.infolist():
        n = i.filename
        if n in seen or n not in want: bad += 1; print("лишний/дубль", n); continue
        seen.add(n); h, c, s = want[n]
        got = E if i.file_size == 0 else hashlib.sha256(z.read(n)).hexdigest()
        if (i.file_size, f"{i.CRC:08x}", got) != (s, c, h): bad += 1; print("несовпадение", n)
    bad += len(set(want) - seen)
    return bad, len(seen)
wo, wn = man(OMAN), man(NMAN)
print("вход:", check(OLD, wo))
pz = zipfile.ZipFile(PATCH)
put = [i for i in pz.infolist() if not i.filename.startswith("_b7patch/") and not i.filename.endswith("/")]
dele = [l.strip() for l in pz.read("_b7patch/DELETE.txt").decode().splitlines() if l.strip()]
names = {i.filename for i in put} | set(dele)
zi = zipfile.ZipFile(OLD)
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zo:
    for i in zi.infolist():
        if i.filename not in names: zo.writestr(i, zi.read(i.filename))
    for i in put: zo.writestr(i, pz.read(i.filename), compress_type=zipfile.ZIP_DEFLATED)
print("патч: записано", len(put), "удалено", len(dele))
r = check(OUT, wn); print("результат:", r, "PASS" if r[0] == 0 and r[1] == len(wn) else "FAIL")
