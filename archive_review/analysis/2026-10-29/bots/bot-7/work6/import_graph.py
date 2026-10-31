"""Достижимые модули expds из inference/main.py (AST, включая ленивые импорты внутри функций)."""
import ast, sys
from pathlib import Path
R = Path(sys.argv[1])
mods = {p.stem: p for p in (R / "expds").glob("*.py")}
def deps(p):
    out = set()
    for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
        if isinstance(n, ast.ImportFrom):
            m = n.module or ""
            if n.level >= 1 and m == "":
                out |= {a.name for a in n.names}
            elif m.startswith("expds."): out.add(m.split(".")[1])
            elif n.level >= 1: out.add(m.split(".")[0])
            elif m == "expds": out |= {a.name for a in n.names}
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name.startswith("expds."): out.add(a.name.split(".")[1])
    return {d for d in out if d in mods}
seen, todo = set(), set(deps(R / "inference" / "main.py")) | {"__init__"}
while todo:
    m = todo.pop()
    if m in seen: continue
    seen.add(m); todo |= deps(mods[m]) - seen
print("reachable", len(seen), "of", len(mods))
print("unreachable:", sorted(set(mods) - seen))
