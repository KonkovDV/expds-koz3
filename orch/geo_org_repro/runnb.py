"""Run selected code cells of an organizer notebook headless; print last-expression values."""
import ast, json, sys, time, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: plt.close("all")
def run(nb, cells, g):
    cs = json.load(open(nb))["cells"]
    for i in cells:
        src = "".join(cs[i]["source"]); t = time.time()
        tree = ast.parse(src); last = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last = ast.Expression(tree.body.pop().value)
        exec(compile(tree, f"cell{i}", "exec"), g)
        if last is not None:
            v = eval(compile(last, f"cell{i}", "eval"), g)
            if v is not None:
                try: print(v.to_string())
                except Exception: print(v)
        print(f"[cell {i} done {time.time()-t:.0f}s]", flush=True)
