"""Same image-blind sim as pooled_metric_sim.py, scored with OFFICIAL KOZ3 koz3-1.5.0 rules
(/workspace/expds-koz3/materials/compute_metrics.py, F1Calculator, read-only import) vs their KOZ2-style metric."""
import sys, importlib.util, io, contextlib
spec = importlib.util.spec_from_file_location("pm", "/workspace/koz3_shared/bot-3/mata_sanity/pooled_metric_sim.py")
with contextlib.redirect_stdout(io.StringIO()):
    pm = importlib.util.module_from_spec(spec); spec.loader.exec_module(pm)
mspec = importlib.util.spec_from_file_location("cm", "/workspace/expds-koz3/materials/compute_metrics.py")
cm = importlib.util.module_from_spec(mspec); mspec.loader.exec_module(cm)
def tag(fc):
    for f in fc["features"]: f["properties"]["region_name"] = "r"
    return fc
for name, shift in (("pooled", 0), ("per-patch", 1000)):
    G = tag(pm.fc(pm.gt(shift)))
    for pname, cls in (("prior-sampled class", pm.blind_cls), ("all povrezhdennye", ["kurgany_povrezhdennye"] * len(pm.v))):
        P = tag(pm.fc(pm.blind(shift, cls)))
        koz3 = cm.F1Calculator().calculate_f1(P, G)["final_score"]
        their = pm.competition_like_f1(P, G)[0]
        print(f"{name:10s} | {pname:22s} | their(KOZ2-style)={their:.4f} | official KOZ3={koz3:.4f}", flush=True)
