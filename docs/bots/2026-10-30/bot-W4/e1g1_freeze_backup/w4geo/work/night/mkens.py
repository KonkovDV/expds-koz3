"""mean OOF prob maps over probs dirs -> W/<out>/<reg>/<raster>.npz (+prep.pkl copied from first)"""
import sys, glob, os, shutil, numpy as np
W="/workspace/koz3/w4geo/work/"; out=sys.argv[1]; P=sys.argv[2:]
for fz in sorted(glob.glob(W+P[0]+"/*/*.npz")):
    rel=os.path.relpath(fz,W+P[0]); o=W+out+"/"+rel; os.makedirs(os.path.dirname(o),exist_ok=True)
    if os.path.exists(o): continue
    acc=None
    for p in P:
        z=np.load(W+p+"/"+rel); a=z["prob"].astype(np.float32); acc=a if acc is None else acc+a; v=z["valid"]
    np.savez_compressed(o,prob=(acc/len(P)).astype(np.float16),valid=v)
    shutil.copy(fz.replace(".npz",".prep.pkl"),o.replace(".npz",".prep.pkl")); print(rel,flush=True)
