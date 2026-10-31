import sys; from pathlib import Path
sys.path.insert(0, sys.argv[1])
from expds import sp_dfine as S
p = Path(sys.argv[2]); sc = S.Scene(p, S.utm_for(p, None), 0.5)
try:
    a = sc.read(0, 0, 1024); print("PLAIN_SCENE_OK", a.shape)
except Exception as e:
    print("PLAIN_SCENE_FAIL", type(e).__name__, str(e)[:300])
