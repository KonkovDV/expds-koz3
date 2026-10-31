import sys,copy,collections; sys.path.insert(0,"/tmp/bot8_half_src")
import torch; torch.set_num_threads(2)
from pathlib import Path
from expds import pipeline, candidates
from expds.lidar import build_dem, hillshade_set
from expds.infer_nn import Segmenter
seg=Segmenter("/tmp/bot8_half_src/models/li_r18_hard2.pt",device="cpu")
cfg=pipeline.load_config(Path("/tmp/bot8_half_src/configs/default.toml"))
cp=pipeline._params(candidates.CandidateParams,cfg.get("candidates"))
d=build_dem(Path("/workspace/koz3_shared/speedup/sample/016_Хога/01_Хога_La_точки/01_Хога_La_точки.laz"),res=0.1,fill="delaunay",crs="EPSG:32632")
for res in (0.5,1.0):
  s=copy.copy(seg); s.res=res
  ff=pipeline._li_with_model(s,d,"016","x",hillshade_set,cp,cfg.get("li"),pipeline.output_floors(cfg,"li"))
  kept=pipeline._kept(s,ff)
  print(res,len(ff),len(kept),collections.Counter(f["class_name"] for f in ff),collections.Counter(f["class_name"] for f in kept), [round(f["confidence"],3) for f in ff if f["class_name"].startswith("kurg")][:8])
