# bot-8 lidar_a (чистая ветка bot8-cpu-a-clean @95bc0a6, git archive, только чтение) на том же LAS: DTM 0.25 + relief_stack(extra=True) на CPU.
import sys, json, time, numpy as np
sys.path.insert(0, "b8_95bc0a6")
from expds.lidar_a import ground, terrain
t = time.time()
dtm, valid, origin = ground.make_dtm(sys.argv[1])
st, inside = terrain.relief_stack(dtm, valid, origin[2], extra=True, unclip=False, dev="cpu")
np.savez_compressed("theirs.npz", stack=st, inside=inside, origin=np.array(origin), dtm_shape=np.array(dtm.shape))
print(json.dumps(dict(origin=origin, dtm_shape=dtm.shape, stack_shape=st.shape, mode=ground.ground_mode(sys.argv[1]), t=round(time.time()-t,1))))
