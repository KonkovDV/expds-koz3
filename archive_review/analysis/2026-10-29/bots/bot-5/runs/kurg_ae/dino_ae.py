"""bot-5. Вариант 2 (ТОЛЬКО замер, предобученные веса — не для сдачи до письма организаторов):
frozen DINOv2-S на 64-м виньетках (64x64 px при 1 м, нормированная яркость из extract_ae.py) -> 112 px,
выход CLS ⊕ mean(patch) = 768. Порядок строк = конкатенация wins по sorted(feat/*.npz). Метки не используются."""
import glob, sys, time, numpy as np, torch
import torch.nn.functional as Fn
torch.set_num_threads(2)
MD = glob.glob('/workspace/koz3_shared/bot-5/tmp/hf/hub/models--facebook--dinov2-small/snapshots/*/')[0]
sys.path.insert(0, '/workspace/koz3_shared/bot-5/repo/scripts')
from transformers import Dinov2Model
from det_train_hf import assert_no_meta
m = Dinov2Model.from_pretrained(MD).eval(); assert_no_meta(m, 'dinov2-small')
MEAN = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]; STD = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
import os
OUT = os.environ.get('DINO_FILE', '/workspace/koz3_shared/bot-5/analysis/kurg_ae/dino_feats.npy')
files = [f for d in os.environ.get('FEAT_DIRS', '/workspace/koz3_shared/bot-5/tmp/kurg_ae/feat').split(',') for f in sorted(glob.glob(d + '/*.npz'))]
n = sum(np.load(f)['wins'].shape[0] for f in files); print('windows', n, flush=True)
F = np.lib.format.open_memmap(OUT + '.tmp.npy', mode='w+', dtype=np.float16, shape=(n, 768)); k = 0; t0 = time.time()
ONLY_STACK_DIR = os.environ.get('ONLY_STACK_DIR')  # в этом каталоге считаем только виньетки строк стека (src1), прочие строки = 0
F[:] = 0
for f in files:
    z = np.load(f); W = z['wins']; need = np.arange(len(W))
    if ONLY_STACK_DIR and f.startswith(ONLY_STACK_DIR):
        need = np.flatnonzero(z['rec'][z['widx'], 2] == 1) if len(W) else need
    base = k
    for s in range(0, len(need), 64):
        nn = need[s:s + 64]
        x = torch.from_numpy(W[nn]).float()[:, None] / 255.0
        x = Fn.interpolate(x, size=(112, 112), mode='bilinear', align_corners=False).repeat(1, 3, 1, 1)
        x = (x - MEAN) / STD
        with torch.no_grad():
            o = m(pixel_values=x).last_hidden_state
        F[base + nn] = torch.cat([o[:, 0], o[:, 1:].mean(1)], 1).numpy().astype(np.float16)
    k = base + len(W)
    print(f.split('/')[-1], k, f'{time.time()-t0:.0f}s', flush=True)
F.flush(); del F
import os; os.replace(OUT + '.tmp.npy', OUT); print('DONE', k)
