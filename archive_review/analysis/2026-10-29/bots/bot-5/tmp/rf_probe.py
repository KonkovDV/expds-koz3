import os, time, resource, logging
logging.basicConfig(level=logging.WARNING)
import torch; torch.set_num_threads(2)
from rfdetr.config import RFDETRMediumConfig, TrainConfig
from rfdetr.models.lwdetr import build_model_from_config
mc = RFDETRMediumConfig(pretrain_weights=None, num_classes=5, device='cpu')
tc = TrainConfig(dataset_dir='.', output_dir='.')
m = build_model_from_config(mc, tc)
enc = m.backbone[0].encoder.encoder
print(type(enc).__name__, sum(p.numel() for p in m.parameters())/1e6, 'M')
sd = enc.state_dict(); ks = list(sd)
print(len(ks)); [print(k, tuple(sd[k].shape)) for k in ks[:8]+ks[-4:]]
from huggingface_hub import snapshot_download
p = snapshot_download('facebook/dinov2-small', allow_patterns=['*.json','*.safetensors','README.md','LICENSE*'])
print(p, os.listdir(p))
from safetensors.torch import load_file
hf = load_file(os.path.join(p, 'model.safetensors'))
hk = list(hf); print(len(hk)); [print(k, tuple(hf[k].shape)) for k in hk[:8]+hk[-4:]]
print('missing in hf', [k for k in ks if k not in hf][:20]); print('extra in hf', [k for k in hk if k not in sd][:20])
print('rss', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
