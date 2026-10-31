import torch, sys, glob, json, hashlib
for f in sorted(glob.glob('models/*.pt')):
    d = torch.load(f, map_location='cpu', weights_only=False)
    info = {}
    if isinstance(d, dict):
        info['keys'] = [k for k in d.keys()][:12]
        for k, v in d.items():
            if not hasattr(v, 'shape') and k not in ('state_dict', 'model', 'model_state', 'weights'):
                s = repr(v); info[k] = s[:300]
        sd = d.get('state_dict') or d.get('model') or d.get('model_state') or d.get('weights') or {}
        if isinstance(sd, dict):
            ks = list(sd.keys()); info['n_tensors'] = len(ks); info['first'] = ks[:3]; info['last'] = ks[-3:]
    else:
        info['type'] = type(d).__name__
    print(f, json.dumps(info, ensure_ascii=False, default=str)[:1500]); sys.stdout.flush()
    del d
