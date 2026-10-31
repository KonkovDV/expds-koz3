"""bot-6 ORGATE_S: замер мс/кроп как в bot-4/work/cropverif/bench.py (2 потока, батч 32, вход 112 px), S и B/reg подряд при одной нагрузке."""
import time, json, torch, resource, os
torch.set_num_threads(2)
from transformers import AutoModel
res = {}
for name, path in (('dinov2-with-registers-small', 'dinov2s_reg'),):  # B не грузится при prlimit 2 ГБ (mmap ENOMEM, bench_s_1.log)
    m = AutoModel.from_pretrained(path, local_files_only=True).eval(); x = torch.randn(32, 3, 112, 112)
    with torch.inference_mode():
        m(pixel_values=x[:4]); t = time.perf_counter(); k = 0
        while k < 256: m(pixel_values=x); k += 32
    res[name + '@112'] = round((time.perf_counter() - t) / k * 1000, 2); print(name, res[name + '@112'], 'ms/crop', flush=True)
res['maxrss_mb'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024; res['threads'] = torch.get_num_threads(); res['torch'] = torch.__version__
res['loadavg'] = os.getloadavg(); json.dump(res, open('bench_s.json', 'w'), indent=1); print(res)
