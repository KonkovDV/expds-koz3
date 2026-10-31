"""Копия фолдового каталога с весами fp16 (≈½ размера). Точность на CUDA не проверена; на CPU — compare в smoke."""
import sys, json, shutil
from pathlib import Path
from safetensors.torch import load_file, save_file
src, dst = Path(sys.argv[1]), Path(sys.argv[2]); dst.mkdir(parents=True, exist_ok=True)
sd = load_file(str(src / "model.safetensors"))
sd = {k: (v.half() if v.is_floating_point() else v).contiguous() for k, v in sd.items()}
save_file(sd, str(dst / "model.safetensors"), metadata={"format": "pt"}); shutil.copy(src / "config.json", dst / "config.json")
print(json.dumps({"src_B": (src / "model.safetensors").stat().st_size, "dst_B": (dst / "model.safetensors").stat().st_size, "tensors": len(sd)}))
