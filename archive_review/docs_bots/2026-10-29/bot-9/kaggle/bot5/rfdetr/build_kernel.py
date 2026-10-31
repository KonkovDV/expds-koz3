from pathlib import Path
import hashlib, json, ast
root = Path("/workspace/koz3_shared/bot-9/kaggle/bot5/rfdetr")
tar_sha = (root / "TAR_SHA.txt").read_text().split()[0]
a = (root / "tpl/main_a.py").read_text().replace(", base64", "").replace('SRC_B64 = "@@SRC_B64@@"\n', "")
start = a.find('SRC_DIR = TMP / "src"')
end = a.find('log("src OK"')
assert start > 0 and end > 0
