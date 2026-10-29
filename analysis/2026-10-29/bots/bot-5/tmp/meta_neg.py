import sys,torch;sys.path.insert(0,'repo/scripts')
from det_train_hf import assert_no_meta
m=torch.nn.Linear(2,2); assert_no_meta(m,'ok'); print('real: ok')
with torch.device('meta'): mm=torch.nn.Linear(2,2)
try: assert_no_meta(mm,'neg')
except RuntimeError as e: print('meta: raised:',e)
