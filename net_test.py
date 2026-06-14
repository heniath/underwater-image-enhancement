from thop import profile
import torch
import time
from net import UNet5ch


model = UNet5ch(in_channels=5).to('cuda')
input = torch.rand(1, 5, 256, 256).to('cuda')
torch.cuda.synchronize()
model.eval()
time_start = time.time()
_ = model(input)
time_end = time.time()
torch.cuda.synchronize()
time_sum = time_end - time_start
print(f"Time: {time_sum}")
n_param = sum([p.nelement() for p in model.parameters()])
n_paras = f"n_paras: {(n_param / 2**20):.3f}M"
print(n_paras)
macs, params = profile(model, inputs=(input,))
print(f'FLOPs: {macs / (2**30):.3f}G')
