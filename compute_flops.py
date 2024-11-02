import torch

# from basic_CNN3 import Net
from DNet import *
from thop import profile
model = DNet()
input = torch.randn((3, 44, 257, 127))
flops, params = profile(model, inputs=(input, ))

# print(flops,params)
