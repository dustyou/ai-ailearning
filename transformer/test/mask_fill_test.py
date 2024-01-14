import torch
from torch.autograd import Variable

x = Variable(torch.randn(5,5))
print(x)
mask = Variable(torch.zeros(5,5))
print(mask)
y = x.masked_fill(mask==0, 1e-9)
print(y)

import torch

x = torch.randn(5, 5)
mask = torch.zeros(5, 5)
y = x.clone()  # 创建一个x的副本以避免修改原始数据

y[mask == 0] = 1e-9  # 使用布尔索引来替换值

print(x)
print(mask)
print(y)