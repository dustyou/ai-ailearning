import torch
from torch import nn

# 01
# randn
# x = torch.randn(4,4)
# print(x.size())
# y = x.view(2,2,2,2)
# print(y.size())
# a = torch.randn(1,2,3,4)
# print(a.size(), a)
# b = a.transpose(1,2)
# print(b.size(), b)
# c = a.view(1,3,2,4)
# print(c.size(), c)

# # embedding
# embedding = nn.Embedding(10, 3)
# input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
# print(embedding(input))
# print(embedding.size())


# nn.Dropout演示
m=nn.Dropout(p=0.2)
input = torch.randn(4,5)
output=m(input)
print(output)



# torch.unsqueeze

x= torch.tensor([1,2,3,4])
print(x.shape)
y= torch.unsqueeze(x,0)
print(y.shape, y)
z =torch.unsqueeze(x,1)
print(z.shape, z)