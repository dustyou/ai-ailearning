import torch
from torch import nn

print('start')
embedding = nn.Embedding(10, 3)
input1 = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
print(embedding(input1))

embedding = nn.Embedding(10, 3, padding_idx=0)
input1 = torch.LongTensor([[0, 2, 3, 5]])
print(embedding(input1))