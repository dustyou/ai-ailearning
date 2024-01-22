import torch
from torch import nn

# # embedding
embedding = nn.Embedding(10, 3)
input = torch.LongTensor([[1,2,4,5,5],[4,3,2,9,8]])
print(embedding(input))
