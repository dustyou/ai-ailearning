import torch
import math

import torch.nn.functional as F
from torch.autograd import Variable

from transformer.positional_encoding import pe_result


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn

query = key = value = pe_result
attn, p_attn = attention(query, key, value)
print('attn:', attn)
print(attn.shape)
print('p_attn:', p_attn)
print(p_attn.shape)

mask = Variable(torch.zeros(2,4,4))
attn, p_attn = attention(query, key, value, mask=mask)
print('attn:', attn)
print(attn.shape)
print('p_attn:', p_attn)
print(p_attn.shape)