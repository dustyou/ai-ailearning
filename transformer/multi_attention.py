import copy  # 导入copy模块，用于深度复制对象。
import math

import torch
from torch import nn  # 从torch库中导入神经网络模块nn。
from torch.autograd import Variable
from torch.autograd.grad_mode import F

# 导入 PyTorch 框架，这是用于深度学习的流行框架
import torch


# 定义一个名为 "Embeddings" 的类，该类继承自 "torch.nn.Module"。
# 在 PyTorch 中，自定义的神经网络模块需要继承自 "torch.nn.Module"。


def clones(module, N):  # 定义一个名为clones的函数，用于复制一个模块N次。
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.trasnpos(-2, -1) / math.sqrt(d_k))
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


# 这个代码实现了一个多头注意力机制的类，其中使用了四个线性层（self.linears），这些线性层可能用于实现查询、键、值和输出变换。在前向传播方法中，对输入的查询、键、值向量进行了变换和整形，然后调用了attention函数进行多头注意力计算，最后对输出向量进行了转置和整形操作。
class MultiHeadedAttention(nn.Module):  # 定义一个名为MultiHeadedAttention的类，继承自nn.Module。
    def __init__(self, head, embedding_dim, dropout=0.1):  # 初始化方法，用于设置模型的参数。
        super(MultiHeadedAttention, self).__init__()  # 调用父类的初始化方法。
        assert embedding_dim % head == 0, "Embedding dimension must be divisible by number of heads."  # 断言：嵌入维度必须能被头数整除。
        self.d_k = embedding_dim // head  # 计算每个头的维度（每个头的特征数）。
        self.head = head  # 记录头数。
        self.embedding_dim = embedding_dim  # 记录嵌入向量的维度。
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)  # 使用clones函数复制线性层4次，创建4个线性层。
        self.attn = None  # 初始化注意力权重为None。
        self.dropout = nn.Dropout(p=dropout)  # 创建dropout层，用于防止过拟合。

    def forward(self, query, key, value, mask=None):  # 定义前向传播方法。
        if mask is not None:  # 如果提供了遮盖（mask）：
            mask = mask.unsqueeze(1)  # 扩展mask的维度。
        batch_size = query.size(0)  # 获取查询向量的第一个维度的大小，即批次大小。
        query, key, value = \
            [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
             for model, x in zip(self.linears, (query, key, value))]  # 对查询、键、值向量进行变换和整形。
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)  # 调用attention函数计算输出向量和注意力权重。
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)  # 对输出向量进行转置、连接和整形。
        return self.linears[-1](x)  # 使用最后一个线性层对输出向量进行变换，并返回结果。

#
# # 实例化参数
# head = 8
# embedding_dim = 512
# dropout = 0
#
# # 假设我们要创建一个形状为 [batch_size, sequence_length, embedding_dim] 的张量
# batch_size = 8
# sequence_length = 4
#
# pe_result = torch.randn(batch_size, sequence_length, embedding_dim)  # 使用随机数初始化
# # 若干输入参数的初始化
# query = key = value = pe_result
#
# mask = Variable(torch.zeros(8, 4, 4))
# # multiheadattention
# mha = MultiHeadedAttention(head, embedding_dim, dropout)
# mha(query, key, value, mask=mask)
# # print(mha_result)
# # print(mha_result.shape)
#
#












