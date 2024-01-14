import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import matplotlib.pyplot as plt
import numpy as np
import copy


# 定义一个名为 "Embeddings" 的类，该类继承自 "torch.nn.Module"。
# 在 PyTorch 中，自定义的神经网络模块需要继承自 "torch.nn.Module"。
class Embeddings(torch.nn.Module):
    # 初始化方法，当创建 "Embeddings" 类的新实例时会被调用。
    def __init__(self, d_model, vocab):
        # 调用父类 "torch.nn.Module" 的初始化方法。这是必需的，以确保实例可以正确地注册子模块等。
        super(Embeddings, self).__init__()
        # 创建一个嵌入层，将词汇表中的每个单词映射到一个维度为 "d_model" 的向量。
        # "vocab" 是词汇表的大小，即词汇表中的单词数量。
        self.lut = nn.Embedding(vocab, d_model)  # 创建一个嵌入层
        # 保存嵌入向量的维度 "d_model"，以便在后续的计算中使用。
        self.d_model = d_model

        # 定义前向传播的方法。当对输入数据进行计算时，这个方法会被调用。

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)  # 对输入的整数序列进行前向传播，得到嵌入向量序列





# 词嵌入微度是512维
d_model = 512
# 词表大小是1000
vocab = 1000
# 输入x是一个使用Variable封装的长整形张量, 形状是2x4
x = Variable(torch.LongTensor([[100,2,421,508], [491,998,1,221]]))

emb = Embeddings(d_model, vocab)
embr = emb(x)
print('embr: ', embr)
print('embr: ', embr.shape)