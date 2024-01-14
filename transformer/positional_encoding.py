# P6
# ·位置编码器的作用：
# ·因为在Transformer的编码器结构中，并没有针对调汇位置信息的处理，因此需要在
# Embedding层后加入位置编码器，将词汇位置不同可能会产生不同语义的信息加入到词嵌
# 入张量中，以弥补位置信息的缺失

import math

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.tensorboard import _embedding

from transformer.embedding import Embeddings


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        # d_model:f代表词嵌入的维度
        # dropout:代Dropout层的置零比事
        # max_len:代表每个句子的最大长度
        super(PositionalEncoding, self).__init__()
        # 实例化Dropout层
        self.dropout = nn.Dropout(p=dropout)
        # 初始化一个位置编码炬阵，大小是nax_len*d_model
        pe = torch.zeros(max_len, d_model)
        # 切始化一个绝对位置矩阵，max_len*1
        position = torch.arange(0, max_len).unsqueeze(1)
        # 定义一个变化矩阵d1 vterm,跳跃式的切始化
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        # 将前而定义的变化矩阵进行奇数，得数的分属值
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 将二维张量扩充成三维张量
        pe = pe.unsqueeze(0)
        # 将位置编码矩阵注册成模型的处uffr,这个buffer不是模型中的参数，不盟随优化器同
        # 注册成buff「后我财可以在横型保存后重新加覆的时候，将这个位置综码图和模型的数
        self.register_buffer('pe', pe)

    def forward(self, x):
        # X:代表文本序列的词波入表示
        # 首先明确p的编码太长了，将第二个维度，也就是maX1n对应的那个维度缩小2x的句
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


pe = PositionalEncoding(512, 0.1, max_len=60)

# 词嵌入微度是512维
d_model = 512
dropout = 0.1
max_len = 60
# 词表大小是1000
vocab = 1000
# 输入x是一个使用Variable封装的长整形张量, 形状是2x4
x = Variable(torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]]))

emb = Embeddings(d_model, vocab)
embr = emb(x)

pe = PositionalEncoding(d_model, dropout)
pe_result = pe(embr)
print("pe_result:", pe_result)
print('pe_result.shape:', pe_result.shape)
