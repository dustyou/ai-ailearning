import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable

from transformer.positional_encoding import PositionalEncoding
#创建一张15×5大小的画布
plt.figure(figsize=(15,5))
#实例化PositionalEncoding类得到pe对象，输入参数是2g和g
pe = PositionalEncoding(20,0.05)
# 然后向pe传入被Variable封装的tensor,这样pe会直接执行forward函数，
# 且这个tensor里的数值都是0，被处理后相当于位置编码张量
y = pe(Variable(torch.zeros(1,100,20)))
plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
#然后定义画布的横纵坐标，横坐标到100的长度，纵坐标是某一个词汇中的某维特征在不同长度下对应的值
# 因为总共有20维之多，我们这里只查看4,5,6,7维的值.
plt.legend(["dim %d"% p for p in [4,5,6,7]])
plt.show()
