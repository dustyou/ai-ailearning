import matplotlib.pyplot as plt
import numpy as np
import torch

# a = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
# print(np.triu(a, k=0))
# print(np.triu(a, k=1))
# print(np.triu(a, k=-1))

# 定义一个下三角矩阵
def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k = 1).astype('uint8')
    return torch.from_numpy(1 - subsequent_mask)

size = 5
sm = subsequent_mask(size)
print('sm: ', sm)

plt.figure(figsize=(5,5))
plt.imshow(subsequent_mask(20)[0])
plt.show()

print(1e-9)
print(-1e9)