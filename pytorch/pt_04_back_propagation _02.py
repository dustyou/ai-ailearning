import torch
from matplotlib import pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


w = torch.tensor([1.0])  # w的初值为1.0
w.requires_grad = True  # 需要计算梯度


def forward(x):
    return x * w  # w是一个Tensor


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

epoch_list = []
cost_list = []
print("predict (before training)", 4, forward(4).item())

for epoch in range(10):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)  # l是一个张量，tensor主要是在建立计算图 forward, compute the loss
        l.backward()  # backward,compute grad for Tensor whose requires_grad set to True
        print('\tx:{}, y:{}, grad:{}, loss: {}'.format(x, y, w.grad.item(), l.item()))
        print('\tw: {}'.format(w.data))
        w.data = w.data - 0.01 * w.grad.data  # 权重更新时，注意grad也是一个tensor
        w.grad.data.zero_()  # after update, remember set the grad to zero

    print('progress:{}, loss: {}'.format(epoch, l.item()))  # 取出loss使用l.item，不要直接使用l（l是tensor会构建计算图）
    epoch_list.append(epoch)
    cost_list.append(l.item())
print("predict (after training)", 4, forward(4).item())

plt.plot(epoch_list, cost_list)
plt.ylabel('cost')
plt.xlabel('epoch')
plt.show()