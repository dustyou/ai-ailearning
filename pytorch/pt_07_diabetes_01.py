import numpy as np
import torch
import matplotlib.pyplot as plt

# prepare dataset
xy = np.loadtxt('diabetes.csv', delimiter=',', dtype=np.float32, skiprows=1)
x_data = torch.from_numpy(xy[:, :-1])  # 第一个‘：’是指读取所有行，第二个‘：’是指从第一列开始，最后一列不要
print("input data.shape", x_data.shape)
y_data = torch.from_numpy(xy[:, [-1]])  # [-1] 最后得到的是个矩阵
epoch_list = []
cost_list = []
acc_list = []

# print(x_data.shape)
# design model using class

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 2)
        self.linear4 = torch.nn.Linear(2, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        x = self.sigmoid(self.linear4(x))
        return x

print('start')
model = Model()

# construct loss and optimizer
# criterion = torch.nn.BCELoss(size_average = True)
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# training cycle forward, backward, update
for epoch in range(100001):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
        plt.close('all')
        y_pred_label = torch.where(y_pred >= 0.5, torch.tensor([1.0]), torch.tensor([0.0]))

        acc = torch.eq(y_pred_label, y_data).sum().item() / y_data.size(0)
        print("loss = ", loss.item(), "acc = ", acc)
        epoch_list.append(epoch)
        cost_list.append(loss.item())
        acc_list.append(acc)
        plt.figure()
        plot = plt.plot(epoch_list, cost_list, label='cost')
        plot = plt.plot(epoch_list, acc_list, label='acc')
        plt.legend()

        plt.show()



plot = plt.plot(epoch_list, cost_list)
plt.ylabel('loss')
plt.xlabel('epoch')


print('end')
