import torch
import matplotlib.pyplot as plt

x = torch.tensor([[1.0], [2.0], [3.0]])
y = torch.tensor([[2.0], [4.0], [6.0]])

class Linear2(torch.nn.Module):
    def __init__(self):
        super(Linear2, self).__init__()
        self.linear = torch.nn.Linear(1,1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


model = Linear2()
criterion = torch.nn.MSELoss(reduction='sum')
print(str(model.parameters()))
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

for epoch in range(101):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 5 == 0:
        print(epoch, loss.item())
x_test = torch.tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)




