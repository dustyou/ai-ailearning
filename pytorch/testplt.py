import matplotlib.pyplot as plt
import numpy as np

# 创建一些数据
x = np.linspace(0, 10, 100)  # 生成0到10之间的100个等间距的点
y = np.sin(x)  # 对每个x值计算对应的sin值

# 创建一个新的图形
fig = plt.figure()

# 在图形上画线
plt.plot(x, y)

# 设置标题和轴标签
plt.title("Simple Line Plot")
plt.xlabel("X Axis")
plt.ylabel("Y Axis")

# 显示图形
plt.show()
x = np.linspace(0, 20, 100)  # 生成0到10之间的100个等间距的点
y = np.sin(x)  # 对每个x值计算对应的sin值
plt.plot(x, y)  # 在新图形上画线
fig.clear()
plt.show()
