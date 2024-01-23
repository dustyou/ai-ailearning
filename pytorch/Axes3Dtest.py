import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('TkAgg')



# 创建一个新的图形
fig = plt.figure()

# 创建一个三维坐标轴
ax = Axes3D(fig)

# 定义X, Y
X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)

# 将X和Y转换为二维网格
X, Y = np.meshgrid(X, Y)

# 定义Z
Z = np.sin(np.sqrt(X**2 + Y**2))
path = 'D:/test/test.png'
# 绘制三维曲面
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot')
plt.show()



