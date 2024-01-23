import tensorflow as tf
from tensorflow.keras.datasets import mnist

# 加载MNIST数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 显示一张图片
import matplotlib.pyplot as plt

plt.imshow(train_images[0], cmap=plt.cm.gray)
plt.show()