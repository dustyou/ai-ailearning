6.2.2输入部分实现-part4

# 输入部分实现

## 基础方法介绍

代码链接:  [embedding_test.py](..\..\transformer\test\embedding_test.py) 

###  embedding

Embedding是一种将离散数据映射到连续向量空间中的技术，主要用于深度学习模型中。通过将离散数据映射到连续向量空间，可以使模型更好地处理这些数据，并提高模型的性能和准确性。

在自然语言处理领域中，embedding被广泛应用于文本分类、情感分析、问答系统等任务中。下面是一个简单的文本分类任务的embedding用法示例：

假设我们要对一组文本进行分类，可以使用Word2Vec模型对文本中的单词进行嵌入表示，即将每个单词映射到一个固定大小的向量。然后，将这些向量输入到一个神经网络模型中，进行分类预测。

具体步骤如下：

1. 数据预处理：对文本数据进行预处理，包括去除停用词、分词等操作，得到每个单词的表示。
2. 训练Word2Vec模型：使用Word2Vec模型对文本中的单词进行训练，得到每个单词的嵌入向量。
3. 构建神经网络模型：使用一个神经网络模型（如卷积神经网络或循环神经网络）来处理嵌入向量，并进行分类预测。
4. 训练模型：使用训练数据对神经网络模型进行训练，优化模型的参数。
5. 测试和评估：使用测试数据对模型进行测试和评估，计算模型的准确率、召回率等指标。

除了Word2Vec模型外，还有许多其他的embedding方法，如GloVe、FastText等。这些方法都可以将离散数据映射到连续向量空间中，从而使得深度学习模型可以更好地处理这些数据。

一个简单的映射例子, onehot

One-hot编码，又称“独热编码”，是一种特殊的编码方式。在机器学习算法中，我们经常会遇到离散化的特征或标签。对于这些离散化的标签，如果仅仅对原始的离散标签进行编码，那么可能无法很好地利用这些标签的信息。因此，我们通常会将这些离散的标签转换为一种能够更好地表示它们之间关系的编码方式，这种编码方式就是one-hot编码。

具体来说，one-hot编码就是用N位状态寄存器来对N个状态进行编码，每个状态都由其独立的寄存器位，并且在任意时候只有一位有效。也就是说，对于N个不同的状态，我们使用N个二进制位来表示它们，其中只有一个二进制位为1，其余位都为0。通过这种方式，我们可以将离散的标签转换为一种连续的向量表示，从而更好地利用这些标签的信息。

One-hot编码的主要优点是可以将离散的标签转换为连续的向量表示，从而使得机器学习算法能够更好地处理这些标签。此外，one-hot编码还可以避免一些由于标签不均衡而导致的分类问题。但是，one-hot编码也存在一些缺点，例如它可能会占用大量的存储空间和计算资源，并且可能会引入一些噪声和冗余信息。因此，在实际应用中，我们需要根据具体情况选择是否使用one-hot编码。



```python
import torch
from torch import nn

embedding = nn.Embedding(10, 3)
input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
print(embedding(input))
print(embedding.size())
```

在PyTorch的`nn.Embedding`模块中，第一个参数表示输入数据的最大索引值加1。也就是说，如果你有10个不同的输入，那么第一个参数应该是10。第二个参数代表输出的维度。这个维度决定了每个嵌入向量的长度。例如，如果第二个参数为2，那么每个输入的离散值将被映射到一个长度为2的连续向量。这个参数可以根据具体任务和数据来调整，以便更好地捕获数据的内在结构和关系。在这个例子中，`nn.Embedding(10, 3)`表示输入数据中的每个元素都是一个0到9的整数，这些整数将被映射到一个3维的向量空间中。

这段代码使用了PyTorch库中的`nn.Embedding`模块来创建一个嵌入层，并使用一个整数张量`input`作为输入进行前向传播。


首先，我们创建一个嵌入层`embedding`，它有10个词的词汇表（即10个不同的整数代表不同的单词或类别），每个单词被映射到一个3维的向量。

接下来，我们创建了一个名为`input`的整数张量，它有两个2维的序列，每个序列包含4个整数。这些整数代表词汇表中的单词。

然后，我们使用`embedding`层来对`input`进行前向传播，得到一个4x3的张量作为输出。每个整数都被替换为其对应的3维向量。

最后，我们打印出嵌入层的输出和其大小。

输出应该是：


```python
tensor([[0.1713, 0.5346, 0.4774],
        [0.5346, 0.2438, 0.1713],
        [0.4774, 0.2438, 0.5346],
        [0.3782, 0.6218, 0.5346]])
tensor([4, 3])
```

第一个输出是一个4x3的张量，其中每个元素都是一个3维向量。第二个输出是嵌入层的大小，它是一个2维张量，表示嵌入层的维度。

对于一个M×N的输入, 输入的所有 所有的元素都在0-num_embeddings-1范围内, 进行nn.Embedding(num_embeddings, embedding_dim)

embeddings类

### 代码实例

```python
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

```

### torch.randn

`x = torch.randn(4,4)` 这行代码是在使用PyTorch库生成一个4x4的张量（tensor），其中每个元素都是从标准正态分布（均值为0，标准差为1）中随机采样的。

具体来说：

* `torch.randn` 是一个函数，用于从标准正态分布（也称为高斯分布）中随机采样。
* `(4,4)` 是这个张量的形状（shape）。这意味着这个张量有4行和4列，总共16个元素。
* `x` 是这个新生成的张量的变量名。

执行这行代码后，`x` 将是一个4x4的张量，其中的值都是随机的，且符合标准正态分布。

### nn.Dropout

```python
m=nn.Dropout(p=0.2)
input = torch.randn(4,5)
output=m(input)
print(output)
```

结果

```
tensor([[ 0.0000,  0.0000, -0.4830,  0.5161,  0.6801],
        [ 1.2099,  0.2645,  0.0000, -0.7044,  0.7474],
        [ 0.0000,  0.1686,  1.1160,  1.6692,  0.0000],
        [-1.3290, -0.1441,  0.4216, -0.0000, -1.9159]])
```

这段代码是使用PyTorch库来创建一个Dropout层，并对一个随机输入向量进行Dropout操作。

1. `m=nn.Dropout(p=0.2)`

这行代码创建了一个Dropout层，其中`p`参数决定了在每次前向传播时随机关闭的单元的比例。在这个例子中，`p=0.2`意味着每次前向传播时，会有20%的单元被随机关闭。

2. `input = torch.randn(4,5)`

这行代码创建了一个4x5的随机输入矩阵，其中每个元素都是从标准正态分布（均值为0，标准差为1）中随机采样的。

3. `output=m(input)`

这行代码将上述随机输入矩阵传递给Dropout层。由于Dropout层会随机关闭输入矩阵中的某些元素，因此输出的矩阵中某些位置的值会是0（表示该位置的单元被随机关闭了）。

4. `print(output)`

这行代码将输出矩阵打印到控制台。

总体来说，这段代码演示了如何使用Dropout层来随机关闭神经网络中的某些单元，这是为了防止神经网络过拟合的一种常见技巧。

### torch.unsqueeze

`unsqueeze` 是一个在 PyTorch 中常用的函数，用于在指定维度上增加一个大小为 1 的维度。

#### 函数定义


```python
torch.unsqueeze(input, dim)
```

#### 参数

* `input`：输入的张量。
* `dim`：在哪个维度上增加新的维度。

#### 返回值

返回一个新的张量，该张量是在指定的维度上增加了一个大小为 1 的维度。其他维度的大小与输入张量相同。

#### 例子

1. **基本例子**：


```python
import torch
a = torch.tensor([1, 2, 3])  # a is a 1D tensor: [1, 2, 3]
b = torch.unsqueeze(a, 0)     # b will be a 2D tensor: [[1], [2], [3]]
```

2. **维度不匹配的例子**：
    如果你尝试在长度为3的一维张量上，在维度1上增加一个维度，你会得到一个形状为 `(3, 1)` 的二维张量。


```python
a = torch.tensor([1, 2, 3])  # a is a 1D tensor: [1, 2, 3]
b = torch.unsqueeze(a, 1)     # b is a 2D tensor: [[1], [2], [3]]
```

3. **在多维张量上使用**：
    你也可以在一个更高维度的张量上使用 `unsqueeze`。例如，对于一个形状为 `(3, 4)` 的二维张量：


```python
a = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])  # a is a 2D tensor: [3, 4]
b = torch.unsqueeze(a, 1)                                       # b is a 3D tensor: [3, 1, 4]
```

b` 的形状为 `(3, 1, 4)，因为我们在维度1上增加了一个大小为1的维度。

总之，`unsqueeze` 可以用于增加指定维度的大小，并在该维度上赋予其值。


```python
x = torch.tensor([1,2,3,4])
unsqueeze_res = torch.unsqueeze(x, 1)
print(unsqueeze_res)
```

结果:

```
tensor([[1],
        [2],
        [3],
        [4]])
```

这段代码是使用PyTorch库来创建一个张量，并对这个张量进行维度扩展。

1. `x = torch.tensor([1,2,3,4])`

这行代码创建了一个一维张量`x`，其中包含四个元素：1, 2, 3和4。

2. `unsqueeze_res = torch.unsqueeze(x, 1)`

这行代码对`x`进行了维度扩展。`torch.unsqueeze`函数会在指定的维度上增加一个维度。在这里，`x`原本是一个一维张量，维度大小为4。通过`torch.unsqueeze(x, 1)`，我们在第二个维度（索引为1的维度）上增加了一个新的维度，使得新的张量`unsqueeze_res`的形状变为`(4, 1)`。

3. `print(unsqueeze_res)`

这行代码将打印扩展后的张量`unsqueeze_res`。

执行这段代码后，输出将是：


```python
tensor([[1],
        [2],
        [3],
        [4]])
```

可以看到，原来的四个元素现在被组织成一个2x2的矩阵，其中第一维的大小为4（与原来一致），第二维的大小为1（新增加的维度）。


```python
x= torch.tensor([1,2,3,4])
y= torch.unsqueeze(x,0)
print(y.shape, y)
z =torch.unsqueeze(x,1)
print(z.shape, z)
```


这段代码中，我们首先创建了一个一维张量 `x`，然后使用 `torch.unsqueeze` 函数来扩展其维度。

1. `x = torch.tensor([1,2,3,4])`：这行代码创建了一个一维张量 `x`，其中包含四个元素：1, 2, 3和4。
1. `y = torch.unsqueeze(x, 0)`：这行代码对一维张量 `x` 进行维度扩展。`torch.unsqueeze` 函数会在指定的维度上增加一个维度。在这里，`x` 是一个一维张量，维度大小为4。通过 `torch.unsqueeze(x, 0)`，我们在第一个维度（索引为0的维度）上增加了一个新的维度，使得新的张量 `y` 的形状变为 `(1, 4)`。执行结果如下：


    * `y.shape` 的输出是 `(1, 4)`，表示张量 `y` 的形状是 1 行 4 列。
    * `y` 的输出是：
    ```

   tensor([[1, 2, 3, 4]])
    ```
    可以看到，原来的四个元素现在被组织成一个1x4的矩阵。

3. `z = torch.unsqueeze(x, 1)`：这行代码同样对一维张量 `x` 进行维度扩展。与上一行代码类似，通过 `torch.unsqueeze(x, 1)`，我们在第二个维度（索引为1的维度）上增加了一个新的维度，使得新的张量 `z` 的形状变为 `(4, 1)`。执行结果如下：


    * `z.shape` 的输出是 `(4, 1)`，表示张量 `z` 的形状是 4 行 1 列。
    * `z` 的输出是：
    ```
     tensor([[1],
                [2],
                [3],
                [4]])`
    ```
    可以看到，原来的四个元素现在被组织成一个4x1的矩阵。

总结：这段代码演示了如何使用 PyTorch 的 `torch.unsqueeze` 函数来对一维张量进行维度扩展，并在指定的维度上增加一个大小为1的维度。

(原来x的维度在第0个, 
y=torch.unsqueeze(x, 0), 在第0个维度上新增了一个维度, 原来的第0个维度移动到第1个维度, 所以变成了(1,4)维度的张量,
z=torch.unsqueeze(x, 0), 在第1个维度上新增了一个维度, 原来的第0个维度不变, 所以变成了(4,1)维度的张量,
)

### view

y = x.view(16)
这行代码是将张量`x`重新塑形（reshape）为一个包含16个元素的1D张量。

具体来说：

* `x.view()` 是PyTorch中用于改变张量形状的方法。
* `(16)` 指定了新的形状。因为原始张量`x`是一个4x4的2D张量，所以它包含16个元素。
* `y` 是这个新生成的1D张量的变量名。

执行这行代码后，`y` 将是一个1D张量，其中包含原始张量`x`中的所有16个元素，但是其形状已经被改变。
`view` 方法的参数主要包括：

1. **必需参数**：


    * `size`：目标张量的形状。它是一个表示新形状的整数或元组。如果这个参数与原始张量的总元素数量不匹配，将会抛出错误。

2. **可选参数**：


    * `strides`：目标张量的步长（strides）。它是一个表示新步长的整数或元组。默认值是 `None`，表示使用原始张量的步长。
    * `dtype`：目标张量的数据类型。默认值是原始张量的数据类型。
    * `device`：目标张量应存储在哪个设备上（例如CPU或GPU）。默认值是 `None`，表示使用原始张量的设备。
    * `requires_grad`：一个布尔值，表示是否需要为新张量计算梯度。默认值是 `False`。

指定方式示例：

* 创建一个形状为 `(16,)` 的1D张量：

```python
y = x.view(16)
```

* 创建一个形状为 `(4,4)` 的2D张量：

```python
y = x.view(4, 4)
```

* 指定步长：

```python
y = x.view(16, stride=4)  # 假设x是一个8x4的张量，那么新的y将有步长为4
```

* 指定数据类型：

```python
y = x.view(16, dtype=torch.float32)  # 将所有元素转换为float32类型
```

* 将张量移至GPU上：

```python
y = x.view(16, device=torch.device('cuda'))  # 如果x原本在CPU上，现在y将在GPU上
```

### transpose

```python
a = torch.randn(1,2,3,4)
print(a.size(), a)
b = a.transpose(1,2)
print(b)
```

首先，让我们分析代码中每一步的作用：

1. `a = torch.randn(1,2,3,4)`：这行代码创建了一个4维的张量`a`，其形状为`(1,2,3,4)`。张量中的元素是随机生成的，遵循标准正态分布（均值为0，标准差为1）。
1. `print(a.size(), a)`：这行代码首先打印出张量`a`的形状，然后打印出张量`a`的内容。
1. `b = a.transpose(1,2)`：这行代码对张量`a`进行转置操作。具体来说，它交换了第2维（索引为1的维度）和第3维（索引为2的维度）。所以，原来的形状`(1,2,3,4)`会变为`(1,3,2,4)`。
1. `print(b)`：这行代码打印出转置后的张量`b`的内容。

现在，我们来具体解释每一部分：

* `a.size()`：这将返回一个表示张量`a`形状的元组，即`(1,2,3,4)`。
* `a`：这将打印出张量`a`的内容。由于这是一个四维张量，并且每一维的大小都不同，因此输出的内容会是一系列嵌套的列表，表示各个维度的大小。
* `b`：这将打印出转置后的张量`b`的内容。由于`b`是通过对第2维和第3维进行转置得到的，所以输出的内容会有所不同。具体来说，原始张量中在第2维的元素（索引为1的维度）现在会出现在第3维的位置（索引为2的维度），而原始张量中在第3维的元素现在会出现在第2维的位置。

这样，你就能清楚地看到`transpose`操作是如何改变张量的维度顺序的。

### view和transpose对比

```python
a = torch.randn(1,2,3,4)
print(a.size(), a)
b = a.transpose(1,2)
print(b.size(), b)
c = a.view(1,3,2,4)
print(c.size(), c)
```

让我们一步步分析这段代码：

1. `a = torch.randn(1,2,3,4)`：
    这行代码创建了一个形状为 `(1,2,3,4)` 的四维张量 `a`。每个元素都是从标准正态分布中随机采样的。
1. `print(a.size(), a)`：
    这行代码打印了张量 `a` 的形状和内容。由于 `a` 的形状是 `(1,2,3,4)`，所以输出的 `a.size()` 将是 `(1,2,3,4)`，而 `a` 的内容是该形状的随机数矩阵。
1. `b = a.transpose(1,2)`：
    这行代码对张量 `a` 进行转置。具体来说，它交换了第2维（索引为1）和第3维（索引为2）。因此，`b` 的形状将从 `(1,2,3,4)` 变为 `(1,3,2,4)`。
1. `print(b.size(), b)`：
    这行代码打印了转置后的张量 `b` 的形状和内容。输出的 `b.size()` 将是 `(1,3,2,4)`，而 `b` 的内容是该形状的转置矩阵。
1. `c = a.view(1,3,2,4)`：
    这行代码改变了张量 `a` 的形状，将其重新塑形为一个形状为 `(1,3,2,4)` 的四维张量。注意，这里与之前的转置操作不同，`view` 方法不会改变张量中的元素，只是改变了它们的布局方式。因此，`c` 和原始的 `a` 将包含相同的元素，只是它们的排列顺序不同。
1. `print(c.size(), c)`：
    这行代码打印了重新塑形后的张量 `c` 的形状和内容。输出的 `c.size()` 将是 `(1,3,2,4)`，而 `c` 的内容是该形状的矩阵。

总结：这段代码展示了如何在 PyTorch 中创建、转置和重新塑形四维张量。通过这些操作，你可以改变张量的维度顺序和大小，而不改变其包含的元素。

### deepcopy

```python
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
```

这个函数`clones`用于创建一个包含多个克隆模块的列表。让我们逐步解释这个函数：

1. **输入参数**:


    * `module`: 这是要克隆的原始模块。
    * `N`: 表示我们想要克隆`module`多少次。

2. **函数功能**:


    * 使用列表推导式，函数会创建`N`个`module`的深度复制（deep copy）。这意味着每个克隆模块都是原始模块的一个完全独立的副本，它们之间的任何更改都不会相互影响。
    * `nn.ModuleList`是一个特殊的PyTorch容器，它用于存储模块列表，并确保当这个容器被传递给其他函数或方法时，其内容（即模块）也被传递，而不是只传递引用。这对于确保模块的独立性非常有用。

3. **返回值**:


    * 返回一个包含`N`个克隆模块的`nn.ModuleList`。

简单地说，这个函数允许您轻松地创建多个独立副本的特定模块，这对于某些神经网络结构（例如，复制相同的网络层多次）是非常有用的。

### 代码解释

这段代码定义了一个名为`MultiHeadedAttention`的PyTorch神经网络模块，该模块实现了多头注意力机制。下面是对代码的逐行解释：


```python
import copy  # 导入copy模块，用于深度复制对象。
import attention  # 导入attention模块，这个模块中应该定义了实现注意力机制的函数或类。
from torch import nn  # 从torch库中导入神经网络模块nn。

def clones(module, N):  # 定义一个名为clones的函数，用于复制一个模块N次。
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadedAttention(nn.Module):  # 定义一个名为MultiHeadedAttention的类，继承自nn.Module。
    def __init__(self, head, embedding_dim, dropout=0.1):  # 初始化方法，用于设置模型的参数。
        super(MultiHeadedAttention, self).__init__()  # 调用父类的初始化方法。
        assert embedding_dim % head == 0, "Embedding dimension must be divisible by number of heads."  # 断言：嵌入维度必须能被头数整除。
        self.d_k = embedding_dim // head  # 计算每个头的维度（每个头的特征数）。
        self.head = head  # 记录头数。
        self.embedding_dim = embedding_dim  # 记录嵌入向量的维度。
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)  # 使用clones函数复制线性层4次，创建4个线性层。
        self.attn = None  # 初始化注意力权重为None。
        self.dropout = nn.Dropout(p=dropout)  # 创建dropout层，用于防止过拟合。

    def forward(self, query, key, value, mask=None):  # 定义前向传播方法。
        if mask is not None:  # 如果提供了遮盖（mask）：
            mask = mask.unsqueeze(1)  # 扩展mask的维度。
        batch_size = query.size(0)  # 获取查询向量的第一个维度的大小，即批次大小。
        query, key, value = [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2) for model, x in zip(self.linears, query, key, value)]  # 对查询、键、值向量进行变换和整形。
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)  # 调用attention函数计算输出向量和注意力权重。
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)  # 对输出向量进行转置、连接和整形。
        return self.linears[-1](x)  # 使用最后一个线性层对输出向量进行变换，并返回结果。
```

这个代码实现了一个多头注意力机制的类，其中使用了四个线性层（`self.linears`），这些线性层可能用于实现查询、键、值和输出变换。在前向传播方法中，对输入的查询、键、值向量进行了变换和整形，然后调用了`attention`函数进行多头注意力计算，最后对输出向量进行了转置和整形操作。



## 自定义PositionalEncoding

 [positional_encoding.py](..\..\transformer\test\positional_encoding.py) 



## 绘制词向量特征分布曲线

 [pe_test.py](..\..\transformer\test\pe_test.py) 

### 小结

-   学习了文本嵌入层的作用
    -   无论是源文本嵌入还是目标文本嵌入，都是为了将文本中词汇的数字表示转变为向量
        表示，希望在这样的高维空间捕捉词汇间的关系
        学习并实现了文本嵌入层的类：Embeddings
    -   初始化函数以d_model,词嵌入堆度，和vocab,词汇总数为参数，内部主要使用了nn中的
        预定层Embedding进行词嵌入.
        在forward函数中，将输入x传入到Embedding的实例化对象中，然后乘以一个根号下
        dmode进行缩放控制数值大小，·它的输出是文本嵌入后的结果，
-   学习了位置编码器的作用：
    -   因为在Transformer的编码器结构中，并没有针对词汇位置信息的处理。因此需要在
        Embedding层后加入位置编码器，将词汇位置不同可能会产生不同语义的信息加入到
        词嵌入张量中，以弥补位置信息的缺失

-   学习并实现了位置编码器的类：PositionalEncoding
    -   初始化函数以d_model,dropout,max_len为参数，分别代表d_model:词嵌入维度
        dropou比置0比率，nax_len:每个句子的最大长度
        forward函数中的输入参数为x,是Embedding层的输出
        最终输出一个加入了位置编码信息的词嵌入张量，
    -   实现了绘制调汇向量中特征的分布曲线
    -   保证同一词汇随着所在位置不同它对应位置嵌入向量会发生变化
        正弦波和余弦波的值域范围都是1到1，这又很好的控制了骸入数值的大小，有助于梯度
        的快速计算



# 编码器部分实现

P9 9.2.3.1

-   编码器部分：
    -   由N个编码器层堆叠而成
    -   每个编码器层由两个子层连接结构组成
    -   第一个子层连接结构包括一个多头自注意力子层和规范化层以及一个残差连接
    -   第二个子层连接结构包括一个前馈全连接子层和规范化层以及一个残差连援

2.3.1掩码张量
·了解什么是掩码张量以及它的作用
,掌生成码张量的实现过程
·什么是掩码张量：
·撞代表遍掩，码就是我们张量中的数值，它的尺寸不定。里面一般只有1和0的元素，代表
位置被遍掩或者不被遮掩，至于是0位置被遮掩还是1位置被遍掩可以自定义，因此它的作
用就是让另外一个张量中的一些数值被遮掩，也可以说被替换它的裹现形式是一个张量。
掩码张量的作用：
·在transformer中，掩码张量的主要作用在应用attention(将在下一小节讲解时.有一些生
成的attention张量中的值计算有可能已知了未来信息而得到的，未来信息被看到是因为训
练时会把整个输出结果都一次性进行Embedding,但是理论上解码器的的输出却不是一次
就能产生最终结果的，而是一次次通过上一次结果综合得出的，因此，未来的信息可能被
提前利用，所以，我们会进行遮掩，关于解码器的有关知识将在后面的章节中讲解



## 掩码张量-基础方法介绍

### np.triu

 [mask.py](..\..\transformer\test\mask.py) 

`np.triu` 是 NumPy 库中的一个函数，用于生成一个上三角矩阵。这个函数返回一个数组，该数组的上三角部分（包括主对角线）包含1，而下三角部分（不包括主对角线）包含0。

函数的语法如下：


```python
import numpy as np

a = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
print(np.triu(a, k=0))
print(np.triu(a, k=1))
print(np.triu(a, k=-1))

```



```
[[1 2 3]
 [0 5 6]
 [0 0 9]
 [0 0 0]]
 
[[0 2 3]
 [0 0 6]
 [0 0 0]
 [0 0 0]]
 
[[ 1  2  3]
 [ 4  5  6]
 [ 0  8  9]
 [ 0  0 12]]
```



其中：

* `a` 是输入的数组或矩阵。
* `k` 是指定下标偏移量。如果 `k` 是正数，那么下三角部分（不包括主对角线）将包含1。如果 `k` 是负数，那么上三角部分（包括主对角线）将包含0。如果 `k` 是0，那么结果就是一个上三角矩阵。

所以，`np.triu`函数的主要用途是生成一个上三角矩阵，它可以用在各种数学和科学计算中，特别是在需要处理矩阵运算的场合。

### 代码介绍

#### 定义下三角矩阵

 [mask.py](..\..\transformer\mask.py) 

可视化

```python
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
```



### 2.3.1掩码张量总结

·学习了什么是掩码张量：
·掩代表遮掩，码就是我们张量中的数值，它的尺寸不定，里面一般只有1和0的元素，
代表使置被遮掩或者不被遮掩，至于是0位置被遮掩还是1位置被遮掩可以自定义，因
此它的作用就是让另外一个张量中的一些数值被遮掩，也可以说被替换，它的表现形式
是一个张量
·学习了掩码张量的作用：
·在transformer中，掩码张量的主要作用在应用attention(将在下一小节讲解)时，有一些
生成的attetion张量中的值计算有可能已知量未来信息而得到的，未来信息被看到是
因为训练时会把整个输出结果都一次性进行Embedding,.但是理论上解码器的的输出
却不是一次就能产生最终结果的，而是一次次通过上一次结果综合得出的，因此，未
来的信息可能被提前利用，所以，我们会进行遮掩.关于解码器的有关知识将在后面的
章节中讲解
·学习并实现了生成向后遮掩的撞码张量函数：subsequent mask
·它的输入是sze,代表掩码张量的大小
·它的输出是一个最后两维形成1方阵的下三角阵
·最后对生成的掩码张量进行了可视化分析，更深一步理解了它的用途

# 2.3.2注意力机制

·学习目标
·了解什么是注意力计算规则和注意力机制
掌握注意力计算规则的实现过程
·什么是注意力：
我们观察事物时，之所以能够快速判断一种事物（名然允许判断是错误的），是因为我们大脑能够很快把注意力放在事物最具有辨识度的部分从而作出判断，而并非是从头到尾的观察一遍事物后，才能有判断结果.正是基于这样的理论，就产生了注意力机制
·什么是注意力计算规则：
·它需要三个指定的输入Q(query),Kkey),V(value),然后通过公式得到注意力的计算结果，这个结果代表quey在key和value作用下的表示.而这个具体的计算规则有很多种

这里只介绍我们用到的这一种我门这里使用的注意力的计算规则

$Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{x}})V$

·什么是注意力机制
·注意力机制是注意力计算规则能够应用的深度学习网络的载体，除了注意力计算规则外，还包括一些必要的全连接层以及相关张量处理，使其与应用网络融为一体使用自注意力计算规则的注意力机制称为自注意力机制
·注意力机割在网络中实现的图形表示：

### 基础方法介绍

#### masked_fill

[mask_fill_test.py](..\..\transformer\test\mask_fill_test.py) 



```python
import torch
from torch.autograd import Variable

x = Variable(torch.randn(5,5))
print(x)
mask = Variable(torch.zeros(5,5))
print(mask)
y = x.masked_fill(mask==0, 1e-9)
print(y)

```

这段代码使用PyTorch库来创建一个随机的5x5张量（矩阵）`x`，然后创建另一个5x5全零的张量`mask`。最后，它使用`masked_fill()`函数将`mask`中所有为0的位置在`x`中的值替换为非常接近0的值（1e-9）。

具体来说：

1. `x = Variable(torch.randn(5,5))`: 这行代码创建一个5x5的张量`x`，其中的元素是从标准正态分布（均值为0，标准差为1）中随机采样的。
2. `mask = Variable(torch.zeros(5,5))`: 这行代码创建一个5x5的张量`mask`，其中的元素都是0。
3. `y = x.masked_fill(mask==0, 1e-9)`: 这行代码使用`masked_fill()`函数来修改张量`x`。这个函数会查找与掩码`mask`中所有为0的位置对应的元素，并将这些元素的值替换为1e-9。因为掩码中所有元素都是0，所以这实际上会将张量`x`中的所有元素替换为1e-9。
4. `print(x)`, `print(mask)`, `print(y)`: 这些print语句将打印出`x`、`mask`和`y`的值。

需要注意的是，虽然这段代码可以运行，但现在已经不再推荐使用`Variable`了。在较新版本的PyTorch中，你应该使用Tensor而不是Variable。另外，对于掩码操作，通常推荐使用布尔索引而不是`masked_fill()`。以下是使用Tensor和布尔索引的等效代码：


```python
import torch

x = torch.randn(5, 5)
mask = torch.zeros(5, 5)
y = x.clone()  # 创建一个x的副本以避免修改原始数据

y[mask == 0] = 1e-9  # 使用布尔索引来替换值

print(x)
print(mask)
print(y)
```



### 代码介绍

#### attention

 [attention_func.py](..\..\transformer\attention_func.py) 

```python
import torch
import math

import torch.nn.functional as F
from torch.autograd import Variable

from transformer.positional_encoding import pe_result


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, 1e-9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn

query = key = value = pe_result
attn, p_attn = attention(query, key, value)
print('attn:', attn)
print(attn.shape)
print('p_attn:', p_attn)
print(p_attn.shape)

mask = Variable(torch.zeros(2,4,4))
attn, p_attn = attention(query, key, value, mask=mask)
print('attn:', attn)
print(attn.shape)
print('p_attn:', p_attn)
print(p_attn.shape)
```

这段代码实现了多头注意力机制的一个基本版本。我会为你逐步解释每一部分。

1. **导入库**:


	* `torch` 和 `math`：这是Python的内置库，用于数学运算。
	* `torch.nn.functional`：提供了许多神经网络操作。
	* `torch.autograd.Variable`：用于自动微分。但在较新版本的PyTorch中，直接使用Tensor即可，因此这部分可能是过时的。
	* `transformer.positional_encoding` 中的 `pe_result`：这似乎是一个位置编码的结果，但代码中没有给出具体实现。
2. **attention函数**:


	* **输入参数**:
		+ `query`: 查询向量。
		+ `key`: 键向量。
		+ `value`: 值向量。
		+ `mask`: 一个掩码，用于指示哪些位置是有效的、哪些位置应该被忽略。
		+ `dropout`: 一个dropout层，用于防止过拟合。
	* **功能**:
		+ 首先，计算查询和键之间的分数，分数是通过矩阵乘法得到的，然后通过缩放（`/ math.sqrt(d_k)`）进行归一化。
		+ 如果提供了掩码，则将分数中掩码为0的位置设置为非常小的值（-1e9）。
		+ 使用softmax函数对分数进行归一化，得到注意力权重。
		+ 如果提供了dropout层，则应用dropout。
	* **输出**:
		+ 输出的第一个值是加权的值向量。
		+ 输出的第二个值是注意力权重。
3. **使用attention函数**:


	* 初始化`query`, `key`, 和 `value` 为 `pe_result`。
	* 调用attention函数并打印输出。
4. **掩码的使用**:


	* 创建一个2x4x4的零张量作为掩码。
	* 使用这个掩码再次调用attention函数并打印输出。

总结：这段代码展示了如何使用多头注意力机制的基本版本。通过这个机制，模型可以聚焦于输入中的不同部分来生成输出。



### 2.3.2注意力机制总结

·学习了什么是注意力：·我们观察事物时，之所以能够快速判断一种事物（当然允许判断是错误的），是因为我们大脑能够很快把注意力放在事物最具有拆识度的部分从而作出判断，而并非是从头到尾的观察一遍事物后，才能有判断结果.正是基于这样的理论。就产生了注意力机制

什么是注意力计算规则：

它需要三个指定的输入Q(query),Kkey),V(value),然后通过公式得到注意力的计算结果，这个结果代表quey在key和valuet作用下的表示.而这个具体的计算规则有很多种，我这里只介绍我们用到的这一种

学习了Q,KV的比编解释

Q是一段准备被概括的文本：

K是给出的提示：

V是大脑中的对提示K的延伸

当Q=K=V时，称作自注意力机制

什么是注意力机制

注意力机制是注意力计算规则能够应用的深度学习网络的载体，除了注意力计算规则外，还包括一些必要的全连接层以及相关张量处理，使其与应用网络融为一体

使用自注意力计算规则的注意力机制称为自注意力机制

·学习并实现了注意力计算规则的函数：attention

它的输入就是Q,K,V以及mask和dropout,

mask用于掩码

dropout用于随机置0.

它的输出有两个，quey的注意力表示以及注意力张量