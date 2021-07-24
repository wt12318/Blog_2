---
title: Pytorch 基础
date: 2021-07-21 20:59:00
tags: 编程
categories:
  - DL
index_img: img/pytorc.jpg
---

 Pytorch 官网教程 学习

<!-- more -->

学习 Pytorch 官网教程，主要包括以下内容：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210721075853969.png)

## Tensors

Tensor 类似于 Numpy 的 ndarray，不同之处在于张量可以在 GPU 或其他硬件加速器上运行。 实际上，张量和 NumPy 数组通常可以共享相同的底层内存，从而无需复制数据；另外张量也针对自动微分进行了优化。

``` python
##首先导入需要的库
import torch
import numpy as np
```

### 初始化 Tensor

可以以多种方式初始化 Tensor：

1.  直接从数据创建，数据类型可以自动判断

``` python
data = [[1,2],[3,4]]
x_data = torch.tensor(data)
x_data.type()
>> 'torch.LongTensor'
x_data.dtype
>> torch.int64
```

2. 从 numpy 数组创建

``` python
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
```

3. 从另一个 Tensor 创建，`torch.**_like` 函数，会保留作为参数的 Tensor 的形状和数据类型，当然也可以进行覆盖：

``` python
x_ones = torch.ones_like(x_data)##全为1的Tensor，形状和数据类型一致
x_ones.dtype
>> torch.int64
x_ones = torch.ones_like(x_data, dtype=torch.float)
x_ones.dtype
>> torch.float32
x_rand = torch.rand_like(x_data) 
>> Error in py_call_impl(callable, dots$args, dots$keywords): RuntimeError: "check_uniform_bounds" not implemented for 'Long'
x_rand = torch.rand_like(x_data, dtype=torch.float) 
```

### Tensor 的属性

Tensor 的属性包括形状（shape），数据类型（dtype）和存储的设备（device）：

``` python
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
>> Shape of tensor: torch.Size([3, 4])
print(f"Datatype of tensor: {tensor.dtype}")
>> Datatype of tensor: torch.float32
print(f"Device tensor is stored on: {tensor.device}")
>> Device tensor is stored on: cpu
```

### Tensor 操作

对 Tensor 的操作有多种，包括转置，索引，切片，数学计算，随机抽样等，更具体的描述[在这里](https://pytorch.org/docs/stable/torch.html)，这些操作都可以在 GPU 上运行，查看 GPU 信息的方法：

``` python
torch.cuda.is_available()#cuda是否可用
>> True
torch.cuda.device_count()#返回gpu数量
>> 1
torch.cuda.get_device_name(0)#返回gpu名字，设备索引默认从0开始
>> 'NVIDIA GeForce GTX 1060'
torch.cuda.current_device()#返回当前设备索引
>> 0
torch.cuda.get_device_capability()##查看内存
>> (6, 1)
```

默认情况下 tensor 是在 CPU 上创建的，需要使用 `.to` 方法将其移到 GPU 上：

``` python
if torch.cuda.is_available():
  tensor = tensor.to('cuda')

tensor.device
>> device(type='cuda', index=0)
```

下面是一些对 tensor 操作的实列：

``` python
##索引和切片
tensor = torch.randn(4,4)
tensor
>> tensor([[-1.3974,  0.2943, -0.5416, -0.9584],
>>         [-0.5710, -0.6704, -0.2144, -0.8668],
>>         [ 0.2884,  1.0564,  0.1403, -0.4693],
>>         [-0.4992,  0.6969,  0.1507,  1.9419]])
tensor[0]##第一行
>> tensor([-1.3974,  0.2943, -0.5416, -0.9584])
tensor[:,0]##第一例
>> tensor([-1.3974, -0.5710,  0.2884, -0.4992])
tensor[...,-1]##或者tensor[:,-1] 最后一列
>> tensor([-0.9584, -0.8668, -0.4693,  1.9419])
tensor[:,1] = 0
tensor

##合并tensor
>> tensor([[-1.3974,  0.0000, -0.5416, -0.9584],
>>         [-0.5710,  0.0000, -0.2144, -0.8668],
>>         [ 0.2884,  0.0000,  0.1403, -0.4693],
>>         [-0.4992,  0.0000,  0.1507,  1.9419]])
t0 = torch.cat([tensor,tensor],dim=0)
t0
>> tensor([[-1.3974,  0.0000, -0.5416, -0.9584],
>>         [-0.5710,  0.0000, -0.2144, -0.8668],
>>         [ 0.2884,  0.0000,  0.1403, -0.4693],
>>         [-0.4992,  0.0000,  0.1507,  1.9419],
>>         [-1.3974,  0.0000, -0.5416, -0.9584],
>>         [-0.5710,  0.0000, -0.2144, -0.8668],
>>         [ 0.2884,  0.0000,  0.1403, -0.4693],
>>         [-0.4992,  0.0000,  0.1507,  1.9419]])
t0.shape
>> torch.Size([8, 4])
t1 = torch.cat([tensor,tensor],dim=1)
t1
>> tensor([[-1.3974,  0.0000, -0.5416, -0.9584, -1.3974,  0.0000, -0.5416, -0.9584],
>>         [-0.5710,  0.0000, -0.2144, -0.8668, -0.5710,  0.0000, -0.2144, -0.8668],
>>         [ 0.2884,  0.0000,  0.1403, -0.4693,  0.2884,  0.0000,  0.1403, -0.4693],
>>         [-0.4992,  0.0000,  0.1507,  1.9419, -0.4992,  0.0000,  0.1507,  1.9419]])
t1.shape
>> torch.Size([4, 8])
t2 = torch.stack([tensor,tensor],dim=0)
t2
>> tensor([[[-1.3974,  0.0000, -0.5416, -0.9584],
>>          [-0.5710,  0.0000, -0.2144, -0.8668],
>>          [ 0.2884,  0.0000,  0.1403, -0.4693],
>>          [-0.4992,  0.0000,  0.1507,  1.9419]],
>> 
>>         [[-1.3974,  0.0000, -0.5416, -0.9584],
>>          [-0.5710,  0.0000, -0.2144, -0.8668],
>>          [ 0.2884,  0.0000,  0.1403, -0.4693],
>>          [-0.4992,  0.0000,  0.1507,  1.9419]]])
t2.shape
>> torch.Size([2, 4, 4])
t3 = torch.stack([tensor,tensor],dim=1)
t3
>> tensor([[[-1.3974,  0.0000, -0.5416, -0.9584],
>>          [-1.3974,  0.0000, -0.5416, -0.9584]],
>> 
>>         [[-0.5710,  0.0000, -0.2144, -0.8668],
>>          [-0.5710,  0.0000, -0.2144, -0.8668]],
>> 
>>         [[ 0.2884,  0.0000,  0.1403, -0.4693],
>>          [ 0.2884,  0.0000,  0.1403, -0.4693]],
>> 
>>         [[-0.4992,  0.0000,  0.1507,  1.9419],
>>          [-0.4992,  0.0000,  0.1507,  1.9419]]])
t3.shape
>> torch.Size([4, 2, 4])
t4 = torch.stack([tensor,tensor],dim=2)
t4
>> tensor([[[-1.3974, -1.3974],
>>          [ 0.0000,  0.0000],
>>          [-0.5416, -0.5416],
>>          [-0.9584, -0.9584]],
>> 
>>         [[-0.5710, -0.5710],
>>          [ 0.0000,  0.0000],
>>          [-0.2144, -0.2144],
>>          [-0.8668, -0.8668]],
>> 
>>         [[ 0.2884,  0.2884],
>>          [ 0.0000,  0.0000],
>>          [ 0.1403,  0.1403],
>>          [-0.4693, -0.4693]],
>> 
>>         [[-0.4992, -0.4992],
>>          [ 0.0000,  0.0000],
>>          [ 0.1507,  0.1507],
>>          [ 1.9419,  1.9419]]])
t4.shape
>> torch.Size([4, 4, 2])
```

注意 `stack` 和 `cat` 的区别：`cat` 在哪个维度上合并就增加哪个维度的大小，而 `stack` 则新增一个维度。

``` python
##数学操作
###矩阵相乘
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y1 == y2
>> tensor([[True, True, True, True],
>>         [True, True, True, True],
>>         [True, True, True, True],
>>         [True, True, True, True]])
y3 = torch.rand_like(tensor) ##4*4 * 4*4 = 4*4
torch.matmul(tensor,tensor.T,out=y3) == y1

###按元素相乘
>> tensor([[True, True, True, True],
>>         [True, True, True, True],
>>         [True, True, True, True],
>>         [True, True, True, True]])
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z1 == z2
>> tensor([[True, True, True, True],
>>         [True, True, True, True],
>>         [True, True, True, True],
>>         [True, True, True, True]])
z3 = torch.rand_like(tensor)
torch.mul(tensor,tensor,out=z3) == z1

##单元素tensor可以使用 item 转换为python数值
>> tensor([[True, True, True, True],
>>         [True, True, True, True],
>>         [True, True, True, True],
>>         [True, True, True, True]])
agg = tensor.sum()
agg_item = agg.item()
agg_item,type(agg_item)

##原位操作 带下划线的函数
>> (-2.996830940246582, <class 'float'>)
tensor = torch.tensor([1,2])
tensor.add(4)
>> tensor([5, 6])
tensor
>> tensor([1, 2])
tensor.add_(4)
>> tensor([5, 6])
tensor
>> tensor([5, 6])
```

### 和 Numpy 数组间共享内存

CPU 和 NumPy 数组上的 tensor 可以共享它们的底层内存位置，改变一个也会改变另一个：

``` python
##Tensor to NumPy array
t = torch.ones(5)
t
>> tensor([1., 1., 1., 1., 1.])
n = t.numpy()
n

###tensor 改变 numpy array也会改变
>> array([1., 1., 1., 1., 1.], dtype=float32)
t.add_(1)
>> tensor([2., 2., 2., 2., 2.])
t
>> tensor([2., 2., 2., 2., 2.])
n

##NumPy array to Tensor
>> array([2., 2., 2., 2., 2.], dtype=float32)
n = np.ones(5)
t = torch.from_numpy(n)

np.add(n,1,out=n)
>> array([2., 2., 2., 2., 2.])
n
>> array([2., 2., 2., 2., 2.])
t
>> tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
```

## 数据集和数据载入（DATASETS & DATALOADERS）

pytorch 提供了两个数据读入和获取的接口：`torch.utils.data.DataLoader` 和
`torch.utils.data.Dataset`；`Dataset` 存储了样本和对应的标签，`DataLoader` 封装了一个对 `Dataset`
的迭代器，可以方便的获取样本。

### 载入数据集

Pytorch 提供了一些提前已经读入的常用数据集，比如 FashionMNIST；下面是读入 FashionMNIST 数据集的示例：

``` python
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

training_data
>> Dataset FashionMNIST
>>     Number of datapoints: 60000
>>     Root location: data
>>     Split: Train
>>     StandardTransform
>> Transform: ToTensor()
```

主要有以下的参数：

- `root`：训练集/测试集存放的位置 
- `train`：是训练集还是测试集
- `download`：当在 `root` 中找不到数据时是否从网络上下载 
- `transform`（转化样本）/`target_transform`（转化标签）：如何对数据进行转化处理

### 迭代和可视化数据集

我们可以像列表一样索引 `Datasets`:

``` python
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()##随机抽一个图像
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
>> <AxesSubplot:>
>> Text(0.5, 1.0, 'Ankle Boot')
>> (0.0, 1.0, 0.0, 1.0)
>> <matplotlib.image.AxesImage object at 0x000001CF910C5220>
>> <AxesSubplot:>
>> Text(0.5, 1.0, 'T-Shirt')
>> (0.0, 1.0, 0.0, 1.0)
>> <matplotlib.image.AxesImage object at 0x000001CF911009A0>
>> <AxesSubplot:>
>> Text(0.5, 1.0, 'Ankle Boot')
>> (0.0, 1.0, 0.0, 1.0)
>> <matplotlib.image.AxesImage object at 0x000001CF91139160>
>> <AxesSubplot:>
>> Text(0.5, 1.0, 'Trouser')
>> (0.0, 1.0, 0.0, 1.0)
>> <matplotlib.image.AxesImage object at 0x000001CF911688E0>
>> <AxesSubplot:>
>> Text(0.5, 1.0, 'Pullover')
>> (0.0, 1.0, 0.0, 1.0)
>> <matplotlib.image.AxesImage object at 0x000001CF911A20A0>
>> <AxesSubplot:>
>> Text(0.5, 1.0, 'Trouser')
>> (0.0, 1.0, 0.0, 1.0)
>> <matplotlib.image.AxesImage object at 0x000001CF911DDC10>
>> <AxesSubplot:>
>> Text(0.5, 1.0, 'Trouser')
>> (0.0, 1.0, 0.0, 1.0)
>> <matplotlib.image.AxesImage object at 0x000001CF9121A3D0>
>> <AxesSubplot:>
>> Text(0.5, 1.0, 'Bag')
>> (0.0, 1.0, 0.0, 1.0)
>> <matplotlib.image.AxesImage object at 0x000001CF9123E790>
>> <AxesSubplot:>
>> Text(0.5, 1.0, 'T-Shirt')
>> (0.0, 1.0, 0.0, 1.0)
>> <matplotlib.image.AxesImage object at 0x000001CF9126EF10>
plt.show()
```

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/unnamed-chunk-26-1.png" width="768" />

### 创建自定义数据集

一个自定义的 Dataset 类需要实现 3 个函数：`__int__`, `__len__`, 和 `__getitem__`；下面是自定义的 FashionMNIST 数据集，图片存在 `img_dir` 目录下，标签存在 `annotations_file` 文件中：

``` python
import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])##annotations_file第一列是文件名
        image = read_image(img_path)##read_image 可以自动转tensor
        label = self.img_labels.iloc[idx, 1]##第二列是标签
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
```

`__init__` 函数会在实例化 Dataset 对象时运行，初始化包含图片和标签的路径以及对样本和标签的转化，这里标签数据第一列为样本数据的文件名，第二列是对应的标签，类似：

``` python
tshirt1.jpg, 0
tshirt2.jpg, 0
......
ankleboot999.jpg, 9
```

`__len__` 函数返回的是数据集的大小（有多少个样本）；`__getitem__` 函数读入并返回给定索引（idx）的样本。

### 使用 `DataLoader` 准备训练的数据

`Dataset` 每次只能返回一个样本的数据和标签，但是在训练模型时，通常的情况是：一次性传入一个
`minibatch` 的样本，在一个 `epoch` 结束时打乱数据再进行下一个 `epoch`来减少模型的过拟合，`DataLoader` 可以简化这个流程：

``` python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
```

#### 通过 `DatatLoader` 进行迭代

通过 `DatatLoader` 载入数据后就可以对该数据进行迭代，每一次迭代会返回大小为 `batch_size` 的训练样本和相应的标签，`shuffle=True` 表示在迭代完整个数据后（一个 epoch）对数据进行打乱再进行下一个 epoch：

``` python
# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
>> Feature batch shape: torch.Size([64, 1, 28, 28])
print(f"Labels batch shape: {train_labels.size()}")
>> Labels batch shape: torch.Size([64])
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
>> <matplotlib.image.AxesImage object at 0x000001CF9241E5E0>
plt.show()
```

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/unnamed-chunk-26-3.png" width="768" />

``` python
print(f"Label: {label}")
>> Label: 3
```

## 转化（Transforms）

大多情况下，数据要经过一定的预处理才可以作为模型的输入进行训练，这个过程就是 transform。下面以FashionMNIST 数据集的 transform 为例：

FashionMNIST 的特征是 PIL 格式的，而标签是整数，因此需要将特征转化为 tensor，标签转化为 one-hot 编码的 tensor，分别使用 torchvision.transforms 中的 `ToTensor` 和 `Lambda`：

``` python
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)
```

`ToTensor` 除了转化 tensor 外，还将图像的像素值缩放到 \[0,1\] 上；`Lambda` 函数可以用 lambda 匿名函数作为参数进行操作。

## 构建神经网络模型

神经网络是由许多层或者叫模块构成的，`torch.nn` 提供了所有构建神经网络所需的组件，在 Pytorch 中每一个模块都是 `nn.Module` 的子类，并且一个深度网络自身就是一个由其他模块组成的模块。

``` python
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

##查看是否有GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
>> Using cuda device
```

### 定义类

通过继承 `nn.Module` 来定义我们自己的神经网络，在 `__init__` 中初始化神经网络层，每一个 `nn.Module` 的亚类都需要实现对输入数据的 `forward` 方法：

``` python
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()##调用父类中的初始化
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )##神经网络架构

    def forward(self, x):##前向传播操作
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
      
```

接下来就需要创建一个 `NeuralNetwork` 实例，并将其移到 `device` 上（如果有GPU）：

``` python
model = NeuralNetwork().to(device)
print(model)
>> NeuralNetwork(
>>   (flatten): Flatten(start_dim=1, end_dim=-1)
>>   (linear_relu_stack): Sequential(
>>     (0): Linear(in_features=784, out_features=512, bias=True)
>>     (1): ReLU()
>>     (2): Linear(in_features=512, out_features=512, bias=True)
>>     (3): ReLU()
>>     (4): Linear(in_features=512, out_features=10, bias=True)
>>     (5): ReLU()
>>   )
>> )
```

当将数据传入模型时，模型会自动执行 `forword` 方法，上面的模型返回的是 1\*10 的 tensor，表示每类的原始预测值，需要使用 `nn.Softmax` 将其转化为概率值：

``` python
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits) ##在维度1 上使得所有值和为1
y_pred = pred_probab.argmax(dim=1)##在维度1上找到最大值的索引
print(f"Predicted class: {y_pred}")
>> Predicted class: tensor([7], device='cuda:0')
```

### 模型层

现在我们来看一下上面构建的模型的各个层，以大小为 3 的 minibatch 为例：

``` python
input_image = torch.rand(3,28,28)
print(input_image.size())
>> torch.Size([3, 28, 28])
```

首先是 `nn.Flatten` 层，将每一个二维的 28\*28 的图片转化为连续的 784 个像素值（batch 维度维持不变，也就是只对每个图片操作）：

``` python
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())
>> torch.Size([3, 784])
```

接着是线性层，对输入数据使用其存储的权重和偏置进行线性转化：

``` python
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())
>> torch.Size([3, 20])
```

然后是非线性的激活函数，这里使用的是 `nn.ReLu` :

``` python
print(f"Before ReLU: {hidden1}\n\n")
>> Before ReLU: tensor([[ 0.0523, -0.1083,  0.2858,  0.1203,  0.0551, -0.1314,  0.3096, -0.5887,
>>          -0.6137,  0.2250, -0.0138, -0.3449,  0.1049, -0.3507, -0.1467,  0.1690,
>>          -0.1686,  0.5031, -0.7191,  0.0630],
>>         [-0.0315, -0.2344,  0.2104,  0.0119,  0.3207,  0.1959,  0.2715, -0.3763,
>>          -0.6927,  0.1180,  0.0326, -0.0591,  0.1555, -0.3325, -0.0991,  0.0538,
>>          -0.0325,  0.0603, -0.6778,  0.1263],
>>         [ 0.0870, -0.0861,  0.1531,  0.1096,  0.1239,  0.2402, -0.0497, -0.3567,
>>          -0.4211,  0.0606, -0.0933, -0.2004, -0.1123, -0.1470, -0.1331, -0.2290,
>>          -0.2558,  0.2107, -0.5791,  0.3495]], grad_fn=<AddmmBackward>)
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")
>> After ReLU: tensor([[0.0523, 0.0000, 0.2858, 0.1203, 0.0551, 0.0000, 0.3096, 0.0000, 0.0000,
>>          0.2250, 0.0000, 0.0000, 0.1049, 0.0000, 0.0000, 0.1690, 0.0000, 0.5031,
>>          0.0000, 0.0630],
>>         [0.0000, 0.0000, 0.2104, 0.0119, 0.3207, 0.1959, 0.2715, 0.0000, 0.0000,
>>          0.1180, 0.0326, 0.0000, 0.1555, 0.0000, 0.0000, 0.0538, 0.0000, 0.0603,
>>          0.0000, 0.1263],
>>         [0.0870, 0.0000, 0.1531, 0.1096, 0.1239, 0.2402, 0.0000, 0.0000, 0.0000,
>>          0.0606, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.2107,
>>          0.0000, 0.3495]], grad_fn=<ReluBackward0>)
```

最后这些线性层和非线性层以一定的顺序被放在 `nn.Sequential` 容器中：

``` python
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)
```

### 模型参数

`nn.Module` 的子类会自动追踪在模型对象中定义的所有字段，并且可以使用 `parameters()` 和 `named_parameters()` 方法来获取模型的参数：

``` python
print("Model structure: ", model, "\n\n")
>> Model structure:  NeuralNetwork(
>>   (flatten): Flatten(start_dim=1, end_dim=-1)
>>   (linear_relu_stack): Sequential(
>>     (0): Linear(in_features=784, out_features=512, bias=True)
>>     (1): ReLU()
>>     (2): Linear(in_features=512, out_features=512, bias=True)
>>     (3): ReLU()
>>     (4): Linear(in_features=512, out_features=10, bias=True)
>>     (5): ReLU()
>>   )
>> )
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
>> Layer: linear_relu_stack.0.weight | Size: torch.Size([512, 784]) | Values : tensor([[-0.0164, -0.0271, -0.0350,  ...,  0.0217,  0.0249,  0.0046],
>>         [ 0.0234, -0.0130, -0.0324,  ...,  0.0176,  0.0005, -0.0079]],
>>        device='cuda:0', grad_fn=<SliceBackward>) 
>> 
>> Layer: linear_relu_stack.0.bias | Size: torch.Size([512]) | Values : tensor([-0.0350, -0.0063], device='cuda:0', grad_fn=<SliceBackward>) 
>> 
>> Layer: linear_relu_stack.2.weight | Size: torch.Size([512, 512]) | Values : tensor([[ 0.0141,  0.0015,  0.0391,  ..., -0.0240,  0.0044, -0.0037],
>>         [ 0.0396,  0.0321, -0.0239,  ..., -0.0277, -0.0196, -0.0199]],
>>        device='cuda:0', grad_fn=<SliceBackward>) 
>> 
>> Layer: linear_relu_stack.2.bias | Size: torch.Size([512]) | Values : tensor([0.0313, 0.0194], device='cuda:0', grad_fn=<SliceBackward>) 
>> 
>> Layer: linear_relu_stack.4.weight | Size: torch.Size([10, 512]) | Values : tensor([[ 0.0255, -0.0100,  0.0135,  ..., -0.0103, -0.0210,  0.0378],
>>         [-0.0066,  0.0378,  0.0037,  ...,  0.0146,  0.0090, -0.0166]],
>>        device='cuda:0', grad_fn=<SliceBackward>) 
>> 
>> Layer: linear_relu_stack.4.bias | Size: torch.Size([10]) | Values : tensor([0.0075, 0.0051], device='cuda:0', grad_fn=<SliceBackward>)
```

## 自动微分

Pytorch 内置的微分引擎是 `torch.autograd` ,支持任何计算图的梯度自动计算，下面是一个简单的单层神经网络的例子，输入是 `x`，输出是 `y`，参数是 `w` 和 `b`:

``` python
import torch

x = torch.ones(5)  # 输入
y = torch.zeros(3)  # 标签
w = torch.randn(5, 3, requires_grad=True)##权重
b = torch.randn(3, requires_grad=True)##偏置
z = torch.matmul(x, w)+b##前向传播
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)##损失函数
```

这个神经网络的计算图如下：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/comp-graph.png)

`w` 和 `b` 是我们需要优化的参数，所以要计算损失函数对这些参数的梯度，因此这些参数的 `requires_grad` 就要设置为 True（也可以在创建 tensor 后使用 `x.requires_grad_(True)` 来设置）。

在计算图中每个节点的计算（如加/减，乘等）的函数都是 `Funtion` 类的一个对象，这个对象知道如何进行前向运算以及在反向传播中如何计算导数（比如在反向传播中加法节点将上游的值原封不动地输出到下游，见[深度学习入门](https://wutaoblog.com.cn/2021/01/03/deep_learning/)），这些函数对象被存储在 tensor 的 `grad_fn` 属性中：

``` python
print('Gradient function for z =',z.grad_fn)
>> Gradient function for z = <AddBackward0 object at 0x000001CF917BE730>
print('Gradient function for loss =', loss.grad_fn)
>> Gradient function for loss = <BinaryCrossEntropyWithLogitsBackward object at 0x000001CF917BE7F0>
```

那么怎么计算损失函数对这些参数的梯度呢？在 Pytorch 中很简单：只需要调用`loss.backward` 就可以了：

``` python
loss.backward()
print(w.grad)
>> tensor([[0.3163, 0.3305, 0.0401],
>>         [0.3163, 0.3305, 0.0401],
>>         [0.3163, 0.3305, 0.0401],
>>         [0.3163, 0.3305, 0.0401],
>>         [0.3163, 0.3305, 0.0401]])
print(b.grad)
>> tensor([0.3163, 0.3305, 0.0401])
```

需要注意的是：在给定的计算图上只能使用 `backward` 进行一次梯度计算：

``` python
loss.backward()
>> Error in py_call_impl(callable, dots$args, dots$keywords): RuntimeError: Trying to backward through the graph a second time (or directly access saved variables after they have already been freed). Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved variables after calling backward.
```

如果想要在相同的计算图上多次调用 `backward`，就需要在第一次调用 `backward` 时加上 `retain_graph=True` 的选项：

``` python
x = torch.ones(5)  # 输入
y = torch.zeros(3)  # 标签
w = torch.randn(5, 3, requires_grad=True)##权重
b = torch.randn(3, requires_grad=True)##偏置
z = torch.matmul(x, w)+b##前向传播
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
loss.backward(retain_graph=True)
loss.backward(retain_graph=True)
```

另外，当第二次调用 `backward` 时，梯度会累加，因此要计算正确的梯度需要将之前的梯度清零（在实际训练模型中，`optimizer` 会帮我们干这件事）：

``` python
print(w.grad)
>> tensor([[0.5726, 0.2259, 0.6100],
>>         [0.5726, 0.2259, 0.6100],
>>         [0.5726, 0.2259, 0.6100],
>>         [0.5726, 0.2259, 0.6100],
>>         [0.5726, 0.2259, 0.6100]])
loss.backward(retain_graph=True)
print(w.grad)
>> tensor([[0.8589, 0.3389, 0.9150],
>>         [0.8589, 0.3389, 0.9150],
>>         [0.8589, 0.3389, 0.9150],
>>         [0.8589, 0.3389, 0.9150],
>>         [0.8589, 0.3389, 0.9150]])
```

设置了 `requires_grad=True` 的参数默认都会记录其计算图的路径并支持梯度的计算，但是有些时候我们并不想要这个属性，比如已经训练好了模型，想要将其应用到测试集上就只需要前向的计算就行了（不追踪梯度的前向运算更有效率）；可以将计算代码放到 `torch.no_grad()` 块中就会停止追踪计算：

``` python
z = torch.matmul(x, w)+b
print(z.requires_grad)
>> True
with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)
>> False
```

另一种选择是对 tensor 使用 `detach` 方法：

``` python
z = torch.matmul(x, w)+b
print(z.requires_grad)
>> True
z_det = z.detach()
print(z_det.requires_grad)
>> False
```

## 优化模型参数

通过前面的步骤，我们已经有了数据和模型，接下来就要通过在数据上优化参数来训练（train），验证（validate）和检测（test）。

### 超参数

超参数是可以控制模型优化过程的可调节的参数，超参数是需要人为调整的，不能从数据中自动学习得到，不同的超参数会影响模型训练和收敛速度（关于超参数调试见后面）；这里定义 3 个超参数：数据迭代的次数（epoch），批量大小（batch size）和学习率（learning rate）：

``` python
learning_rate = 1e-3
batch_size = 64
epochs = 5
```

### 优化循环

当我们设置好了超参数之后就可以通过一个循环来训练优化模型，每一个优化循环的迭代称为一个 epoch；每一个 epoch 都有两个步骤：

-   训练循环：对整个训练数据进行迭代，优化参数
-   验证步骤：在验证集上进行测试，检查模型性能是否提升

下面是这两个部分的代码实现：

``` python
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()##每次优化前要清零，不然梯度会累加
        loss.backward()##反向传播，计算梯度
        optimizer.step()##更新参数

        if batch % 100 == 0:##每100个batch打印一下信息
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()##每个batch的loss相加
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches##取平均
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```

由于这是个多分类问题，所以损失函数选择交叉熵损失：

``` python
# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()
```

优化器选择随机梯度下降（SGD）：`torch.optim.SGD`，需要提供要优化的模型参数和学习率：

``` python
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
```

接下来就可以训练了：

``` python
epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
>> Epoch 1
>> -------------------------------
>> loss: 2.306926  [    0/60000]
>> loss: 2.297123  [ 6400/60000]
>> loss: 2.291222  [12800/60000]
>> loss: 2.284291  [19200/60000]
>> loss: 2.266220  [25600/60000]
>> loss: 2.266951  [32000/60000]
>> loss: 2.247694  [38400/60000]
>> loss: 2.223377  [44800/60000]
>> loss: 2.203081  [51200/60000]
>> loss: 2.202767  [57600/60000]
>> Test Error: 
>>  Accuracy: 52.3%, Avg loss: 2.200312 
>> 
>> Epoch 2
>> -------------------------------
>> loss: 2.203640  [    0/60000]
>> loss: 2.207824  [ 6400/60000]
>> loss: 2.145190  [12800/60000]
>> loss: 2.144182  [19200/60000]
>> loss: 2.161016  [25600/60000]
>> loss: 2.118622  [32000/60000]
>> loss: 2.149072  [38400/60000]
>> loss: 2.105933  [44800/60000]
>> loss: 2.085976  [51200/60000]
>> loss: 1.978651  [57600/60000]
>> Test Error: 
>>  Accuracy: 53.3%, Avg loss: 2.048256 
>> 
>> Epoch 3
>> -------------------------------
>> loss: 2.045080  [    0/60000]
>> loss: 2.035074  [ 6400/60000]
>> loss: 1.953637  [12800/60000]
>> loss: 1.910037  [19200/60000]
>> loss: 2.020058  [25600/60000]
>> loss: 1.988134  [32000/60000]
>> loss: 2.021082  [38400/60000]
>> loss: 1.978815  [44800/60000]
>> loss: 1.801217  [51200/60000]
>> loss: 1.817578  [57600/60000]
>> Test Error: 
>>  Accuracy: 55.2%, Avg loss: 1.841357 
>> 
>> Epoch 4
>> -------------------------------
>> loss: 1.791114  [    0/60000]
>> loss: 1.774542  [ 6400/60000]
>> loss: 1.920177  [12800/60000]
>> loss: 1.805154  [19200/60000]
>> loss: 1.847489  [25600/60000]
>> loss: 1.698445  [32000/60000]
>> loss: 1.779692  [38400/60000]
>> loss: 1.583771  [44800/60000]
>> loss: 1.377952  [51200/60000]
>> loss: 1.661736  [57600/60000]
>> Test Error: 
>>  Accuracy: 57.9%, Avg loss: 1.632644 
>> 
>> Epoch 5
>> -------------------------------
>> loss: 1.743047  [    0/60000]
>> loss: 1.628604  [ 6400/60000]
>> loss: 1.772539  [12800/60000]
>> loss: 1.722733  [19200/60000]
>> loss: 1.706047  [25600/60000]
>> loss: 1.456557  [32000/60000]
>> loss: 1.356653  [38400/60000]
>> loss: 1.608426  [44800/60000]
>> loss: 1.525072  [51200/60000]
>> loss: 1.375649  [57600/60000]
>> Test Error: 
>>  Accuracy: 60.2%, Avg loss: 1.468750 
>> 
>> Epoch 6
>> -------------------------------
>> loss: 1.545837  [    0/60000]
>> loss: 1.597548  [ 6400/60000]
>> loss: 1.662136  [12800/60000]
>> loss: 1.390470  [19200/60000]
>> loss: 1.871670  [25600/60000]
>> loss: 1.244840  [32000/60000]
>> loss: 1.359863  [38400/60000]
>> loss: 1.307094  [44800/60000]
>> loss: 1.402576  [51200/60000]
>> loss: 1.375248  [57600/60000]
>> Test Error: 
>>  Accuracy: 60.9%, Avg loss: 1.345098 
>> 
>> Epoch 7
>> -------------------------------
>> loss: 1.422368  [    0/60000]
>> loss: 1.434490  [ 6400/60000]
>> loss: 1.286219  [12800/60000]
>> loss: 1.351996  [19200/60000]
>> loss: 1.357883  [25600/60000]
>> loss: 1.230724  [32000/60000]
>> loss: 1.410875  [38400/60000]
>> loss: 1.027635  [44800/60000]
>> loss: 1.359871  [51200/60000]
>> loss: 1.364859  [57600/60000]
>> Test Error: 
>>  Accuracy: 61.9%, Avg loss: 1.255917 
>> 
>> Epoch 8
>> -------------------------------
>> loss: 1.235713  [    0/60000]
>> loss: 1.044062  [ 6400/60000]
>> loss: 1.162858  [12800/60000]
>> loss: 1.220107  [19200/60000]
>> loss: 1.291631  [25600/60000]
>> loss: 0.966984  [32000/60000]
>> loss: 1.059310  [38400/60000]
>> loss: 1.355727  [44800/60000]
>> loss: 1.302162  [51200/60000]
>> loss: 1.444875  [57600/60000]
>> Test Error: 
>>  Accuracy: 63.0%, Avg loss: 1.195613 
>> 
>> Epoch 9
>> -------------------------------
>> loss: 0.986301  [    0/60000]
>> loss: 1.011468  [ 6400/60000]
>> loss: 1.181765  [12800/60000]
>> loss: 1.380456  [19200/60000]
>> loss: 1.152233  [25600/60000]
>> loss: 1.146567  [32000/60000]
>> loss: 1.255965  [38400/60000]
>> loss: 1.355518  [44800/60000]
>> loss: 1.049245  [51200/60000]
>> loss: 1.128320  [57600/60000]
>> Test Error: 
>>  Accuracy: 63.6%, Avg loss: 1.151816 
>> 
>> Epoch 10
>> -------------------------------
>> loss: 1.179832  [    0/60000]
>> loss: 1.270620  [ 6400/60000]
>> loss: 1.147824  [12800/60000]
>> loss: 1.163859  [19200/60000]
>> loss: 1.227105  [25600/60000]
>> loss: 1.204818  [32000/60000]
>> loss: 1.303981  [38400/60000]
>> loss: 1.032624  [44800/60000]
>> loss: 1.061971  [51200/60000]
>> loss: 1.084357  [57600/60000]
>> Test Error: 
>>  Accuracy: 64.3%, Avg loss: 1.116730
print("Done!")
>> Done!
```

## 储存和读取模型

第一种方法就是使用 `torch.save` 存储模型的参数（state_dict）：

``` python
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
>> Layer: linear_relu_stack.0.weight | Size: torch.Size([512, 784]) | Values : tensor([[-0.0164, -0.0271, -0.0350,  ...,  0.0217,  0.0249,  0.0046],
>>         [ 0.0234, -0.0130, -0.0324,  ...,  0.0173,  0.0004, -0.0079]],
>>        device='cuda:0', grad_fn=<SliceBackward>) 
>> 
>> Layer: linear_relu_stack.0.bias | Size: torch.Size([512]) | Values : tensor([-0.0350, -0.0040], device='cuda:0', grad_fn=<SliceBackward>) 
>> 
>> Layer: linear_relu_stack.2.weight | Size: torch.Size([512, 512]) | Values : tensor([[ 0.0141,  0.0018,  0.0393,  ..., -0.0243,  0.0044, -0.0037],
>>         [ 0.0396,  0.0318, -0.0239,  ..., -0.0278, -0.0201, -0.0202]],
>>        device='cuda:0', grad_fn=<SliceBackward>) 
>> 
>> Layer: linear_relu_stack.2.bias | Size: torch.Size([512]) | Values : tensor([0.0341, 0.0185], device='cuda:0', grad_fn=<SliceBackward>) 
>> 
>> Layer: linear_relu_stack.4.weight | Size: torch.Size([10, 512]) | Values : tensor([[ 0.0248, -0.0103,  0.0497,  ..., -0.0248, -0.0431,  0.0532],
>>         [-0.0075,  0.0374, -0.0252,  ...,  0.0573,  0.0476, -0.0564]],
>>        device='cuda:0', grad_fn=<SliceBackward>) 
>> 
>> Layer: linear_relu_stack.4.bias | Size: torch.Size([10]) | Values : tensor([0.0304, 0.0557], device='cuda:0', grad_fn=<SliceBackward>)
torch.save(model.state_dict(), 'model_weights.pth')
```

在载入模型的时候需要先创建一个相同模型的实例，然后在使用
`load_state_dict()` 方法载入模型的参数：

``` python
model = NeuralNetwork()
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
    
>> Layer: linear_relu_stack.0.weight | Size: torch.Size([512, 784]) | Values : tensor([[ 0.0238, -0.0081,  0.0160,  ..., -0.0083, -0.0010, -0.0024],
>>         [ 0.0100,  0.0321, -0.0217,  ...,  0.0226, -0.0264,  0.0076]],
>>        grad_fn=<SliceBackward>) 
>> 
>> Layer: linear_relu_stack.0.bias | Size: torch.Size([512]) | Values : tensor([-0.0054,  0.0092], grad_fn=<SliceBackward>) 
>> 
>> Layer: linear_relu_stack.2.weight | Size: torch.Size([512, 512]) | Values : tensor([[-0.0226,  0.0203, -0.0223,  ...,  0.0067, -0.0291, -0.0129],
>>         [-0.0428,  0.0410, -0.0325,  ...,  0.0424,  0.0010, -0.0367]],
>>        grad_fn=<SliceBackward>) 
>> 
>> Layer: linear_relu_stack.2.bias | Size: torch.Size([512]) | Values : tensor([ 0.0299, -0.0382], grad_fn=<SliceBackward>) 
>> 
>> Layer: linear_relu_stack.4.weight | Size: torch.Size([10, 512]) | Values : tensor([[ 0.0154, -0.0093,  0.0043,  ...,  0.0269,  0.0015,  0.0145],
>>         [-0.0316, -0.0201, -0.0111,  ..., -0.0192, -0.0140, -0.0011]],
>>        grad_fn=<SliceBackward>) 
>> 
>> Layer: linear_relu_stack.4.bias | Size: torch.Size([10]) | Values : tensor([-0.0373, -0.0201], grad_fn=<SliceBackward>)
            
model.load_state_dict(torch.load('model_weights.pth'))
>> <All keys matched successfully>

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
>> Layer: linear_relu_stack.0.weight | Size: torch.Size([512, 784]) | Values : tensor([[-0.0164, -0.0271, -0.0350,  ...,  0.0217,  0.0249,  0.0046],
>>         [ 0.0234, -0.0130, -0.0324,  ...,  0.0173,  0.0004, -0.0079]],
>>        grad_fn=<SliceBackward>) 
>> 
>> Layer: linear_relu_stack.0.bias | Size: torch.Size([512]) | Values : tensor([-0.0350, -0.0040], grad_fn=<SliceBackward>) 
>> 
>> Layer: linear_relu_stack.2.weight | Size: torch.Size([512, 512]) | Values : tensor([[ 0.0141,  0.0018,  0.0393,  ..., -0.0243,  0.0044, -0.0037],
>>         [ 0.0396,  0.0318, -0.0239,  ..., -0.0278, -0.0201, -0.0202]],
>>        grad_fn=<SliceBackward>) 
>> 
>> Layer: linear_relu_stack.2.bias | Size: torch.Size([512]) | Values : tensor([0.0341, 0.0185], grad_fn=<SliceBackward>) 
>> 
>> Layer: linear_relu_stack.4.weight | Size: torch.Size([10, 512]) | Values : tensor([[ 0.0248, -0.0103,  0.0497,  ..., -0.0248, -0.0431,  0.0532],
>>         [-0.0075,  0.0374, -0.0252,  ...,  0.0573,  0.0476, -0.0564]],
>>        grad_fn=<SliceBackward>) 
>> 
>> Layer: linear_relu_stack.4.bias | Size: torch.Size([10]) | Values : tensor([0.0304, 0.0557], grad_fn=<SliceBackward>)
```

注意：如果我们要在测试集进行计算，需要先 `model.eval()` 转换为评价模型而不是训练模式（在训练与测试中有些设置不一样，比如 dropout 是在训练中使用，防止模型过拟合，但是如果在评价中也使用 dropout，会造成结果的不稳定）。

另一种方法是直接使用 `torch.save` 和 `torch.load` 来存储和读入模型（使用了 python 的 pickle 模块）。

