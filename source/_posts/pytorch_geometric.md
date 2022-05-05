---
title: 利用 Pytorch geometric 构建图神经网络
date: 2022-04-21 19:14:18
tags: 深度学习
index_img: img/pyg2.png
categories:
  - 深度学习
---



Pytorch geometric 学习及实践

<!-- more -->

## 安装 Pytorch geometric

需要先安装 `pytorch` ，检查安装的版本：

```
import torch
import os
print("PyTorch has version {}".format(torch.__version__))
#PyTorch has version 1.10.1+cu102
```

根据 Pytorch 和 cuda 的版本选择相应的 `torch-scatter` 和 `torch-sparse` 包 （网络原因安装不了，因此手动下载 whl 文件安装）：

```python
!pip install torch_scatter-2.0.9-cp36-cp36m-linux_x86_64.whl #-f https://data.pyg.org/whl/torch-1.10.1+cu102.html
!pip install torch_sparse-0.6.12-cp36-cp36m-linux_x86_64.whl #-f https://data.pyg.org/whl/torch-1.10.1+cu102.html
!pip install torch-geometric
```

## 基础用法

### 图的数据操作

在 `PyG` 中一个图就是 `torch_geometric.data.Data` 对象的一个实例，该对象默认有下面的一些属性：

* `data.x` : 节点的特征矩阵，形状是 [节点的数量，节点特征的数量]
* `data.edge_index` : 以 COO 格式存储的图的连接性，形状是 [2，边的数量]，数据类型是 `torch.long` ，两个维度上相应的元素构成边连接的两个节点

下面以一个 3 个节点，4条边构成的无权重的无向图为例：

```python
import torch
from torch_geometric.data import Data

edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)
data
#Data(x=[3, 1], edge_index=[2, 4])
```

这个 `edge_index` 表示 0 和1有边相连，1和2有边相连，`x` 表示每个节点的特征，这里每个节点的特征只有一个元素，可以用下图来表示：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/graph-20220422211728-1te7ub8.svg)

注意除了使用这种在不同维度相应位置表示边的两个节点，也可以使用一个节点对构成的列表表示一个边，此时需要将上面的格式进行转置后再 `contiguous` 操作：

```python
data = Data(x=x, edge_index=edge_index.t().contiguous())
data 
#Data(x=[3, 1], edge_index=[4, 2])

edge_index.t().contiguous()
#tensor([[0, 1],
#        [1, 0],
#        [1, 2],
#        [2, 1]])
```

{% note warning %}
为什么要.contiguous()? 因为转置后的数据在内存中是不连续的，后续如果使用 view改变shape就会报错
{% endnote %}

`Data` 对象还提供了一些有用的函数：

```python
print(data.keys)## Data 和 python 中字典的行为类似，因此有键和值
#['x', 'edge_index']

print(data['x'])##相当于打印字典的值
#tensor([[-1.],
#        [ 0.],
#        [ 1.]])

for key, item in data:
    print(f'{key} found in data')##也可以像字典一样遍历
#x found in data
#edge_index found in data

'edge_attr' in data
#False

##一些图属性的函数
data.num_nodes
#3

data.num_edges
#4

data.num_node_features
#1

data.has_isolated_nodes()
#False

data.has_self_loops()
#False

data.is_directed()
#False
```

### 常用数据集

PyG 包含一些常用的图数据集，初始化一个数据集会自动下载数据并处理成上面所说的 `Data` 格式，比如下载 `ENZYMES` 数据集（包含 6个类的600 个图）：

```python
from torch_geometric.datasets import TUDataset
dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
```

可能由于这个数据在国外，由于网络的原因下不下来，因此我手动下载了这个数据（数据在[这里](https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/) 下载）然后建了个 Gitee 仓库存放这个数据，接着指定 `TUDataset` 的 URL，就可以成功下载并载入数据了 ：

```python
from torch_geometric.datasets import TUDataset
TUDataset.url = 'https://gitee.com/wt12318/graphkerneldatasets/raw/master/'
dataset = TUDataset(root='./', name='ENZYMES')

#Downloading https://gitee.com/wt12318/graphkerneldatasets/raw/master//ENZYMES.zip
#Extracting ./ENZYMES/ENZYMES.zip
#Processing...
#Done!
```

可以查看这个数据集的一些属性：

```python
len(dataset)##数据集大小
#600
dataset.num_classes##数据集类别
#6
dataset.num_node_features##节点特征维度
#3
```

可以直接用索引获取数据集中单个的图：

```python
data = dataset[0]
data
#Data(edge_index=[2, 168], x=[37, 3], y=[1])## 1 表示这个图的类别

##这个 data 就是 Data 对象了
data.is_undirected()
#True
data.num_edges
#168
data.num_nodes
#37
```

也可以使用切片来获取数据集中的多个图，这样我们就可以方便的划分训练集和测试集：

```python
train_dataset = dataset[:540]
test_dataset = dataset[540:]
train_dataset 
#ENZYMES(540)
test_dataset 
#ENZYMES(60)

##还可以进行随机打乱
dataset = dataset.shuffle()
##和下面操作一样
perm = torch.randperm(len(dataset))
dataset = dataset[perm]
```

这个数据集是用来对图进行分类任务的，因为每个图有一个标签，下面再看一个节点任务的数据集 `Cora` (半监督节点分类问题，数据在[这里](https://github.com/kimiyoung/planetoid)：

```python
from torch_geometric.datasets import Planetoid
Planetoid.url = "https://gitee.com/wt12318/graphkerneldatasets/raw/master/Planetoid/data/"
dataset = Planetoid(root='./', name='Cora')

Downloading https://gitee.com/wt12318/graphkerneldatasets/raw/master/Planetoid/data//ind.cora.x
Downloading https://gitee.com/wt12318/graphkerneldatasets/raw/master/Planetoid/data//ind.cora.tx
Downloading https://gitee.com/wt12318/graphkerneldatasets/raw/master/Planetoid/data//ind.cora.allx
Downloading https://gitee.com/wt12318/graphkerneldatasets/raw/master/Planetoid/data//ind.cora.y
Downloading https://gitee.com/wt12318/graphkerneldatasets/raw/master/Planetoid/data//ind.cora.ty
Downloading https://gitee.com/wt12318/graphkerneldatasets/raw/master/Planetoid/data//ind.cora.ally
Downloading https://gitee.com/wt12318/graphkerneldatasets/raw/master/Planetoid/data//ind.cora.graph
Downloading https://gitee.com/wt12318/graphkerneldatasets/raw/master/Planetoid/data//ind.cora.test.index
Processing...
Done!
```

查看数据集：

```python
len(dataset)
#1
dataset.num_classes
#7
dataset.num_node_features
#1433

##可以看到这个数据集只有一个图
data = dataset[0]
data
#Data(x=[2708, 1433], edge_index=[2, 10556], y=[2708], train_mask=[2708], val_mask=[2708], test_mask=[2708])

data.is_undirected()
#True
```

可以看到这个图是一个无向图，并且不像上面那个图，这个图对每个 x 都有一个 y，也就是节点层面的分类问题，这里多了  3 个 key：

* `train_mask` : 用来训练的节点
* `val_mask` ：用来验证的节点（进行 early stopping）
* `test_mask` ：用来测试的节点

这些都是布尔值的列表：

```python
data.train_mask
##tensor([ True,  True,  True,  ..., False, False, False])

##看看有多少个
data.train_mask.sum().item()
#140
data.val_mask.sum().item()
#500
data.test_mask.sum().item()
#1000
```

### 小批量

神经网络训练时一般是以批量作为单位，PyG 通过  `torch_geometric.loader.DataLoader` 将 `edge_index` ，特征和目标值连接在一起形成特定的批量大小：

```python
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
TUDataset.url = 'https://gitee.com/wt12318/graphkerneldatasets/raw/master/'
dataset = TUDataset(root='./', name='ENZYMES',use_node_attr=True)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in loader:
    print(batch)

DataBatch(edge_index=[2, 3590], x=[904, 21], y=[32], batch=[904], ptr=[33])
DataBatch(edge_index=[2, 4218], x=[1081, 21], y=[32], batch=[1081], ptr=[33])
DataBatch(edge_index=[2, 3810], x=[962, 21], y=[32], batch=[962], ptr=[33])
DataBatch(edge_index=[2, 4276], x=[1221, 21], y=[32], batch=[1221], ptr=[33])
DataBatch(edge_index=[2, 3714], x=[952, 21], y=[32], batch=[952], ptr=[33])
DataBatch(edge_index=[2, 3670], x=[933, 21], y=[32], batch=[933], ptr=[33])
DataBatch(edge_index=[2, 4142], x=[1133, 21], y=[32], batch=[1133], ptr=[33])
DataBatch(edge_index=[2, 4180], x=[1058, 21], y=[32], batch=[1058], ptr=[33])
DataBatch(edge_index=[2, 3902], x=[1033, 21], y=[32], batch=[1033], ptr=[33])
DataBatch(edge_index=[2, 4004], x=[1084, 21], y=[32], batch=[1084], ptr=[33])
DataBatch(edge_index=[2, 4136], x=[1115, 21], y=[32], batch=[1115], ptr=[33])
DataBatch(edge_index=[2, 3710], x=[938, 21], y=[32], batch=[938], ptr=[33])
DataBatch(edge_index=[2, 3764], x=[1036, 21], y=[32], batch=[1036], ptr=[33])
DataBatch(edge_index=[2, 3842], x=[982, 21], y=[32], batch=[982], ptr=[33])
DataBatch(edge_index=[2, 4088], x=[1052, 21], y=[32], batch=[1052], ptr=[33])
DataBatch(edge_index=[2, 4342], x=[1246, 21], y=[32], batch=[1246], ptr=[33])
DataBatch(edge_index=[2, 4196], x=[1080, 21], y=[32], batch=[1080], ptr=[33])
DataBatch(edge_index=[2, 3974], x=[1014, 21], y=[32], batch=[1014], ptr=[33])
DataBatch(edge_index=[2, 3006], x=[756, 21], y=[24], batch=[756], ptr=[25])
```

`torch_geometric.data.Batch` 对象继承自 `Data` 对象，并且多了一个 `batch`  的属性，`batch` 是一个向量，和该批量中节点的大小是一致的，每一个元素表示相应的节点属于该批量中哪个图：

```python
batch["batch"]##最后一个批量大小为24，也就是有24个图

tensor([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
         2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  3,  3,  3,
         3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
         3,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,
         4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,
         4,  4,  4,  4,  4,  4,  4,  4,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,
         5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,
         5,  5,  5,  5,  5,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,
         6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,
         6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  7,  7,  7,  7,  7,  7,  7,
         7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,
         7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  8,  8,  8,  8,  8,  8,
         8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,
         8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,  8,
         8,  8,  8,  8,  8,  8,  8,  8,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,
         9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9, 10,
        10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
        10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11,
        11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
        11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
        12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
        12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13,
        13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14,
        14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
        14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
        15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
        16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17,
        17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18,
        18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18,
        18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
        19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,
        19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20,
        20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
        20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 21,
        21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
        21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21,
        21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 22,
        22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23,
        23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23])
```

比如我们可以根据这个特征来计算每个图中每个特征的均值：

```python
from torch_scatter import scatter_mean
x = scatter_mean(batch.x, batch.batch, dim=0)
x.size()
#torch.Size([24, 21])
```

`scatter_mean` 就是对 batch 中相同的值（也就是每个图）计算 x 在维度 0 上的均值（也就是每个特征对所有节点的均值）。

### GNN

以 Cora 引用数据集作为例子，实现一个2层的图卷积神经网络（GCN）：

```python
from torch_geometric.datasets import Planetoid

##载入数据
Planetoid.url = "https://gitee.com/wt12318/graphkerneldatasets/raw/master/Planetoid/data/"
dataset = Planetoid(root='./', name='Cora')

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

##构建 GCN
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
```

注意 GCN 和 CNN 不同，层数和表达能力并不一定相关，GCN 的层数只是指收集离节点多远的邻居节点的信息。并且图神经网络最终得到的是节点的 embedding，因此在构建GCN 的时候变化的 embedding 的维度（`num_node_features` , 16, `num_classes`），输入的是节点以及图的结构（`edge_index`）。接下来进行训练 200 个 epochs：

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
```

在测试节点上进行评估模型：

```python
model.eval()
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')
##Accuracy: 0.7970
```

## 创建信息传播网络

网络中的信息传播可以类比于图像上的卷积操作，都是收集邻居节点（像素）的信息：
$$
x_i^{(k)} = \gamma^{(k)} ( x_i^{(k-1)}, \square_{j \in N(i)} \, \phi^{(k)}(x_i^{(k-1)}, x_j^{(k-1)},e_{j,i}) ),
$$
$\square$ 表示可微的置换不变函数（也就是打乱元素的次序不改变函数的输出，比如 sum，mean，max 等），$\gamma$ 和 $\phi$ 都表示可微的函数（比如用多层感知机拟合的函数）。

### MessagePassing 基础类

PyG 提供了 `MessagePassing` 基础类，通过自动的进行信息传播从而构建神经网络；用户只需要定义函数 $\phi$，也就是 `message()`，$\gamma$，也就是 `update()`，和聚合的方法，也就是上面的 $\square$，可以是：`aggr="add"`, `aggr="mean"` 或者 `aggr="max"`。这个过程在下面一些方法的帮助下完成：

- `MessagePassing(aggr="add", flow="source_to_target", node_dim=-2)`：定义了聚合的方式（add，mean，max）和信息流动的方向（source_to_target 或者 target_to_source，中心节点也就是需要 embedding 的节点是 target，邻居节点为 source）以及 node_dim 为信息聚合的维度
- `MessagePassing.propagate(edge_index, size=None, **kwargs)`：信息传播，在内部依次调用 `message`，`aggregate` 和 `update`
- `MessagePassing.message(...)`：相当于 $\phi$ 函数，注意传递给 `propagate` 的张量会先先添加 i 和 j 的下标转化成相应的节点然后再传递给 `message` （i 表示中心节点，j 表示邻居节点）
- `MessagePassing.update(aggr_out, ...)`：相当于 $\gamma$ ，接受 `aggregation` 的输出和其他传给 `propagate` 的参数

下面看两个例子：

#### GCN layer

GCN 层可以定义为：
$$
x_i^{(k)} = \sum_{j \in N(i) \cup \{ i \}} \frac{1}{\sqrt{\deg(i)} \cdot \sqrt{\deg(j)}} \cdot ( \Theta^{\top} \cdot x_j^{(k-1)} ),
$$
邻居节点首先由 $\Theta$ 转化，然后根据度进行标准化，接着使用求和作为汇聚函数，可以分成一下几个步骤：

- 对邻接矩阵添加自身的边（source 和 target 都是自己，因为上面的式子中 j 包括了 $N(i)\cup{i}$）
- 对节点的特征矩阵进行线性转化
- 计算标准化系数
- 依据标准化系数来标准化节点特征
- 求和汇聚

1-3 步是信息传播之前就可以计算好的：

```python
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index##将edge_index 拆分成 source (col) 和 target (row)
        deg = degree(col, x.size(0), dtype=x.dtype)##x.size(0)就是节点的数量
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        #print(x.shape)
        #print(x)
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        #print(x_j.shape)
        #print(x_j)
        return norm.view(-1, 1) * x_j
```

```python
edge_index = dataset[0]["edge_index"]
edge_index
tensor([[   0,    0,    0,  ..., 2707, 2707, 2707],
        [ 633, 1862, 2582,  ...,  598, 1473, 2706]])
        
row, col = edge_index
row,col
(tensor([   0,    0,    0,  ..., 2707, 2707, 2707]),
 tensor([ 633, 1862, 2582,  ...,  598, 1473, 2706]))
```

需要注意的是我们传入 `propagate` 的是 x ，也就是所有节点的 embedding，但是在其内部进行了一个 `lifted` 操作转化成了 `lifted` 张量 `x_j` ，这个张量包括每个节点的邻居节点的 embedding，便于进行信息传递，因此我们运行上面加了 `print` 的代码：

```python
conv = GCNConv(1433, 32)##实例化
x = conv(dataset[0]["x"], dataset[0]["edge_index"])
torch.Size([2708, 32])##图中有 2708 个节点
tensor([[ 0.0352, -0.0465,  0.0304,  ..., -0.0498,  0.0052, -0.0351],
        [ 0.0118, -0.0057, -0.0229,  ..., -0.0921,  0.0561, -0.0387],
        [-0.1651, -0.0569, -0.0675,  ...,  0.0006,  0.0174, -0.1239],
        ...,
        [ 0.0200,  0.0012, -0.0240,  ..., -0.0073,  0.0137, -0.0065],
        [ 0.0939,  0.0371, -0.0670,  ...,  0.0017, -0.0460,  0.0491],
        [-0.0430, -0.0073,  0.0501,  ..., -0.0168,  0.0014, -0.0537]],
       grad_fn=<AddmmBackward0>)
torch.Size([13264, 32])
tensor([[ 0.0352, -0.0465,  0.0304,  ..., -0.0498,  0.0052, -0.0351],
        [ 0.0352, -0.0465,  0.0304,  ..., -0.0498,  0.0052, -0.0351],
        [ 0.0352, -0.0465,  0.0304,  ..., -0.0498,  0.0052, -0.0351],
        ...,
        [ 0.0200,  0.0012, -0.0240,  ..., -0.0073,  0.0137, -0.0065],
        [ 0.0939,  0.0371, -0.0670,  ...,  0.0017, -0.0460,  0.0491],
        [-0.0430, -0.0073,  0.0501,  ..., -0.0168,  0.0014, -0.0537]],
       grad_fn=<IndexSelectBackward0>)
```

#### Edge Convolution

EdgeConv 的特点就是在每一层中节点的邻居都可能变化（动态图），也就是每一层都根据上一层更新的节点 embedding 利用 KNN 计算节点的邻居（embedding 距离而不是实际图的连接），除了这个动态图结构之外，EdgeConv 的信息传播和汇聚可以表示为：
$$
x_i^{(k)} = \max_{j \in {N}(i)} h_{\Theta} ( x_i^{(k-1)}, x_j^{(k-1)} - x_i^{(k-1)} )
$$
$h_{\mathbf{\Theta}}$ 表示 MLP，汇聚函数使用 max：

```python
import torch
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing

class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='max') #  "Max" aggregation.
        self.mlp = Seq(Linear(2 * in_channels, out_channels),
                       ReLU(),
                       Linear(out_channels, out_channels))

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]

        tmp = torch.cat([x_i, x_j - x_i], dim=1)  # tmp has shape [E, 2 * in_channels]
        return self.mlp(tmp)
```

上面讲过，EdgeConv 在每层都会利用 KNN 来选择邻居节点：

```python
from torch_geometric.nn import knn_graph

class DynamicEdgeConv(EdgeConv):##继承上面的类
    def __init__(self, in_channels, out_channels, k=6):
        super().__init__(in_channels, out_channels)
        self.k = k

    def forward(self, x, batch=None):
        edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)##选择邻居节点
        return super().forward(x, edge_index)##调用EdgeConv 的forward
```













参考：

- **Pytorch geometric** [文档](https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html)
- [GNN Project YouTube](https://www.youtube.com/watch?v=QLIkOtKS4os&list=PLV8yxwGOxvvoNkzPfCx2i8an--Tkt7O8Z&index=8)
- [Google Colab](https://colab.research.google.com/drive/1DIQm9rOx2mT1bZETEeVUThxcrP1RKqAn)
- [AntonioLonga/PytorchGeometricTutorial: Pytorch Geometric Tutorials (github.com)](https://github.com/AntonioLonga/PytorchGeometricTutorial)