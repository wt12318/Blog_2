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

![](C:\Users\wutao\Downloads\Pytorch geometric\Pytorch geometric\深度学习\笔记\图神经网络\assets\graph-20220422211728-1te7ub8.svg)

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





参考：

- **Pytorch geometric** [文档](https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html)
- [GNN Project YouTube](https://www.youtube.com/watch?v=QLIkOtKS4os&list=PLV8yxwGOxvvoNkzPfCx2i8an--Tkt7O8Z&index=8)
- [Google Colab](https://colab.research.google.com/drive/1DIQm9rOx2mT1bZETEeVUThxcrP1RKqAn)