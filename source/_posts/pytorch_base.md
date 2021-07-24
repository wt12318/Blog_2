---
title: 【Pytorch】Tensor 的基本操作
date: 2021-06-23 20:59:00
tags: 编程
categories:
  - DL
index_img: img/pytorc.jpg
---

Tensor 的基本操作和自动求导

<!-- more -->

对tensor的操作以作用的对象分有两类：

-   troch function 以torch开头：torch.sum
-   tensor function 以tensor开头：tensor.view

按照修改方式也可以分成两类：

-   不修改自身数据，返回新的tensor
-   修改自身数据，一般是带下划线的操作，如x.add\_(y)

``` python
import torch
a = torch.tensor([1,2])
b = a.add(2)
print(a,b)
>> tensor([1, 2]) tensor([3, 4])
c = a.add_(2)
print(a,c)
>> tensor([3, 4]) tensor([3, 4])
```

## 基本操作

tensor的构建：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/Untitled.png)

``` python
torch.Tensor([1,2,3])
##根据指定的形状
>> tensor([1., 2., 3.])
torch.Tensor(2,3)##注意是随机生成的
>> tensor([[0.0000, 1.8750, 0.0000],
>>         [2.0000, 0.0000, 2.1250]])
torch.Tensor(2,3)
##和小写的tensor的区别：
>> tensor([[0.0000, 1.9844, 0.0000],
>>         [2.0000, 0.0000, 2.0156]])
torch.Tensor([2,3]).type()
>> 'torch.FloatTensor'
torch.tensor([2,3]).type()
>> 'torch.LongTensor'
torch.tensor([2.,3.]).type()
##也就是说tensor是推断类型，而Tensor是默认FloatTensor

##特殊类型的tensor
>> 'torch.FloatTensor'
torch.eye(2,2)##单位矩阵
>> tensor([[1., 0.],
>>         [0., 1.]])
torch.zeros(2,3)##全为0
>> tensor([[0., 0., 0.],
>>         [0., 0., 0.]])
torch.linspace(1,10,4)##和numpy类似，从1到10分成4份
>> tensor([ 1.,  4.,  7., 10.])
torch.rand(2,3)##均匀分布
>> tensor([[0.2096, 0.3077, 0.3466],
>>         [0.4751, 0.9939, 0.1684]])
torch.randn(2,3)##标准正态分布
>> tensor([[ 0.2854, -1.6030,  0.3589],
>>         [ 0.0345, -0.1393,  0.7295]])
torch.zeros_like(torch.rand(2,3))##形状相同，但全为0
>> tensor([[0., 0., 0.],
>>         [0., 0., 0.]])
```

修改 Tensor 的形状：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/b78a5eb320ca85865498ef331663acd6.png)

``` python
x = torch.rand(2,3)
x
>> tensor([[0.6279, 0.2673, 0.6451],
>>         [0.9874, 0.3448, 0.3950]])
x.shape
>> torch.Size([2, 3])
x.size() ##返回shape属性
>> torch.Size([2, 3])
x.dim()
>> 2
x.numel()
>> 6
y = torch.unsqueeze(x,0)##增加一个维度
y.size()
>> torch.Size([1, 2, 3])
y.numel()
>> 6
```

这里面需要注意的是 `view()` 和 `reshape()` 的区别：`view()` 只能作用于连续的 tensor，并且只会返回视图（也就是改变返回的数据，原数据也会改变），而 `reshape()` 则可以作用于连续和非连续的 tensor，对于连续的tensor 返回视图，对于非连续的 tensor 则返回拷贝：

``` python
#view
##连续的
t1 = torch.rand(2,3)
t2 = t1.view(3,2)
t2[0,1] = 0
t1,t2

##非连续的使用 view 会报错
>> (tensor([[0.7285, 0.0000, 0.8832],
>>         [0.3564, 0.8119, 0.9175]]), tensor([[0.7285, 0.0000],
>>         [0.8832, 0.3564],
>>         [0.8119, 0.9175]]))
t3 = t1.t()
t3
>> tensor([[0.7285, 0.3564],
>>         [0.0000, 0.8119],
>>         [0.8832, 0.9175]])
t3.view(2,3)

#reshape
##连续的
>> Error in py_call_impl(callable, dots$args, dots$keywords): RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
t1 = torch.rand(3,4)
t2 = t1.reshape(2,6)
t2[0,1] = 0
t2,t1

##非连续的
>> (tensor([[0.2901, 0.0000, 0.1992, 0.0292, 0.2516, 0.5446],
>>         [0.9478, 0.1281, 0.3004, 0.1900, 0.8300, 0.4039]]), tensor([[0.2901, 0.0000, 0.1992, 0.0292],
>>         [0.2516, 0.5446, 0.9478, 0.1281],
>>         [0.3004, 0.1900, 0.8300, 0.4039]]))
t3 = t1.t()
t4 = t3.reshape(2,6)
t4[1,1] = 0
t4,t1##返回的是copy
>> (tensor([[0.2901, 0.2516, 0.3004, 0.0000, 0.5446, 0.1900],
>>         [0.1992, 0.0000, 0.8300, 0.0292, 0.1281, 0.4039]]), tensor([[0.2901, 0.0000, 0.1992, 0.0292],
>>         [0.2516, 0.5446, 0.9478, 0.1281],
>>         [0.3004, 0.1900, 0.8300, 0.4039]]))
```

索引操作：和 numpy 类似，也有一些专用的函数：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/ca5b6ebfbff3968e030d3c7b92c445f0.png)

``` python
x = torch.randn(2,3)
x
>> tensor([[-0.2166,  0.3223, -0.5408],
>>         [-1.4433,  0.6585, -0.2060]])
x[0,]##和 x[0,:] 一样
>> tensor([-0.2166,  0.3223, -0.5408])
x[:,-1] #这个冒号不能省
>> tensor([-0.5408, -0.2060])
torch.masked_select(x,x>0)#根据第二个参数来选择
>> tensor([0.3223, 0.6585])
torch.nonzero(x)#非0元素的索引
>> tensor([[0, 0],
>>         [0, 1],
>>         [0, 2],
>>         [1, 0],
>>         [1, 1],
>>         [1, 2]])
index = torch.LongTensor([[0,1,1]])
index.type()
>> 'torch.LongTensor'
torch.gather(x,0,index)
>> tensor([[-0.2166,  0.6585, -0.2060]])
index = torch.LongTensor([[0,1,1],[1,1,1]])
torch.gather(x,1,index)

##scatter 补充元素
>> tensor([[-0.2166,  0.3223,  0.3223],
>>         [ 0.6585,  0.6585,  0.6585]])
a = torch.gather(x,1,index)
y = torch.zeros(2,3)
y.scatter_(1,index,a)
>> tensor([[-0.2166,  0.3223,  0.0000],
>>         [ 0.0000,  0.6585,  0.0000]])
```

广播机制：

``` python
import numpy as np
a = np.arange(0, 40, 10).reshape(4, 1)
b = np.arange(0, 3)
 
ta = torch.from_numpy(a) ##4 * 1
tb = torch.from_numpy(b) ## 3

ta.size()
>> torch.Size([4, 1])
tb.size()
>> torch.Size([3])
ta+tb
>> tensor([[ 0,  1,  2],
>>         [10, 11, 12],
>>         [20, 21, 22],
>>         [30, 31, 32]], dtype=torch.int32)
```

按元素操作：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/3fbded1e5a9c9c5330566b41cf6f20fd.png)
上面这些操作都是创建新的 tensor，可以使用加上下划线的版本来创建视图。

``` python
t1 = torch.randn(2,3)
t2 = torch.randn(2,3)
t1,t2
>> (tensor([[ 1.8564,  0.2339, -0.6054],
>>         [-1.0644,  0.1839, -0.2998]]), tensor([[ 0.6240, -1.2900, -1.1661],
>>         [-0.3400,  0.8373,  1.2089]]))
torch.mul(torch.abs(t1),torch.abs(t2))
>> tensor([[1.1584, 0.3017, 0.7059],
>>         [0.3619, 0.1540, 0.3624]])
a = torch.abs(t1)
b = torch.abs_(t1)

a[0,1] = 0
b[1,1] = 0

a
>> tensor([[1.8564, 0.0000, 0.6054],
>>         [1.0644, 0.1839, 0.2998]])
b
>> tensor([[1.8564, 0.2339, 0.6054],
>>         [1.0644, 0.0000, 0.2998]])
t1
>> tensor([[1.8564, 0.2339, 0.6054],
>>         [1.0644, 0.0000, 0.2998]])
```

归并操作：

可以按照某个轴或者整个 tensor 进行，一般有两个参数：dim 参数指定计算的维度，keepdim 参数指输出结果中是否保留维度 1

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/625f6b4e0a1446f77392fc93a847f5a1.png)

``` python
a = torch.linspace(0,10,6)
a = a.view(2,3)
a.sum(dim=0)##0轴，也就是列
>> tensor([ 6., 10., 14.])
a.sum(dim=0).shape
>> torch.Size([3])
a.sum(dim=0, keepdim=True)
>> tensor([[ 6., 10., 14.]])
a.sum(dim=0, keepdim=True).shape
>> torch.Size([1, 3])
```

比较操作：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/c8d86d5a90943e7be3ecc22885bb28f0.png)

``` python
x = torch.linspace(0,10,6).reshape(2,3)
x
>> tensor([[ 0.,  2.,  4.],
>>         [ 6.,  8., 10.]])
torch.max(x)
>> tensor(10.)
torch.max(x,dim=0) ##返回每列的最大值以及其index
>> torch.return_types.max(
>> values=tensor([ 6.,  8., 10.]),
>> indices=tensor([1, 1, 1]))
torch.topk(x,2,dim=0)#求最大的两个值
>> torch.return_types.topk(
>> values=tensor([[ 6.,  8., 10.],
>>         [ 0.,  2.,  4.]]),
>> indices=tensor([[1, 1, 1],
>>         [0, 0, 0]]))
y = torch.tensor([6,8,10])
torch.eq(x,y)##支持广播
>> tensor([[False, False, False],
>>         [ True,  True,  True]])
torch.equal(x,y)###数据结构要一致
>> Error in py_call_impl(callable, dots$args, dots$keywords): RuntimeError: Expected object of scalar type float but got scalar type __int64 for argument 'other'
y = y.float()
torch.equal(x,y)
>> False
```

矩阵操作：

主要有两种：按元素相乘和点积

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/883ab00d77e328c75eae26fc46abf6ad.png)

注意：在 torch 中 dot 只能对向量（一维张量）做点积运算（numpy 中没有限制）；另外 torch 中矩阵按元素相乘是用 `mul`（numpy 中是 multiply）；转置会使 tensor 不连续，需要使用 `contiguous` 方法转为连续

``` python
a = torch.rand(2,3)
b = torch.tensor([2,3])
c = torch.rand(2,3)
d = torch.rand(3,2)

a,b,c
>> (tensor([[0.5224, 0.5259, 0.5960],
>>         [0.6903, 0.5222, 0.6591]]), tensor([2, 3]), tensor([[0.7300, 0.9068, 0.1036],
>>         [0.3268, 0.5661, 0.0630]]))
torch.dot(b,b)
>> tensor(13)
torch.dot(a,c)##只能一维
>> Error in py_call_impl(callable, dots$args, dots$keywords): RuntimeError: 1D tensors expected, but got 2D and 2D tensors
torch.mul(a,c)
>> tensor([[0.3814, 0.4769, 0.0617],
>>         [0.2256, 0.2956, 0.0415]])
torch.mm(a,c)##维度要对应
>> Error in py_call_impl(callable, dots$args, dots$keywords): RuntimeError: mat1 and mat2 shapes cannot be multiplied (2x3 and 2x3)
torch.mm(a,d)
>> tensor([[0.9200, 0.7083],
>>         [1.0955, 0.8583]])
x = torch.randint(10,(2,2,3))
y = torch.randint(10,(2,3,4))
x,y
>> (tensor([[[0, 4, 4],
>>          [4, 0, 5]],
>> 
>>         [[7, 0, 7],
>>          [2, 1, 4]]]), tensor([[[3, 8, 6, 3],
>>          [5, 2, 4, 1],
>>          [6, 2, 5, 7]],
>> 
>>         [[2, 6, 5, 5],
>>          [9, 4, 7, 3],
>>          [2, 5, 2, 7]]]))
torch.bmm(x,y)##必须是3维
>> tensor([[[44, 16, 36, 32],
>>          [42, 42, 49, 47]],
>> 
>>         [[28, 77, 49, 84],
>>          [21, 36, 25, 41]]])
```

pytorch 与 numpy 的比较：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/995d56a6c02fd906dcbd71e3e09c703b.png)

## 自动求导

自动求导的核心是计算图，通过将复杂的函数拆分成计算图中每个节点处的简单运算，并在节点处保留简单运算的梯度，然后通过反向传播运用链式法则求得复杂函数的梯度（计算图的概念见 [深度学习入门](https://wutaoblog.com.cn/2021/01/03/deep_learning/)。

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/74d0cafe66686492ce6228e5fa0f81fc.png)

图中 x，w，b 为叶子节点（叶子节点的张量参数 requires_grad 需要设置为True），z 为根节点

``` python
import torch
 
 
#定义输入张量x
x=torch.Tensor([2])
#初始化权重参数W,偏移量b、并设置require_grad属性为True，为自动求导
w=torch.randn(1,requires_grad=True)
b=torch.randn(1,requires_grad=True) 
#实现前向传播
y=torch.mul(w,x)  #等价于w*x
z=torch.add(y,b)  #等价于y+b
#查看x,w，b页子节点的requite_grad属性
x.requires_grad,w.requires_grad,b.requires_grad
>> (False, True, True)
z
>> tensor([1.0761], grad_fn=<AddBackward0>)
z.backward()
print(w.is_leaf,x.is_leaf,b.is_leaf,y.is_leaf,z.is_leaf)##查看是否为叶子节点
>> True True True False False
print(w.grad,b.grad,x.grad,y.grad)
>> tensor([2.]) tensor([1.]) None None
>> 
>> <string>:1: UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the gradient for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more informations.
```

在反向传播结束之后，非叶子节点的梯度会被释放掉（以及没有设置requires_grad 的叶子节点），可以通过 `retain_grad()` 来保留任意节点的梯度：

``` python
y=torch.mul(w,x)  
y.retain_grad()
z=torch.add(y,b)

z,z.backward()
>> (tensor([1.0761], grad_fn=<AddBackward0>), None)
print(w.grad,b.grad,x.grad,y.grad)
>> tensor([4.]) tensor([2.]) None tensor([1.])
```

tensor 还有一个 `grad_fn` 属性，来记录创建该 tensor 时所使用的函数类型，因为在计算图中可以根据不同的计算类型来使用不同的求导法则（比如加法节点的求导不改变输入信号，直接输出；而乘法节点则需要输入的两个乘子互换作为梯度的输出等）：

``` python
print(y.grad_fn,z.grad_fn)
>> <MulBackward0 object at 0x000001DE3BD8F250> <AddBackward0 object at 0x000001DE40B1F550>
```

上面讲到的是标量对标量求导，也可以标量对张量求导，但是不能直接张量对张量求导:

``` python
x=torch.Tensor([2,3])##张量
w=torch.randn(2,requires_grad=True)
b=torch.randn(2,requires_grad=True) 

y=torch.mul(w,x)  
z=torch.add(y,b)

k = z.mean()
k
>> tensor(4.9734, grad_fn=<MeanBackward0>)
k.backward()##标量对张量

z
>> tensor([5.9739, 3.9728], grad_fn=<AddBackward0>)
z.backward()##非标量张量对非标量张量求导会报错
>> Error in py_call_impl(callable, dots$args, dots$keywords): RuntimeError: grad can be implicitly created only for scalar outputs
```

张量对张量求导，需要在 `backward()` 中加上 `gradient` 参数，该参数的维数要和目标张量维数一致，然后对目标张量和该参数进行加权求和（按元素相乘再相加）得到一个标量后再进行求导：

``` python
import torch

#定义叶子节点张量x，形状为1x2
x= torch.tensor([[2, 3]], dtype=torch.float, requires_grad=True)

y = x*3
z = torch.Tensor([[1,1]])
y.backward(gradient=z)
x.grad
>> tensor([[3., 3.]])
```
