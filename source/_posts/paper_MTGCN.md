---
title: 使用多任务图卷积神经网络改进癌症驱动基因识别
date: 2022-05-07 19:14:18
tags: paper
index_img: img/BIF.jpg
categories:
  - 深度学习
---

论文阅读：Improving cancer driver gene identification using multi-task learning on graph convolutional network

<!-- more -->

预测癌症驱动基因传统的方法一般是基于突变频率的方法：先计算背景突变率，然后再比较某些位点的突变频率是否显著高于背景突变率，这些方法存在的问题：1. 不能够准确的估计背景突变率，2.有一些 driver 突变有着低的突变率，这种方法就不能发现。

一些研究发现 driver 基因通常在共同的通路或者蛋白复合体中发挥作用，将这些通路或者蛋白复合体的亚单元破坏就会导致癌症表型，因此可以利用 PPI （蛋白互作）网络来帮助寻找癌基因，但是单纯 PPI 网络的可信度并不高（可能某些研究比较多的基因在网络中的度就比较高），因此还需要结合其他的特征，比如基因表达，表观状态等（即整合多组学的信息）。这篇文献的主要特点在于：

* 增加了基因相关的特征
* 构建了多任务的图卷积网络学习框架来鉴别 driver 基因，多任务包括节点预测和边预测
* 在 GCN 中增加了 skip connection （类似残差连接）
* 使用贝叶斯任务权重学习器来学习调整多任务中的不同任务间的权重
* 在训练中使用 drop out 技术（drop out 一些 PPI 网络中的边）

数据：

* PPI 网络 ：数据来自 Consensus Path DB，移除互作分数小于 0.5 的边，还剩 13627 个节点和 504378 条边
* TCGA 的基因突变，DNA 甲基化和基因表达数据

 对于每种癌症计算 3 种基因特征：

* 基因突变率：一个癌症类型中所有样本某个基因中发生的 SNV 和 CNA 的平均数量
* 差异 DNA 甲基化率：一个癌症类型中所有样本某个基因在癌症样本和正常样本的甲基化信号差异的均值
* 差异基因表达率：一个癌症类型中所有样本某个基因在正常样本和癌症样本中表达值的 log2 Fold change 的均值

最后将每个基因 16 种癌症的 3 个特征合并，并进行最小最大标准化，因此每个基因有 16 * 3 = 48 个特征。正例样本来自 NCG（包括 pan-cancer 和 癌症类型特异的），负例样本是从所有基因中剔除在 NCG, COSMIC, Online Mendelian Inheritance in Man (OMIM) database , Kyoto Encyclopedia of Genes and Genomes (KEGG) cancer pathway 中的基因，一共正例样本 796 个，负例样本 2187 个。

### 方法

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220507183147-humnmrk.png" style="zoom:67%;" />

每个基因的特征包括上面提到的 48 个生物学相关的特征以及 PPI 网络中的网络结构特征，MTGCN 的输入是节点的特征矩阵（X）和 PPI 网络（A）。MTGCN 使用多任务学习来预测 driver 基因，主任务为节点标签预测，辅助任务是边标签预测；这两个任务共享两个 Chebyshev GCN 层（1，2），并使用不同的目标函数进行优化，最后使用贝叶斯任务权重学习器对两个任务的 loss 进行平衡得到最后的 loss，用来训练整个模型。

#### 节点的特征

节点的特征包括两种，第一个就是上面讲到的 48 个生物学特征，突变 + 基因表达 + 甲基化；第二类是特征是 PPI 网络结构，通过 DeepWalk 算法提取每个节点的 embedding （16 维的 embedding），可以保留网络的拓扑特征（DeepWalk 算法可以见之前的笔记：[node embedding](https://wutaoblog.com.cn/2022/03/13/gnn/#Node-embedding))，因此最后的特征有 64 维：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220505142814-ujpvk7w.png" style="zoom: 50%;" />

#### Main task

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220505140108-jssach31-20220505143444-3hrgl3z.png" style="zoom:50%;" />

MTGCN 的主要任务是预测一个基因是否是 driver 基因，任务流程为上图：节点的特征矩阵和 PPI 网络输入经过两个 Chebyshev GCN 层，另外特征矩阵还通过一个全连接层，将第二个 Chebyshev GCN 层的输出 embedding 和全连接层输出相加，然后和 PPI 网络一起输入第三个 Chebyshev GCN 层得到最终的 embedding 进行节点标签的预测。

#### Chebyshev GCN 层

经典的 GCN 学习步骤有三步：信息传播，聚合，更新（embedding），这篇文章使用的是 Chebyshev 层，每层定义为：

$$
H=f(\sum_{k=1}^KZ^{(k)} \Theta^{(k)})
$$

$\Theta \in R^{g\times{h}}$ 是权重矩阵，g 是基因数量，h 是隐藏层维度；Z 通过下面计算得到（没有看懂）：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220507170600-7pa92cz.png)

f 是激活函数，在训练过程中使用 drop out 技术，即随机丢弃一部分 PPI 网络中的边。

#### 跳转连接

最初的输入特征经过两层 Chebyshev 层之后会得到每个基因的 embedding 向量，但是在将这些 embedding 输入两个任务之前，作者将最初的输入向量和 embedding 向量相加从而减少过拟合（类似于ResNet 的残差连接，残差连接可以避免梯度消失的问题，但是这里的网络并不是很深，另外残差连接可以一定程度保留原始的信息，还起到模型融合的作用）；为了在输入特征的维度和得到的 embedding 特征维度之间保持一致（不然就加不了），将原始特征输入两个独立的全连接层（main task 和 auxiliary task 各一个）输出的特征维度和 embedding 一致。所谓的跳转连接可以表示为：

$$
H_p=H_2+relu(Wx+b)
$$

$H_2$ 表示经过两个 Chebyshev 层学习到的 embedding，W 和 b 是全连接的权重和偏置。

#### 主要任务的损失函数

在经过 main task 的 Chebyshev 层之后使用一个 sigmoid 函数来得到 driver 基因的预测。Loss 函数使用修改后的交叉熵 loss：

$$
L_{pre}(\theta)=-\frac{1}{n}\sum_{i=1}^n[\omega y_ilog(\hat{y_i})+(1-y_i)log(1-\hat{y_i})]
$$

$\hat{y_i}$ 表示预测值，$y_i$ 表示真实的基因标签（0 或 1），n 是基因的数目，$\omega$ 控制着正例样本的权重（这里面取 1）。

#### 辅助任务：边预测

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220507183147-humnmrk1-20220507202952-x736v96.png" style="zoom:50%;" />

边预测的目的是辅助网络进行节点的预测，更进一步的提供网络的拓扑信息（文中说可以利用未标注的数据点，但是实际上只有标注的基因才可以得到其 embedding）；在上面的跳转连接之后会得到一个节点 embedding 的矩阵 $H_p$ ，我们可以使用内积来“重构”一个邻接矩阵（如果两个节点的 embedding 内积比较大，就说明这两个节点的 embedding 相似，因此更可能有边连接）：

$$
\hat{A}=\sigma(H_pH_p^T)
$$

$\sigma$ 是激活函数，边预测的 Loss 函数如下，类似交叉熵：

$$
L_{rec}(\theta)=-\frac{1}{m}[\sum_{i,j\in E}loga_{i,j}+\sum_{i,j\in Neg}(1-loga_{i,j})]
$$

E 是 PPI 网络中的边的集合，Neg 是负采样的边的集合（PPI 中没有的边抽样得到），E 和 Neg 的大小都是 m，$a_{i,j}$ 是 $\hat{A}$ 中的元素。

#### 联合两个任务的 Loss

结合节点预测和边预测的 loss 得到整个模型的 loss：

$$
L_{total}= w_{pre}L_{pre}+w_{rec}L_{rec}
$$

这篇文章使用了[贝叶斯权重学习器](https://arxiv.org/abs/1705.07115)来结合两个 loss：

基于一个简化假设：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220507210413-6d004yl.png" style="zoom:50%;" />

可以将下面两个 sigmoid 激活函数进行转化：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220507210452-0setejr.png" style="zoom:50%;" />

之前的节点预测任务的 loss 可以写成：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220507210517-636605f.png" style="zoom:50%;" />

$h^3$ 是第三个 Chebyshev 层输出的节点 embedding，接着引入一个因子 $\alpha$  再根据上面的 10 和 11 可以得到:

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220507210649-fwf6zj7.png" style="zoom: 67%;" />

通过类似的方法可以得到边预测任务的 loss：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220507210826-lf0goi0.png" style="zoom:50%;" />

然后再把两个 loss 相加得到最终的联合 loss：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220507210907-81naywq.png" style="zoom:67%;" />

因此，整个 MTGCN 的算法伪代码如下：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220507211008-yyg7ygj.png" style="zoom: 50%;" />

### 实验

Github 地址在 [weiba/MTGCN (github.com)](https://github.com/weiba/MTGCN)

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220507220013-jbxrxw4.png)

安装相应的包，设置环境：

```python
conda create -n MTGCN python=3.7
conda activate MTGCN
conda install pytorch=1.6.0 torchvision cpuonly -c pytorch ##安装pytorch
pip install torch_scatter-2.0.6-cp37-cp37m-linux_x86_64.whl #在这里下载 whl https://data.pyg.org/whl/torch-1.6.0%2Bcpu.html
pip install torch_sparse-0.6.9-cp37-cp37m-linux_x86_64.whl 
pip install torch-geometric==1.7.0

conda install ipykernel##使环境可以被jupyter-lab识别
```