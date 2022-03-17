---
title: 图机器学习
date: 2022-03-13 19:14:18
tags: 深度学习
index_img: img/GNN.png
categories:
  - 深度学习
---



图机器学习 CS224W 课程笔记

<!-- more -->

[(4) CS224W: Machine Learning with Graphs | 2021 | Lecture 1.1 - Why Graphs - YouTube](https://www.youtube.com/watch?v=JAB_plj2rbA&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn)

## 第一课

**Graphs connect things**

为什么图深度学习比较难？因为图是作为网络的形式展现的，网络是比较复杂的：

* 大小不固定，拓扑结构复杂（没有类似于序列或者图像的空间局部性）

  <img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220305211603-kn6ivwg.png" alt="" style="zoom:50%;" />
* 节点的次序不固定，或者说没有参考点 reference point
* 动态性，多维特征

图深度学习的一般构造：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220305211903-jx7ltl0.png" style="zoom:50%;" />

这门课主要覆盖：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220305211942-cmiexbm.png" alt="" style="zoom:50%;" />

### 图学习的任务

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220305212050-7qr5zne.png" alt="" style="zoom:50%;" />

* 节点层面的任务：预测节点的属性，比如蛋白质折叠预测，节点是氨基酸，边是氨基酸的临近程度（节点的属性就是其坐标）
* 边层面的任务：预测在两个节点之间是否有缺失的连接，比如推荐系统，节点是用户和物品，边就是用户物品之间的作用，也就是预测用户和物品间有无联系（购买的倾向）；再比如预测药物的副作用：

  <img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220305213609-rwjjwnr.png" style="zoom:50%;" />
* 子图层面的任务：对不同的图进行分类，比如分子气味的分类；行程时间的预测（谷歌地图）：

  <img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220305213922-wzy2iix.png" style="zoom:67%;" />
* 图层面的任务：比如预测抗生素分子（2020 Cell paper：A Deep Learning Approach to Antibiotic Discovery），优化已有的小分子（生成模型）等；物理模拟属于图演化任务，节点是粒子，边是粒子之间的相互作用（距离），预测图是如何演化的，也就是下一时刻粒子的位置：

  <img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220305214336-y540vf8.png" style="zoom: 67%;" />

### 图展示的选择

一个网络主要有3个部分构成：**对象**，对象之间的**相互作用**，以及这些相互作用和对象构成的**系统**：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220305214554-wyayxl6.png" style="zoom:50%;" />

对于一个问题，选择一种合适的图展示方法对于解决问题有很大帮助

图（网络）可以分为有向图，无向图，**异质性图**，加权图，不加权图，有自联结环的图，多图：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220305214948-jkqll8t.png" alt="" style="zoom:67%;" />

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220305220709-mstia0u.png" alt="" style="zoom:67%;" />

**异质性图表示在同一个图中有不同类型的节点和不同类型的边**：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220305215050-ev2hg8x.png" alt="image.png" style="zoom: 67%;" />

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220305215057-689lsgr.png" style="zoom:67%;" />

节点的自由度：

* 对于无向图为有多少边与该节点相连，平均自由度为:  $\frac{2E}{N}$
* 对于有向图，分入自由度（指向该节点的边）和出自由度（该节点指向别的节点的边数目），平均自由度为 $\frac{E}{N}$

还有一种特殊的图，叫做**二部图（Bipartite）**，也就是将图的节点分为2部分，每部分之间的节点没有连接，只有不同部分之间的节点有连接，比如上面那个推荐系统的图：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220305220256-1uh5aom.png" alt="" style="zoom:67%;" />

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220305220318-ben1cez.png" style="zoom:67%;" />

图结构的展示方法：

* 邻接矩阵：比较**稀疏**

  <img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220305220407-d4dbxiw.png" alt="" style="zoom: 67%;" />
* 边列表：不好分析，比如计算自由度

  <img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220305220454-cn6qqxy.png" style="zoom:67%;" />
* 邻接列表：

  <img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220305220541-vzvuskw.png" alt="image.png" style="zoom:67%;" />

图的**连接性**：

* 对于无向图来说，如果任意两个节点都可以被一条路径连接，那么这个图就叫做连接无向图（Connected undirected graph），那么不连接的图至少有两个或两个以上的连接组分构成：

  <img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220305221022-aan5o7t.png" alt="image.png" style="zoom:67%;" />
* 对于有向图，又可以分成**强连接**和**弱连接**，强连接表示任意两个节点间都有有向的路径连接，弱连接表示把方向去掉后是一个连接的无向图：

  <img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220305221258-1f1lyym.png" alt="image.png" style="zoom:67%;" />

## 第二课

这一节主要是介绍经典机器学习中如何**人工抽取图的特征**，分为节点，边，和图层面的特征构建

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220306161422-gblapoz.png" alt="" style="zoom:50%;" />

经典机器学习的一般步骤为：提取特征，训练模型，应用模型进行预测，对于图来说关键在于特征的设计，能否有效的表示图的组分：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220306161701-y08kjjy.png" alt="" style="zoom:67%;" />

### 节点特征

用来表示节点的特征应该可以反应网络中节点的结构和位置，通常考虑 4 种类型的节点特征：

* 节点自由度
* 节点中心性
* 聚类系数
* Graphlets，一种子图结构（一种模式）

这四种特征可以分为两类：基于重要性的和基于网络结构的：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307155404-xf5bi97.png" style="zoom:67%;" />

#### 节点自由度

**节点自由度**是一种比较简单的特征，仅仅考虑了邻居节点的数量，并且所有的邻居节点是等同的，因此没有考虑到这些节点的重要性，比如下图的 C 和 E，自由度都是3：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220306162529-obdq6h3.png" alt="" style="zoom:67%;" />

#### 节点中心性

**节点中心性**（centrality）考虑了图中节点的重要性，有一些方法可以用来表示“重要性”：

* 特征向量中心性（Engienvector centrality）
* 介数中心性（Betweenness centrality）<br />
* 临接中心性（Closeness centrality）

**特征向量中心性**的思想是如果一个节点的邻居节点是重要的，那么这个节点也比较重要；一个节点的中心性可以表示为其邻居节点中心性的和，这个迭代的形式可以写成矩阵，进一步可以用特征向量来表示一个节点的邻居节点的中心性：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220306163130-ghw32oa.png" style="zoom:67%;" />

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220306164418-xzx5t26.png" alt="" style="zoom:67%;" />

**介数中心性**的思想是如果一个节点频繁的出现在连接其他节点的最短路径上，那么这个节点就是比较重要的，在数值上为通过该节点的其他所有节点对之间的最短路径数量除以其他所有节点对之间的最短路径数量：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307150410-17609z7.png" style="zoom:67%;" />

**临接中心性**的思想是如果一个节点离其他节点都比较近，那么这个节点就比较重要，数值上为该节点到其他所有节点的最短路径和的倒数：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307150632-b22rn53.png" alt="" style="zoom:67%;" />

#### 聚类系数

聚类系数衡量节点 v 的邻居之间连接有**多紧密**，为节点邻居的实际边除以所有可能形成的边数目：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307151003-6zzpeco.png" alt="" style="zoom:50%;" />

其实从这个图也可以看出来在计算聚类系数时，实际计算的是以感兴趣的节点为中心旁边自由度为1的子网络中三角形的数目（这个子网络又叫ego-network）：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307151336-nst4015.png" alt="" style="zoom:50%;" />

将这个概念进行拓展→可以预定义一些图像（子图），然后在网络中数有多少这样的子图，这种预定于的子图就叫做 **graphlet** （类似于一种模式，motif），因此**描述了节点 u 周围的网络结构**：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307151649-n3gnfsj.png" alt="" style="zoom:67%;" />

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307152224-07tslps.png" style="zoom: 67%;" />

从图的类型上来看，**graphlet 是 Rooted connected  induced non-isomorphic subgraphs**，将这些概念拆开：

* Rooted 在图中某个节点被特殊标记以区分于其他的节点，这个节点叫做根节点，这个图叫做有根的图
* connected 连接图指的是图中任意两个节点之间都有路径连接
* induced subgraph 诱导子图指的是从一个大网络中拿出来的一个子图，但是这个子图保留了原来网络中拿出来的这些节点之间的边，区别于一般的部分子图，部分子图可能只含有原来网络的一部分边：

  <img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307154833-utv1khe.png" style="zoom:67%;" />
* non-isomorphic 同型图指的是两个图有相同数量的节点，并且节点之间连接的方式也是一样的：

  <img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307155110-uhhot51.png" alt="" style="zoom:67%;" />

对于不同的节点数量有不同的 graphlet：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307155212-fu67pk7.png" style="zoom:67%;" />

因此可以用包含感兴趣节点的 graphlet 数目来作为该节点的特征向量，这个向量叫做：**Graphlet Degree Vector**

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307155257-2v7hk2d.png" alt="" style="zoom:67%;" />

### 边特征

回顾一下边层面的预测任务就是基于已有的连接去预测新的连接，在测试阶段，没有连接的节点对按照某种规则排序，top K 的节点对之间就被预测有连接，因此重要问题就是**对于节点对如何设计特征**？

有3种比较重要的连接层面的特征：

* 基于距离的特征
* Local neighborhood overlap
* Global neighborhood overlap

#### 基于距离的特征

最简单的基于距离的特征就是两个节点之间的最短路径的长度，但是这个特征没有关注到节点对的共同邻居节点的数量，比如下图的B,H和 A,B节点对最短路径都是2，但是AB只有一个共同的邻居节点，而BH有两个，共同邻居越多 两个节点有连接的可能性就越大：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307160358-hplzpb5.png" alt="" style="zoom:67%;" />

#### Local neighborhood overlap

Local neighborhood overlap 就是考虑了两个节点的共同邻居的数量，可以用 **Jaccard 系数**和 **Adamic-Adar指数**来表示：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307160750-eia02uc.png" alt="" style="zoom:67%;" />

#### Global neighborhood overlap

Local neighborhood overlap 的缺点是如果两个节点没有**直接**的共同节点，那么上面的指标算出来就是0，但是这样的两个节点还是有可能连接的，比如图中的A 和 E 节点：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307161001-89d1q7n.png" alt="" style="zoom:67%;" />

因此 Global neighborhood overlap 通过考虑整个图来解决这个问题。

其中一种方法就是计算 **Katz 指数**，Katz 指数计算**给定一对节点间所有长度路径的数量**，现在的问题就是如何计算这个数量？

##### **Katz 指数**

**两个节点之间长度为 K 的路径的数量就为图的邻居矩阵的K次方的相应位置的数值**：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307161912-mmx0a3j.png" alt="" style="zoom:67%;" />

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307161940-kc4lanj.png" style="zoom:67%;" />

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307162008-17bs1py.png" style="zoom:67%;" />

可以用矩阵的几何级数来计算 Katz 指数矩阵的解析解:

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307162312-fw0aujq.png" alt="" style="zoom:67%;" />

### 图特征

图特征表征的是整个图的结构，整个图的特征一般用**核方法**（kernel method）进行构建，核可以用来比较两个数据（图）的**相似性**（可以理解为将数据通过某个函数映射到高维空间，然后对映射后的向量或矩阵做内积，这个内积就是所谓的核，而内积可以表示相似性，核技巧就是定义这个核函数，而不显式的定义两个数据的映射函数，这里貌似没有用到核技巧）。常用的图的核有：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307163928-693hxsp.png" style="zoom: 67%;" />

Bag-of-Words (BoW) 指的是对于一段文本使用词的计数作为文本的特征，那么对于图来说，则可以把节点当作词，比如使用节点的数量来作为特征，但是这就带来一个问题，如果两个图的节点数量一致，那么这种特征就不能区分两个图。但是我们可以使用其他的一些图的特征，比如可以使用 **Bag of  node degree**, 看一个图中不同自由度的节点的数量：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307165151-d0lg7vo.png" alt="" style="zoom:67%;" />

上面提到的两种方法： Graphlet Kernel and Weisfeiler-Lehman (WL) Kernel 都使用了 Bag of * 的图展示方法，这里面的 * 可以有多种表示，不止节点的自由度。

#### Graphlet kernel

Graphlet kernel 是基于 Graphlet 的，而这里的Graphlet 和之前提到的 Graphlet 有不同的地方：

* 这里的Graphlet不必要是连接的，也就是可以有独立的，和其他节点没有路径连接的节点
* 这里的Graphlet是无根的

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307165925-k3jt90h.png" alt="" style="zoom:50%;" />

给定一个图和一个 graphlet 列表，可以定义  graphlet 数量向量，也就是每个 **graphlet 的出现数量作为向量的元素**：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307170100-mtb3ijt.png" alt="" style="zoom:67%;" />

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307170112-h8mi1ds.png" alt="" style="zoom:50%;" />

给定两个图，**Graphlet 核就是两个图的Graphlet 向量的内积**，但是如果两个图的大小差异比较大，那么所得到的向量中的值差异也就比较大，因此对于每个Graphlet 向量都用其大小进行归一化：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307170340-659o39w.png" style="zoom:67%;" />

但是计算一个图中的graphlets是一个NP难问题，时间复杂度比较高，因此需要更高效的图核的计算方法。

#### Weisfeiler-Lehman Kernel (Color refinement)

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307183253-c97cczp.png" style="zoom:67%;" />

这个算法是一种迭代的算法，逐步更新节点的颜色，在每次迭代中有两步：

1. 收集邻居节点的颜色数值
2. 根据一个 hash 表将收集到的数值转化成新的颜色

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307184131-7egjho2.png" alt="" style="zoom:67%;" />

一个例子：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307184149-1xbqkl9.png" alt="" style="zoom:67%;" />

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307184202-gju1jyu.png" style="zoom:67%;" />

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307184213-vc81p9r.png" style="zoom:67%;" />

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307184226-owq5tla.png" style="zoom:67%;" />

经过K步的更新后，计算每个颜色数值的出现次数，然后 WL 核就是两个图的颜色数量向量的内积：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307184348-8y80rp8.png" alt="" style="zoom:67%;" />

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307184408-c57lh27.png" alt="" style="zoom:67%;" />

这种迭代的计算方法是非常高效的，时间复杂度和边的数量成线性关系，并且和后面要见到的图神经网络非常类似，总结一下：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307184616-b34vahy.png" alt="" style="zoom:67%;" />

## 第三课 

上一节讲的都是人工提取节点，边和图的特征（展示），更有效的方法可能是进行任务不依赖的特征展示（也就是embedding）：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220309152025-0rmp015.png" alt="" style="zoom:67%;" />

将节点embed到embedding 空间后就可以用节点的embedding 之间的相似性来衡量原来网络中节点的相似性（比如节点之间的距离），另外这种embedding 的向量适合进行下游的各种预测任务：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220309153031-ruxe1pv.png" alt="" style="zoom:67%;" />

### Node embedding

我们现在有一个图 G，V是其节点集合，A是其邻接矩阵；目标就是得到节点的embedding向量，并且在embedding空间中的embedding 向量的相似性（可以用向量的点积来衡量）可以近似为图中节点的相似性：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220309153420-qrauked.png" alt="" style="zoom:67%;" />

我们可以使用 encoder-decoder 框架来分析这个问题，分为以下几个步骤：

* encoder 将节点映射到 embedding
* 定义节点的相似性函数，也就是如何度量原来网络中节点的相似性
* decoder 将embedding映射到一个相似性值
* 我们的目标就是优化 encoder 的参数使得在 embedding 空间的相似性（点积）和网络中节点的相似性近似相等

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220309153937-k3wquu0.png" alt="" style="zoom:67%;" />

最简单的 encoder 策略就是对每个节点都映射到一个 embedding 向量（embedding lookup），很多方法是使用这种策略的，比如 DeepWalk 和 node2vec：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220309154228-2yd5bw3.png" alt="" style="zoom:67%;" />

因此这种方法是非监督或者是半监督的（不会使用节点的标签，特征），目的是估计节点的 embedding 向量来保留网络的部分结构。

关键问题就是**如何定义节点间的相似性**，简单的方法就是根据节点是否连接，节点共享的邻居数目等，但是这里使用的是更具表现力的一种方法：随机游走（**random walk**），随机游走指的是给定一个图，从某个节点出发，随机选择该节点的邻居节点然后移动到该邻居节点，重复这个过程，以这种方法访问到的节点序列称为图上的随机游走：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220309155116-a3l45a3.png" alt="" style="zoom:67%;" />

**我们可以使用 u 和 v 节点在同一个以 u 为出发节点的随机游走上的概率来定义两个节点的相似性**，因此可以根据这个概率来优化我们的 embedding 向量，使得 embedding 向量的内积接近于这个概率：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220309155442-0kj3d7e.png" alt="" style="zoom:67%;" />

我们学习到的节点embedding 应该使得在网络中邻近的节点在embedding空间也邻近，可以通过随机游走来收集某个节点的邻近节点，然后计算从该节点到其邻近节点的似然（所有两两节点的上面那个概率的乘积）然后优化log似然函数：

* 使用固定长度的随机游走策略从节点 u 进行随机游走
* 对于图中的每个节点 u ，收集 $N_R(u)$，也就是从 u 出发通过随机游走得到的节点集合（这个集合是 multiset，因为可以有重复的元素）
* 优化**对数似然函数**：

  $$
  L = \sum_{u\in V} \sum_{v\in N_R(u)} -log(P(v|z_u))
  $$

在实践中一般使用 **softmax 函数**来表示 $P(v|z_u)$:

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220309161344-2grjvna.png" style="zoom:67%;" />

> 为什么使用 softmax？
>
> 因为我们想要最相似的突出出来
>

但是这个函数优化起来比较困难，因为有两个求和，计算复杂度比较高；一种解决方法是使用**负采样（negative sampling）**，负采样就是不用所有的节点来标准化（分母），而是选择 k 个随机的负样本（不在random work 上的样本，但是实际操作的适合一般使用任何节点）：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220309162346-f7tjua3.png" alt="" style="zoom:67%;" />

这个采样的方法采用的是baised采样，**可以根据节点的自由度赋予采样的概率**，对于采样的数目：高的k会带来更稳定的估计，但是计算也更复杂，并且大的k会使得结果偏向于负样本，一般k选择 **5-20**个；**那么对于这个负采样后的对数似然函数可以使用随机梯度下降的方法进行优化**：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220309162839-s0fdb34.png" alt="" style="zoom:67%;" />

那么还有一个问题：**如何选择随机游走的策略 R**？对于 **DeepWalk**，采取的是最简单的方法：**固定长度，没有偏向的随机游走**。

#### node2vec

node2vec 使用的是有偏的游走，**可以在局部和全局的网络视角间进行平衡**（对比 deepwalk，使用的仅仅是固定长度的随机游走）。定义给定节点 u 的邻居节点的经典策略有两个：

* BFS：局部
* DFS：全局

<img src="assets/image-20220309171239-x5ner7v.png" alt="" style="zoom:67%;" />

这种策略有两个超参数：

* 返回参数 p，返回到之前的节点
* In-out 参数 q，移出（DFS）还是移入（BFS），q可以直观的理解为 BFS 和 DFS 的比值

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220309173425-cl78bki.png" alt="" style="zoom:67%;" />

因此 node2vec 算法步骤为（与 deepwalk 不同处就是如何产生 $N_R(u)$）:

* 计算随机游走概率
* 对每个节点进行 r 次长度为 l 的随机游走
* 使用随机梯度下降优化目标函数（对数似然）

还有一些其他的随机游走方法：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220309173809-15ragxx.png" alt="" style="zoom:67%;" />

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220309173847-482bhrq.png" alt="" style="zoom:67%;" />

### Graph embedding

也可以对整个图进行 embedding，相对应的任务就是对整个图进行分类，比如识别分子的毒性，识别异常图等，可以有如下的方法：

1. 利用节点的embedding得到图的embedding

* 对图（或子图）进行上述的节点的embedding
* 然后对节点的embedding进行加和或者平均：

  $$
  Z_G=\sum_{v\in G}Z_v
  $$

2. 引入一个虚拟的节点来代替图（或者子图），然后对该节点进行 embedding：

    <img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220309223742-plaxwjt.png" alt="" style="zoom:67%;" />

3. anonymous walk embedding

不记名walk的状态就是在随机游走中第一次访问节点的索引，比如：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220309224032-xtfhj6b.png" alt="" style="zoom:67%;" />

在上图中，左边两个随机游走代表的序列是一样的，对于 random walk1：首先访问 A，记其索引为1，第二个访问的节点是B， 记其索引为2，第三个访问的是节点C，记其索引为3，然后又是节点B，其第一次被访问的索引为2，然后是节点C，其第一次被访问的索引是3，因此这个状态序列为1-2-3-2-3。因为这样我们不能从这个序列上推断出访问的节点身份，所以叫做**匿名游走**。

匿名游走的数量是随着其长度指数增长的：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220309224638-z8cy6ud.png" alt="" style="zoom:67%;" />

基于这种匿名游走，如何去得到图的 embedding 呢？有两种想法：

第一种简单的想法是**使用随机游走的概率分布来进行图的 embedding**：

* 随机产生 m 个长度为 l 的匿名游走
* 图的 embedding 为这些匿名游走的概率分布

比如设随机游走的长度为 3，因此可以将图表示为一个 5 维的向量（因为长度为 3 的匿名游走有 5 种：111，112，121，122，123），然后随机生成 m 个这样的随机游走，统计每种匿名游走的的数量，计算概率分布。这里有一个问题，我们需要生成多少个随机游走（也就是 m 是多少）：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220313164248-oasfccc.png" alt="" style="zoom:67%;" />

第二种想法是**学习匿名游走 $w_i$ 的 embedding $z_i$：**使得可以根据前面固定大小的 window 中已有的游走 embedding 来预测下一个游走，比如下图根据 w1 和 w2 来预测 w3（window 为 2 ）：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220313165835-kwb547l.png" alt="" style="zoom:67%;" />

因此目标函数为 (T 为随机游走的总数量，$\Delta$ 为 window 大小）：

$$
max_{z_G}\sum_{t=\Delta+1}^TlogP(w_t|w_{t-\Delta},...,w_{t-1},z_G)
$$

具体步骤：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220313170348-eapj54q.png" alt="" style="zoom:67%;" />

## 第四课

这一课主要是将图视作矩阵，进行图的分析。

互联网可以看作一个有向图，节点是网页，边是超链接；但是不是所有的节点的重要性都是一样的，比如 thispersondoesnotexist.com 和 www.stanford.edu，对于网络构成的图，节点的连接性的差异是非常大的（比如下图的**蓝色**和**红色**节点），因此可以使用网络图的连接结构来对网页进行排序。

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220315151830-cjxn872.png" style="zoom: 67%;" />

本章学习下面的 3 种连接分析方法来计算节点的重要性：

* PageRank
* Personalized PageRank （PPR）
* Random Walk with Restart

### PageRank

一个简单的想法是我们可以使用网页的链接来给网页投票：一个网页如果有更多的链接，那么这个网页更重要，使用指向网页的链接还是该网页指出的链接？使用 in-link 可能更好，因为别的网页指向该网页的 in-link 不容易造假，out-link 容易造假，那么现在问题就是所有的 in-link 都是等同的吗？**显然从重要节点指向该网页的 in-link 权重应该更大**，从这个描述可以看出这个问题是一个**递归**的问题。

PageRank 使用的是 `Flow` 的模型即从更重要的网页来源的指向（边）投票更多：如果一个节点 i 有着重要性 $r_i$，同时有 $d_i$ 个出边（out-link），那么每个出边有 $r_i/d_i$ 的票数（权重），对于节点 j，其重要性 $r_j$ 是其所有入边的票数和，比如下面的例子：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220315154100-xgtv87v.png" alt="" style="zoom:67%;" />

因此节点 j 的排序 $r_j$ 可以定义为：

$$
r_j=\sum_{i\rightarrow j}\frac{r_i}{d_i}
$$

$d_i$ 是节点 i 的出度（out-degree），对于一个简单的图，我们可以使用这个定义来列出方程：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220315154813-gz9r72o.png" alt="" style="zoom: 67%;" />

但是直接去解这个方程（高斯消元）不是一个方法，因为不能简单的迁移到大的数据集上。对于这个问题，pagerank引入了一种**随机邻接矩阵（stochastic adjacency matrix）M**：$d_i$ 是节点 i 的出度，如果节点 i 指向节点 j，那么 $M_{ji}$ 为 $\frac{1}{d_i}$，因此 M 是一个列随机矩阵，其每列加起来为 1：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202022-03-15%20155429-20220315155527-x9naqjq.png" alt="" style="zoom:50%;" />

再定义一个**排序向量 r，其中的元素 $r_i$ 为第 i 个节点的重要性值**，并且：

$$
\sum_ir_i=1
$$

因此上面的 flow equation 可以写成：

$$
r=M\cdot r
$$

还以刚才的那个简单的图为例：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220315155929-zfxh2ct.png" style="zoom:67%;" />

现在想象一个场景：随机网上冲浪，在任意时刻 t，我们在某个网页 i，然后在 t+1 时刻我们从网页 i 指向别的网页的链接中随机选一个（均匀分布）到达网页 j，重复这个过程，设 $p(t)$  为时间 t 网页的概率分布，也就是 p(t) 的第 i 个元素为在时刻 t 在网页 i 上的概率；那么可以得到：

$$
p(t+1)=M\cdot p(t)
$$

(因为是均匀随机选择边)

那么如果假设这个随机游走达到一种状态，此时：

$$
p(t+1)=M\cdot p(t)=p(t)
$$

称这个 p(t) 为随机游走的稳定分布，也就是下一个时刻网页的概率分布和这一个时刻一样，整个系统达到一种平衡。把这个式子和之前的 $r=M\cdot r$ 比较，可以看出两者的形式是一样的，因此 **r 也是这个随机游走的稳定分布**。

回顾一下第二课中节点的特征向量中心性

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220315162222-jqh5aq7.png" alt="" style="zoom:50%;" />

我们可以将 flow equation 写成：

$$
1\cdot r = M\cdot r
$$

因此**秩向量 r 也可以看为随机邻接矩阵 M 在特征值为 1 时的特征向量**

我们也可以将上面的**稳定分布看成从任意向量 u 开始，不停的左乘矩阵 M，其极限为 r**，那么这个 r 就是 M 的 principal eigenvector（最大特征值的特征向量），通过这种方式可以有效的解出 r，这个方法叫做 Power iteration（幂迭代），总结：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220315214033-dxjtyaa.png" style="zoom:50%;" />

实际操作时可以分为三步：

* 初始化每个节点的重要性为 1/N：$r^{0}=[1/N,..., 1/N]^T$
* 进行迭代：$r^{t+1}=M\cdot r^{t}$
* 当两次迭代的误差小于某个值时停止（这个误差可以使用 L1 范数）：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220315215528-j56twlp.png" style="zoom:50%;" />

举个例子：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220315215601-yr5skbb.png" alt="" style="zoom:67%;" />

上面这个过程可能会出现两个问题：

* 一些节点是 dead end，也就是没有指出的边，有这种节点进行上面的迭代时会造成所有的节点都为0的情况：

  <img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220315215811-15jbqap.png" style="zoom:50%;" />

* 第二种情况为 spider traps，也就是有一个节点其所有的指出的边都指向自己，迭代时就会出现该节点是 1 其余都是 0 的情况：

  <img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220315220009-r8mc3g2.png" alt="" style="zoom:67%;" />

对于 spider-trap 来说在数学上看着是没有问题的，但是结果不是我们想要的，因为被困在 b 节点并不能说明 b 是最重要的，因此对于这种情况可以采用**有一定概率直接跳到其他节点** (**teleport**)，使得在一定步骤后可以摆脱困在某个节点的情况：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220315220420-nk9mmi6.png" alt="" style="zoom:50%;" />

对于 dead end，这种情况下的随机邻接矩阵就不符合我们的设定，因为某一（些）列加起来是 0 而不是 1，因此我们对这个矩阵可以调整，如果有一列全为 0 则对每个元素赋予同样值：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220315220732-zxhs54i.png" alt="" style="zoom:50%;" />

Google 采取的 PageRank 算法：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220315220836-yyjnedf.png" alt="" style="zoom:50%;" />

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220315220849-tycz1ds.png" alt="" style="zoom:50%;" />

例子：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220315220909-7nr4jta.png" alt="" style="zoom:50%;" />

### Personalized PageRank & Random Walk with Restart

上面讲到的 teleport 是随机的跳向任意节点，但是根据这个 teleport 的目标节点的不同，pageRank 有一些不同的变种。

通过推荐任务来引入问题：有一个二部图代表用户和商品的相互作用（购买）：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220316145048-nwmuryw.png" alt="" style="zoom:50%;" />

我们想要预测的问题是，如果用户和商品 Q 互作，那么我们应该推荐什么商品给这个用户；问题就变成了哪些节点与 Q 最相关，也就是我们需要基于与节点集 S 的邻近性对其他节点进行排序（之前是直接根据节点的重要性进行排序）【这里的S = {Q}】，这个问题可以用 Random Walk with Restart 算法来解决：

* 给定一个 Query-Nodes 集合（可以只有一个节点），开始模拟随机游走
* 随机走向一个邻居节点，记录其被访问的次数（visit count）
* 以概率 alpha 重启游走，也就是直接回到 Query-Nodes
* 重复以上过程，最后有着最高的 visit count 的节点就是和 Query-Nodes 最近的节点

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220316145923-xvr2ivu.png" style="zoom:50%;" />

为什么这个方法可以奏效？原因可能是：考虑了节点间的多种连接，多个路径，有向和无向的路径，还有节点的自由度（也就是节点的边）。

Personalized PageRank ，Random Walk with Restart 和 PageRank 之间的区别就在于如何定义这个重启节点集合 S：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220316150210-p2akpwb.png" alt="" style="zoom: 50%;" />

