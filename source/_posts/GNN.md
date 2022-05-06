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



![图机器学习](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/%E5%9B%BE%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0_backup_301789.png)

## 第一课

**Graphs connect things**

为什么图深度学习比较难？因为图是作为网络的形式展现的，网络是比较复杂的：

* 大小不固定，拓扑结构复杂（没有类似于序列或者图像的空间局部性）

  <img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220305211603-kn6ivwg.png" alt="" style="zoom:50%;" div align=center/>
* 节点的次序不固定，或者说没有参考点 reference point
* 动态性，多维特征

图深度学习的一般构造：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220305211903-jx7ltl0.png" style="zoom:50%;" div align=center/>

这门课主要覆盖：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220305211942-cmiexbm.png" alt="" style="zoom:50%;" div align=center/>

### 图学习的任务

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220305212050-7qr5zne.png" alt="" style="zoom:50%;" div align=center />

* 节点层面的任务：预测节点的属性，比如蛋白质折叠预测，节点是氨基酸，边是氨基酸的临近程度（节点的属性就是其坐标）
* 边层面的任务：预测在两个节点之间是否有缺失的连接，比如推荐系统，节点是用户和物品，边就是用户物品之间的作用，也就是预测用户和物品间有无联系（购买的倾向）；再比如预测药物的副作用：

  <img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220305213609-rwjjwnr.png" style="zoom:50%;" div align=center/>
* 子图层面的任务：对不同的图进行分类，比如分子气味的分类；行程时间的预测（谷歌地图）：

  <img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220305213922-wzy2iix.png" style="zoom:67%;" div align=center/>
* 图层面的任务：比如预测抗生素分子（2020 Cell paper：A Deep Learning Approach to Antibiotic Discovery），优化已有的小分子（生成模型）等；物理模拟属于图演化任务，节点是粒子，边是粒子之间的相互作用（距离），预测图是如何演化的，也就是下一时刻粒子的位置：

  <img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220305214336-y540vf8.png" style="zoom: 67%;" div align=center/>

### 图展示的选择

一个网络主要有3个部分构成：**对象**，对象之间的**相互作用**，以及这些相互作用和对象构成的**系统**：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220305214554-wyayxl6.png" style="zoom:50%;" div align=center/>

对于一个问题，选择一种合适的图展示方法对于解决问题有很大帮助

图（网络）可以分为有向图，无向图，**异质性图**，加权图，不加权图，有自联结环的图，多图：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220305214948-jkqll8t.png" alt="" style="zoom:67%;" div align=center/>



<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220305220709-mstia0u.png" alt="" style="zoom:67%;" div align=center/>

**异质性图表示在同一个图中有不同类型的节点和不同类型的边**：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220305215050-ev2hg8x.png" alt="image.png" style="zoom: 67%;" div align=center/>



<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220305215057-689lsgr.png" style="zoom:67%;" div align=center/>

节点的自由度：

* 对于无向图为有多少边与该节点相连，平均自由度为:  $\frac{2E}{N}$
* 对于有向图，分入自由度（指向该节点的边）和出自由度（该节点指向别的节点的边数目），平均自由度为 $\frac{E}{N}$

还有一种特殊的图，叫做**二部图（Bipartite）**，也就是将图的节点分为2部分，每部分之间的节点没有连接，只有不同部分之间的节点有连接，比如上面那个推荐系统的图：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220305220256-1uh5aom.png" alt="" style="zoom:67%;" div align=center/>



<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220305220318-ben1cez.png" style="zoom:67%;" div align=center/>

图结构的展示方法：

* 邻接矩阵：比较**稀疏**

  <img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220305220407-d4dbxiw.png" alt="" style="zoom: 67%;" div align=center/>
* 边列表：不好分析，比如计算自由度

  <img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220305220454-cn6qqxy.png" style="zoom:67%;" div align=center/>
* 邻接列表：

  <img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220305220541-vzvuskw.png" alt="image.png" style="zoom:67%;" div align=center/>

图的**连接性**：

* 对于无向图来说，如果任意两个节点都可以被一条路径连接，那么这个图就叫做连接无向图（Connected undirected graph），那么不连接的图至少有两个或两个以上的连接组分构成：

  <img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220305221022-aan5o7t.png" alt="image.png" style="zoom:67%;" div align=center/>
* 对于有向图，又可以分成**强连接**和**弱连接**，强连接表示任意两个节点间都有有向的路径连接，弱连接表示把方向去掉后是一个连接的无向图：

  <img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220305221258-1f1lyym.png" alt="image.png" style="zoom:67%;" div align=center/>

## 第二课

这一节主要是介绍经典机器学习中如何**人工抽取图的特征**，分为节点，边，和图层面的特征构建

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220306161422-gblapoz.png" alt="" style="zoom:50%;" div align=center/>

经典机器学习的一般步骤为：提取特征，训练模型，应用模型进行预测，对于图来说关键在于特征的设计，能否有效的表示图的组分：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220306161701-y08kjjy.png" alt="" style="zoom:67%;" div align=center/>

### 节点特征

用来表示节点的特征应该可以反应网络中节点的结构和位置，通常考虑 4 种类型的节点特征：

* 节点自由度
* 节点中心性
* 聚类系数
* Graphlets，一种子图结构（一种模式）

这四种特征可以分为两类：基于重要性的和基于网络结构的：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307155404-xf5bi97.png" style="zoom:67%;" div align=center/>

#### 节点自由度

**节点自由度**是一种比较简单的特征，仅仅考虑了邻居节点的数量，并且所有的邻居节点是等同的，因此没有考虑到这些节点的重要性，比如下图的 C 和 E，自由度都是3：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220306162529-obdq6h3.png" alt="" style="zoom:67%;" div align=center/>

#### 节点中心性

**节点中心性**（centrality）考虑了图中节点的重要性，有一些方法可以用来表示“重要性”：

* 特征向量中心性（Engienvector centrality）
* 介数中心性（Betweenness centrality）<br />
* 临接中心性（Closeness centrality）

**特征向量中心性**的思想是如果一个节点的邻居节点是重要的，那么这个节点也比较重要；一个节点的中心性可以表示为其邻居节点中心性的和，这个迭代的形式可以写成矩阵，进一步可以用特征向量来表示一个节点的邻居节点的中心性：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220306163130-ghw32oa.png" style="zoom:67%;" div align=center/>



<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220306164418-xzx5t26.png" alt="" style="zoom:67%;" div align=center/>

**介数中心性**的思想是如果一个节点频繁的出现在连接其他节点的最短路径上，那么这个节点就是比较重要的，在数值上为通过该节点的其他所有节点对之间的最短路径数量除以其他所有节点对之间的最短路径数量：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307150410-17609z7.png" style="zoom:67%;" div align=center/>

**临接中心性**的思想是如果一个节点离其他节点都比较近，那么这个节点就比较重要，数值上为该节点到其他所有节点的最短路径和的倒数：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307150632-b22rn53.png" alt="" style="zoom:67%;" div align=center/>

#### 聚类系数

聚类系数衡量节点 v 的邻居之间连接有**多紧密**，为节点邻居的实际边除以所有可能形成的边数目：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307151003-6zzpeco.png" alt="" style="zoom:50%;" div align=center/>

其实从这个图也可以看出来在计算聚类系数时，实际计算的是以感兴趣的节点为中心旁边自由度为1的子网络中三角形的数目（这个子网络又叫ego-network）：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307151336-nst4015.png" alt="" style="zoom:50%;" div align=center/>

将这个概念进行拓展→可以预定义一些图像（子图），然后在网络中数有多少这样的子图，这种预定于的子图就叫做 **graphlet** （类似于一种模式，motif），因此**描述了节点 u 周围的网络结构**：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307151649-n3gnfsj.png" alt="" style="zoom:67%;" div align=center/>

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307152224-07tslps.png" style="zoom: 67%;" div align=center/>

从图的类型上来看，**graphlet 是 Rooted connected  induced non-isomorphic subgraphs**，将这些概念拆开：

* Rooted 在图中某个节点被特殊标记以区分于其他的节点，这个节点叫做根节点，这个图叫做有根的图
* connected 连接图指的是图中任意两个节点之间都有路径连接
* induced subgraph 诱导子图指的是从一个大网络中拿出来的一个子图，但是这个子图保留了原来网络中拿出来的这些节点之间的边，区别于一般的部分子图，部分子图可能只含有原来网络的一部分边：

  <img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307154833-utv1khe.png" style="zoom:67%;" div align=center/>
* non-isomorphic 同型图指的是两个图有相同数量的节点，并且节点之间连接的方式也是一样的：

  <img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307155110-uhhot51.png" alt="" style="zoom:67%;" div align=center/>

对于不同的节点数量有不同的 graphlet：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307155212-fu67pk7.png" style="zoom:67%;" div align=center/>

因此可以用包含感兴趣节点的 graphlet 数目来作为该节点的特征向量，这个向量叫做：**Graphlet Degree Vector**

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307155257-2v7hk2d.png" alt="" style="zoom:67%;" div align=center/>

### 边特征

回顾一下边层面的预测任务就是基于已有的连接去预测新的连接，在测试阶段，没有连接的节点对按照某种规则排序，top K 的节点对之间就被预测有连接，因此重要问题就是**对于节点对如何设计特征**？

有3种比较重要的连接层面的特征：

* 基于距离的特征
* Local neighborhood overlap
* Global neighborhood overlap

#### 基于距离的特征

最简单的基于距离的特征就是两个节点之间的最短路径的长度，但是这个特征没有关注到节点对的共同邻居节点的数量，比如下图的B,H和 A,B节点对最短路径都是2，但是AB只有一个共同的邻居节点，而BH有两个，共同邻居越多 两个节点有连接的可能性就越大：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307160358-hplzpb5.png" alt="" style="zoom:67%;" div align=center/>

#### Local neighborhood overlap

Local neighborhood overlap 就是考虑了两个节点的共同邻居的数量，可以用 **Jaccard 系数**和 **Adamic-Adar指数**来表示：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307160750-eia02uc.png" alt="" style="zoom:67%;" div align=center/>

#### Global neighborhood overlap

Local neighborhood overlap 的缺点是如果两个节点没有**直接**的共同节点，那么上面的指标算出来就是0，但是这样的两个节点还是有可能连接的，比如图中的A 和 E 节点：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307161001-89d1q7n.png" alt="" style="zoom:67%;" div align=center/>

因此 Global neighborhood overlap 通过考虑整个图来解决这个问题。

其中一种方法就是计算 **Katz 指数**，Katz 指数计算**给定一对节点间所有长度路径的数量**，现在的问题就是如何计算这个数量？

##### **Katz 指数**

**两个节点之间长度为 K 的路径的数量就为图的邻居矩阵的K次方的相应位置的数值**：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307161912-mmx0a3j.png" alt="" style="zoom:67%;" div align=center/>



<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307161940-kc4lanj.png" style="zoom:67%;" div align=center/>



<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307162008-17bs1py.png" style="zoom:67%;" div align=center/>

可以用矩阵的几何级数来计算 Katz 指数矩阵的解析解:

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307162312-fw0aujq.png" alt="" style="zoom:67%;" div align=center/>

### 图特征

图特征表征的是整个图的结构，整个图的特征一般用**核方法**（kernel method）进行构建，核可以用来比较两个数据（图）的**相似性**（可以理解为将数据通过某个函数映射到高维空间，然后对映射后的向量或矩阵做内积，这个内积就是所谓的核，而内积可以表示相似性，核技巧就是定义这个核函数，而不显式的定义两个数据的映射函数，这里貌似没有用到核技巧）。常用的图的核有：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307163928-693hxsp.png" style="zoom: 67%;" div align=center/>

Bag-of-Words (BoW) 指的是对于一段文本使用词的计数作为文本的特征，那么对于图来说，则可以把节点当作词，比如使用节点的数量来作为特征，但是这就带来一个问题，如果两个图的节点数量一致，那么这种特征就不能区分两个图。但是我们可以使用其他的一些图的特征，比如可以使用 **Bag of  node degree**, 看一个图中不同自由度的节点的数量：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307165151-d0lg7vo.png" alt="" style="zoom:67%;" div align=center/>

上面提到的两种方法： Graphlet Kernel and Weisfeiler-Lehman (WL) Kernel 都使用了 Bag of * 的图展示方法，这里面的 * 可以有多种表示，不止节点的自由度。

#### Graphlet kernel

Graphlet kernel 是基于 Graphlet 的，而这里的Graphlet 和之前提到的 Graphlet 有不同的地方：

* 这里的Graphlet不必要是连接的，也就是可以有独立的，和其他节点没有路径连接的节点
* 这里的Graphlet是无根的

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307165925-k3jt90h.png" alt="" style="zoom:50%;" div align=center/>

给定一个图和一个 graphlet 列表，可以定义  graphlet 数量向量，也就是每个 **graphlet 的出现数量作为向量的元素**：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307170100-mtb3ijt.png" alt="" style="zoom:67%;" div align=center/>

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307170112-h8mi1ds.png" alt="" style="zoom:50%;" />

给定两个图，**Graphlet 核就是两个图的Graphlet 向量的内积**，但是如果两个图的大小差异比较大，那么所得到的向量中的值差异也就比较大，因此对于每个Graphlet 向量都用其大小进行归一化：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307170340-659o39w.png" style="zoom:67%;" div align=center/>

但是计算一个图中的graphlets是一个NP难问题，时间复杂度比较高，因此需要更高效的图核的计算方法。

#### Weisfeiler-Lehman Kernel (Color refinement)

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307183253-c97cczp.png" style="zoom:67%;" div align=center/>

这个算法是一种迭代的算法，逐步更新节点的颜色，在每次迭代中有两步：

1. 收集邻居节点的颜色数值
2. 根据一个 hash 表将收集到的数值转化成新的颜色

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307184131-7egjho2.png" alt="" style="zoom:67%;" div align=center/>

一个例子：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307184149-1xbqkl9.png" alt="" style="zoom:67%;" div align=center/>



<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307184202-gju1jyu.png" style="zoom:67%;" div align=center/>



<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307184213-vc81p9r.png" style="zoom:67%;" div align=center/>



<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307184226-owq5tla.png" style="zoom:67%;" div align=center/>

经过K步的更新后，计算每个颜色数值的出现次数，然后 WL 核就是两个图的颜色数量向量的内积：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307184348-8y80rp8.png" alt="" style="zoom:67%;" div align=center/>



<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307184408-c57lh27.png" alt="" style="zoom:67%;" div align=center/>

这种迭代的计算方法是非常高效的，时间复杂度和边的数量成线性关系，并且和后面要见到的图神经网络非常类似，总结一下：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220307184616-b34vahy.png" alt="" style="zoom:67%;" div align=center/>

## 第三课 

上一节讲的都是人工提取节点，边和图的特征（展示），更有效的方法可能是进行任务不依赖的特征展示（也就是embedding）：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220309152025-0rmp015.png" alt="" style="zoom:67%;" div align=center/>

将节点embed到embedding 空间后就可以用节点的embedding 之间的相似性来衡量原来网络中节点的相似性（比如节点之间的距离），另外这种embedding 的向量适合进行下游的各种预测任务：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220309153031-ruxe1pv.png" alt="" style="zoom:67%;" div align=center/>

### Node embedding

我们现在有一个图 G，V是其节点集合，A是其邻接矩阵；目标就是得到节点的embedding向量，并且在embedding空间中的embedding 向量的相似性（可以用向量的点积来衡量）可以近似为图中节点的相似性：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220309153420-qrauked.png" alt="" style="zoom:67%;" div align=center/>

我们可以使用 encoder-decoder 框架来分析这个问题，分为以下几个步骤：

* encoder 将节点映射到 embedding
* 定义节点的相似性函数，也就是如何度量原来网络中节点的相似性
* decoder 将embedding映射到一个相似性值
* 我们的目标就是优化 encoder 的参数使得在 embedding 空间的相似性（点积）和网络中节点的相似性近似相等

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220309153937-k3wquu0.png" alt="" style="zoom:67%;" div align=center/>

最简单的 encoder 策略就是对每个节点都映射到一个 embedding 向量（embedding lookup），很多方法是使用这种策略的，比如 DeepWalk 和 node2vec：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220309154228-2yd5bw3.png" alt="" style="zoom:67%;" div align=center/>

因此这种方法是非监督或者是半监督的（不会使用节点的标签，特征），目的是估计节点的 embedding 向量来保留网络的部分结构。

关键问题就是**如何定义节点间的相似性**，简单的方法就是根据节点是否连接，节点共享的邻居数目等，但是这里使用的是更具表现力的一种方法：随机游走（**random walk**），随机游走指的是给定一个图，从某个节点出发，随机选择该节点的邻居节点然后移动到该邻居节点，重复这个过程，以这种方法访问到的节点序列称为图上的随机游走：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220309155116-a3l45a3.png" alt="" style="zoom:67%;" div align=center/>

**我们可以使用 u 和 v 节点在同一个以 u 为出发节点的随机游走上的概率来定义两个节点的相似性**，因此可以根据这个概率来优化我们的 embedding 向量，使得 embedding 向量的内积接近于这个概率：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220309155442-0kj3d7e.png" alt="" style="zoom:67%;" div align=center/>

我们学习到的节点embedding 应该使得在网络中邻近的节点在embedding空间也邻近，可以通过随机游走来收集某个节点的邻近节点，然后计算从该节点到其邻近节点的似然（所有两两节点的上面那个概率的乘积）然后优化log似然函数：

* 使用固定长度的随机游走策略从节点 u 进行随机游走
* 对于图中的每个节点 u ，收集 $N_R(u)$，也就是从 u 出发通过随机游走得到的节点集合（这个集合是 multiset，因为可以有重复的元素）
* 优化**对数似然函数**：

  $$
  L = \sum_{u\in V} \sum_{v\in N_R(u)} -log(P(v|z_u))
  $$

在实践中一般使用 **softmax 函数**来表示 $P(v|z_u)$:

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220309161344-2grjvna.png" style="zoom:67%;" div align=center/>

> 为什么使用 softmax？
>
> 因为我们想要最相似的突出出来
>

但是这个函数优化起来比较困难，因为有两个求和，计算复杂度比较高；一种解决方法是使用**负采样（negative sampling）**，负采样就是不用所有的节点来标准化（分母），而是选择 k 个随机的负样本（不在random work 上的样本，但是实际操作的适合一般使用任何节点）：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220309162346-f7tjua3.png" alt="" style="zoom:67%;" div align=center/>

这个采样的方法采用的是baised采样，**可以根据节点的自由度赋予采样的概率**，对于采样的数目：高的k会带来更稳定的估计，但是计算也更复杂，并且大的k会使得结果偏向于负样本，一般k选择 **5-20**个；**那么对于这个负采样后的对数似然函数可以使用随机梯度下降的方法进行优化**：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220309162839-s0fdb34.png" alt="" style="zoom:67%;" div align=center/>

那么还有一个问题：**如何选择随机游走的策略 R**？对于 **DeepWalk**，采取的是最简单的方法：**固定长度，没有偏向的随机游走**。

#### node2vec

node2vec 使用的是有偏的游走，**可以在局部和全局的网络视角间进行平衡**（对比 deepwalk，使用的仅仅是固定长度的随机游走）。定义给定节点 u 的邻居节点的经典策略有两个：

* BFS：局部
* DFS：全局

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220309171239-x5ner7v.png" style="zoom:50%;" div align=center/>

这种策略有两个超参数：

* 返回参数 p，返回到之前的节点
* In-out 参数 q，移出（DFS）还是移入（BFS），q可以直观的理解为 BFS 和 DFS 的比值

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220309173425-cl78bki.png" alt="" style="zoom:67%;" div align=center/>

因此 node2vec 算法步骤为（与 deepwalk 不同处就是如何产生 $N_R(u)$）:

* 计算随机游走概率
* 对每个节点进行 r 次长度为 l 的随机游走
* 使用随机梯度下降优化目标函数（对数似然）

还有一些其他的随机游走方法：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220309173809-15ragxx.png" alt="" style="zoom:67%;" div align=center/>



<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220309173847-482bhrq.png" alt="" style="zoom:67%;" div align=center/>

### Graph embedding

也可以对整个图进行 embedding，相对应的任务就是对整个图进行分类，比如识别分子的毒性，识别异常图等，可以有如下的方法：

1. 利用节点的embedding得到图的embedding

* 对图（或子图）进行上述的节点的embedding
* 然后对节点的embedding进行加和或者平均：

  $$
  Z_G=\sum_{v\in G}Z_v
  $$

2. 引入一个虚拟的节点来代替图（或者子图），然后对该节点进行 embedding：

    <img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220309223742-plaxwjt.png" alt="" style="zoom:67%;" div align=center/>

3. anonymous walk embedding

不记名walk的状态就是在随机游走中第一次访问节点的索引，比如：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220309224032-xtfhj6b.png" alt="" style="zoom:67%;" div align=center/>

在上图中，左边两个随机游走代表的序列是一样的，对于 random walk1：首先访问 A，记其索引为1，第二个访问的节点是B， 记其索引为2，第三个访问的是节点C，记其索引为3，然后又是节点B，其第一次被访问的索引为2，然后是节点C，其第一次被访问的索引是3，因此这个状态序列为1-2-3-2-3。因为这样我们不能从这个序列上推断出访问的节点身份，所以叫做**匿名游走**。

匿名游走的数量是随着其长度指数增长的：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220309224638-z8cy6ud.png" alt="" style="zoom:67%;" div align=center/>

基于这种匿名游走，如何去得到图的 embedding 呢？有两种想法：

第一种简单的想法是**使用随机游走的概率分布来进行图的 embedding**：

* 随机产生 m 个长度为 l 的匿名游走
* 图的 embedding 为这些匿名游走的概率分布

比如设随机游走的长度为 3，因此可以将图表示为一个 5 维的向量（因为长度为 3 的匿名游走有 5 种：111，112，121，122，123），然后随机生成 m 个这样的随机游走，统计每种匿名游走的的数量，计算概率分布。这里有一个问题，我们需要生成多少个随机游走（也就是 m 是多少）：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220313164248-oasfccc.png" alt="" style="zoom:67%;" div align=center/>

第二种想法是**学习匿名游走 $w_i$ 的 embedding $z_i$：**使得可以根据前面固定大小的 window 中已有的游走 embedding 来预测下一个游走，比如下图根据 w1 和 w2 来预测 w3（window 为 2 ）：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220313165835-kwb547l.png" alt="" style="zoom:67%;" div align=center/>

因此目标函数为 (T 为随机游走的总数量，$\Delta$ 为 window 大小）：

$$
max_{z_G}\sum_{t=\Delta+1}^TlogP(w_t|w_{t-\Delta},...,w_{t-1},z_G)
$$

具体步骤：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220313170348-eapj54q.png" alt="" style="zoom:67%;" div align=center/>

## 第四课

这一课主要是将图视作矩阵，进行图的分析。

互联网可以看作一个有向图，节点是网页，边是超链接；但是不是所有的节点的重要性都是一样的，比如 `thispersondoesnotexist.com` 和 `www.stanford.edu`，对于网络构成的图，节点的连接性的差异是非常大的（比如下图的**蓝色**和**红色**节点），因此可以使用网络图的连接结构来对网页进行排序。

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220315151830-cjxn872.png" style="zoom: 67%;" div align=center/>

本章学习下面的 3 种连接分析方法来计算节点的重要性：

* PageRank
* Personalized PageRank （PPR）
* Random Walk with Restart

### PageRank

一个简单的想法是我们可以使用网页的链接来给网页投票：一个网页如果有更多的链接，那么这个网页更重要，使用指向网页的链接还是该网页指出的链接？使用 in-link 可能更好，因为别的网页指向该网页的 in-link 不容易造假，out-link 容易造假，那么现在问题就是所有的 in-link 都是等同的吗？**显然从重要节点指向该网页的 in-link 权重应该更大**，从这个描述可以看出这个问题是一个**递归**的问题。

PageRank 使用的是 `Flow` 的模型即从更重要的网页来源的指向（边）投票更多：如果一个节点 i 有着重要性 $r_i$，同时有 $d_i$ 个出边（out-link），那么每个出边有 $r_i/d_i$ 的票数（权重），对于节点 j，其重要性 $r_j$ 是其所有入边的票数和，比如下面的例子：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220315154100-xgtv87v.png" alt="" style="zoom:67%;" div align=center/>

因此节点 j 的排序 $r_j$ 可以定义为：

$$
r_j=\sum_{i\rightarrow j}\frac{r_i}{d_i}
$$

$d_i$ 是节点 i 的出度（out-degree），对于一个简单的图，我们可以使用这个定义来列出方程：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220315154813-gz9r72o.png" alt="" style="zoom: 67%;" div align=center/>

但是直接去解这个方程（高斯消元）不是一个方法，因为不能简单的迁移到大的数据集上。对于这个问题，pagerank引入了一种**随机邻接矩阵（stochastic adjacency matrix）M**：$d_i$ 是节点 i 的出度，如果节点 i 指向节点 j，那么 $M_{ji}$ 为 $\frac{1}{d_i}$，因此 M 是一个列随机矩阵，其每列加起来为 1：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202022-03-15%20155429-20220315155527-x9naqjq.png" alt="" style="zoom:50%;" div align=center/>

再定义一个**排序向量 r，其中的元素 $r_i$ 为第 i 个节点的重要性值**，并且：

$$
\sum_ir_i=1
$$

因此上面的 flow equation 可以写成：

$$
r=M\cdot r
$$

还以刚才的那个简单的图为例：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220315155929-zfxh2ct.png" style="zoom:67%;" div align=center/>

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

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220315162222-jqh5aq7.png" alt="" style="zoom:50%;" div align=center/>

我们可以将 flow equation 写成：

$$
1\cdot r = M\cdot r
$$

因此**秩向量 r 也可以看为随机邻接矩阵 M 在特征值为 1 时的特征向量**

我们也可以将上面的**稳定分布看成从任意向量 u 开始，不停的左乘矩阵 M，其极限为 r**，那么这个 r 就是 M 的 principal eigenvector（最大特征值的特征向量），通过这种方式可以有效的解出 r，这个方法叫做 Power iteration（幂迭代），总结：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220315214033-dxjtyaa.png" style="zoom:50%;" div align=center/>

实际操作时可以分为三步：

* 初始化每个节点的重要性为 1/N：$r^{0}=[1/N,..., 1/N]^T$
* 进行迭代：$r^{t+1}=M\cdot r^{t}$
* 当两次迭代的误差小于某个值时停止（这个误差可以使用 L1 范数）：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220315215528-j56twlp.png" style="zoom:50%;" div align=center/>

举个例子：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220315215601-yr5skbb.png" alt="" style="zoom:67%;" div align=center/>

上面这个过程可能会出现两个问题：

* 一些节点是 dead end，也就是没有指出的边，有这种节点进行上面的迭代时会造成所有的节点都为0的情况：

  <img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220315215811-15jbqap.png" style="zoom:50%;" div align=center/>

* 第二种情况为 spider traps，也就是有一个节点其所有的指出的边都指向自己，迭代时就会出现该节点是 1 其余都是 0 的情况：

  <img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220315220009-r8mc3g2.png" alt="" style="zoom:67%;" div align=center/>

对于 spider-trap 来说在数学上看着是没有问题的，但是结果不是我们想要的，因为被困在 b 节点并不能说明 b 是最重要的，因此对于这种情况可以采用**有一定概率直接跳到其他节点** (**teleport**)，使得在一定步骤后可以摆脱困在某个节点的情况：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220315220420-nk9mmi6.png" alt="" style="zoom:50%;" div align=center/>

对于 dead end，这种情况下的随机邻接矩阵就不符合我们的设定，因为某一（些）列加起来是 0 而不是 1，因此我们对这个矩阵可以调整，如果有一列全为 0 则对每个元素赋予同样值：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220315220732-zxhs54i.png" alt="" style="zoom:50%;" div align=center/>

Google 采取的 PageRank 算法：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220315220836-yyjnedf.png" alt="" style="zoom:50%;" div align=center/>

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220315220849-tycz1ds.png" alt="" style="zoom:50%;" div align=center/>

例子：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220315220909-7nr4jta.png" alt="" style="zoom:50%;" div align=center/>

### Personalized PageRank & Random Walk with Restart

上面讲到的 teleport 是随机的跳向任意节点，但是根据这个 teleport 的目标节点的不同，pageRank 有一些不同的变种。

通过推荐任务来引入问题：有一个二部图代表用户和商品的相互作用（购买）：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220316145048-nwmuryw.png" alt="" style="zoom:50%;" div align=center/>

我们想要预测的问题是，如果用户和商品 Q 互作，那么我们应该推荐什么商品给这个用户；问题就变成了哪些节点与 Q 最相关，也就是我们需要基于与节点集 S 的邻近性对其他节点进行排序（之前是直接根据节点的重要性进行排序）【这里的S = {Q}】，这个问题可以用 Random Walk with Restart 算法来解决：

* 给定一个 Query-Nodes 集合（可以只有一个节点），开始模拟随机游走
* 随机走向一个邻居节点，记录其被访问的次数（visit count）
* 以概率 alpha 重启游走，也就是直接回到 Query-Nodes
* 重复以上过程，最后有着最高的 visit count 的节点就是和 Query-Nodes 最近的节点

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220316145923-xvr2ivu.png" style="zoom:50%;" div align=center/>

为什么这个方法可以奏效？原因可能是：考虑了节点间的多种连接，多个路径，有向和无向的路径，还有节点的自由度（也就是节点的边）。

Personalized PageRank ，Random Walk with Restart 和 PageRank 之间的区别就在于如何定义这个重启节点集合 S：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220316150210-p2akpwb.png" alt="" style="zoom: 50%;" div align=center/>

​    

## 第五课

这一课的主要问题就是：**给定一个一些节点有标签的网络，如何给这个网络其他节点也打上标签**？这个问题是一个半监督学习问题，因为只有一部分样本有标签。我们可以利用前面第 3 课讲到的 node embedding 方法来处理这个问题，也就是先学习节点的 embedding，然后用这个 embedding 向量去预测节点的标签。但是这一课讨论的是另一个处理这类问题的框架：**信息传递（message passing）**该框架的想法就是：网络中存在着相关性，也就是说相似的节点倾向于连接在一起。

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220320162151-sholsef.png" style="zoom:50%;" div align=center/>

两种类型的依赖会导致网络中存在相关性：

* 同质性："物以类聚，人以群分"
* 影响："社会关系会影响个人的特性"，比如我向朋友推荐我喜欢的音乐，长此以往，他们可能和我的品味变得类似

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220320162639-zp5lmna.png" style="zoom:50%;" div align=center/>

如何使用这种网络中的相关性来帮助我们预测节点的标签？可以想到的是除了用到节点本身的特征之外还需要节点邻居节点的特征和标签（这种方法叫做 **collective classification**）。

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220320215355-net8f5c.png" style="zoom:50%;" div align=center/>

collective classification 一个重要的假设（除了网络中存在同质性）是马尔可夫假设，也就是一个节点的标签和其一阶邻居相关，（也就是和邻居的邻居不相关）：

$$
P(Y_v)=P(Y_v|N_v)
$$

$Y_v$ 为节点 v 的标签，$N_v$ 为节点 v 的邻居节点。

collective classification 一般有3个步骤：

* 训练局部分类器（local classifier）：给每个节点初始的标签，基于节点的属性/特征训练分类器，没有用到网络的信息
* 训练相关分类器（relational classifier）：该分类器可以捕捉节点间的相关性，基于邻居的节点标签或者（和）属性训练分类器，用到了网络的信息
* 进行集体推理（collective inference）：将相关性在网络间进行“传播”（propagate），对每个节点迭代运用相关分类器直到标签收敛

collective classification 有 3 种经典的方法：

* Relational classification
* Iterative classification
* Belief propagation

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220320221256-4cxpzai.png" style="zoom:50%;" div align=center/>

### Relational classifiers

基本想法是：**节点的类概率是其邻居类概率的加权平均**，具体步骤为：

1. 对于有标签的节点，初始化的标签为其真实的标签，对于没有标签的节点，初始化其概率为 0.5

2. 以随机的顺序使用加权平均来迭代更新节点的类概率，直到收敛或者达到最大的迭代次数：

   <img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220320222227-i9y8m1e.png" style="zoom:50%;" div align=center/>

例子：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220320222330-x1c78tn.png" style="zoom:50%;" div align=center/>

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220320222346-fwagdrs.png" style="zoom:50%;" div align=center/>

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220320222355-3i0lhu0.png" style="zoom:50%;" div align=center/>

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220320222403-exes9pu.png" style="zoom:50%;" div align=center/>

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220320222415-1pi7pto.png" style="zoom:50%;" div align=center/>

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220320222424-uq5ks5w.png" style="zoom:50%;" div align=center/>

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220320222435-pyng4ia.png" style="zoom:50%;" div align=center/>

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220320222444-7mk05ml.png" style="zoom:50%;" div align=center/>

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220320222456-3faw0j3.png" style="zoom:50%;" div align=center/>


### Iterative classification

上面的 Relational classifiers 初始化时直接给没有标签的节点概率 0.5，其实并没有collective classification 的第一步 local classifier，也就是没有使用节点的特征。那么iterative classification就是既考虑了节点的特征也考虑了邻居节点的标签。该方法主要是训练两个分类器：

* $\phi_1(f_v)$ 基于节点的特征 $f_v$ 训练分类器预测节点的标签
* $\phi_2(f_v,z_v)$ 基于节点的特征 $f_v$ 以及 邻居节点标签的汇总变量 $z_v$ 来训练模型更新节点的标签

$z_v$ 如何计算？可以有几种选择：

* 邻居节点的每种标签的分布（直方图）
* 最多数量的标签
* 标签的种类
* ...

步骤：

1. 在训练集上（节点都有标签）训练两个模型：只使用节点属性，以及使用节点属性和汇总变量
2. 在测试集上应用第一个模型得到每个节点的标签
3. 计算每个节点的汇总变量 $z_v$
4. 用第二个模型 基于节点的特征和汇总变量更新节点的标签
5. 重复3，4步直到收敛或者达到最大迭代次数

总结：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220320224610-kbc0qbi.png" style="zoom:50%;" div align=center/>

### Belief Propagation （loopy BP 算法）

信念传播是一个迭代过程，邻居节点之间相互“交谈”，传递信息:

> “I (node v) believe you (node u) belong to class 1 with likelihood …”

每个节点收集其邻居节点传递的信息，然后更新自己的信念（比如属于某个类的概率），然后再将这种信息传递给下一个邻居节点。

用数图中节点的数量来引入信息传播的概念：前提是每个节点只能和其邻居相互作用：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220320230930-3fzjkus.png" style="zoom:50%;" div align=center/>

这个里面更新的 belif 就是图上有多个节点，进一步可扩展到树形结构的图上（从叶子节点到根节点）：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220320231050-urotjqw.png" style="zoom:50%;" div align=center/>

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220320231100-urwvfzy.png" style="zoom:50%;" div align=center/>

这个过程就是局部的信息计算，节点收集信息，进行转换，然后再传递给别的节点。

loopy 信念传播算法是一种迭代算法，节点 i 传递给邻居节点 j 的信息是 i 对 j 的状态的信念（比如节点 i 认为 j 是某个类的概率），而节点 i 所接受的信息又来自于其邻居，因此可以将上面那句话写成：

> I (node i) believe that you (node j) belong to class Y_j with probability ... 

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220322201017-9ivd6hf.png" style="zoom: 80%;" div align=center/>

该算法用到的一些记号：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220322201100-rv5xxc5.png" style="zoom:50%;" div align=center/>

算法步骤：

* 初始化所有的信息为 1

* 对每个节点进行迭代：假设现在在节点 i 上，节点 i 传递给节点 j 的信息为 (即节点 i 认为节点 j 属于 $Y_j$ 的 belief)：

  <img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220322225819-xl91k7b.png" style="zoom:80%;" div align=center/>

也就是说节点 i 收集整合来自下游的信息（下游节点认为节点 i 属于 $Y_i$ 的 belief，即最后一项连乘）乘以其自己认为自己属于某一类的概率（先验），而 label-label potential 表示节点 i 的标签如何影响节点 j 的标准，因此乘以这个值就是要向节点 j 传递的信息（i 认为 j 应该是某类的 belief），求和是对所有可能的 i 节点的类求和。

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220322230718-2wm1wdh.png" style="zoom:67%;" div align=center/>

但是当图中出现环的时候就可能有问题：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220322230807-35mlurt.png" style="zoom:67%;" div align=center/>

因为此时一个节点的下游分支就有可能不是独立的（用乘法就有问题），最终导致错误的信息被放大，但是实际情况这个影响不大（实际情况中环比较大，使得这个影响减弱），总结：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220322231147-2vzflci.png" style="zoom:50%;" div align=center/>

## 第六课

先来回顾一下之前学习的 node embedding 方法：

node embedding 就是把图中的节点映射到 d 维的 embedding 空间，使得在图中相似的节点在这个 embedding 空间中是距离较近的，问题就是如何学习这样的映射函数。之前讲过了 encoder + decoder 策略，encoder 将节点映射到 d 维的向量，decoder 就是衡量映射后的向量间的相似性（可以使用点积），然后这个 encoder + decoder 框架的目标就是使得两个节点 embedding 向量 decoder 后的值（即两个向量的内积）和这两个节点在图中的相似性值接近，那么定义这种相似性又有多种方法，比如 DeepWalk 和 nod2vec 的随机游走方法；之前讲的 encoder 是一种最简单的 “shallow” encoding 的方法，也就是 embedding 矩阵的每一列是一个节点的 embedding，这种 shallow encoder 的缺点有：

* 需要学习的参数数量和节点的数量相关，因为每个节点都需要学习一个 embedding 向量
* 不能迁移到在训练过程中没有见过的节点上，也就是不能对这些节点生成 embedding
* 这种方法没有考虑节点或图的特征

这一节讲的是通过图神经网络来构造编码器（encoder）：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220326154720-68gy4ix.png" style="zoom:50%;" />

图和经典的深度学习输入的数据类型（图片或者序列信息）有什么不同：

* 大小可变，拓扑结构复杂，不像图片和序列一样，局部拓扑结构是可变的，而图片和序列的局部拓扑结构是类似的（图片的每个局部结构都是一个方块，序列的局部还是一个线性的结构）
* 没有固定的节点次序或者参考点，图片可以从左到右，从上到下；序列从左到右
* 图的结构是可变的，并且有多模态的特征，比如节点的特征可以多样性

### 图深度学习

一些符号标记：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220326155547-tzz6ws6.png" alt="" style="zoom:50%;" />

 一种简单的方法就是讲邻接矩阵和特征拼接起来作为一个矩阵，然后喂给一个深度神经网络：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220326163511-fn3trcm.png" alt="" style="zoom: 67%;" />

这种方法的问题是：

* 参数仍然和节点的数量有关
* 不能迁移到不同大小的图上
* 对节点的次序敏感，也就是即使保持网络结构不变，将节点的标记改变，最后得到的矩阵也会不一样，导致学习到的网络参数也会不同

我们可以从图片的卷积神经网络上获得一些启发，在CNN中我们是使用一个滑框（卷积核）来对图片进行操作，将滑框内的像素整合成下一个卷积层的新的像素：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220326164038-9ehy6dg.png" alt="" style="zoom:67%;" />

在图上就行这种操作会有问题：还是之前讲过的图的局部拓扑结构是变化的，因此我们不能使用类似滑框的方法（也就是在图片上的操作必须要满足平移不变性，而在图上的操作要满足扰动不变性，permutation invariant，即改变节点的次序不影响操作的结果），但是我们可以借鉴 CNN 的思想：将一定范围内的元素进行整合，在图上就是**将一个节点的邻居节点的信息进行整合**：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220326164503-s1gl662.png" alt="" style="zoom:50%;" />

因此图神经网络的关键想法就是基于局部的邻居节点信息来产生节点的embedding：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220326165611-lsx5x8s.png" alt="" style="zoom:50%;" />

所以每个节点都有自己的计算图（computation graph），并且我们可以创建任意深度的模型，在每一层节点都有一个 embedding；在 0 层时，节点的 embedding 就是该节点的输入特征，在第 k 层的 embedding 就可以获得离该节点 k 步的节点的信息：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220326165929-v486kpd.png" alt="" style="zoom:50%;" />

不同的图神经网络的关键区别就在于：如何去聚合和转化邻近节点的信息？一种基本的方法就是对来自各个邻近节点的信息取个平均，然后再用一个神经网络来转化这个聚合的信息：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220326170227-fbjz29g.png" style="zoom:50%;" />

数学化的形式：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220326170351-vtpywpt.png" alt="" style="zoom:50%;" />

需要学习的参数就是上式中的 $W_l$ 和 $B_l$ ,前者是对邻近节点进行转化，后者是对自己的embedding向量进行转化。

接下来就是如何去训练这个模型，可以有两种方法：

* 监督式的训练，直接用节点的 embedding z 和节点的标签来进行监督训练即可：

  <img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220326171308-lqblal6.png" alt="" style="zoom:67%;" />

* 非监督式的训练，利用图的结构，也就是相似的节点有着相似的 embedding，节点的相似性可以用第三课中的方法衡量，比如随机游走的方法，embedding 的相似性可以使用点积来衡量：

  <img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220326171454-rnaerl0.png" alt="" style="zoom:50%;" />

可以看到这个**训练的参数对一层中所有的节点来说都是共享的**，也就是说即使对于没有训练过的节点甚至是另一个新的图中的节点，我们也可以得到其 embedding，总结一下在图中应用神经网络得到节点 embedding 的过程：

1. 定义邻居节点信息汇总函数
2. 定义节点 embedding 的 loss 函数
3. 对节点的批次进行训练
4. 得到节点的 embedding

### GraphSAGE

前面的方法是通过平均得到邻居节点信息的聚合，而GraphSAGE 则进一步拓展了这一点，并不一定需要平均的操作，下式的 AGG 可以有多种选择，而且是直接将转换后的邻居节点的信息和自身的信息进行连接，不是上面的加和：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220326212546-qthx1jx.png" alt="" style="zoom:50%;" />

比如 AGG 可以选择平均，和上面的GCN一样，也可以选择池化操作（min/max），甚至可以用更复杂的 LSTM，但是需要注意的是：在使用 LSTM时需要将邻居节点的次序打乱，从而避免模型记住了次序（也就是我们需要的是次序无关的模型）：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220326212838-b5hkb5x.png" alt="" style="zoom:50%;" />

总结：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220326212917-qekkdwe.png" alt="" style="zoom:50%;" />

## 第七课

一般的 GNN 架构分成 5 个部分，即GNN 层，包括信息的转换和整合；GNN 层之间如何连接；图的增强，包括特征和图结构的增强；学习目标：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220326213542-bujxnww.png" alt="" style="zoom:50%;" />

### GNN 层

一个单独的 GNN 层的作用是将一组向量（来自邻居节点的embedding 和自己的embedding向量）给压缩成一个向量:

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220326213948-h6lsmxe.png" alt="" style="zoom:50%;" />

那么一个 GNN 层可以分成两个过程：

* 信息的计算
* 信息的汇聚

信息的计算就是每个节点会计算自己的信息，然后传递给其他的节点，一个简单的例子就是线性转化，将节点的特征（embedding）乘以一个权重，对于多个节点（一层）来说就是乘以一个权重矩阵：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220326214649-zl7mpkk.png" alt="" style="zoom:50%;" />

信息的汇聚就是将来自节点转化后的信息进行整合，这个整合的要去就是对节点次序不敏感，可以使用求和，平均或者最大/最小操作，由于在上述计算过程中我们并没有考虑目标节点自身的信息，所以需要将这个信息加入：在信息计算步骤对自身节点单独赋予参数进行计算，然后在汇聚阶段将邻居的信息和自身的信息进行联合：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220326215214-j1c8h3z.png" alt="" style="zoom:50%;" />

为了增加模型的表达能力还需要增加非线性的激活函数，这个激活函数可以添加在信息计算或者信汇聚步骤。下面来看上节中讲过的 GCN 和 GraphSAGE 如何用这种信息计算和信息汇聚的框架来理解：

对于 GCN 原始的表示为：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220326215653-f67pjcs.png" alt="" style="zoom:50%;" />

可以将 W 写进去，就可以看成计算（乘以 W）和汇聚步骤（求和）：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220326215735-5migblj.png" style="zoom:50%;" />

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220326215805-upl5m4g.png" alt="" style="zoom:50%;" />

对于 GraphSAGE ，原始的形式为：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220326220132-bipkrgj.png" alt="" style="zoom:50%;" />

信息计算是在 AGG 函数内部进行的，比如上节讲过的 3 种选择：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220326220304-og2bqrk.png" alt="" style="zoom:50%;" />

然后信息汇聚过程分为两步，第一步为 AGG 函数汇聚邻居节点的信息，第二步是和自己的信息进行合并，然后乘上个 W ：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220326220505-wjb24av.png" alt="" style="zoom:50%;" />

GraphSAGE 一般还包括一个 L2 标准化的过程，对每一层的所有节点的 embedding 向量进行 L2 标准化，使得每个向量的范围差不多（有相同的 L2 范数）。

##### GAT

图注意力网络应用了注意力机制，也就是对每个邻居节点的关注度不一样，GAT 的一般形式为：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220326222122-v3v7msd.png" alt="" style="zoom:50%;" />

通过比较这个式子和上面的 GCN的形式，可以发现 GCN 中 $\alpha_{vu}$ 就是 $1/|N(v)|$ 在这里面每个节点的重要性都是一样的，但是实际情况更可能是节点的每个邻居不是同等重要的，所以我们可以使得这个 $\alpha_{vu}$ 成为一个可学习得参数，来赋予不同的节点不同的权重。这个注意力权重是通过注意力机制 $\alpha$ 计算出来的：

* 先通过 $\alpha$ 基于一对节点转化后的信息计算注意力系数 $e_{vu}$ :

  <img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220326223249-5i8fnhk.png" alt="" style="zoom:50%;" />

* 然后通过 softmax 函数将注意力系数标准化为最终的注意力权重 $\alpha_{vu}$

那么现在的问题就是这个注意力机制 $\alpha$ 是什么？这个有多种选择，一种方法就是通过神经网络来训练这个参数：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220326223555-2b0fwsz.png" alt="" style="zoom:50%;" />

NLP 中的多头注意力机制也可以在这里面应用，可以训练多个注意力机制，得到多个权重，最后将这些embedding 聚合起来：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220326223724-92zc4m6.png" alt="" style="zoom:50%;" />

### 实践中的 GNN 层

现代的一些深度学习的模块也可以加到 GNN 层中：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220326224116-ah7fiy9.png" alt="" style="zoom:50%;" />

* Batch Normalization，批次标准化可以稳定神经网络的训练过程，给定一个批次的输入（节点 embedding），将这些 embedding 向量归一化到均值为0 ，方差为1：

  <img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220326224327-2eentle.png" alt="" style="zoom:50%;" />

* Dropout，减轻过拟合现象，**在GNN 中 dropout是应用在线性层的**，比如转化信息的 w 操作：

  <img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220326224530-m0f972c.png" alt="" style="zoom:50%;" />

* 激活函数：

  <img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220326224738-150qndo.png" alt="" style="zoom:50%;" />

### GNN 层的叠加

接下来就需要将不同的 GNN 层叠加在一起，最直接的方法就是顺序叠加 GNN 层，输入是原始的节点特征，在 L 层的 GNN 后输出是节点的 embedding 向量：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220327141929-75jtckx.png" style="zoom:50%;" />

但是这样叠加很多 GNN 层可能会出现 over-smoothing 问题，也就是最后不同节点的 embedding 向量是非常类似的，而在不是我们想要的，我们需要的是不同的节点有不同的可分辨的 embedding，那么为什么会出现这种问题呢？首先需要了解感受野的概念（receptive field）和 CNN 中的类似，感受野指的是**决定一个节点 embedding 的一组节点**，也就是在 GNN 层的信息流动过程中，感兴趣的节点是从这组节点中收集信息的，在有 K 层的 GNN 中每个节点的感受野是距离其 k 步内（k-hops away）的邻居节点，比如下面的例子，黄色节点的感受野随着 GNN 层数的增多也越来越大，在 3 层 GNN 中其感受野已经是几乎所有节点了：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220327142544-ty8jyc6.png" style="zoom:50%;" />

因此如果我们看两个节点感受野的重合就可以知道为什么 over-smooth 问题会出现，如下图的两个黄色节点，在 3-hop 内的共同邻居几乎覆盖整个网络的节点：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220327142715-1yj89hl.png" style="zoom:50%;" />

所以**如果两个节点的感受野有较大的重合，那么这两个节点收集到的信息几乎是一样的，并且每个 GNN 层的参数也是在节点间是共享的，那么就可能会导致最终两个节点的 embedding 向量是类似的**，也就是所谓的 over-smooth 问题。那么如何去减轻这种 over-smooth 问题呢？

第一个考虑的问题就是谨慎地添加更多的 GNN 层，从上面的描述可以看出GNN 的层和 CNN 的层有所不同，GNN的层的深度表示想要获取多少步远的节点信息 （hops） 不一定越深的网络的表达能力就越好。因此在设计网络层数的时候可以考虑：

* 分析解决问题所必须的感受野
* 设置 GNN 的层数略大于我们所需要的感受野（不要一味的增大层数）

第二个问题就是在 GNN 层数比较小的情况下，如何增加 GNN 的表达能力？

* 可以增加每一层 GNN 的表达能力，比如可以在信息计算和汇聚步骤使用更加深的神经网络

* 可以把 GNN 层和其他的非 GNN 层结合起来：

  <img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220327143750-72coapk.png" style="zoom:50%;" />

如果我们的问题仍然需要比较大的 GNN 层数，如何在不减少 GNN 层的情况下减轻 over-smooth 问题呢？这里可以借鉴**残差连接**的思想，也就是 over-smooth 是由于 GNN 过深导致，那么在较浅的 GNN 层中节点的 embedding 可以更好的区分节点，因此我们可以通过 skip connections 直接在后面较深的 GNN 层中加入之前浅层 GNN 中的这些信息，：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220327144218-m7ts3fh.png" style="zoom:50%;" />

这种方法类似于创建了一个混合模型，将之前的GNN 模型和该层的 GNN 进行加权组合：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220327144419-iooy8wd.png" style="zoom:50%;" />



## 第八课

下一个问题就是进行节点或者图特征的增强，这一点可以类比 CNN 中的图像增强操作，对应下图中的第 4 点：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220326213542-bujxnww.png" style="zoom:50%;" />

需要图增强的原因是：并不是在所有情况下我们都可以直接将原始的图转化成 GNN 所需要的计算图，下面是一些例子：

* 原始图缺少节点的特征，我们可以进行**特征增强**
* 图过于稀疏，进行信息传播时效率低下，可以**添加虚拟的节点或者边**来解决
* 图过于稠密，在进行信息传播时计算就比较复杂，比如一个节点的度非常高，如果直接将原始的图转化为计算图，那么就需要整合该节点所有邻居节点的信息，这个计算就比较耗时，可以**对邻居节点采样**来进行信息计算
* 图太大，无法将计算图导入 GPU 进行运算，**对图进行采样**，利用子图进行计算 embedding

### 特征增强

进行特征增强通常是由于原始图缺少节点的特征，比如我们只有图的邻接矩阵信息；常用的方法有：

* 给每个节点赋予相同的常数
* 给每个节点赋予独特的 ID，然后可以将这些 ID 转化为 one hot 向量

下面是两者的比较：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220328180230-ur2ft9j.png" alt="" style="zoom:50%;" />

还有一种情况下需要特征增强：只通过 GNN 难以学习某些特征，比如某个节点所在的环的长度，如下面两个图中节点 v1 所在环的长度一个是 3，一个是 4，但是这两个图中所有节点的度都是 2，因此两个图的计算图是完全相同的：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220328180520-6nv5ckw.png" alt="" style="zoom:50%;" />

所以通过 GNN 无法分辨这两种节点的区别，但是在某些情况下这种区别是重要的，比如分子结构中长度为 3 的环和长度为  4 的环的功能可能完全不一样。一种解决方法就是添加一个向量来表示节点所在环的长度：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220328180800-sivt9ed.png" alt="" style="zoom:50%;" />

另外在第二课中学习的节点特征编码方法都可以使用，比如聚类系数，PageRank，节点中心性等

### 图增强

对于比较稀疏的图，我们可以增加虚拟的节点或者边；对于边，常用的方法是**增加长度为 2 的虚拟边**，从而连接两个距离为 2 的节点，因此在计算 GNN 时可以使用的不再是原始图的邻接矩阵 A，而是 $A+A^2$ (之前讲过两个节点之间长度为 K 的路径的数量就为图的邻居矩阵的K次方的相应位置的数值)，一个典型的例子就是 “作者-论文” 二部图，添加这样的虚拟节点后就可以表示作者的协作关系，也就是如果两个作者节点之间有这种虚拟边的连接就表明这两个作者是这个虚拟边所通过的论文的共同作者：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220328182325-88bqc4u.png" alt="" style="zoom: 67%;" />

对于节点，可以添加一个和所有节点都有连接的虚拟节点，比如现在有一个非常稀疏的图，两个节点间最短的路劲长度都有 10，通过添加这样的虚拟节点后，所有的节点间都有距离为 2 的路径了，可以有效的提高信息传递的效率：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220328182630-2qo12qw.png" alt="" style="zoom:67%;" />

对于过于稠密的图，为了计算的高效，需要对邻居节点进行采样操作：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220328183104-v0il80r.png" alt="" style="zoom:50%;" />

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220328183118-g29rdxo.png" alt="" style="zoom:50%;" />

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220328183136-79wpcja.png" alt="" style="zoom:50%;" />

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220328183147-y12actw.png" style="zoom:50%;" />

（这里的 next layer 也可以表示在下一个 epoch 的操作）

总计一下目前讲过的东西：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220328183355-7j4xayp.png" style="zoom:50%;" />

### GNN 预测

下一个问题就是使用 GNN 进行预测，不同的图任务需要不同的预测方法：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220328183606-p5ap93i.png" alt="" style="zoom:50%;" />

对于**节点**，可以直接使用 GNN 产生的节点 embedding 进行预测：在 GNN 计算后我们可以得到 d 维的节点 embedding（$h_v^{L}$)，需要使用一个矩阵来将节点的 embeeding 从 d 维的 embedding 空间映射到 k 维的预测空间（假设分类任务的类别有 k 个，对于回归就是 1）：

$$
\hat{y_v}=Head_{node}(h_v^{L})=W^Hh_v^L
$$

对于**边**，预测需要使用节点对:

$$
\hat{y_{uv}}=Head_{edge}(h_u^L,h_v^L)
$$

这个 $Head_{edge}$ 可以有多种选择，比如：

* 将两个节点的 embedding 直接连接起来，然后输进一个线性层，这个线性层的作用就是将维度为 2d 的 embedding 映射成 k 维的 embedding

* 也可以直接将两个节点的 embedding 进行点积运算，适用于单个值的预测（比如预测两个节点之间有没有连接），这种方法可以进一步扩展到 k 维的预测：类似于多头注意力，使用多个矩阵参数得到 k 个点积，最后把这 k 个点积拼起来得到 k 维的向量：

  <img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220328185519-nlzvyr2.png" alt="" style="zoom:67%;" />

对于图的预测，使用的是所有节点的 embedding，类似于 GNN 层中的汇聚函数 AGG，同样的这种汇聚操作可以是各种池化（平均，最小/最大，求和等），但是直接对所有节点 embedding 进行池化操作可能会损失信息，下面是一个例子：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220328185844-wa7snke.png" alt="" style="zoom:50%;" />

一种解决方法是通过**层次汇聚**来聚合节点的信息，比如还是上面那个例子，我们可以先对前两个节点和后3个节点进行汇聚，然后再对得到的两个结果进行汇聚：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220328211626-qvajezk.png" alt="" style="zoom: 67%;" />

另一种有意思的层次汇聚方法是 **DiffPool**，通过网络中的 community 分析对节点进行分层汇聚，而这个 community 检测也可以通过另一个 GNN 来完成，这两个 GNN 可以并行运算，这样就不需要另外的方法进行 community 检测了：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220328211918-b6esloq.png" alt="" style="zoom: 67%;" />

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220328211957-lr0m8rw.png" style="zoom:67%;" />

### GNN 训练

这一部分和经典的深度学习没有什么区别：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220328212121-5aj9nyu.png" style="zoom: 67%;" />

### 数据集分割

还有一个关键的问题就是我们在训练 GNN 时如果设置训练集，验证集和测试集？

一般来说机器学习中数据集的划分有两种方式：

* 固定划分，也就是把数据分成固定的三份：

  * 训练集用来优化 GNN 模型参数
  * 验证集用来选择模型超参数
  * 测试集用来检测并报告模型的性能
* 随机划分，也就是随机多次划分训练集，验证集和测试集，报告多次平均的性能

但是对于图数据来说，划分数据集和一般的机器学习任务有所不同，比如如果是图片数据集，那么每个数据点是一张图片，每个数据点之间是独立的，但是如果对图的节点进行预测，那么每个数据点是图中的一个节点，这样数据点之间就不是独立的关系了，比如下图的节点5会影响节点1的预测，因为其参与节点1 的信息传递过程，那么如果把节点1 划分到训练集，把节点5划分到测试集，就会造成信息泄露：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220328222817-t95xwnc.png" style="zoom:67%;" />

对图的数据集划分有两类方法：

* Transductive：对于所有的数据集划分，输入的图都可以可被观测的，只划分标签，比如对上图而言，在训练步骤使用整个图计算节点的 embedding，但是计算 Loss 时只使用节点 1和2的标签；在验证步骤也是使用整个图基于训练步骤训练的 GNN 模型计算节点的embedding，但是在评估时使用节点 3 和4的标签

* Inductive：破坏划分数据集之间的连接，从而得到不同的独立的图，在训练步骤使用节点 1和2的图，并用节点 1和2 的标签来计算 Loss；在验证步骤基于训练步骤的模型在节点3和4构成的图中计算 embedding，并使用 3和4的标签来评估模型：

  <img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220328223635-z261man.png" style="zoom:67%;" />

总结：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220328223659-ew09eev.png" alt="" style="zoom:50%;" />

需要注意的是预测图中的边，也就是预测两个节点之间是否存在边，这种问题是一种自监督问题，因为不需要外部的标签，也就是在训练过程中隐藏部分节点之间的边，然后让 GNN 模型来预测这些边（supervision edge训练步骤不会输入GNN）。

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220328224810-rfffafh.png" alt="" style="zoom:80%;" />

预测边同样也有两种方法：

* Inductive：在训练集，验证集和测试集中都有两种类型的边：

  <img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220328224838-homc0i5.png" alt="" style="zoom:67%;" />

* Transductive：逐步的过程：

  <img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220328225104-e8oaree.png" alt="" style="zoom: 50%;" />

  <img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220328225131-t9mcwcx.png" alt="" style="zoom:50%;" />

总结整个的 GNN 训练流程：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220328225156-30tniaq.png" alt="" style="zoom:50%;" />

## 第九课

这一课主要是讲 GNN 的表达能力，以及如何设计表达能力更强的 GNN 模型

### GIN

前面讲过 GNN 的通用架构就是使用神经网络从邻居节点收集信息：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220413224547-66zba1u.png" style="zoom:50%;" />

不同的 GNN 的神经网络不一样，比如GCN是按元素的平均池化+线性层+ ReLu激活层，而GraphSAGE 是MLP + 按元素的最大池化操作。

GNN 的表达能力指的是：对于局部邻接结构不同的节点，GNN 能否产生不同的 node embedding？通过前面的学习我们知道 GNN 是通过计算图的方式来收集邻居节点的信息从而得到节点的 embedding，因此**如果两个节点的计算图是完全一样的，那么GNN 就不能分辨这两个节点**，比如下图中的节点 1和2：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220413225417-jnoohoa.png" style="zoom:50%;" />

节点颜色表示特征，这里所有节点的特征都是一样的。

因此在这个限制下，如果 GNN 能将不同的有根子树映射到不同的 node embedding，那么这个GNN就是最具表现力的 GNN （也就是只要局部网络结构不同，GNN 就能分辨出来）：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220413230027-bo4m4qq.png" alt="" style="zoom:50%;" />

这个想法和单射函数（injective）的概念类似，单射函数指的是将不同的输入映射到不同的输出，也就是这种函数保留了输入的全部信息：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220413230225-nqv5f8n.png" alt="" style="zoom:50%;" />

因此**最具表现力的GNN应该将子树单射到node embedding**。

GNN 是由多层构成的，在每一层中节点收集邻居节点的信息，如果GNN的每一层的汇聚步骤能够完全保留邻接信息，那么最终得到的 embedding也可以区分整个树结构：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220413230824-6twol7j.png" alt="" style="zoom:50%;" />

也就是说每一层的**汇聚函数也要是单射的**，下面就来分析这个汇聚函数。

汇聚函数可以看出一个输入是 multi-set 的函数（multi-set 也就是有重复元素的集合，比如在某一层有节点的特征是一样的），下面来看一下 GCN 和 GraphSAGE 的汇聚函数：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220413231318-dz8e50q.png" alt="" style="zoom:50%;" />

GCN 使用的是平均池化，因此如果multi-set相同特征的节点的比例一样多，那么这个汇聚函数就不能分辨不同的multi-set，比如假设黄色绿色节点特征为 one-hot（黄色(1,0)，绿色(0,1)）:

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220413232159-pr3xfon.png" alt="" style="zoom:50%;" />

GraphSAGE 的汇聚函数有多种选择，这里以最大池化为例，对于最大池化，如果multi-set 中相同特征的节点集合是一样的，那么也不能分辨：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220413232344-ncyoqnr.png" alt="" style="zoom:50%;" />

因此 GCN 和GraphSAGE 的汇聚函数都不是单射函数，所以这两种 GNN 都不是最具表达力的 GNN。那么如何设计这样的 GNN 呢？一种好的解决方法是利用神经网络学习到这种单射的汇聚函数。

我们可以将单射的 multi-set 函数表示为：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220414091431-ew3t2z9.png" alt="" style="zoom:50%;" />

f 进行非线性转化（可以假设转化后的是 one-hot），sum操作可以保留转换后的输入信息，接着 $\phi$ 再进行一个非线性转化，而对 f 和 $\phi$ 则可以使用 MLP 进行近似（一层的MLP就可以逼近任何连续函数），这样得到了最具表现力的 GNN 模型：Graph Isomorphism Network (GIN)，通过 MLP + sum + MLP 学习单射汇聚函数。

### GIN VS WL

在第二课的图特征中讲过 WL 核，简要的步骤是：初始化每个节点的颜色；收集每个节点邻居节点的颜色并用预定义的 HASH 函数将收集的颜色映射到新的颜色；迭代收集-映射步骤，在 K步迭代后，每个节点就可以收集 K-hop 的邻居节点信息。在达到稳定状态后，如果两个图有个一样的节点颜色集合，那么这两个图就是同构的（isomorphic）。从这个描述我们可以看到 GIN 就是使用神经网络来学习这个 HASH 函数，WL 在收集信息时是将邻居的颜色和自己节点的颜色合并在一起，因此 GIN 也可以这么做：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220414093747-wtvl7k1.png" alt="" style="zoom:50%;" />

由于 GIN 和 WL 的这种关联使得 GIN 和 WL 图核的表达能力是相似的，而WL已经被理论和实践证明可以区分大部分实践的图结构，因此 GIN 也具有区分大部分图结构的能力。

## 第十课

目前遇见的图的边都只有一种类型（虽然可以有权重），这一课主要是讲有着**多种边和节点类型的有向图**的处理方法，这种图也叫异质性图 (heterogeneous graphs)。

### 异质性图和相关GCN（RGCN）

异质性图定义为：

$$
G = (V,E,R,T)
$$

* V：节点，$v_i \in V$
* E ：边（边类型 r），$(v_i,r,v_j) \in E$
* T：节点类型
* R：关系（边）类型

现实世界有很多异质性图的例子，比如下面的生物医学知识图：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220502220107-1g02ryx.png" style="zoom:50%;" />

不同的节点形状表示不同的节点类型，比如药物，疾病，蛋白等；边的类型也不一样，比如有 target，cause 等。

我们可以将 GCN 拓展到有着不同边类型的异质性图上，先看单一边类型的有向图，假设我们想要得到下图 A 的 embedding：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220502221528-co4mlt1.png" style="zoom:50%;" />

需要将之前的信息传递过程变成沿着图中边的方向的信息传递：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220502221743-myocld5.png" alt="" style="zoom:50%;" />

接下来将其拓展到有着多种边类型的图：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220502221825-fa9ib8s.png" style="zoom: 67%;" />

可以在每一个 GCN 层中对不同的边类型使用不同的权重（也就是一层中参数不是在所有节点中共享的）

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220502222038-3n215541-20220502222135-qgujyt9.png" style="zoom:50%;" />

上图不同的颜色表示不同的权重矩阵，因此 RGCN 的信息传递和汇聚过程可以表示为：

* 信息传递：分为两部分

  * 一个是节点的边类型为 r 的邻居节点（$c_{u,r}$ 表示类型为 r 的边所定义的自由度）：

    $$
    m_{u,r}^{(l)}=\frac{1}{c_{v,r}}W_r^{(l)}h_u^{(l)}
    $$

  * 一个是节点自身：

    $$
    m_v^{(l)} = W_0^{(l)}h_v^{(l)}
    $$

* 信息汇聚：求和，再进行激活函数操作

#### RGCN regularize the weights

但是这样的拓展会带来一种问题：如果一层中边的类型非常多，那么就需要**不同的权重矩阵**，造成参数的激增；有两种方法来缓解这种问题：

* 使用分块对角矩阵
* 基本学习或者叫字典学习

原本对每个边类型在每一层都会有一个矩阵，每个矩阵的大小是 $d^{(l+1)} * d^{(l)}$，而使用分块对角矩阵可以使得权重矩阵稀疏化，因此减少参数（*没有明白*）：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220502223658-0zwaq74.png" alt="" style="zoom:50%;" />

而字典学习则是将不同边类型的矩阵转化为基本矩阵 ($V_b$) 的不同线性组合（$\sum_{b=1}^Ba_{rb}$），而这个 V 则是对于不同的边类型是相同的，所以这个 $a_{rb}$ 可以看出基本矩阵的重要性（权重），因此我们只需要学习这个权重就行了，大大减少了参数量。

#### RGCN example

对于节点标签的预测和之前没有什么区别，都是用最后一层的 embedding 连接一个 softmax，得到 k 类的概率。

对于边预测任务的数据集划分在第八课中讲过了，这里每个边又有不同的类型，这个类型是独立于在第八课中的 Transductive 划分方法中的四类边的（Training message, Traning supervision, validation 和 test）(这里不使用 Inductive ，因为如果随机划分的话，不同类型的边的 message 和 supervision 可能不一样多)。因此对于不同类型的边，分别划分四类边，最后将不同类型的边相应的划分合并：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220503200409-plb1fpf1-20220503200518-708kzka.png" style="zoom: 50%;" />

下面来看一个例子：

* 在训练步骤，假设合并后使用 $(E,r_3,A)$ 作为 supervision 边，其他的都是 message 边：

  <img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220503201208-5h36fg3.png" alt="" style="zoom:50%;" />

  1. 使用 RGCN 对 supervision 边打分（取最后一层的 E 和 A 的 embedding，传给某个打分函数，比如直接 $h_E^T W_{r_1}h_A$）
  2. 通过对 supervision 边的打乱 （比如取 $(E,r_3,B)$）构建负例边，注意负例边不能是 supervision 或者 message 边（比如 C）
  3. 使用 RGCN 对负例边进行打分
  4. 通过交叉熵 loss 优化模型参数（最大化supervision边，最小化负例边）

* 在评估步骤（validation 或者 test）validation 边为 $(E,r_3,D)$  

  <img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220503203346-kxqvpf8.png" style="zoom:50%;" />

  1. 使用上一步训练的模型计算边 $(E,r_3,D)$ （使用 supervision 和 message 进行 RGCN 计算，然后预测 ED）
  2. 计算所有负例边的值，负例边不能是 supervision 或者 message 边，因此是 EB 和 EF
  3. 计算 validation  边的 Rank
  4. 计算评估指标，可以有两个选择：

     1. Hits：validation 边有多少比例在 top k 的边里面
     2. $\frac{1}{Rank}$：validation 边的 rank 越高，这个值就越大

### 知识图：KG completion with embeddings

知识图是异质性图的一种，节点是实体（entities），节点有类别标签，连接两个节点之间的边代表着节点之间的关系（relationships）：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220503222508-5krntw4.png" alt="" style="zoom:50%;" />

比如一个书目网络，节点的类型是文章，标题，作者，会议，年份等；而节点之间的边可以表示文章发表在哪里，发表的年份，有什么样的标题，作者是谁这些关系：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220503223140-lybzw0e.png" style="zoom:50%;" />

现在已经有很多的知识图的数据，特点是数据量比较大；另外信息不是很完整，比如很多真实的边是丢失的，因此一个重要的任务就是对这些缺失边的填补（**Knowledge Graph Completion**）：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220503223407-t39ysfi.png" style="zoom:50%;" />

KG 补全和边预测任务还不是一样的，KG补全是给定一个起始节点（head）和边的类型，预测终止节点（tail），比如下图中给定作者 J.K. Rowling 和边 genre (体裁) 预测尾节点 Science Fiction:

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220504212618-79vdsnd.png" style="zoom:50%;" />

在知识图谱中的边可以表示成三元组 $(h,r,t)$ ，h 表示 head，t 表示 tail，r 表示 relation；在这个任务中我们使用的是最开始讲过的 shallow embedding，也就是对每个节点学习一个 embedding，而不是使用 GNN（GNN 是对一层所有节点共享参数，而 shallow embedding 是每个节点都有系列参数）。主要想法就是对于一个实际存在的 $(h,r,t)$，$(h,r)$ 的 embedding 应该和 t 的 embedding 接近，问题就是如何得到 $(h,r)$ 的 embedding 以及怎么定义“接近”

#### TransE

一个想法就是如果能从 h 节点沿着边 r 移动到 t 节点，那么说明 h 和 t 之间就是有这样的连接的：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220506222440-rgvk0g4.png" style="zoom:50%;" />

因此我们可以使用这样的打分函数：

$$
f_r(h,t)=-||h+r-t||
$$

TransE 的学习算法：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220506222750-81bp56g.png" alt="" style="zoom:50%;" />

关键有 3 步：

* 节点（实体）和边（关系）首先初始化和标准化
* 进行负采样，产生一些不在 KG 中的三元组（比如可以固定 head 随机定义没有边连接的 tail）
* 依据上面蓝色方框中的 loss 进行更新 embedding -- 最小化这个 loss 就是需要前面的正例样本的距离比较小，后面的负例样本的距离比较大

在异质性的 KG 中关系有着不同的模式：

* 对称（反对称）关系：$r(h,t) \Rightarrow r(t,h)$ 或者 $r(h,t) \Rightarrow \lnot r(t,h)$；比如同桌关系，A 是 B 的同桌，B 肯定也是 A 的同桌，反对称关系比如上位词和下位词对应
* 相反关系：两个节点之间当边的方向相反，关系也会颠倒，$r_2(h,t) \Rightarrow r_1(t,h)$ ；比如导师和学生指导和被指导的关系
* 可传递的关系：$r_1(x,y) \land r_2(y,z) \Rightarrow r_3(x,z)$ ，比如我的母亲的丈夫是我的父亲
* 1 对 N 关系，从一个节点有连接多个节点的关系，$r(h,t_1),r(h,t_2)$，比如一个老师和班里的所有学生

先来看 TransE 可以处理上述关系中的哪些。

1. 反对称关系 ✅，h 可以经过 r 移动到 t，但是 t 不能继续移动 r 到 h：

   <img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220506225639-q5vbz92.png" style="zoom:50%;" />

2. 相反关系 ✅，h 可以通过 r2 移动到 t，我们可以将 r1 设成负的 r2，这样 t 就能通过 r1 回到 h（下面的图画反了）：

   <img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220506225934-6dyppt9.png" style="zoom: 67%;" />

3. 可传递关系 ✅，x 可以通过 r1 到达 y，然后再通过 r2 到达 z，根据向量的加法我们可以将 r3 设为 r1 + r2，那么就可以通过 x 直接从 r3 到 z：

   <img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220506230209-axr16dq.png" alt="image.png" style="zoom:50%;" />

4. 对称 ❎，如果对于 h，t 要同时满足 $r(h,t),r(t,h)$ 都存在，那么 $||h+r-t|| =0$ 并且 $||t+r-h|| =0$，因此 r = 0 并且 h = t ，但是 h 和 t 是两个不同的节点，所以 TransE 不能对对称关系建模

5. 一对多关系 ❎ 因为 t1 和 t2 会映射到同一个节点，但实际上并不是，所以 TransE 不能对一对多关系建模：

   <img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220506231759-vdgnnd3.png" style="zoom: 50%;" />

#### TransR

