---
title: 一致性聚类（Consensus Clustering）
author: wutao
date: 2021-08-31 17:03:30
slug: lagrange_duality
categories:
  - R
  - Statistics
tags:
  - notes
index_img: img/cc.png
---



一致性聚类概念及R实现

<!-- more -->

## 什么是一致性聚类

聚类方法一般分为两大类：划分聚类（如 k-means，k-medoids 等）和层次聚类，对于划分聚类我们需要指定聚类的数量，并且初始值的随机选取也会带来聚类结果的不稳定；对于层次聚类，尽管不需要随机选取初始值，但是我们仍然需要选择如何切割聚类树得到最终的聚类结果，因此聚类分析中一个重要的过程就是如何选择聚类的数量以及聚类结果的稳定性。

一致性聚类并不是一种聚类方法，而是通过在数据子集上多次迭代运行我们选择的聚类方法，从而提供关于聚类稳定性和参数选择（比如类的数量）的一种指标。一致性聚类的一般步骤为：

-   选择想要测试的一系列 k 值（类别数），和迭代的次数
-   对于每个 k，在每次迭代中选择一定比例的样本子集（如 80%），在该子集上运行需要测试的聚类方法（k-means 等）
-   对每个 k，迭代完后创建一个一致性矩阵（consensus matrix）
-   基于一致性分布选择最优的矩阵

下面举一个简单的例子说明上述过程：假设现在想要使用 k-means 聚类来对四个用户进行聚类，迭代次数为 4 次，K=2 时结果如下（0 表示在这次迭代中该用户没有被抽到）：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/1K80zNAiLjMAO6WcgNqh1GA.png)

可以看到 Brian 和 Sally 在第 1 次和第 4 次迭代中被聚在一类，Brian 和 James 在第 3 次迭代中聚在一类，而 Sally 和 James 在第 2 次中聚在一类，Alice 在四次迭代中没有和任何用户聚在一类。接下来就要生成一致性矩阵，这个矩阵表示两两样本间的关系；矩阵中的每个值表示对于两个样本，在他们同时被抽到的迭代中有几次是被聚在一类中的，比如 Brian 和 Sally 在三次迭代中同时被抽中（1，2，4）并且在这三次中有两次是被聚到一类的（第 1 和 4 次迭代），所以对应矩阵的值就是 2/3 = 0.67：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/1SD5g5G3Sv2-HXxgZLlIsw.png)

然后对于每个 k，我们都可以生成一个这样的矩阵 M；一致性矩阵可以提供另一种样本间相似性的度量，因此我们可以将 M 视为新的距离矩阵进行层次聚类来将样本进行分类，下图是 k=3 时的例子：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/t8fuK051LqMw0wtjw8wayw.png)

接着检查每个矩阵的一半（上三角或下三角）中的值的分布，选择分布 “最好” 的矩阵；“最好” 指的是大部分值接近 0 或者 1，这个结果说明比较一致，比如下面的分布：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/UvsYV8moEE9SnPfx89ufQQ.png)
对于一致性聚类还有一些重要的汇总统计量： 

- 聚类一致性（cluster consensus）：在每个类别（*I*<sub>*k*</sub>）中，样本对的平均一致性值 Nk是该类的样本数)，聚类一致性可以用来衡量聚类的稳定性：

$$
m(k)=\frac{1}{N(k)(N(k)-1)/2}\sum_{i,j\in I_k,i<j}M(i,j)
$$

- 样本一致性（item consensus）：某个样本与在某个类别中的其他所有样本的平均一致性值，样本一致性可以用来对类别中样本进行排序，找到有代表性的样本有哪些：

$$
m_i(k)=\frac{1}{N_k-1\{ei ∈ I_k \}}\sum_{j\in I_k,j\neq i}   M(i,j)
$$

## 用 R 包 ConsensusClusterPlus  实现一致性聚类

### 准备输入数据

输入数据需要是一个矩阵，列是样本，行是特征，这里使用 ALL 的基因表达数据：

``` r
library(ALL)
data(ALL)
d=exprs(ALL)
d[1:5,1:5]
>>              01005    01010    03002    04006    04007
>> 1000_at   7.597323 7.479445 7.567593 7.384684 7.905312
>> 1001_at   5.046194 4.932537 4.799294 4.922627 4.844565
>> 1002_f_at 3.900466 4.208155 3.886169 4.206798 3.416923
>> 1003_s_at 5.903856 6.169024 5.860459 6.116890 5.687997
>> 1004_at   5.925260 5.912780 5.893209 6.170245 5.615210
```

为了选择更有信息的基因进行样本的分类，我们依据 MAD（median absolute deviation，中位数绝对偏差，每个样本点离中位数的绝对值的中位数）来选择变化最大的前5000个基因（也可以使用其他的筛选指标）：

``` r
mads=apply(d,1,mad)
d=d[rev(order(mads))[1:5000],]

##中位数中心化
d = sweep(d,1, apply(d,1,median,na.rm=T))
```

### 运行 `ConsensusClusterPlus`

接下来就需要选择参数来运行一致性聚类：

-   pItem：重抽样样本的比例，设置为 80%
-   pFeature：重抽样特征的比例，设置为 80%
-   maxK：最大评估的类别数（k），设为 6
-   reps：对每个 k 的迭代次数，选择 50
-   clusterAlg：聚类算法，选择层次聚类
-   distance：聚类距离，选择 Pearson 相关距离
-   title：图和文件存放的位置

在实际情况中需要设置更高的迭代次数（一般可以设 1000）和更大的聚类数量。

``` r
library(ConsensusClusterPlus)
title="./fig"
results = ConsensusClusterPlus(d,maxK=6,reps=50,pItem=0.8,pFeature=1,title=title,clusterAlg="hc",distance="pearson",seed=1262118388.71279,plot="png")
>> end fraction
>> clustered
>> clustered
>> clustered
>> clustered
>> clustered
```

输出是一个列表，其中的元素是每个 k 的运行结果，包括一致性矩阵，一致性聚类结果等：

``` r
##我们可以获取其一致性矩阵
results[[2]][["consensusMatrix"]][1:5,1:5]
>>           [,1]      [,2]      [,3]      [,4]     [,5]
>> [1,] 1.0000000 1.0000000 0.8947368 1.0000000 1.000000
>> [2,] 1.0000000 1.0000000 0.9142857 1.0000000 1.000000
>> [3,] 0.8947368 0.9142857 1.0000000 0.8857143 0.969697
>> [4,] 1.0000000 1.0000000 0.8857143 1.0000000 1.000000
>> [5,] 1.0000000 1.0000000 0.9696970 1.0000000 1.000000

##一致性聚类结果
results[[2]][["consensusClass"]][1:5]
>> 01005 01010 03002 04006 04007 
>>     1     1     1     1     1
```

### 计算一致性汇总统计量

汇总统计量包括：聚类一致性（cluster consensus）和样本一致性（item consensus）：

``` r
icl = calcICL(results,title=title,plot="png")

##返回有两个元素的列表
icl[["clusterConsensus"]]
>>       k cluster clusterConsensus
>>  [1,] 2       1        0.7681668
>>  [2,] 2       2        0.9788274
>>  [3,] 3       1        0.6176820
>>  [4,] 3       2        0.9190744
>>  [5,] 3       3        1.0000000
>>  [6,] 4       1        0.8446083
>>  [7,] 4       2        0.9067267
>>  [8,] 4       3        0.6612850
>>  [9,] 4       4        1.0000000
>> [10,] 5       1        0.8175802
>> [11,] 5       2        0.9066489
>> [12,] 5       3        0.6062040
>> [13,] 5       4        0.8154580
>> [14,] 5       5        1.0000000
>> [15,] 6       1        0.7511726
>> [16,] 6       2        0.8802040
>> [17,] 6       3        0.7410730
>> [18,] 6       4        0.8154580
>> [19,] 6       5        0.7390864
>> [20,] 6       6        1.0000000
icl[["itemConsensus"]][1:5,]
>>   k cluster  item itemConsensus
>> 1 2       1 28031     0.6173782
>> 2 2       1 28023     0.5797202
>> 3 2       1 43012     0.5961974
>> 4 2       1 28042     0.5644619
>> 5 2       1 28047     0.6259350
```

### 可视化

在我们运行上面的 `ConsensusClusterPlus` 时指定输出的路径（title）和 plot 不为 None 时，就会在指定的路径中生成图片：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210831191516691.png)

第一个图（consensus001）是依据一致性矩阵生成的热图的图例，后面（002-006）是不同 k 的一致性矩阵热图：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210831192023694.png)

### 决定聚类的数目

前面也讲过，对于一个 “完美” 的一致性矩阵，里面的值应该只有 0 和 1，因此我们可以看在不同的 k 时得到的一致性矩阵偏离这个完美状态的程度来衡量聚类的稳定性。对于一致性矩阵中值的分布可以使用累计密度函数（CDF）来衡量（生成文件 consensus007）：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/consensus007.png)
更进一步量化可以计算随着聚类数量的增加 CDF 的曲线下面积的相对变化值，然后根据 “拐点法” 来选择最优的聚类数量（consensus008)：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/consensus008.png)
上图说明可以选择 4 或者 5 类作为最终的聚类数量。

另一个图是 Tracking Plot（consensus009），列是样本，行是每个 k 值，颜色表示在一致性矩阵中的类别，因此这个图展示了在不同的 k 的情况下，样本所属的类的变化，如果样本的类别经常变化，说明该样本是不稳定的样本，含有大量这种样本的类也是不稳定的类：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/consensus009.png)

还有两个图就是前面展示的汇总统计量，聚类一致性和样本一致性的可视化：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/icl003.png)

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/icl001.png)
