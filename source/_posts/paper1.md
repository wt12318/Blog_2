---
title: 【文献阅读】001
date: 2022-08-25 09:14:18
tags: paper
index_img: img/paper.png
categories:
  - paper
---



文献阅读笔记

<!-- more -->

### Mutational signatures are markers of drug sensitivity of cancer cells （2022-5-NC）

Umap k36 criterion ：定义了基因组的 mappability （Umap and Bismap: quantifying genome and methylome mappability）

构建样本*种系突变的矩阵（样本由细胞系和TCGA的正常样本构成，种系突变就是特征），对这个矩阵进行 PCA 降维，得到 150 个 PC；然后使用这些 150 个PC 构建随机森林模型来区分 TCGA 和 细胞系的样本，得到特征的重要性，将最重要的前10个特征去掉（因为这些 PC 可能是 batch effect，来自不同类型样本的差异），对剩下的 140 个 PC 进行聚类分析（robust 聚类，使用 `tclust` 算法，去掉离群样本）。利用**模拟的细胞系外显子组**得到最佳的聚类数量（13个），将聚成一类的里面的 TCGA 正常样本叫做 ancestors，用这个 ancestors 来辅助种系 SNV 的筛选：对TCGA正常样本和细胞系样本按照一般流程筛选剩下的 SNV 构建三核苷酸突变谱（2个矩阵，96\*细胞系样本+ 96 * TCGA正常样本），对于细胞系的 96 个特征都减去该细胞系匹配的 ancestors 样本中相应特征的中位数（如果减的结果是负值就归0），整个流程如下：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220613160007-w34ax90.png" style="zoom:67%;" />

所谓的模拟细胞系外显子组的做法：对细胞系和 TCGA 数据集匹配的癌症类型，在 TCGA 癌症样本中随机抽 450 个做 mutation calling（也就是不用正常样本）将这个结果作为模拟的细胞系 variant calls，然后走上面的流程得到模拟的 96 突变谱矩阵；然后将这个矩阵和实际 TCGA 样本得到的 SNV （tumor normal 配对）构建的 96 突变谱比较，也就是看是否能够重建突变谱（用Absolute Error 衡量，两个矩阵相应位置的差值总和）。

耐药预测：建立随机森林回归模型（因变量是 LnIC50），使用 **RRMSE** 来评估（relative root-mean-square-error），也就是模型的 RMSE 除以 default 模型的 RMSE，default 模型的所有样本的 IC50 都是一样的常数（训练集 lnIC50 的均值），因此 RRMSE 小于1表示模型的性能比无信息模型要好。

评估模型的性能差异是否显著：corrected **Friedman test**  + post-hoc **Nemenyi test** ；Friedman test 是 two way ANOVA 的非参版本，这里就是检验考虑 dataset（block 变量）时不同的 predictor （group 变量）的性能是否有差异（R 中使用 `friedman.test(y, groups, blocks)` 来计算）；Friedman test 只能告诉我们是否有差异，比较 group 变量之间的两两差异需要使用 `Nemenyi test` ：对于每个 drug-cancer 数据集中的不同的 predictor 进行排序（1是最好的），然后对每个 predictor 取不同数据集的 rank 均值，再对不同的 predictor 两两比较其 rank 均值的差异，如果这个差异大于给定显著性水平下的 CD 值（critical difference）则拒绝原假设（两个 predictor 没有差异），下图中如果两个变量之间有连接则说明没有显著差异：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220613234135-bl70104.png" style="zoom:67%;" />

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220613234237-42996q11-20220613234348-unqjw3w.png" style="zoom:67%;" />

为了检验 marker （signature ，CNA，甲基化等）和药物敏感性的相关在不同的数据集上是否可重复，作者的方法：考虑了3 个不同的数据集，分别是PRISM中的同一种药物，GDSC中有着相同target的药物和Project SCORE中的药物靶标基因的 fitness score（CRISPR/Cas9敲除后细胞的适应度）；这里使用 Cohen's d 统计量来评估不同的feature和药物反应的相关性：有着这种 feature的细胞系的平均 lnIC50 与没有这种feature的细胞系的平均lnIC50的差值除以所有细胞系的lnIC50的标准差；负的Cohen's d表示sensitivity，因为有这这种 feature的细胞比没有这种feature的细胞的IC50要小（论文中的表述应该有点问题），正的Cohen's d则表示resistance，下图中的柱状图表示了这个值。为了检验在不同数据集上这个统计量的 effect size 是否显著比随机的大，作者分别对sensitivity和resistance取 min 和 max，然后应用了随机化来获得 p 值：随机打乱 feature 的标签（有这个feature还是没有这个feature）计算 Cohen's d 统计量获得 min 和 max 值，接着根据这个随机的空分布计算 p 值（下面图中所示）：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220614100206-x44p1h4.png" style="zoom:67%;" />

参考：

[(2) R - Friedman post-hoc: Nemenyi - YouTube](https://www.youtube.com/watch?v=g1TKACbl_O8)

[(2) Brandon Foltz - YouTube](https://www.youtube.com/c/BrandonFoltz/search?query=ANOVA)

### Network-based machine learning approach to predict immunotherapy response in cancer patients （2022-6-NC）

NetBio 基于网络发现 ICI 的生物标志物

主要挑战：标志物在多个病人的 cohort 中都能较稳健的预测药物反应

> Strikingly, however, other studies have reported no signifificant correlation between PD-L1 expression and the ICI treatment response, and some studies have even revealed that ICI responders display low PD-L1 expression levels

> For example, Hofree et al.showed that patients with somatic mutations in similar network regions displayed similar clinical outcomes, although many clinically identical patients share no more than a single mutation

虽然两个病人可能有的突变不一样，但是 network alteration 可能是一样的，因此造成的表型是相似的，network--gene module 比单个基因含有的生物学意义的层级可能更高。

---

作者提到了他们之前发表的一篇文章：Network-based machine learning in colorectal and bladder organoid models predicts anti-cancer drug efficacy in patients

利用 PPI 网络和已知的药物靶点来识别和药物反应相关的潜在生物通路，具体的流程如下图:

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220719103122-caxz9x9.png" style="zoom:50%;" />

这篇文章的一个主要的创新点就是特征挑选过程，选择和已知的 drug target 距离近的 pathway，再利用岭回归对 IC50 和这些 pathway 基因的表达进行建模预测药物反应。距离的计算使用的是 pathway 基因和药物相关基因之间最短路径的长度的平均值：

$$
d_c=\frac{1}{|T|}\sum_{t\in T}min_{s\in S}d(s,t)
$$

T 就是药物相关基因，S 是 pathway 里面的基因，$d(s,t)$ 是某个药物相关基因和某个 pathway 基因的最短路径长度；也就是对每个药物相关基因找到和其距离最近的 pathway 基因并计算其距离，对所有的药物相关基因都计算一遍，然后取个均值。这个距离 $d_c$ 的显著性计算是利用 bootstrap 得到的：随机选择和药物相关基因以及 pathway 基因相同数量的基因集，并且随机选择的基因节点的度和实际基因集的度也要一样，再计算上面的距离，这个随机过程重复 1000 次，如果实际的距离小于 90% 的随机模拟距离，那么就认为是显著的结果（结果都进行 zcore 转化）。

其他的亮点：

* 在分析病人的反应时进行了 negative control，作者首先对接受药物治疗的病人用模型预测有反应和不反应，然后比较这两类病人的生存；对于 negative control 则是用没有接受治疗的病人，发现接受治疗的病人中模型预测的两类病人之间的生存差异是显著的，但是在没有接受治疗的病人中则没有显著差异，说明这种生物标志物是和药物反应相关，而不是简单的和生存相关
* 多方面的比较，包括整合多个 pathway 的预测能力（整合多个 pathway 效果并没有提升），多种模型的预测能力（岭回归，线性回归，支持向量机，深度学习模型），多种特征选择方法的预测能力（所有的 pathway 数据，也就是所有的表达数据，没有经过特征筛选；基于网络中心性挑选特征；选择药物靶点的直接邻居节点作为特征；选择表达水平和药物反应高度相关的基因作为特征）

---

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220719141541-fjl54qb.png" style="zoom:50%;" />

这篇文章的基本想法和之前的差不多，都是寻找和药物靶点在 PPI 网络上接近的基因，方法有所不同。这篇文章用的是网络传播算法（page-rank 算法，见[图机器学习 - wutao`&#39;'`s Blog (wutaoblog.com.cn)](https://wutaoblog.com.cn/2022/03/13/gnn/#PageRank)），从免疫治疗的靶点基因（PD1/PDL1）出发进行传播，计算邻居节点的影响力分数，选择排在前 200 的基因，再对这些基因做通路的富集分析（超几何检验）得到富集的通路，使用这些通路的基因表达作为特征进行后续的建模，使用的是逻辑回归模型。做的比较：

* 训练和测试数据：

  * within-study，训练集和测试集来自同一个 cohort

    * 留一交叉验证，也就是只留一个样本来验证，最后每个样本都有一个预测值
    * Monte Carlo 交叉验证，每次随机选择 80% 的样本训练，剩下的 20% 验证，重复 100 次
  * across-study，两个独立的数据集作为训练集和测试集，测试泛化能力

    * 使用训练集的全部样本训练
    * 抽训练集的 80% 样本训练，看看数据量减少的训练集是否会影响模型性能
* 特征比较

  * 免疫治疗靶基因的表达（PD1 PDL1 CTLA4）
  * 基于肿瘤微环境的 biomarker
  * 数据驱动的机器学习方法挑选特征
* 性能比较

  * 免疫治疗反应（RECIST）AUROC， F1，accuracy
  * 病人生存
  * IMvigor210 数据集的 tumor proportion scores (TPS)，这个数据集即测了 RNA-seq 也有 TPS

作者还依据在免疫治疗数据集上训练的模型来预测相应癌症类型的 TCGA 肿瘤样本，比较了预测的值和免疫的相关性。结合了 TMB 和作者依据 PPI 提取的基因训练一个整合的模型的效果有所提升：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220719152029-ak0aozz.png" style="zoom:50%;" />

注意上图中两者都有差异，因此作者在文中比较的是一年生存差异（从 18% 提高到 22%）和 Log-Rank 的 t 统计量差异。

### Graph Neural Networks for Double-Strand DNA Breaks Prediction（2022-1-arxiv）

这篇文章使用图神经网络基于 DNA 序列特征和染色体结构特征来预测 DNA 双链断裂。

首先对染色体结构进行图建模：

* 节点：将基因组分成 5kb 的 bins，每个 bins 就是一个节点，每个节点的标签为这个 5 kb 的区域有没有 DSB（1/0）

* 节点特征：将序列分成 k-mers 然后计算每个 kmer 序列在 5kb 的bin 里面的频率作为特征，还有每个 bin 里面的 ATCG 的频率分布：

  <img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220604095429-2g8a0sm.png" alt="" style="zoom: 50%;" />

  <img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220604095855-5bianzb.png" alt="" style="zoom: 33%;" />

* 边：bin 之间的相互作用强度，HiC 接触图谱中的相应位置的互作频率数值

图模型架构：

* GAT 加入了边的特征，一般的 GAT 是使用两个节点的特征得到注意力分数，这里面另外加上了边的特征（边的特征是通过对 bins 节点之间的相互作用强度进行线性转化得到的 edge encoding）：

  <img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220604101539-6ictc8b.png" style="zoom:50%;" />

* Centrality encoding and positional encoding：节点中心性编码考虑了节点的度和 PageRank 的结果（参考了 Graphormer）；位置编码使用的是 bin 节点的 ID（在基因组上的先后顺序），对于这些编码分别进行线性转化（度，PageRank，Position），使得其维度和节点特征的维度一样；然后将节点特征与这些表示节点重要性和位置的编码相加得到最终 GAT 模型的输入

* Jumping Knowledge (JK) network：最终的节点表示整合了之前每个层的节点表示

整体的模型框架：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220604101749-ufad6nc.png" style="zoom: 33%;" />

关于 Centrality encoding：先初始化两个矩阵，分别和图中节点的出度和入度相关，对一个节点而言根据其出度和入度在这两个矩阵中检索（比如这个节点的入度是 2，出度是 3，那么就选择下面两个向量，分别和该节点的 embedding 相加）：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220825101839736.png" alt="" style="zoom:50%;" />

参考：

[Graphormer — the generalization of the transformer architecture to graphs | by Cheng Jing | MLearning.ai | Medium](https://medium.com/mlearning-ai/graphormer-the-generalization-of-the-transformer-architecture-to-graphs-4838c55b38ae)

[(20条消息) JK-Nets在引文网络上的应用【jumping knowledge】_智慧的旋风的博客-CSDN博客](https://blog.csdn.net/weixin_41650348/article/details/112960544)



### KG4SL: knowledge graph neural network for synthetic lethality prediction in human cancers (2021-7-Bioinformatics)

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220823083609653.png" style="zoom:50%;" />

有合成致死效应的基因对在功能上是互补的，任何一个基因发生了功能性突变并不会对表型（细胞活力）造成显著的影响，但是基因对上的基因全部失活的话会导致细胞活力显著下降。因此合成致死在癌症治疗上是一种有前景的治疗策略，当癌细胞中有个特定的基因发生了突变时，使用药物来抑制这个基因的合成致死的伙伴基因就会使得癌细胞死亡，而不伤害正常细胞。

SL 预测的计算方法可分为三类：

* 使用代谢网络模型模拟敲除
* 使用已知的知识进行特征工程
* 基于机器学习的方法

目前基于 GNN 的方法一般将每个 SL 对视为独立的样本，忽视了他们之间可能共有的生物学机制。本文提出了一种新的基于知识图的预测方法，可以部分解决样本独立性的问题（通过知识图为 SL 对发现添加额外的生物学知识）。

数据来源：SynLethDB，包含 SL 基因对（72 804 gene pairs between 10 004 genes）和一个知识图（11 种实体和24种关系）。

方法分为三个步骤：

* 从知识图中提取 SL 相关基因的子图：分成两步，选择节点和计算边的权值；第一步对每个基因，在每个 Hop 都抽取 k 个邻居（小于 k 就进行有放回的抽样）构成子图；对于每个基因子图对边赋予权重来表示关系的重要性：对于一个 SL 基因对 $(e_i,e_j)$ ，**$e_i$ 子图中边的权重由该边（关系）的 embedding 和基因 $e_j$ 的 embedding 内积计算得到**；通过这种方式来对不同基因的生物学作用之间的关系建模
* 节点表示的更新：和一般的 GNN 框架一样，先是聚合邻居节点的信息，然后更新节点表示；聚合采用的是对邻居节点表示的加权平均，权重是对上面边的权重进行 softmax 归一化后的值（也就是对子图中的一个节点，其连接的边的权重归一化后加起来是 1，然后用这个归一化后的权重进行加权平均）；更新就是将当前节点的表示和聚合得到的邻居节点表示相加后进行 MLP。
* SL 预测分数：当通过上面的步骤对知识图中的两个子图进行 GNN 学习后（对应两个基因）可以得到两个基因的 embedding，然后就可以计算这两个基因之间的互作概率：$\hat{s_{i,j}}=\phi(f(\hat{e_i},\hat{e_j}))$ f 是内积，$\phi$ 是 sigmoid 函数，将值压缩到 0-1。可以将阈值设置为 0.5，来预测这两个基因是不是 SL 对。

最后的损失函数是预测的标签（是不是 SL 对）和实际标签之间的交叉熵再加上个 L2 的正则项（用来限制基因，关系的 embedding 以及边的权重大小）。

### Multi-omic machine learning predictor of breast cancer therapy response (2021-12-Nature)

收集了 180 个乳腺癌的病人样本，在治疗前进行测序，包括 DNA-seq，RNA-seq 以及病理切片的染色图像数据，接着进行化疗或者靶向治疗，使用 RCB （Residual cancer burden）来评估病人的反应：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220823214349-yhy8llo.png" style="zoom:50%;" />

这篇文章前面都是一些常规的分析，包括：

* 不同临床特征的反应的关系，不同基因组特征和反应的关系，这些基因组特征有癌症驱动基因的突变，肿瘤突变负荷，突变 signature，拷贝数变异，HLA 位点的 LOH
* 对反应和不反应的病人做基因表达差异分析然后对差异基因进行通路富集分析
* 进行 GSVA 分析，看增殖，免疫相关基因集和反应的关系，用到了 Genomic Grade Index (GGI) gene set，embryonic stem-cell metagene，taxane response metagene，STAT1 gene expression module（免疫反应）；还对病理切片数据进行分析得出淋巴细胞密度
* 在 45 个 GGI 和 STAT1 高的肿瘤中有 26 个仍然是没有 pathological complete response (pCR)，用 TIDE 分析了 T 细胞 dysfunction 和 exclusion 发现 dysfunction 是富集的

不同组学数据的整合就是简单的将不同的组学特征添加进模型里面，依次训练 6 个预测器：(1) clinical features only, and adding (2) DNA, (3) RNA, (4) DNA and RNA, (5) DNA, RNA and digital pathology, and (6) DNA。每一个预测器都是由三个模型聚合而来，也就是预测值是三个分类器的预测值的平均。这三个模型都有四个步骤构成，前面的三个数据预处理处理步骤是一样的：

1. 对变量之间两两做相关分析，对于 Person 相关系数大于 0.8 的两个变量，只保留和响应变量相关性最好的一个变量
2. 根据两组比较（反应 VS 不反应）的 ANOVA F 值来保留排名前 k 的变量（使用的是 scikit-learn 中的 SelectKBest 类）
3. 对剩下的变量进行 Z score 标准化

最后一步就是进行模型的训练，三个模型分别是：逻辑回归，支持向量机和随机森林。

这篇文章的 sWGS 测序深度（中位数）是 0.1×，使用 sWGS 来获得拷贝数变异用到的工具有：QDNAseq--binning(100-kb window)+GC矫正和匹配率低的区域矫正，DNAcopy--Segmentation 使用 CBS 算法，ASCAT--拷贝数，肿瘤纯度和倍性估计，使用的输入是来自 QDNAseq 的 log ratio 和来自 HaplotypeCaller 的种系 SNP。





