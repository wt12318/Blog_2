---
title: 【文献阅读】002
date: 2022-09-10 09:14:18
tags: paper
index_img: img/paper.png
categories:
  - paper

---



文献阅读笔记

<!-- more -->

### Integration of tumor extrinsic and intrinsic features associates with immunotherapy response in non-small cell lung cancer （NC-2022-7）

尽管 HLA LOH 被认为是一种免疫逃逸的机制，很多有着 HLA LOH 的病人仍然有较好的 ICB 反应（在 NSCLC 中有 40% 的病人有 HLA-LOH）。

经典的效应 CD4+ T 细胞通过授权树突状细胞和分泌促炎细胞因子来帮助 CD8+ T 细胞发挥杀伤功能；也有研究表明，识别肿瘤表达的 HLA-II 类新抗原的 CD4 T 细胞也能够直接杀伤肿瘤细胞（参考文献：Naive tumor-specific CD4+ T cells differentiated in vivo
eradicate established melanoma；Tumor-reactive CD4+ T cells develop cytotoxic activity and eradicate large established melanoma after transfer into lymphopenic
hosts.）

表达 II 类 MHC 分子的肿瘤细胞可以刺激 CD4 细胞，但是 CD4 细胞中也有一些是抑制型的，比如 Treg，那么表达 II 类 HLA 的肿瘤细胞是否倾向于呈递 Treg 的新抗原，而不是杀伤性的 CD4 T 细胞可以识别的抗原？如何证明这一点？这一点是否也是一种免疫逃逸？肿瘤演化过程中是否存在这样的 shift，早期没有偏向性，晚期偏向呈递 Treg 的抗原？

常用的 PDL1 组化表达，TMB 和 HLA LOH 与预后生存没有显著的关系，并且 HLA LOH 病人中也有对免疫治疗反应的。

We performed single-cell profifiling on **10** dissociated tumor samples obtained from patients with NSCLC who had **never received treatment**，单细胞实验设计：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220830082006-2bnd7y4.png" style="zoom:67%;" />

**CD45 由一类结构相似、分子量较大的跨膜蛋白组成，广泛存在于白细胞表面。** CD45 分子在所有白细胞上都有表达，是细胞膜上信号传导的关键分子，在淋巴细胞的发育成熟，功能调节及信号传递中具有重要意义，其分布可作为某些 T 细胞亚群的分类标志。

计算 TMB：非同义突变的数量除以 panel 的大小（靶向测序或者 WES 测序，MB），非同义突变的定义：All non-silent somatic coding mutations, including missense, indel, and stop-loss variants with coverage >100× and an allelic fraction >5% for targeted gene panels and coverage >30× and an allelic fraction >10% for the whole exome, were counted as non-synonymous mutations.

HLA LOH 的检测参考：A pan-cancer organoid platform for precision medicine；Detection of human leukocyte antigen class I loss of heterozygosity in solid tumor types by next generation DNA sequencing

单细胞转录组分析时使用 BBKNN 来进行批次矫正，使用细胞表明蛋白表达的 marker 来区分 CD4 和 CD8 细胞，再去掉高可变基因和标准化基因表达后，使用 Leiden 聚类来识别 CD8 和 CD4 细胞群体中的亚群（有着不同的转录 profile）。

在 CD8 细胞群中鉴定出 6 个亚群，其中 5 个高表达一些典型的细胞杀伤基因（GZMA，GZMB，GZMH，NKG7 等等）然后就描述了这些细胞群体的一些特征以及结合已有文献的推断。接下来用 partition-based graph abstraction (PAGA，PAGA: graph abstraction reconciles clustering with trajectory inference through a topology preserving map of single cells) 方法解析不同 CD8 细胞群体之间的关系（也就是轨迹推断，**Trajectory analysis**）；发现 Prolif 群只和 GZMB 亚群相连，而这个 Prolif 群的细胞周期是活跃的（S 和 G2M score 比较大）说明 suggesting that GZMB expressing cells are the primary CD8+ T cell population undergoing proliferation and clonal expansion in the tumor microenvironment.

接下来作者分析了 CD4 群中的杀伤性细胞亚群，发现杀伤性 CD4 细胞表达更高的 IFG，而 IFG 可以诱导肿瘤细胞中的 HLA-II 的表达。

如果肿瘤微环境中有这种杀伤性的 CD4 细胞，那么肿瘤细胞就会被迫表达 MHC-II 从而处于不利地位（受到压力），那么肿瘤细胞有没有这个层面的免疫逃逸？也就是存在杀伤性 CD4 细胞的时候，肿瘤细胞呈递的 II 类新抗原的免疫原性比较低或者呈递那种与 Treg TCR 结合的新抗原来抑制免疫系统？

接下来作者检测了杀伤性的 T 细胞群体在肿瘤微环境中是否有克隆扩增，方法是分析单细胞 RNA-seq 数据中得到的 TCR 序列：**将包含相同的 TCR α 和 β CDR3 序列的 T 细胞群体定义为扩增的 T 细胞克隆**（至少含有 2 个细胞），发现在杀伤性 T 细胞群体中有着广泛的 T 细胞克隆扩增（Prolif 群最多，有 95% 是扩增的 TCR 克隆）。然后作者将 TCR 克隆和不同的细胞亚群进行 overlap 分析（不同细胞群体作为节点，节点之间的边为共享的 TCR 克隆数目）：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220906101736-i7x775u.png" style="zoom:60%;" />

说明两个 GZBM 群体被抗原刺激并进行活跃的克隆扩增，而 GZMK 和 GNLY 则是不活跃的杀伤性 T 细胞群体（而这些群体是 early stage 的）；对于 CD4 细胞而言，克隆扩增则比较少（大概有 54 % 的细胞属于扩增的 TCR 克隆）。

为了检测肿瘤细胞的 HLA 表达，对 CD45 negative 的细胞使用 scHLAcount 来检测细胞的 HLA 表达，然后使用 Leiden clustering 和一些 marker 将这些细胞分为肿瘤，内皮细胞和成纤维细胞：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220906103510-r5qzws6.png" style="zoom:60%;" />

发现有一部分肿瘤细胞表达 HLA-II，并且表达和其分子伴侣 CD74 的表达呈显著的正相关，而且 DRB 的表达比其他的 HLA-II 亚型表达更多，通过荧光染色发现表达 HLA-II 的肿瘤细胞和杀伤性 CD4 T 细胞在位置上比较接近。

通过比较单细胞数据中某个 cluster 和其他 cluster 的基因表达，找到前 25 的差异基因作为该 cluster 的 gene signature；然后对病人的 Bulk RNA 测序计算相应的 gene signature 的表达（基因列表中的基因表达的算术平均）并检测其和免疫治疗反应的关系；发现杀伤性 CD8 和 CD4 细胞的 gene signature 值和好的反应显著相关；然后作者检测了整体的 CD8 和 CD4 细胞的免疫浸润和反应的关系，发现都不显著，说明是杀伤性 T 细胞而不是所有的 T 细胞浸润对病人的 ICB 治疗获益相关。

接下来作者提出了 pan-T 细胞杀伤 signature 来预测 ICB 的反应，其实就是在单细胞数据中找到的在杀伤性 CD4 或者 CD8 细胞中高表达的基因（25 个 log fold change 最高的基因，log fold change 是杀伤性细胞比上非杀伤性细胞），然后计算这些基因表达的算术平均，发现这个 signature score （作者称之为 cytotoxic score）和之前聚类，差异基因得到的 signature 算出来的分数显著正相关，但是和 Xcell 估计的总体 CD8 或者 CD4 浸润没有显著相关性，并且这个分数和 ICB 的反应显著相关。因此认为这个基因 signature 为和免疫治疗反应相关的肿瘤外源因素。

然后作者考虑了可以辅助 T 细胞杀伤的肿瘤内源因素，考虑了 TMB（以 10 MB 为阈值，将 TMB 进行二值化）和上面的 cytotoxic score 进行建模预测 ICB 反应（CoxPH 模型，这个 cytotoxic score 和 TMB 是独立的，因为相关性很低），但是这篇文章用的 risk 分层还是用的所有样本的中位数，有点问题。发现这个模型的预测值和好的预后相关。

这篇文章的亮点在于解析了病人样本中 CD4 细胞的状态，有多种杀伤性 CD4 细胞亚群存在，并且和肿瘤细胞表达的 II 类 HLA 相关，说明这类细胞可能是 HLA-I 受损的病人也可以对免疫治疗有反应的原因；另外还提出了结合杀伤性 CD8 和 CD4 以及 TMB 的免疫治疗预测模型。

### Integration of multiomics data with graph convolutional networks to identify new cancer genes and their associated molecular mechanisms (Nature Machine Intelligence-2021-5)

这篇文章使用图深度学习整合多组学信息来预测癌基因，比较有意思的点在于模型的解释（一种通用的方法 LRP）。

目前发现癌基因的方法局限性：

* 在一些癌症类型中发现的癌基因还是比较少的
* 一些在癌症发生过程中有重要作用的基因并不是通过突变，而是通过功能的失调，这些基因很多是转录或表观的调控因子，可以被一些小分子靶向

这篇文章的癌基因定义：

> cancer genes—broadly defined here as genes that are able to confer a selective growth advantage to the cell when altered at genetic, epigenetic or expression level.

多组学的数据包括：

* 单核苷酸变异（13097个基因），矫正基因长度的突变频率（non-silent SNVs/gene length）
* 拷贝数变异（12088个基因）GISTIC2，癌症类型的该基因所有拷贝数变异聚合在一起
* 基因表达（18898个基因）和正常组织比较的 log2 fold change，癌症类型样本取均值
* 启动子区域的甲基化（12406个基因），信号值为基因启动子区域所有 CpG 位点的 β 值的均值，并使用 ComBat 进行批次效应矫正；在某个癌症类型中，某个基因的甲基化信号为癌症样本与正常样本信号差值的均值

PPI 网络：CPDB，STRING-db，IRefIndex，Multinet，PCNet

PPI 网络中每个基因都是一个节点，每个节点的特征有 64 维（16 种癌症类型，每种癌症类型都有上面的四维特征），缺失值都填充为 0，并进行 min-max 归一化（也就是对每个基因的特征归一化到 -1 到 1）。

正例样本：已知的癌基因，包括从 NCG 数据库中收集的 711 个 KCGs，以及从 PubMed 中通过 DigSEE 挖掘得到的 85 个高可信度的癌基因；负例样本则是最大可能不与癌症相关的基因，采取逐步排除的方法：不是 NCG 中的基因，不和 KEGG 数据库中的 'pathway in cancer' 相关的通路有关系，不存在于 OMIM 疾病数据库，不被 MutSigdb 预测和癌症相关，最后基因表达不与已知的癌基因有相关性。

使用的模型是 Graph Convolutional Networks（[tkipf/gcn: Implementation of Graph Convolutional Networks in TensorFlow (github.com)](https://github.com/tkipf/gcn)）

使用 LRP 进行模型的解释

#### EMOGI accurately identifies KCGs

##### Performance comparison with other methods

和其他不同的方法进行比较并计算在测试集上的 AURRC：

* 在多组学数据上训练随机森林模型
* 在 PPI 网络上训练 PageRank 或者 DeepWalk 模型，得到的节点表示再用 SVM 进行分类建模
* 只使用 PPI 信息的 GCN 模型（没有多组学特征的 EMOGI 模型）
* HotNet2
* DeepWalk + 随机森林，DeepWalk 学习到的节点表示和多组学特征拼接在一起，再使用随机森林进行分类建模
* 常用的预测 driver 基因的方法：MutSigCV 和 20/20+

平均来说，EMOGI 在六个不同的数据集上表现都比其他方法要好，在只使用 PPI 网络的方法中 DeepWalk 的表现最好，比没有使用组学信息的 EMOGI 要好，使用 DeepWalk 得到的节点表示加上组学信息再用随机森林预测的效果在 PCNet 甚至比 EMOGI 还要好，但是在其他几个 PPI 网络中的表现没有 EMOGI 好。

额外的测试数据：NCG 中的潜在的癌基因；OncoKB 数据库中的癌基因；ONGene 数据库中的癌基因和 Bailey 文章中预测的癌基因。模型预测的在这些基因集中的癌基因被认为是 true positive，不在这些基因集中的癌基因是 false positive；在 ONGene 和 OncoKB 中 EMOGI 表现要比其他的方法要好（尽管所有的方法在不同 PPI 网络上的表现不同）

##### EMOGI benefits from different data representations and multiomics integration

什么样的数据类型对 EMOGI 的预测是最有信息的？进行了 perturbation 实验，打乱网络的边或者单个基因节点的特征向量，然后评估模型的表现。

* 对于基因的特征：选择两个节点，交互其特征向量，分别对25%, 50% or 75% and 100%的节点进行操作
* 对于网络：随机选择两个边进行交互（原来是 x-y, u-v 现在变成 x-u, y-v），分别对网络中的25%, 50% or 75% and 100%的边进行操作，在操作的同时保持网络中节点的度不变；另外还考虑一种情况，进行随机的打乱时（100%的边）不保持节点的度不变，而是节点的度服从幂律分别

也比较了使用部分组学数据的模型表现，发现 SNV 是最重要的特征

##### Pan-cancer analysis improves EMOGI’s capability to predict cancer genes

检查在泛癌上训练的模型是不是比只在单个癌症数据上训练的模型的表现效果要好？在两种癌症类型上验证：BRCA 和 THCA；建了两个模型：

* 对特定癌症类型的样本的组学数据取平均，和之前训练的 pan-cancer 模型差不多，之前是一个组学有 16 个值，现在只有 1 个值

* 不进行平均，增加一个维度的信息，表示每个样本：

  <img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220901222121-y9guhcb.png" style="zoom:67%;" />

##### EMOGI recovers distinct omics contributions of predicted cancer genes

提取对模型预测最有用的特征，使用的方法是 LRP：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220902075159-cpqknda.png" style="zoom:60%;" />

Demo：[Explainable AI Demos (fraunhofer.de)](https://lrpserver.hhi.fraunhofer.de/image-classification)

[InDepth: Layer-Wise Relevance Propagation | by Eugen Lindwurm | Towards Data Science](https://towardsdatascience.com/indepth-layer-wise-relevance-propagation-340f95deb1ea)

LRP 是从输入的角度去解释输出，比如如果从一个乳腺组织的图片中预测乳腺癌，那么 LRP 会给出原来图片中的每个像素对预测结果的影响，这种方法是应用于已经训练好的模型，不参与训练过程。

简单来说 LRP 先进行一个正向传播，得到 l + 1 层 zj 神经元的值及其和前一层的权重，然后将 l + 1 层的相关性值进行反向传播，如果权重比较高，反向传播得到的值就比较大，这个过程是一个迭代的过程，因为我们最开始只知道输出层的相关性值。

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220902091118-mvi4e9w.png" style="zoom:60%;" />

上面那个例子是图像识别的例子，可以看到返回的是和原来图片一样大小的矩阵，里面的值表示每个像素点的相关性值，因此对于 GCN 网络，每个基因返回的是两个矩阵，一个是不同组学特征的重要性值，另一个是不同邻居节点的重要性值：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220902091410-ucne11v.png" style="zoom:60%;" />

为了验证模型解释的结果，作者在文献中搜集一些 KCG 基因的分子特征，看看模型解释是否可以识别这些特征；APC 基因在文献中描述是在结肠癌中突变比较多，并激活 Wnt 信号通路，EMOGI 识别到在结直肠组织中突变率是对预测最重要的变量；TWIST1 在多种癌症类型中都有启动子的超甲基化，并且是筛选结肠癌的生物标志物，EMOGI 识别到在肺癌和结肠癌中 DNA 甲基化是对预测最重要的特征；STIL 在多种癌症类型中过表达，EMOGI 识别到在uterine, cervical and other cancers 中基因表达是对预测比较重要的特征；MYC 在多种癌症中是拷贝数扩增的，EMOGI  也识别到在多种癌症类型中 CNV 和基因表达特征对预测最重要。

另外对于提取的相互作用重要性，EMOGI 也提供了一些证据，比如识别到 RB1 的最重要的相互作用的基因是 E2F1 和组蛋白去乙酰化酶 HDAC1。

另外还发现平均来说，突变频率是最重要的预测特征（将所有基因相应组学特征的 LRP 值相加并进行 min-max 归一化）

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220902093403-36yzi8v.png" style="zoom:60%;" />

#### Newly predicted cancer genes

关注分析新预测的癌基因（不在已有数据库中，NPCG），从 6 个 PPI 网络中收集预测排名前 100 个基因，然后将已知的癌基因剔除，最终得到 165 个 NPCG。

##### NPCGs interact with KCGs

发现 EMOGI 的预测值和与已知的癌基因相互作用的数量呈显著的正相关，所有的 NPCG 都至少有一个与 KCG 的相关作用，并且使用节点的度标准化后的相互作用数量要显著高于其他的非 NPCG 基因。与 NPCG 相互作用的基因中排在前面的基因都是一些比较知名的癌基因，如 TP53, EP300, BRCA1 and EGFR；利用 LRP 来分析特征的重要性，发现对于这些 NPCG 基因，PPI 的特征比组学的特征更重要：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220902095801-sgvadn3.png" style="zoom: 33%;" />

##### NPCGs are essential in tumour cell lines

分析 Depmap 的基因敲除数据，发现 NPCG 显著富集在 essential 基因列表中，排名前 20 的 NPCG 可以影响 600 个细胞系并且比 KCG 和 CCG 所影响的细胞系要多。因此这就有一个问题，预测的 NPCG 是否主要是管家基因，在任意细胞中失活都会有致死效应：实际情况是 60% 的 NPCG 细胞影响了小于 10% 的细胞系，而 26% 的 NPCG 影响了超过一半的细胞系：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220902100438-3s95jcn.png" style="zoom: 33%;" />

并且通路富集分析也发现这些基因也不是富集在管家基因功能的基因集中，而是在 

signalling, cell cycle, cancer pathways and development functions：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220902100719-sn2uaa6.png" style="zoom:50%;" />

#### From single-gene feature importance to global model behaviour analysis

基于模型解释的最重要的分子和网络特征对基因进行分组；通过网络重要性分数在 PPI 网络中提取子网络，所进行的分析都是基于下面的 total 矩阵的：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220902110616-qnw6xn7.png" style="zoom: 50%;" />

##### Clustering of feature contributions reveals different groups of cancergenes

使用的方法为 spectral biclustering 

[An Introduction to Biclustering | Kemal Eren](http://www.kemaleren.com/post/an-introduction-to-biclustering/)

双聚类就是同时对一个矩阵的行和列进行聚类：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220902103901-nbcazha.png" style="zoom:60%;" />

这里面的行就是基因，列是不同的特征，矩阵的值是基因这些特征的 LRP 值，聚类的目的是找到基因的类，在这些类中重要的特征是类似的。发现大部分的基因类的重要特征都是突变。通过这个双聚类的分析可以得到由不同特征驱动的癌基因类别：相互作用（1，4，12），突变（2，5，7，10），甲基化（11），表达（3，8）和拷贝数变异（6，9）

##### Cancer-associated strongly connected components from the PPI network

网络模块的识别：使用 LRP 计算相互作用的重要性，对于两个基因 A 和 B，如果基因 A 对基因 B 的分类预测有贡献则添加一条从 A 指向 B 的有向边，边的权重是 LRP 值（移除权重低于 0.14 的边），然后使用 Tarjan's algorithm 在这个网络中检测 SCC（strongly connected components），一共提取到了 45 个至少含有 2 个基因的 module，最大的 SCC 含有 149 个基因，而最小的只有 3 个基因，我们只保留了大于 5 个基因的 SCC 用作后续的分析。

SCC 的定义：在图中存在一条可以经过所有节点的路径

[Tarjan's Algorithm to find Strongly Connected Components - GeeksforGeeks](https://www.geeksforgeeks.org/tarjan-algorithm-find-strongly-connected-components/)

### Predicting and characterizing a cancer dependency map of tumors with deep learning (SCIENCE ADVANCES-2021-8)

这篇文章是用深度学习的模型预测单个样本单个基因的 dependency 值（对细胞的重要性），使用了 Autoencoder 迁移学习以及比较有意思的模型解释方法。

模型的架构：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220925095957-oqgd8sp.png" style="zoom: 50%;" />

前四个 Encoder 都是从 TCGA 训练好的 Autoencoder 之后迁移过来的 encoder，后面的 DepOI 输入是一个 3115 维的独热编码的向量（从 Msigdb 中收集的通路信息，如果基因在这个通路里面就是 1 否则就是 0），选择了 1298 个 DepOI（在 278 个细胞系中的 dependency score 的 SD 大于 0.2 或者是 COSMIC 标注出的癌基因）。

作者比较了预测值和真实值的相关性，平均相关性只有 0.18（虽然比随机的要好），然后又看了 SD 大于 0.3 和 Depmap 定义为高方差的基因，发现平均相关性有所提升。随机（y-scrambling）方法：对于细胞系和检测的基因对，随机打乱其 dependency score，重复 100 次。

模型的比较：没有在 TCGA 上 pre-train 的模型；没有最后的基因 Figureprint 的模型；和其他的机器学习模型的比较（Lasso，Ridge，弹性网络，SVM，都分别使用了原始特征输入和降维后的特征输入，降维方法采用了 PCA 和 NMF）：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220925103838-btvjwpp.png" style="zoom:50%;" />

作者用来验证模型的数据集：

* Depmap 新产生的数据，2018Q3~2020Q2
* Sanger 使用不同的 CRISPR 文库进行的筛选数据集
* 使用 RNA-I 进行筛选的基因集

#### 模型的解释：

##### 表达

建立了简化的模型，只使用了表达特征或者突变+表达特征的模型性能和全部特征的模型性能差不多，但是只使用 CNA 或者 突变的模型性能有下降，因此作者接下来利用只使用了表达信息的模型（Exp-DeepDEP）来尝试理解模型的学习，看基因表达和预测值之间的关系。在 TCGA 中预训练得到的 encoder 经过细胞系上的微调后只有两个神经元的权重是非零的（encoder 的输出层一共 50 个神经元），并且重复训练 10 次后结果也差不多（均值 2.1 变化 1~3）。

对 Encoder 输出层的每个神经元的输出值进行扰动，看对预测值的影响，从所有 CCL 的均值开始，每次扰动 0.25 SD：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220925140853-k32ehp8.png" style="zoom: 67%;" />

为了理解在 encoder 输出层中编码的表达 signature，作者重新构建了一个 autoencoder：encoder 的权重是训练好的 Exp-DeepDEP 模型中的 encoder 权重，而 decoder 则是重新进行训练的（使用 TCGA 数据），然后拿出 decoder，对我们感兴趣的神经元输入 1，其他输入 0，得到 decoder 的输出值（6016 维），将这个向量认为是感兴趣的节点所编码的表达 signature：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220925141829-kybf3x5.png" style="zoom:67%;" />

然后使用对这些输出的向量（6016 个基因）进行**单样本 GSEA 分析**，看看富集在哪些通路，发现了上述两个神经元编码的表达 signature 富集的通路，signature1 主要是增殖相关的通路，而 signature2 主要是微环境和转化相关的通路：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220925142135-25a0ivh.png" style="zoom:67%;" />

然后再用上面的模型去预测每个细胞系-基因的 dependency score，然后对每个基因画等高线图，进行了一系列解释说明产生的基因 “essentiality map”可以对基因的功能提供新的视角。

##### 突变

先构建了只有突变信息的 Mut-DeepDEP 模型，对于每个细胞系-基因对，将该基因的突变状态进行改变（其他基因状态保持不变），然后比较改变前和改变后的预测值：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220925150337-6idmazo.png" style="zoom:67%;" />

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220925150434-pafo9wv.png" style="zoom:67%;" />

因此这个差值（SE）越大，表示该基因的突变形式对该细胞有着更重要的作用，发现大部分 SE 值都是比较小的，并且和细胞系组织类型不相关。

#### 模型验证

接下来作者检测了模型对合成致死效应的预测能力（PTEN/CHD1；BRCA1/PARP1）,在原始的 dependency score 中（实验得出的）CHD1 的 score 在 PTEN 突变和不突变的细胞系中并没有显著差别（BRCA1/PARP1 也是）。但是使用模型预测 PTEN/CHD1 的 SE 值是显著低于 0 的（将 PTEN 的突变状态置为 1 和 0 之间的差别）并且在其他所有突变组合中的 P 值接近于 0（BRCA1/PARP1 也是类似的结果）：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220925151720-nr9sdo0.png" style="zoom:67%;" />

然后作者用模型来预测 KRAS 可能的合成致死基因，发现 KRAS 突变和其他 12 个的突变结合有着更强的 dependency 效应（至少 100 个细胞系中有着最高的负 SE 值），举了两个例子：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220925162438-ic3i9jp.png" style="zoom:67%;" />

接着作者使用模型来预测 TCGA 肿瘤样本的 dependency score，将 M-Dep 定义为于更强或者更弱的 dependency 相关的突变（比较有突变的样本中该基因的预测值与没有突变样本预测值，进行 t 检验），E-Dep（表达），M-Dep（甲基化），C-Dep（CNA） 也是类似定义（表达用相关性）；并看不同的基因这 4 种 Dep 哪个占主导地位（比其他的显著多）：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220925165425-ixk8d75.png" style="zoom:67%;" />

可以看到表达占的比重是最多的，虽然 Me-Dep 占的比例也比较大，但是从 E 图中可以看出大部分 Me-Dep 和 E-Dep 同时出现，所以 DNA 甲基化可能大部分是通过调控基因表达来影响癌细胞的 dependency。

作者还在临床数据上验证了所预测的肿瘤 dependency。首先检查了 BRCA 发现 ER 阳性的肿瘤对 ESR1 基因有着更强的 dependency（和 ER 阴性的相比）；另外还收集一些经过药物治疗的 PDX 样本（2015 Gao）主要关注一种 FGFR2/4 抑制剂 LLM871，发现 CR 的病人比 PD 的病人更依赖于这两个基因，还有研究表明 MSI-H 的肿瘤更依赖 WRN 基因，模型预测结果也证实了这一点：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220925215222-krdx2id.png" style="zoom:67%;" />

 除了靶向治疗外，作者也验证了和化疗反应相关的 dependency（BRCA）；在对化疗有反应和免疫反应的病人之间比较得到有差异的 dependency score 的基因，发现了 71 个基因，大部分是和化疗治疗抵抗相关的，也就是说越负的 dependency 值和对化疗的不良反应相关，并且这些基因显著富集在线粒体和氧化磷酸化相关的 GO 通路上。

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220925220111-ke3veby.png" style="zoom:67%;" />

### Exaggerated false positives by popular differential expression methods when analyzing human population samples (Genome Biology-2022-3)

这篇文章对常用的基因表达差异分析软件和方法进行了对比，发现常用的 Deseq2 和 EdgeR 在大样本时表现没有经典的秩和检验好，推荐在大样本量的时候使用秩和检验。

RNA-seq 的一个关键分析就是寻找不同组别之间的差异基因，目前最流行的方法是 Deseq2 和 EdgeR，但是这两种方法都是在样本量比较少的时候提出来的，所以对数据的分布做了很多有参数的假设；目前测序的样本量已经大大增加了，这两种方法是否仍然适用？

作者在 13 个数据集上比较了这两个工具鉴别出的差异基因的区别（这些数据集的样本量从 100 到 1376 不等），发现两者得到的差异基因差别比较大，大概 23.71–75% 由 Deseq2 发现的差异基因被 EdgeR2 漏掉了，特别是在一个免疫治疗的数据集上只有 8 % 的重叠，那么在这样的数据集上这些工具还能控制比较理想的 FDR 吗（5%）？

测试了 6 种方法 DESeq2, edgeR, limma-voom, NOISeq, dearseq, and the Wilcoxon rank-sum test，比较大的数据集可以允许我们进行 permutation analysis 而不是依赖于特定的模型假设。

**permuted dataset** 的生成：随机打乱样本标签

**semi-synthetic datasets** 的生成：首先使用 6 种方法鉴别 DEG，定义 True DEG 为所有 6 种方法都鉴别为 DEG 的基因（FDR 阈值设置的比较小，0.0001）认为这些是真正的 DEG，然后在这些 True DEG 中随机抽取 k 个基因，保持这 k 个基因的 read counts 不变，对于剩下的基因，在两个 condition 之间随机打乱 read counts，重复 50 次产生 50 个 semi-synthetic datasets

**FDR, power 和模型拟合的好坏**：FDR 是错误发现率的期望，错误发现率也就是在所有 discoveries 中的假阳性的比例；虽然 FDR 不能直接计算，但是 FDP 可以从 benchmark 数据集上计算得到（假阳性和假阴性是已知的）。在上面的半合成的数据集中定义真实的 DEG 为真阳性，剩下的基因为真阴性样本，也就是将 50 个半合成样本计算得到的 **FDP 的均值**作为估计的 FDR。鉴别 DEG 方法的 power 为在某个基因是真实的 DEG 的情况下鉴定为 DEG 的概率（条件概率），也可以被认为是 empirical power 的期望，经验 power 为真实 DEG 被鉴定为 DEG 的比例，也就是将 50 个半合成数据集的**经验 power 的均值**作为鉴定方法的 power 估计。使用 goodness-of-ft 检验来评估基因的 read counts 能否被负二项分布来拟合（使用的是 Deseq2 和 EdgeR 输出的标准化后的 counts，参数为 Deseq2 或 EdgeR 估计的散度）。

对 109 个样本的免疫治疗数据集（51 个治疗前和 58 个在治疗）随机打乱样本标签，生成 1000 个 negative-control 数据集，因此从这些数据集中鉴别的 DEG 就是假阳性的 DEG，Deseq2 和 EdgeR 比原来的数据集发现的差异基因还要多（A 图中红色的菱形就是原来数据集中的 DEG 数量）！并且在原来数据集的差异基因中 Deseq2 和 EdgeR 在至少 50 % 的模拟数据集中仍然能识别出 22 和 194 个差异基因（C），说明这些所谓的差异基因是比较可疑的。

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220921160134-4lpppe0.png" style="zoom:60%;" />

作者还发现一个现象：在原来数据集中 Fold change 越大的基因越容易在模拟的数据集中被识别为差异基因（这和最近发表的一篇文章的结论类似，选择在两种条件下差异大的基因会造成过高的估计 FDR，Inflated false discovery rate due to volcano plots: problem and solutions ）：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220921160334-t2f4x2u.png" style="zoom:60%;" />

然后作者对这些可疑的 DEG 进行富集分析（至少在 10% 的模拟数据集中被鉴别为 DEG），发现富集在免疫相关的通路：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220921161358-0kg5no8.png" style="zoom:60%;" />

为什么这两个工具会产生这么多的假阳性？可能是由于这两个模型的假设不符合实际情况（负二项分布），为了检验这个假设，作者选择了两类基因：在超过 20% 的模拟数据集中被鉴别为 DEG，和在小于 0.1 % 的数据集中被鉴别为 DEG，然后使用 goodness-of-ft  来检查这些基因的 read counts 是否服从负二项分布（log10(p-value)，发现第一类基因的拟合要比第二类要差：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220921162304-0cx8rz5.png" style="zoom:60%;" />

可能使模型假设不成立的原因是离群值的存在，作者接着检测了至少在 10 个模拟数据集中被错误识别为 DEG 的基因，发现这些基因都存在离群值。Deseq2 和 EdgeR 都假设基因在两组中的表达均值相同，而均值对离群值是敏感的，因此离群值的存在对模型的结果影响较大。

作者又在 GTEx 和 TCGA 上进行 benchmark，并加入了其他工具的比较（limma-voom，NOISeq，dearseq，Wilcoxon rank-sum test，limma 也是参数化的模型，后面三个都是非参的模型）依据上面计算 FDR 和 Power 的方法，得到 Wilcoxon rank-sum test 的 FDR 是控制的最好的并且 power 是最高的：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220925222504-fh4p2in.png" style="zoom: 67%;" />

另外还通过降采样来研究不同样本量对 FDR 和 Power 的影响：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220925222655-35ppk93.png" style="zoom:67%;" />

可以看到在低于 8 个样本的时候，Wilcoxon rank-sum test 的表现并不好，但是超过 8 个样本之后 Wilcoxon rank-sum test 的表现就比其他方法要好，特别是 3 个参数化的方法。
