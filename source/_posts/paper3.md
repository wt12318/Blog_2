# Predicting cellular responses to complex perturbations in high-throughput screens

Molecular Systems Biology

> Here, we present the compositional perturbation autoencoder (CPA), which combines the interpretability of linear models with the flexibility of deep-learning approaches for single-cell response modeling

单细胞高通量筛选

预测组合扰乱对细胞的影响（多种不同的因素），有单独因素扰乱的数据，如何去预测新的组合扰乱因素对细胞的影响（OOD 数据，out of distribution）

> Here, we propose the compositional perturbation autoencoder (CPA), a method to predict scRNA-seq perturbation responses across combinations of conditions such as dosage, time, drug, and genetic knock-out. The CPA borrows ideas from interpretable linear models and applies them in a flexible DL model to learn factorized latent representations of both perturbations and covariates.

将神经网络得到的隐空间分解成可解释的，复合的模型，如果这个隐空间是线性的，那么我们就可以将观测到 的基因表达描述成一个因子模型，其中每部分是一个单独的扰动。

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20230706172543-eadbvr0.png)

假设扰动的隐变量（$z_i$）由基础状态，扰动向量（$d_i = (d_{i,1},....,d_{i,M})$，M 个 Drug）以及协变量向量（$c_i =  (c_{i,1},....,c_{i,K})$，K 个协变量）构成，这里的 i 表示细胞：

$$
z_i = z_i^{basal} + V^{perturbation} \cdot(f_1(d_{i,1}),...,f_M(d_{i,M}))+\sum_{j=1,...K}V^{covj} \cdot c_{i,j}c
$$

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20230718192854980.png)



f 表示将扰动进行非线性转化，和 $V^{perturbation}$ 矩阵结合就可以对剂量反应曲线进行建模

训练过程：

* 将细胞表达编码成基础状态 $\hat z^{basal}$ 这个基础状态不含有任何 d 和 c 的信息，这个过程是通过对抗网络来实现的
* 将这个基础状态与关于 d，c 的可学习的 embedding 结合起来形成一个估计的扰乱状态 $\hat z_i$
* 将 $\hat z_i$ 解码成细胞的表达

因此在模型推理的时候就可以进行反事实的推理，也就是将 d 替换成想要进行的扰动 $d'$ 然后预测其基因表达的变化。

训练过程中使用的 Loss 为 3 个 Loss 的结合：

* 对于 Autoencoder 使用的是 Gaussian negative log likelihood loss，decoder 预测的是期望和方差，将细胞的表达视为从预测的期望和方差构成的高斯分布中抽样得到
* 对于对抗的分类器，使用的是两个交叉熵损失，分别预测扰乱 d 和协变量 c



# Tumor aneuploidy predicts survival following immunotherapy across multiple cancers

> **A higher aneuploidy score is associated with poor prognosis following  immunotherapy among tumors with low TMB, but not those with high TMB.**

> Recently, Samstein et al.^^  demonstrated in the largest immunogenomic data set of tumors treated  with immunotherapy that higher nonsynonymous somatic TMB, defined as the  **top 20%** within each cancer type, was associated with improved overall  survival. The following year, the US Food and Drug Administration (FDA)  issued pan-cancer approval of pembrolizumab for patients with a high TMB  tumor, defined as **ten or more mutations per megabase**

异倍体定义为染色体或染色体臂不平衡的数量，有比较矛盾的结果，有些文献发现异倍体和免疫逃逸相关（下调 PDL1 表达，抑制 CD8 T 细胞）但是有些文献表明联用化疗和 ICB 的病人中，高异倍体的肿瘤的反应更佳。

这篇文章使用的数据来自 Tumor mutational load predicts survival after immunotherapy across multiple cancer types. 1660 个病人的免疫基因组数据。[www.synapse.org/#!Synaps...](https://www.synapse.org/#!Synapse:syn7222066/wiki/405659)

这篇文章的异倍体分数定义：

> The aneuploidy score^^  for a sample was defined as the fraction of evaluable arms (ASCETS call  of AMP, DEL, NEUTRAL or NC) afflicted by arm-level somatic copy-number  alterations (AMP or DEL) 即有多少比例的染色体臂是扩增或者删除的

* 96 % 的样本中有异倍体存在，但是得分变化比较大
* 较高的异倍体分数和差的预后显著相关，并且和 TMB 以及临床病理学的指标进行多因素分析时，异倍体分数仍然是显著的；在相关性上和 TMB 相关性也较低 --  说明该分数是独立于 TMB 的
* 异倍体分数和预后的相关性是否被特定的染色体变异事件影响，比如 9p21 位点的丢失（该位点含有编码 PDL1 的基因）-- 没有显著和预后相关的染色体变异事件
* 寻找异倍体分数的最佳阈值（和 Samstein 文章类似的方法）：从 20 分位数到 80 分位数每 10 个分位数为阈值将样本分为高异倍体分数和低异倍体分数进行多因素分析（$Surv \sim AS + TMB + Drug$）AS 表示异倍体状态，TMB 表示 TMB 状态（前 20 分位数为高 TMB），Drug 表示药物类型；进行留一法交叉验证，发现 50 分位数有着最低的 P 值，并且在 BH 矫正后仍然显著（使用FDA 批准的 TMB 阈值为 >= 10/M 也是得到同样的结果）。将样本按照 TMB 和得到异倍体分数的阈值进行分组，KM 分析显示在高的 TMB 病人中异倍体分数和预后没有相关性，但是在低的 TMB 病人中高异倍体分数的病人在接受 ICI 治疗后有着更差的预后；结肠癌和乳腺癌中差异最大

# Cancer aneuploidies are shaped primarily by effects on tumour fitness

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20230713163524-u97wcs3.png)

异倍体--整个染色体或者整个染色体臂水平的 DNA 不平衡，在 90 % 的肿瘤中可以观测到

> Cancer is driven by multiple types of genetic alterations, which range  in size from point mutations to whole-chromosome gains and losses, known  as aneuploidy

异倍体可能的原因：

* 染色体错误分离，重组或者中心体异常（机械偏差）
* 异倍体带来的适应优势（选择偏差）

在比较低的细胞压力情况下，比如酵母，鼠和人类癌细胞系中的异倍体会使得生存率下降，增加细胞衰老；在一些类型的细胞压力或基因缺失情况下，异倍体对酵母和人类细胞是有利的，另外也有一些研究表明异倍体对肿瘤演化和耐药性有作用。

这篇文章主要是解析染色体臂水平的 SCNA 对癌细胞适应性的效应并找到与其相关的位点，通过探索与端粒或着丝粒绑定的 SCNA 的长度分布来实现这一点。

* 臂水平的 SCNA 占了 25 个最频繁发生的体细胞变异事件中的 23 个（TCGA 样本）
* 22.5 % 的癌症样本基因组被臂水平的 SCNA 影响，11.3% 被局部 SCNA 影响
* 发生频率如此高的 SCNA 事件是由于机械原因还是适应性原因是未知的，如果是适应性，哪些位点对臂水平的 SCNA 的适应性效应有贡献也是未知的

  * 有没有可能这些 SCNA 和癌症 Driver 的变异是相关的，这些 Driver 给予了癌细胞适应性？在 23 个发生频率最高的 arm-SCNA 中仅仅有 13 个含有已知的 Driver 基因变异（至少 20 % 的样本中有）-- 10q 在 80 % 的胶质母细胞瘤中丢失，10q 区域中含有抑癌基因 PTEN，但是 PTEN 仅在 40 % 的样本中有双等位的失活，没有双等位失活的样本和没有 10q 丢失的样本的 PTEN 表达水平是差不多的（也就是在剩下的 40 % 的样本中 PTEN 的表达和 10q 正常的样本表达一样，说明这些样本不能用 PTEN 失活来解释其适应性）

起始自端粒或着丝粒的 SCNA 的断点的位置，在特定位置断点的富集可能会提供机械偏好还是适应偏好的信息

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20230717194923-gaofxd5.png)

首先通过计算在染色体臂上的断点和在端粒区域的**断点的密度差别**探索来端粒的机械偏好：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20230717195702-643kta9.png)

> 人类端粒的长度在 0.2 ~ 7 Mb -- https://bionumbers.hms.harvard.edu/bionumber.aspx?id=105402

发现 39 % 的 tel-SCNA 在着丝粒区域结束，在着丝粒区域的断点密度是在臂上的 4 倍，并且密度和着丝粒的长度没有相关性：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20230717200318-2tdd83u.png)​!

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20230717200015-3jhttbn.png)​​

在染色体臂内的 partial-SCNA（与上面的断点在端粒内部不同）可以提供 arm-SCNA 的适应性的信息。和 arm-SCNA 一样 partial-SCNA 更倾向于更低水平的扩增或删除（与局部的 SCNA 相比），partial-SCNA 的长度分布近似于均匀分布：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20230717202110-uwo2e2g.png)

将上面的均匀分布视为背景分布，然后和特定的 partial-SCNA 长度分布进行比较，以此来检测受到选择的位点；如果某个位点可以增加细胞适应性，那么包含这个位点的 partial-SCNA 就会更多，如果某个位点会减少细胞适应性，那么包含这个位点的 partial-SCNA 就会减少，我们就可以看到断点频率的上升或下降：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20230717202937-rv3hlx2.png)

比如如果某个位点可以增加适应性，会受到正选择，包含该位点的 SCNA 就会比较多，也就是在这个位点之后的断点密度会较高，相反，如果是负选择，那么在这个位点之前的断点密度应该较高。下面是实际数据的例子：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20230717203241-s81df6t.png)

作者基于这个想法开发了一个算法：BISCUT（**B**reakpoint **I**dentification of **S**ignificant **C**ancer **U**ndiscovered **T**argets），该算法主要有两个功能，1）检测受到选择的位点，2）计算每个位点的置信区间（peak region），具体步骤：

1. 检测特定染色体臂的 partial-SCNA 长度分布是否和背景分布有显著差异，先将 partial-SCNA 的长度按照升序排列 $T_i = {t_1,t_2,...,t_n}$ ，计算其累积分布，利用 KS 检验比较 $T_i$ 与背景分布的差异（背景分布是所有10872 个肿瘤样本中的 partial-SCNA 长度分布）如果 n > 4 并且 KS 检验 P 值小于 0.05，那么这个臂的 SCNA 是经受选择的

2. 使用不完全 Beta 函数来近似背景分布，不完全 Beta 函数和 Beta 函数（可以理解为分布）的区别在于不完全 Beta 分布的 x 的取值不限制在 0~1 之间 （不定积分取代了 Beta 函数中的定积分），因此不完全 Beta 函数可以写成：（x 是 partial-SCNA 断点的位置，$\alpha$ 和 $\beta$ 是 Beta 分布的参数）

   $$
   B_i = I_x(x;\alpha,\beta)
   $$

   然后对四种 partial-SCNA 分别通过拟合背景分布来得到参数 $\alpha$ 和 $\beta$ 的值（使用 `fitdistrpuls`​ 包中的 `fitdist`​ 函数）。基于这个 $B_i$ 和上面的 $T_i$ 我们可以得到偏离背景分布最大的位点（starting peak）以及其方向：

   $$
   选择方向 = \{正向，如果 T_i -B_i >0,否则负向\}
   $$

   ![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20230718144335-lto3v1f.png)

3. 广义极值分布 $GEV_{left}(peak_0)$ 由从服从背景分布 $B_i$ 的 $n_{left}$ 个随机变量的取值中得到的 1000 个独立的最大值构成，其中 $n_{left}$ 是 peak0 （也就是 starting peak）左边的肿瘤数量

   ![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20230718144314-10sb7jd.png)

4. 重复以上的过程（在检测到选择的位点将染色体分开，然后分别在左右重复），直到观测的数据与经验分布没有统计学上的差异或者剩下的断点数目小于 4 个或者与之前重复过程中找到的 peak 是重叠的

> Generalized extreme value distribution：广义极值分布 -- https://zhuanlan.zhihu.com/p/576922526
>
> ![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20230718141706-7ydtcew.png)

在 10872 个肿瘤样本中使用 BISCUT 检测到 193 个基因组位点有明显的选择信号：

* 90 个正选择 （39 个扩增，51 个删除）
* 103 个负选择（41个扩增，62 个删除）

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20230718150051-kjcj5uo.png)

这些 peak 位点显著富集已知的原癌基因和抑癌基因，并且很多检测到的基因没有被 focal SCNA 影响，这些受选择的基因与 dn/ds 的结果也相符合：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20230718150842-8cn77sd.png)

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20230718150914-ulszkzv.png)

接着作者将 BISCUT 拓展到计算 RF （群体遗传学中用来表示遗传变异的适应效应，为有某个变异的个体的生存率和没有该变异的个体的生存率的比值）‍
