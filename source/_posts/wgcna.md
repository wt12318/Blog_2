---
title: 加权共表达网络分析（WGCNA）
author: wutao
date: 2021-10-14 10:00:00
slug: skill
categories:
  - Bioinformatics
  - R
index_img: img/net.png
---



WGCNA R包学习
<!-- more -->

WGCNA 的主要思想就是将基因聚合成一个一个的模块，然后再计算一个值（eigengene）来代表这些模块，这样就相当于将几万维的基因降维成几十维的模块，然后就可以把这些模块和样本的特征联系起来（通过计算 eigengene与特征的相关性），从而筛选出我们感兴趣的模块，对其中的基因进行研究。

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20211015123442050.png)

## 关键概念

- Co-expression network，共表达网络：网络的节点是基因，边是基因与基因之间表达的相关性，但是这个相关性加上了权重 *β* （所以叫加权基因共表达网络分析）：
  $$
  a_{ij} = |cor(x_i, x_j )|^β
  $$
  在 WGCNA 的分析中，一个关键步骤就是选择这个权重 *β*

-   Scale-free network，无标度网络：大多数 “普通” 节点拥有很少的连接，而少数 “热门” 节点拥有极其多的连接，节点的连接度和其频率之间呈现一种幂律分布（长拖尾分布）：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/Scale-free_network_sample.png)

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/785692-20180807161527827-890797957.png)

判断一个网络是不是无标度网络的方法：对 x 轴的连接度和 y 轴的频率取 log，然后看转化后的这两个值之间是否满足线性关系：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/785692-20180807160901235-1197112370.png)

-   Connectivity，连接度，也叫度（degree）：一个基因和网络中其他基因之间的连接强度的和

-   Module，模块：模块就是基因的聚类，在一个模块内，基因与基因之间的连接度（表达相关性）是比较高的，在两个模块间的基因的连接度比较低

-   Module eigengene：一个基因模块中的基因表达的第一主成分，用这个值来代表该模块的基因表达谱
    
-   Eigengene significance：当我们有样本信息时，我们可以计算 eigengene 与这些样本特征的相关性，相关系数就是 eigengene significance
    
-   Module Membership / eigengene-based connectivity：每一个基因都可以和每一个模块的 eigengene
    做相关性，如果这个相关性是 0，说明这个基因不属于这个模块，如果是 1 或者 -1，那么说明这个基因和该模块是正相关或者负相关的关系，可以用这种方法来寻找模块的 hub 基因
    
-   Hub gene：连接度比较高的基因

## 分析流程

主要流程如下：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20211015094913989.png)

### 读取数据，进行样本和基因筛选

输入数据需要标准化，对于芯片表达数据，可以使用 RMA ，log 的 MAS5 数据（RMA 已经经过 log2 转化了）或其他的标准化方法，对于 RNA-seq 数据，可以使用 FPKM/TPM （需要 log 转化） 或者使用 DESeq2 的
varianceStabilizingTransformation 函数进行标准化后的数据（官方推荐的方法）。一般情况下，我们可以筛选掉一些在大部分样本中表达比较低的基因，对于样本可以进行聚类，然后删除一些离群的样本：

``` r
library(WGCNA)
library(dplyr)
options(stringsAsFactors = FALSE)
femData = read.csv("../test/LiverFemale3600.csv")

##输入格式为行是样本，列是基因
datExpr0 = as.data.frame(t(femData[, -c(1:8)]))
names(datExpr0) = femData$substanceBXH
rownames(datExpr0) = names(femData)[-c(1:8)]

##进行样本和基因的基本筛选，将缺失值过多的样本或基因或方差为0的基因标记
gsg = goodSamplesGenes(datExpr0, verbose = 3);
>>  Flagging genes and samples with too many missing values...
>>   ..step 1
gsg$allOK
>> [1] TRUE

##如果有不符合标准的基因或样本，就要进行筛选
if (!gsg$allOK)
{
  # Optionally, print the gene and sample names that were removed:
  if (sum(!gsg$goodGenes)>0) 
     printFlush(paste("Removing genes:", paste(names(datExpr0)[!gsg$goodGenes], collapse = ", ")));
  if (sum(!gsg$goodSamples)>0) 
     printFlush(paste("Removing samples:", paste(rownames(datExpr0)[!gsg$goodSamples], collapse = ", ")));
  # Remove the offending genes and samples from the data:
  datExpr0 = datExpr0[gsg$goodSamples, gsg$goodGenes]
}

sd_gene <- apply(datExpr0,2,sd,na.rm=T) %>% sort(decreasing = T)
sd_gene[3000]
>> MMT00030800 
>>  0.07940613
need_genes <- sd_gene[1:3000]
datExpr0 <- datExpr0[,which(colnames(datExpr0) %in% names(need_genes))]

##对样本进行聚类
sampleTree = hclust(dist(datExpr0), method = "average");
par(cex = 0.6);
par(mar = c(0,4,2,0))
plot(sampleTree, main = "Sample clustering to detect outliers", sub="", xlab="", cex.lab = 1.5, 
     cex.axis = 1.5, cex.main = 2)

##去掉离群的样本
abline(h = 15, col = "red");
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/unnamed-chunk-105-1.png)

``` r
clust = cutreeStatic(sampleTree, cutHeight = 15, minSize = 10)
table(clust)
>> clust
>>   0   1 
>>   1 134
keepSamples = (clust==1)
datExpr = datExpr0[keepSamples, ]
nGenes = ncol(datExpr)
nSamples = nrow(datExpr)

sampleTree2 = hclust(dist(datExpr), method = "average")
plot(sampleTree2, main = "Sample clustering after remove outliers", sub="", xlab="", cex.lab = 1.5, 
     cex.axis = 1.5, cex.main = 2)
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/unnamed-chunk-105-2.png)

``` r
##读入表型数据
traitData = read.csv("../test/ClinicalTraits.csv");
dim(traitData)
>> [1] 361  38
names(traitData)
>>  [1] "X"                  "Mice"               "Number"             "Mouse_ID"          
>>  [5] "Strain"             "sex"                "DOB"                "parents"           
>>  [9] "Western_Diet"       "Sac_Date"           "weight_g"           "length_cm"         
>> [13] "ab_fat"             "other_fat"          "total_fat"          "comments"          
>> [17] "X100xfat_weight"    "Trigly"             "Total_Chol"         "HDL_Chol"          
>> [21] "UC"                 "FFA"                "Glucose"            "LDL_plus_VLDL"     
>> [25] "MCP_1_phys"         "Insulin_ug_l"       "Glucose_Insulin"    "Leptin_pg_ml"      
>> [29] "Adiponectin"        "Aortic.lesions"     "Note"               "Aneurysm"          
>> [33] "Aortic_cal_M"       "Aortic_cal_L"       "CoronaryArtery_Cal" "Myocardial_cal"    
>> [37] "BMD_all_limbs"      "BMD_femurs_only"

# remove columns that hold information we do not need.
allTraits = traitData[, -c(31, 16)];
allTraits = allTraits[, c(2, 11:36) ];
dim(allTraits)
>> [1] 361  27
names(allTraits)
>>  [1] "Mice"               "weight_g"           "length_cm"          "ab_fat"            
>>  [5] "other_fat"          "total_fat"          "X100xfat_weight"    "Trigly"            
>>  [9] "Total_Chol"         "HDL_Chol"           "UC"                 "FFA"               
>> [13] "Glucose"            "LDL_plus_VLDL"      "MCP_1_phys"         "Insulin_ug_l"      
>> [17] "Glucose_Insulin"    "Leptin_pg_ml"       "Adiponectin"        "Aortic.lesions"    
>> [21] "Aneurysm"           "Aortic_cal_M"       "Aortic_cal_L"       "CoronaryArtery_Cal"
>> [25] "Myocardial_cal"     "BMD_all_limbs"      "BMD_femurs_only"

# Form a data frame analogous to expression data that will hold the clinical traits.

femaleSamples = rownames(datExpr);
traitRows = match(femaleSamples, allTraits$Mice);
datTraits = allTraits[traitRows, -1];
rownames(datTraits) = allTraits[traitRows, 1]

traitColors = numbers2colors(datTraits, signed = FALSE);
# Plot the sample dendrogram and the colors underneath.
plotDendroAndColors(sampleTree2, traitColors,
                    groupLabels = names(datTraits), 
                    main = "Sample dendrogram and trait heatmap")
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/unnamed-chunk-105-3.png)

### 构建共表达网络，发现模块

这一步主要就是选择合适的软阈值 *β* 来构建无标度的基因共表达网络，然后基于网络的邻接矩阵进行聚类来发现模块：

``` r
# Choose a set of soft-thresholding powers
powers = c(c(1:10), seq(from = 12, to=20, by=2))
# Call the network topology analysis function
sft = pickSoftThreshold(datExpr, powerVector = powers, verbose = 5)
>> pickSoftThreshold: will use block size 3000.
>>  pickSoftThreshold: calculating connectivity for given powers...
>>    ..working on genes 1 through 3000 of 3000
>>    Power SFT.R.sq  slope truncated.R.sq mean.k. median.k. max.k.
>> 1      1   0.0402  0.416          0.444  606.00  625.0000  958.0
>> 2      2   0.0976 -0.504          0.797  205.00  205.0000  445.0
>> 3      3   0.2750 -0.861          0.943   90.00   83.3000  245.0
>> 4      4   0.4050 -1.150          0.921   46.40   39.4000  151.0
>> 5      5   0.7500 -1.180          0.926   26.90   21.2000   99.9
>> 6      6   0.8710 -1.590          0.876   17.00   12.2000   86.8
>> 7      7   0.8600 -1.700          0.820   11.50    7.5000   78.1
>> 8      8   0.8190 -1.700          0.791    8.26    4.7800   71.8
>> 9      9   0.7320 -1.640          0.743    6.21    3.1700   66.9
>> 10    10   0.7010 -1.590          0.757    4.85    2.1500   62.7
>> 11    12   0.7340 -1.420          0.849    3.24    1.0500   55.8
>> 12    14   0.8200 -1.320          0.942    2.36    0.5480   50.1
>> 13    16   0.8480 -1.230          0.953    1.82    0.2980   45.4
>> 14    18   0.8860 -1.190          0.976    1.46    0.1650   41.3
>> 15    20   0.9040 -1.150          0.972    1.21    0.0972   37.7
# Plot the results:
par(mfrow = c(1,2));
cex1 = 0.9;
# Scale-free topology fit index as a function of the soft-thresholding power
plot(sft$fitIndices[,1], -sign(sft$fitIndices[,3])*sft$fitIndices[,2],
     xlab="Soft Threshold (power)",ylab="Scale Free Topology Model Fit,signed R^2",type="n",
     main = paste("Scale independence"));
text(sft$fitIndices[,1], -sign(sft$fitIndices[,3])*sft$fitIndices[,2],
     labels=powers,cex=cex1,col="red");
# this line corresponds to using an R^2 cut-off of h
abline(h=0.90,col="red")
# Mean connectivity as a function of the soft-thresholding power
plot(sft$fitIndices[,1], sft$fitIndices[,5],
     xlab="Soft Threshold (power)",ylab="Mean Connectivity", type="n",
     main = paste("Mean connectivity"))
text(sft$fitIndices[,1], sft$fitIndices[,5], labels=powers, cex=cex1,col="red")
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/unnamed-chunk-106-1.png)

选择阈值有两个标准：

-   基于左边的图选择尽可能构建无标度网络的值（R方）
-   基于右边的图选择尽可能高的 mean connectivity，从而在检测 module 和 hub gene 的时候 power比较高

选择好阈值之后就可以来进行构建邻接矩阵（行列都是基因，每个 cell 是基因表达之间的相关性）和 TOM（Topological Overlap）矩阵，并进行聚类：

``` r
softPower = 6;
adjacency = adjacency(datExpr, power = softPower);

# Turn adjacency into topological overlap
##计算 TOM 是最耗时的步骤
TOM = TOMsimilarity(adjacency);
>> ..connectivity..
>> ..matrix multiplication (system BLAS)..
>> ..normalization..
>> ..done.
dissTOM = 1-TOM

###进行聚类
# Call the hierarchical clustering function
geneTree = hclust(as.dist(dissTOM), method = "average");
# Plot the resulting clustering tree (dendrogram)
plot(geneTree, xlab="", sub="", main = "Gene clustering on TOM-based dissimilarity",
     labels = FALSE, hang = 0.04)
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/unnamed-chunk-107-1.png)

``` r

# We like large modules, so we set the minimum module size relatively high:
minModuleSize = 30;
# Module identification using dynamic tree cut:
dynamicMods = cutreeDynamic(dendro = geneTree, distM = dissTOM,
                deepSplit = 2, pamRespectsDendro = FALSE,
                minClusterSize = minModuleSize);
>>  ..cutHeight not given, setting it to 0.995  ===>  99% of the (truncated) height range in dendro.
>>  ..done.
table(dynamicMods)
>> dynamicMods
>>   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19 
>>  88 496 319 316 241 208 188 151 149 115 112  95  91  86  77  67  62  56  49  34
# Convert numeric lables into colors
dynamicColors = labels2colors(dynamicMods)
table(dynamicColors)
>> dynamicColors
>>        black         blue        brown         cyan        green  greenyellow         grey 
>>          151          319          316           77          208           95           88 
>>       grey60    lightcyan   lightgreen  lightyellow      magenta midnightblue         pink 
>>           56           62           49           34          115           67          149 
>>       purple          red       salmon          tan    turquoise       yellow 
>>          112          188           86           91          496          241
# Plot the dendrogram and colors underneath
plotDendroAndColors(geneTree, dynamicColors, "Dynamic Tree Cut",
                    dendroLabels = FALSE, hang = 0.03,
                    addGuide = TRUE, guideHang = 0.05,
                    main = "Gene dendrogram and module colors")
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/unnamed-chunk-107-2.png)

### 计算模块的 eigengene 并与样本特征相关联

接下来就可以计算每个模块的 eigengene，并基于这些 eigengene 进行聚类从而将相似的模块进行合并，接着对新模块的 eigengene 和样本特征计算相关性，来找到我们感兴趣的模块：

``` r
# Calculate eigengenes
MEList = moduleEigengenes(datExpr, colors = dynamicColors)
MEs = MEList$eigengenes
# Calculate dissimilarity of module eigengenes
MEDiss = 1-cor(MEs);
# Cluster module eigengenes
METree = hclust(as.dist(MEDiss), method = "average");
# Plot the result
plot(METree, main = "Clustering of module eigengenes",
     xlab = "", sub = "")

##合并模块
MEDissThres = 0.25
# Plot the cut line into the dendrogram
abline(h=MEDissThres, col = "red")
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/unnamed-chunk-108-1.png)

``` r
# Call an automatic merging function
merge = mergeCloseModules(datExpr, dynamicColors, cutHeight = MEDissThres, verbose = 3)
>>  mergeCloseModules: Merging modules whose distance is less than 0.25
>>    multiSetMEs: Calculating module MEs.
>>      Working on set 1 ...
>>      moduleEigengenes: Calculating 20 module eigengenes in given set.
>>    multiSetMEs: Calculating module MEs.
>>      Working on set 1 ...
>>      moduleEigengenes: Calculating 15 module eigengenes in given set.
>>    Calculating new MEs...
>>    multiSetMEs: Calculating module MEs.
>>      Working on set 1 ...
>>      moduleEigengenes: Calculating 15 module eigengenes in given set.
# The merged module colors
mergedColors = merge$colors;
# Eigengenes of the new merged modules:
mergedMEs = merge$newMEs

plotDendroAndColors(geneTree, cbind(dynamicColors, mergedColors),
                    c("Dynamic Tree Cut", "Merged dynamic"),
                    dendroLabels = FALSE, hang = 0.03,
                    addGuide = TRUE, guideHang = 0.05)
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/unnamed-chunk-108-2.png)

``` r
moduleColors = mergedColors
# Construct numerical labels corresponding to the colors
colorOrder = c("grey", standardColors(50));
moduleLabels = match(moduleColors, colorOrder)-1;
MEs = mergedMEs

# Define numbers of genes and samples
nGenes = ncol(datExpr);
nSamples = nrow(datExpr);
# Recalculate MEs with color labels
MEs0 = moduleEigengenes(datExpr, moduleColors)$eigengenes
MEs = orderMEs(MEs0)
##计算相关性
datTraits <- datTraits[,1:2]
moduleTraitCor = cor(MEs, datTraits, use = "p");
moduleTraitPvalue = corPvalueStudent(moduleTraitCor, nSamples)

##画图
textMatrix =  paste(signif(moduleTraitCor, 2), "\n(",
                           signif(moduleTraitPvalue, 1), ")", sep = "");
dim(textMatrix) = dim(moduleTraitCor)
par(mar = c(6, 8.5, 3, 3));
# Display the correlation values within a heatmap plot
labeledHeatmap(Matrix = moduleTraitCor,
               xLabels = names(datTraits),
               yLabels = names(MEs),
               ySymbols = names(MEs),
               colorLabels = FALSE,
               colors = greenWhiteRed(50),
               textMatrix = textMatrix,
               setStdMargins = FALSE,
               cex.text = 0.5,
               zlim = c(-1,1),
               main = paste("Module-trait relationships"))
>> Warning in greenWhiteRed(50): WGCNA::greenWhiteRed: this palette is not suitable for people
>> with green-red color blindness (the most common kind of color blindness).
>> Consider using the function blueWhiteRed instead.
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/unnamed-chunk-108-3.png)

我们就可以选取和感兴趣的特征相关性强的模块进行后续的研究，比如这里面的 black 模块和病人的体重变量显著正相关，就可以将这个模块中的基因提取出来：

``` r
black <- names(datExpr)[moduleColors=="black"]
annot = read.csv(file = "../test/GeneAnnotation.csv")
annot$gene_symbol[which(annot$substanceBXH %in% black)] %>% na.omit()
>>   [1] "Ngfrap1"       "Nrarp"         "9930023K05Rik" "Unc5b"         "Sulf1"        
>>   [6] "Ltbp1"         "Tusc3"         "Prg4"          "Efemp2"        "2310046G15Rik"
>>  [11] "1200003C23Rik" "Art4"          "Ctps"          "A830073O21Rik" "Sparc"        
>>  [16] "A530057A03Rik" "Timp1"         "Eltd1"         "Jun"           "Mogat1"       
>>  [21] "1200009I06Rik" "Osbpl5"        "Jam2"          "Npn3"          "5031439A09Rik"
>>  [26] "Bmp5"          "D4st1"         "Zfp521"        "Ppic"          "Tagln"        
>>  [31] "3222401M22Rik" "Aard"          "BC029214"      "2600017H02Rik" "Arsa"         
>>  [36] "Ehd2"          "Gal3st1"       "Npdc1"         "Sdcbp2"        "Sh3bp4"       
>>  [41] "Rtn1"          "Kitl"          "Pdir"          "Ggta1"         "9230112O05Rik"
>>  [46] "Fbln2"         "Sgce"          "Serpina1d"     "Col16a1"       "Atp8b2"       
>>  [51] "Zfp503"        "Tnfsf13b"      "0610039N19Rik" "4632428N05Rik" "Cpxm1"        
>>  [56] "Fmo3"          "2810489O06Rik" "Nbea"          "Mat1a"         "Numbl"        
>>  [61] "6230427J02Rik" "9030624L02Rik" "Cnr2"          "Pam"           "1500041B16Rik"
>>  [66] "Akr1b8"        "Loxl1"         "Ppm1l"         "Lum"           "Agpt2"        
>>  [71] "E230026N22Rik" "Gpld1"         "Kcnq1"         "Ian6"          "Itgbl1"       
>>  [76] "E430007K15Rik" "Lgi2"          "Tek"           "C1qr1"         "9030408N13Rik"
>>  [81] "Slc39a14"      "C86987"        "Ccdc3"         "Cxcl14"        "Pcolce"       
>>  [86] "Cldn5"         "Adamts2"       "1300018K11Rik" "Ctla2a"        "Thbd"         
>>  [91] "Adam8"         "Kit"           "Xbp1"          "Plxnb1"        "Sorbs1"       
>>  [96] "Fgd5"          "Rab15"         "Cfh"           "Igfbp7"        "Bcl6b"        
>> [101] "Prss21"        "C8b"           "Bmp2"          "BC025446"      "Itga8"        
>> [106] "Ccl2"          "5730485H21Rik" "Clstn3"        "Cyp2c40"       "F2r"          
>> [111] "Raet1e"        "Pdgfrb"        "Inhbb"         "Sctr"          "Ctsd"         
>> [116] "Gmpr"          "Vtn"           "Socs3"         "Tgm1"          "AI324046"     
>> [121] "Plekha4"       "Tm4sf1"        "Col1a2"        "Zfpm2"         "Gpm6b"        
>> [126] "Itih3"         "5430432M24Rik" "Slc3a1"        "Hspa5bp1"      "Slc38a2"      
>> [131] "Orm2"          "BC014805"      "Pparg"         "Tfpi"          "Frzb"         
>> [136] "Mfng"          "Mest"          "Gja1"          "3110041P15Rik" "Car3"         
>> [141] "Lamb3"         "Cxcl5"         "Tesc"          "Gpx7"          "1200013A08Rik"
>> [146] "Plvap"         "4833409A17Rik" "Gdf10"         "Ramp2"         "Apom"         
>> [151] "AU041783"      "Krt1-23"       "Avpr1a"        "Tuba1"         "Dsip1"        
>> [156] "Ces2"          "Tnc"           "Eml1"          "Rasip1"        "Emilin1"      
>> [161] "Fa2h"          "Vwf"           "Tm4sf6"        "Esam1"         "Mt1"          
>> [166] "Dcn"           "Serpina11"     "Prss11"        "Cbr3"          "E130307J04Rik"
>> [171] "BC011468"      "Cygb"          "Pcdhb17"       "Cyp2g1"        "5730469M10Rik"
>> [176] "Tgfb1i1"       "Tcf21"         "Cyp4b1"        "D10Ucla2"      "Slc9a9"       
>> [181] "D6Ertd32e"     "Sept4"         "Trem2"         "Serpina10"     "Art3"         
>> [186] "Ifitm2"        "Akap12"        "Fsp27"         "Timp3"         "F11"          
>> [191] "Cyp2c40"       "Igfals"        "Postn"         "D19Wsu12e"     "Heph"         
>> [196] "Plk2"          "Upp1"          "Lbp"           "Aox1"          "Sytl2"        
>> [201] "Col6a3"        "Gpihbp1"       "Lxn"           "Mmrn2"         "2610020H15Rik"
>> [206] "Pex11a"        "Stat3"         "BC025600"      "Zfp423"        "Scd2"         
>> [211] "Slc16a10"      "Orm1"          "Armcx1"        "Phlda3"        "1500004A08Rik"
>> [216] "BC024988"      "Gpnmb"         "Spp1"          "C530028O21Rik" "Ccbl1"        
>> [221] "2600006K01Rik" "Kcne3"         "Matn2"         "A230035L05Rik" "Map4k4"       
>> [226] "4631416L12Rik" "1600015H20Rik" "Egfr"          "Scara3"        "CRAD-L"       
>> [231] "Rgs5"          "D330037A14Rik" "Proz"          "Polydom"       "Armcx2"       
>> [236] "Nnmt"          "Daf1"          "Antxr1"        "Synpo"         "Edn1"         
>> [241] "Mrc1"          "Ptprb"         "BC038881"      "Sgk"           "Slc22a7"      
>> [246] "1110039C07Rik" "Serpina3n"     "AA960558"      "Trfr2"         "Defb1"        
>> [251] "Bicc1"         "2310016C16Rik" "Dok4"          "Dnaic1"        "C730007L20Rik"
>> [256] "5430416O09Rik" "Ehd3"          "Fscn1"         "Ctsk"          "Plxnd1"       
>> [261] "Fgb"           "Chst7"         "9330129D05Rik" "1110032E23Rik" "Osbpl3"       
>> [266] "Slc43a1"       "Dll4"          "Fabp4"         "9430059P22Rik" "Ppm1f"        
>> [271] "Arhgap18"      "Lama2"         "Rragd"         "Jam3"          "Cpb2"         
>> [276] "Ntf3"          "Rnase4"        "Hc"            "AI428795"      "1110028A07Rik"
>> [281] "Agxt"          "Flt1"          "Ang1"          "Rcn3"          "Vim"          
>> [286] "Mylip"         "1200009O22Rik" "Cd34"          "Ehhadh"        "Fbn1"         
>> [291] "Cdc42ep3"      "Kng2"          "Cav2"          "Prdc"          "Fetub"        
>> [296] "Stk39"         "Emcn"          "Serpinf2"      "Pdgfra"        "Cml1"         
>> [301] "Ddah2"         "Anxa2"         "Mlp"           "Entpd5"        "Oit3"         
>> [306] "Cd63"          "Fbln5"         "Acyp2"         "Icam2"         "Npl"          
>> [311] "Il1r1"         "0610039P13Rik" "A930021O22"    "2610001E17Rik" "Slc30a2"      
>> [316] "AI428936"      "Col4a2"        "2410004L22Rik" "AI173486"      "Col14a1"      
>> [321] "Col4a1"        "Wisp1"         "Drctnnb1a"     "Calcrl"        "1700021K02Rik"
>> [326] "Plat"          "Itih4"         "Rgs10"         "Slc37a1"       "Fmnl2"        
>> [331] "Epb4.1l2"      "Rgs3"          "Pdlim2"        "Mbl1"          "Scnn1a"       
>> [336] "Notch4"        "Hey1"          "2810004A10Rik" "Thbs2"         "Tm4sf2"       
>> [341] "3732412D22Rik" "Slc6a8"        "Nr2f1"         "Col5a1"        "Casp12"       
>> [346] "Hoxb2"         "Hey2"          "mKIAA1236"     "Maged2"        "Lox"          
>> [351] "Col1a1"        "Ptpn13"        "Serpina3c"     "Crat"          "Kcnj8"        
>> [356] "Mmp2"          "Olfml3"        "F7"            "Cd36"          "D330012D11Rik"
>> [361] "Pde6h"         "Anxa1"         "Fkhl18"        "Sdc4"          "9630050M13Rik"
>> [366] "Ltbp3"         "Cmya4"         "Itih1"         "Fhl2"          "Sdpr"         
>> [371] "Nptxr"         "Dnajc12"       "Lrg1"          "Lrat"          "Boc"          
>> attr(,"na.action")
>>  [1]   1   9  18  20  41  50  52 133 176 177 196 207 236 239 307 324 326
>> attr(,"class")
>> [1] "omit"
```

``` r
# Define variable weight containing the weight column of datTrait
weight = as.data.frame(datTraits$weight_g);
names(weight) = "weight"
# names (colors) of the modules
modNames = substring(names(MEs), 3)

geneModuleMembership = as.data.frame(cor(datExpr, MEs, use = "p"));
MMPvalue = as.data.frame(corPvalueStudent(as.matrix(geneModuleMembership), nSamples));

names(geneModuleMembership) = paste("MM", modNames, sep="");
names(MMPvalue) = paste("p.MM", modNames, sep="");

geneTraitSignificance = as.data.frame(cor(datExpr, weight, use = "p"));
GSPvalue = as.data.frame(corPvalueStudent(as.matrix(geneTraitSignificance), nSamples));

names(geneTraitSignificance) = paste("GS.", names(weight), sep="");
names(GSPvalue) = paste("p.GS.", names(weight), sep="");

module = "black"
column = match(module, modNames);
moduleGenes = moduleColors==module;

par(mfrow = c(1,1));
verboseScatterplot(abs(geneModuleMembership[moduleGenes, column]),
                   abs(geneTraitSignificance[moduleGenes, 1]),
                   xlab = paste("Module Membership in", module, "module"),
                   ylab = "Gene significance for body weight",
                   main = paste("Module membership vs. gene significance\n"),
                   cex.main = 1.2, cex.lab = 1.2, cex.axis = 1.2, col = module)
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/unnamed-chunk-110-1.png)

``` r
##找 hub 基因
MM  <- geneModuleMembership[moduleGenes, column]
MMP <- MMPvalue[moduleGenes, column]
GS <-  geneTraitSignificance[moduleGenes, 1]
GSP <- GSPvalue[moduleGenes, 1]

mydata <- data.frame(moduleGenes=colnames(datExpr)[moduleGenes],MM,MMP,GS,GSP)

mydata <- mydata %>% 
  filter(MMP < 0.05,GSP < 0.05) %>% 
  arrange(desc(MM),desc(GS))
```
