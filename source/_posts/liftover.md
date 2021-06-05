---
title: 基因组版本坐标转化
date: 2021-02-26 10:00:00    
index_img: img/liftover1.png
---





使用R包`rtracklayer` 进行基因组坐标版本转化

<!-- more -->

现在有GRCh37的基因组坐标文件(TCGA的突变记录)，要将其转换成GRCh38的坐标：

``` r
library(dplyr)
>> Warning: package 'dplyr' was built under R version 3.6.3
>> 
>> Attaching package: 'dplyr'
>> The following objects are masked from 'package:stats':
>> 
>>     filter, lag
>> The following objects are masked from 'package:base':
>> 
>>     intersect, setdiff, setequal, union

dt <- readRDS("../test/dt.rds") %>% 
  as.data.frame()

head(dplyr::as_tibble(dt))
>> # A tibble: 6 x 8
>>   Hugo_Symbol Chromosome Start_position End_position sample                Protein_Change Variant_Classifica~ Variant_Type
>>   <chr>            <int>          <int>        <int> <chr>                 <chr>          <chr>               <chr>       
>> 1 SH3PXD2A            10      105614934    105614934 TCGA-OR-A5J1-01A-11D~ .              Intron              SNP         
>> 2 INPP5F              10      121556913    121556913 TCGA-OR-A5J1-01A-11D~ .              Intron              SNP         
>> 3 ITIH2               10        7772149      7772149 TCGA-OR-A5J1-01A-11D~ .              Intron              SNP         
>> 4 OPN4                10       88419681     88419681 TCGA-OR-A5J1-01A-11D~ p.G288D        Missense_Mutation   SNP         
>> 5 TRIM49B             11       49053482     49053482 TCGA-OR-A5J1-01A-11D~ p.L111F        Missense_Mutation   SNP         
>> 6 DNAJC4              11       64001585     64001585 TCGA-OR-A5J1-01A-11D~ p.G219W        Missense_Mutation   SNP
```

转换的方法有多种，可以参考[Converting Genome Coordinates From One Genome
Version To
Another](https://www.biostars.org/p/65558/),这里使用R包`rtracklayer`来转换

安装包：

``` r
if(!("rtracklayer" %in% installed.packages())){
  BiocManager::install("rtracklayer")
}

library(rtracklayer)
```

我们需要使用的是这个包中的`liftOver`函数：

``` r
?liftOver
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210226171357422.png)

该函数需要两个输入：需要转换的基因组区间(**GRanges对象**)；chain文件

首先将数据框转化为GRanges对象(需要使用GenomicRanges包，在library(rtracklayer)中已经载入了)：

``` r
dt$Chromosome <- paste0("chr",dt$Chromosome)

dt_granges <- makeGRangesFromDataFrame(dt,
                         keep.extra.columns=TRUE,
                         ignore.strand=TRUE,
                         seqinfo=NULL,
                         seqnames.field="Chromosome",##染色体所在列的名称
                         start.field="Start_position",##起始位点所在列的名称
                         end.field="End_position",##终止位点所在列的名称
                         starts.in.df.are.0based=FALSE)##是否是0based的
```

这里面需要注意的是：需要转换的基因组坐标起始位点是0based还是1based；对于TCGA的[maf文件](https://docs.gdc.cancer.gov/Encyclopedia/pages/Mutation_Annotation_Format_TCGAv2/)是1-based的，所以这里选FALSE：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210226172456415.png)
需要的第二个文件是Chain文件，需要在[UCSC](https://hgdownload.soe.ucsc.edu/downloads.html)网站上下载：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210226225522742.png)

使用`import.chain`函数导入Chain文件，进行转换：

``` r
chainObject <- import.chain("../test/hg19ToHg38.over.chain")

results <- as.data.frame(liftOver(dt_granges, chainObject))

head(dplyr::as_tibble(results))
>> # A tibble: 6 x 12
>>   group group_name seqnames   start     end width strand Hugo_Symbol sample   Protein_Change Variant_Classif~ Variant_Type
>>   <int> <chr>      <fct>      <int>   <int> <int> <fct>  <chr>       <chr>    <chr>          <chr>            <chr>       
>> 1     1 <NA>       chr10     1.04e8  1.04e8     1 *      SH3PXD2A    TCGA-OR~ .              Intron           SNP         
>> 2     2 <NA>       chr10     1.20e8  1.20e8     1 *      INPP5F      TCGA-OR~ .              Intron           SNP         
>> 3     3 <NA>       chr10     7.73e6  7.73e6     1 *      ITIH2       TCGA-OR~ .              Intron           SNP         
>> 4     4 <NA>       chr10     8.67e7  8.67e7     1 *      OPN4        TCGA-OR~ p.G288D        Missense_Mutati~ SNP         
>> 5     5 <NA>       chr11     4.90e7  4.90e7     1 *      TRIM49B     TCGA-OR~ p.L111F        Missense_Mutati~ SNP         
>> 6     6 <NA>       chr11     6.42e7  6.42e7     1 *      DNAJC4      TCGA-OR~ p.G219W        Missense_Mutati~ SNP
```
