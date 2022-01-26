---
title: 使用 STAR-Fusion 鉴别融合基因
author: wutao
date: 2022-01-24 10:00:00 
categories:
  - 生物信息
index_img: img/star_fusion.png
---



使用 STAR-Fusion 及相关的软件从 RNA-Seq 数据中预测融合基因

<!-- more -->

## 安装 STAR-Fusion

有两种方法可以安装 STAR-Fusion，一种是下载 Github 的 [release]([STAR-Fusion/STAR-Fusion/releases](https://github.com/STAR-Fusion/STAR-Fusion/releases)) （注意下载第一个，其他两个是自动生成的，可能有些组件不全）；第二种方法就是克隆 Github 仓库：

```shell
##下载好解压后,进入文件夹
make
##然后将STAR-Fusion的路径加入Bashrc的 PATH即可
```

STAR-Fusion 依赖一些其他的软件，需要安装：

- STAR

- samtools

- 一些 Perl 模块：

  ```shell
  perl -MCPAN -e shell
  install DB_File
  install URI::Escape
  install Set::IntervalTree
  install Carp::Assert
  install JSON::XS
  install PerlIO::gzip
  ```

先安装 STAR：

```shell
# Get latest STAR source from releases
wget https://github.com/alexdobin/STAR/archive/2.7.10a.tar.gz
tar -xzf 2.7.10a.tar.gz
cd STAR-2.7.10a
cd source
make STAR

##将 STAR 加入 PATH
vi ~/.bashrc
#bashrc添加：
PATH="/home/data/t040201/software/STAR-2.7.10a/bin/Linux_x86_64:$PATH"
source ~/.bashrc

##验证一下是否安装成功
cd 
STAR
#出现以下信息则安装成功
Usage: STAR  [options]... --genomeDir /path/to/genome/index/   --readFilesIn R1.fq R2.fq
Spliced Transcripts Alignment to a Reference (c) Alexander Dobin, 2009-2020

STAR version=2.7.10a
STAR compilation time,server,dir=2022-01-14T18:50:00-05:00 :/home/dobin/data/STAR/STARcode/STAR.master/source
For more details see:
<https://github.com/alexdobin/STAR>
<https://github.com/alexdobin/STAR/blob/master/doc/STARmanual.pdf>

To list all parameters, run STAR --help
```

samtools 也是类似的，下载安装就行：

```shell
tar -xjf samtools-1.13.tar.bz2 
cd samtools-1.13 
./configure --prefix=/home/data/t040201/software/samtools 
make 
make install
```

安装需要的 Perl 模块（通过 conda 安装比较方便）：

```shell
conda create -n STAR_fusion
conda install -c bioconda perl-db-file perl-uri perl-set-intervaltree perl-json-xs perl-perlio-gzip
conda install -c conda-forge perl-carp-assert
```

注意如果这里安装了`perl-set-intervaltree` 可能会和上面 STAR-Fusion make 过程中安装的 `intervaltree` 发生冲突，可以直接将 STAR-Fusion 路径下的 `PerlLib/Set` 目录给删除，这样就可以使用我们用 conda 安装的 `intervaltree` 了。

STAR-Fusion 还需要参考基因组文件以及相关的基因注释文件，可以从[这里](https://data.broadinstitute.org/Trinity/CTAT_RESOURCE_LIB/) 下载，如果下载的是较大的30G左右的文件（GRCh38）就可以直接使用，如果下载的是几个 G 的未经处理的文件，则需要进行进一步的处理才能被 STAR-Fusion 使用，具体方法见 [Github](https://github.com/STAR-Fusion/STAR-Fusion/wiki/installing-star-fusion), 这里直接下载了 30G 的文件（可以使用迅雷下载，比较快），解压就可以使用：

```shell
tar -zxvf GRCh38_gencode_v37_CTAT_lib_Mar012021.plug-n-play.tar.gz
```

如果后面要进行转录本的重构，则需要安装 `Trinity` 和 `GMAP`:

```shell
####安装依赖
###下载预编译的 bowtie2
unzip bowtie2-2.4.5-linux-x86_64.zip 
##安装路径加入 PATH 即可

### 下载安装Jellyfish https://github.com/gmarcais/Jellyfish/releases
./configure --prefix=/home/data/t040201/software/Jellyfish/
make & make install
##bin 加入 PATH 即可

### 下载预编译的 salmon
#https://github.com/COMBINE-lab/salmon/releases
tar -xvzf salmon-1.6.0_linux_x86_64.tar.gz
##bin 加入 PATH 即可

##下载安装 trinity https://github.com/trinityrnaseq/trinityrnaseq/releases
make
##.bashrc中加入：
export TRINITY_HOME=/home/data/t040201/software/trinityrnaseq-v2.13.2

### 下载安装GMAP http://research-pub.gene.com/gmap/
 ./configure --prefix=/home/data/t040201/software/gmap_gsnap/
 ##bin 加入 PATH 即可
```

## 运行 STAR-Fusion

有两种方式来运行 STAR-Fusion，一种就是从 FASTQ 文件开始，另一种是之前已经使用 STAR 进行比对，会得到 `Chimeric.out.junction` 文件，就可以直接使用这个文件作为 STAR-Fusion 的输入 （这种方式具体参照官方文档）。

对于双端测序的 FASTQ 文件，可以运行（这里用 HEC108 细胞系的 RNA-seq 数据为例）：

```shell
 STAR-Fusion --genome_lib_dir /home/data/t040201/data/STAR_data/GRCh38_gencode_v37_CTAT_lib_Mar012021.plug-n-play/ctat_genome_lib_build_dir --left_fq HEC108_1.fastq.gz --right_fq HEC108_2.fastq.gz  --output_dir /home/data/t040201/cell_lines/fusion
```

其中参数：

- `genome_lib_dir` ：刚刚下载的 30 多 G 数据解压后的 `ctat_genome_lib_build_dir` 子目录的路径
- `left_fq` , `right_fq`：双端测序的 fastq 文件
- `output_dir`：输出路径

如果是单端测序，只需要指定 `left_fq` 就行了。

如果是已经有了 `Chimeric.out.junction` 文件则可以直接运行：

```
 STAR-Fusion --genome_lib_dir /home/data/t040201/data/STAR_data/GRCh38_gencode_v37_CTAT_lib_Mar012021.plug-n-play/ctat_genome_lib_build_dir -J ./fusion/Chimeric.out.junction --output_dir /home/data/t040201/cell_lines/fusion
```

## 输出文件

先需要了解几个概念：

- Fragment 和 Read ：对于双端测序，一个 Fragment 会有两个 reads 从两端开始测序，而单端测序只有一个 read：

  ![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/fgene-05-00005-g001.jpg)

- Spanning reads 和 split reads：Spanning reads 指的是双端测序中有一个 read 匹配到与另一个 read 不同的基因上，而 split read 指的是一个 read 横跨两个基因

STAR-Fusion 的输出文件主要有两个：`star-fusion.fusion_predictions.abridged.tsv` 和 `star-fusion.fusion_predictions.tsv`，先来看第一个文件中的内容：

| #FusionName         | JunctionReadCount | SpanningFragCount | est\_J | est\_S | SpliceType        | LeftGene                      | LeftBreakpoint    | RightGene                    | RightBreakpoint   | LargeAnchorSupport | FFPM    | LeftBreakDinuc | LeftBreakEntropy | RightBreakDinuc | RightBreakEntropy | annots                                 |
| ------------------- | ----------------- | ----------------- | ------ | ------ | ----------------- | ----------------------------- | ----------------- | ---------------------------- | ----------------- | ------------------ | ------- | -------------- | ---------------- | --------------- | ----------------- | -------------------------------------- |
| PHACTR4\-\-RCC1     | 128               | 48                | 128    | 48     | ONLY\_REF\_SPLICE | PHACTR4^ENSG00000204138\.13   | chr1:28407463:\+  | RCC1^ENSG00000180198\.16     | chr1:28529858:\+  | YES\_LDAS          | 1\.801  | GT             | 1\.9329          | AG              | 1\.8323           | \["CCLE\_StarF2019"                    |
| LEPROT\-\-LEPR      | 51                | 4                 | 51     | 4      | ONLY\_REF\_SPLICE | LEPROT^ENSG00000213625\.9     | chr1:65425378:\+  | LEPR^ENSG00000116678\.20     | chr1:65565546:\+  | YES\_LDAS          | 0\.5628 | GT             | 1\.8256          | AG              | 1\.9219           | \["GTEx\_recurrent\_StarF2019"         |
| SHISA9\-\-U91319\.1 | 35                | 13                | 35     | 11\.67 | ONLY\_REF\_SPLICE | SHISA9^ENSG00000237515\.9     | chr16:13203549:\+ | U91319\.1^ENSG00000262801\.6 | chr16:13350180:\+ | YES\_LDAS          | 0\.4775 | GT             | 1\.9899          | AG              | 1\.9656           | \["GTEx\_recurrent\_StarF2019"         |
| CRISPLD2\-\-CDH13   | 27                | 9                 | 27     | 9      | ONLY\_REF\_SPLICE | CRISPLD2^ENSG00000103196\.12  | chr16:84854829:\+ | CDH13^ENSG00000140945\.17    | chr16:83486477:\+ | YES\_LDAS          | 0\.3684 | GT             | 1\.9656          | AG              | 1\.9329           | \["CCLE\_StarF2019"                    |
| PFKFB3\-\-LINC02649 | 25                | 11                | 25     | 11     | ONLY\_REF\_SPLICE | PFKFB3^ENSG00000170525\.21    | chr10:6226365:\+  | LINC02649^ENSG00000215244\.3 | chr10:6326546:\+  | YES\_LDAS          | 0\.3684 | GT             | 1\.9329          | AG              | 1\.9329           | \["INTRACHROMOSOMAL\[chr10:0\.02Mb\]"  |
| SEPTIN7P14\-\-PSPH  | 24                | 0                 | 12     | 0      | ONLY\_REF\_SPLICE | SEPTIN7P14^ENSG00000245958\.6 | chr4:119455133:\+ | PSPH^ENSG00000146733\.14     | chr7:56021231:\-  | YES\_LDAS          | 0\.1228 | GT             | 1\.8892          | AG              | 1\.9656           | \["INTERCHROMOSOMAL\[chr4\-\-chr7\]"\] |
| NRIP1\-\-LINC02246  | 13                | 5                 | 13     | 2\.32  | ONLY\_REF\_SPLICE | NRIP1^ENSG00000180530\.11     | chr21:15064745:\- | LINC02246^ENSG00000281903\.2 | chr21:14857708:\- | YES\_LDAS          | 0\.1567 | GT             | 1\.2729          | AG              | 1\.5546           | \["GTEx\_recurrent\_StarF2019"         |
| LINC02643\-\-NEBL   | 8                 | 5                 | 8      | 2\.5   | ONLY\_REF\_SPLICE | LINC02643^ENSG00000230109\.1  | chr10:21369967:\- | NEBL^ENSG00000078114\.19     | chr10:21020201:\- | YES\_LDAS          | 0\.1075 | GT             | 1\.9656          | AG              | 1\.7056           | \["INTRACHROMOSOMAL\[chr10:0\.05Mb\]"  |
| TVP23C\-\-CDRT4     | 6                 | 9                 | 5\.88  | 5\.87  | ONLY\_REF\_SPLICE | TVP23C^ENSG00000175106\.17    | chr17:15540433:\- | CDRT4^ENSG00000239704\.11    | chr17:15440285:\- | YES\_LDAS          | 0\.1203 | GT             | 1\.8323          | AG              | 1\.9899           | \["INTRACHROMOSOMAL\[chr17:0\.03Mb\]"  |
| NRIP1\-\-LINC02246  | 6                 | 10                | 6      | 5\.79  | ONLY\_REF\_SPLICE | NRIP1^ENSG00000180530\.11     | chr21:15014344:\- | LINC02246^ENSG00000281903\.2 | chr21:14857708:\- | YES\_LDAS          | 0\.1206 | GT             | 1\.8892          | AG              | 1\.5546           | \["GTEx\_recurrent\_StarF2019"         |

- FusionName：融合基因的名称，断点上下游的基因名称
- JunctionReadCount：含有 Split read 的 RNA fragment 的数量
- SpanningFragCount：含有 Spanning read 的 RNA fragment 的数量，如果一个融合事件的 JunctionReads 或者 （和）SpanningFrag 比较小，那么这个事件可能是假阳性的融合
- est_J , est_S, FFPM：JunctionReadCount 和 SpanningFragCount 是原始的，直接读出来的 read counts ，而 est_J 和 est_S 是考虑了多重比对和 fusion 转录本的多样性后的估计值，然后基于这些值计算 FFPM （fusion fragments per million total reads），一般使用这个值进行过滤融合事件（大于 0.1 FFPM），这个计算方法类似 RSEM （相关的讨论见[这里](https://groups.google.com/g/star-fusion/c/THb6TxGrSBg?pli=1) )
- SpliceType：表示预测的断点是否发生在提供的参考转录本注释（gencode）的外显子分界处
- LargeAnchorSupport：是否有 split read 提供在断点两侧的较长的比对（设定为 25 个碱基）；如果融合事件仅仅被 split reads 支持（没有 spanning reads）并且缺少LargeAnchorSupport，那么这个融合事件就很可疑，可能是假阳性
- Left/RightBreakEntropy：断点两侧的 15 个碱基的 shannon 熵，低代表不可信（不知道为什么。。。）
- annots：使用 [FusionAnnotator](https://github.com/FusionAnnotator/FusionAnnotator/wiki) 对融合转录本的简单注释

`star-fusion.fusion_predictions.tsv` 就是多了两列：`JunctionReads` 和 `SpanningFrags` 以逗号分隔的方式列出了具体的 split reads 和 spanning reads。

## 使用 FusionInspector 对融合事件进行检查验证

FusionInspector 可以对融合事件进行检查，验证（其他的融合基因检测软件输出的结果也可以）并可视化。FusionInspector 提取融合对的基因组区间并构建包含基因对的 mini-fusion-contigs ，然后原始的 reads 被比对到这个 contig 上：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/FusionInspector-alg_overview.png)

运行的时候只需要加上 `--FusionInspector` 就可以，这个工具有两种模式：

- inspect：只有被 STAR-Fusion 鉴别出支持融合事件的 reads 才会被比对到融合基因的 contig 上
- validate：对所有的 reads 进行重新评估

另外还需要把安装一些依赖的 python 包：

```shell
 conda install -c anaconda numpy requests
 conda install -c bioconda igv-reports
```

接着就可以运行了：

```shell
STAR-Fusion --genome_lib_dir /home/data/t040201/data/STAR_data/GRCh38_gencode_v37_CTAT_lib_Mar012021.plug-n-play/ctat_genome_lib_build_dir --left_fq HEC108_1.fastq.gz --right_fq HEC108_2.fastq.gz  --output_dir /home/data/t040201/cell_lines/fusion --FusionInspector inspect
```

> 出现了一个报错：/bin/sh: bgzip: command not found，因此安装一下htslib：
>
> 下载[SAMtools/BCFtools/HTSlib - Downloads](http://www.htslib.org/download/)
>
> 编译：./configure --prefix=/home/data/t040201/software/htslib
>
> 安装：make & make install
>
> 添加 PATH

主要的输出文件是 `finspector.FusionInspector.fusions.abridged.tsv` （在新生成的目录FusionInspector-inspect 下）：

| FusionName          | JunctionReadCount | SpanningFragCount | est\_J | est\_S | LeftGene                     | LeftLocalBreakpoint | LeftBreakpoint    | RightGene                    | RightLocalBreakpoint | RightBreakpoint   | SpliceType        | LargeAnchorSupport | NumCounterFusionLeft | NumCounterFusionRight | FAR\_left | FAR\_right | LeftBreakDinuc | LeftBreakEntropy | RightBreakDinuc | RightBreakEntropy | microh\_brkpt\_dist | num\_microh\_near\_brkpt |
| ------------------- | ----------------- | ----------------- | ------ | ------ | ---------------------------- | ------------------- | ----------------- | ---------------------------- | -------------------- | ----------------- | ----------------- | ------------------ | -------------------- | --------------------- | --------- | ---------- | -------------- | ---------------- | --------------- | ----------------- | ------------------- | ------------------------ |
| PHACTR4\-\-RCC1     | 117               | 34                | 117    | 34     | PHACTR4^ENSG00000204138\.13  | 2298                | chr1:28407463:\+  | RCC1^ENSG00000180198\.16     | 34901                | chr1:28529858:\+  | ONLY\_REF\_SPLICE | YES                | 0                    | 0                     | 152       | 152        | GT             | 1\.9329          | AG              | 1\.8323           | 1539                | 0                        |
| LEPROT\-\-LEPR      | 48                | 3                 | 48     | 3      | LEPROT^ENSG00000213625\.9    | 2949                | chr1:65425378:\+  | LEPR^ENSG00000116678\.20     | 14955                | chr1:65565546:\+  | ONLY\_REF\_SPLICE | YES                | 0                    | 0                     | 52        | 52         | GT             | 1\.8256          | AG              | 1\.9219           | 57                  | 2                        |
| SHISA9\-\-U91319\.1 | 37                | 10                | 37     | 10     | SHISA9^ENSG00000237515\.9    | 9996                | chr16:13203549:\+ | U91319\.1^ENSG00000262801\.6 | 25888                | chr16:13350180:\+ | ONLY\_REF\_SPLICE | YES                | 0                    | 0                     | 48        | 48         | GT             | 1\.9899          | AG              | 1\.9656           | 3012                | 0                        |
| CRISPLD2\-\-CDH13   | 27                | 9                 | 27     | 9      | CRISPLD2^ENSG00000103196\.12 | 11156               | chr16:84854829:\+ | CDH13^ENSG00000140945\.17    | 50276                | chr16:83486477:\+ | ONLY\_REF\_SPLICE | YES                | 0                    | 0                     | 37        | 37         | GT             | 1\.9656          | AG              | 1\.9329           | 5725                | 0                        |
| PFKFB3\-\-LINC02649 | 22                | 10                | 22     | 10     | PFKFB3^ENSG00000170525\.21   | 16805               | chr10:6226365:\+  | LINC02649^ENSG00000215244\.3 | 31340                | chr10:6326546:\+  | ONLY\_REF\_SPLICE | YES                | 0                    | 0                     | 33        | 33         | GT             | 1\.9329          | AG              | 1\.9329           | 2817                | 0                        |
| NRIP1\-\-LINC02246  | 14                | 4                 | 14     | 2\.24  | NRIP1^ENSG00000180530\.11    | 2192                | chr21:15064745:\- | LINC02246^ENSG00000281903\.2 | 24982                | chr21:14857708:\- | ONLY\_REF\_SPLICE | YES                | 0                    | 0                     | 19        | 19         | GT             | 1\.2729          | AG              | 1\.5546           | 5526                | 0                        |
| NRIP1\-\-LINC02246  | 6                 | 9                 | 6      | 6\.76  | NRIP1^ENSG00000180530\.11    | 4395                | chr21:15014344:\- | LINC02246^ENSG00000281903\.2 | 24982                | chr21:14857708:\- | ONLY\_REF\_SPLICE | YES                | 0                    | 0                     | 16        | 16         | GT             | 1\.8892          | AG              | 1\.5546           | 5568                | 0                        |
| LINC02643\-\-NEBL   | 8                 | 5                 | 8      | 5      | LINC02643^ENSG00000230109\.1 | 2205                | chr10:21369967:\- | NEBL^ENSG00000078114\.19     | 23713                | chr10:21020201:\- | ONLY\_REF\_SPLICE | YES                | 0                    | 0                     | 14        | 14         | GT             | 1\.9656          | AG              | 1\.7056           | 4138                | 0                        |
| TVP23C\-\-CDRT4     | 4                 | 10                | 4      | 10     | TVP23C^ENSG00000175106\.17   | 7982                | chr17:15540433:\- | CDRT4^ENSG00000239704\.11    | 22624                | chr17:15440285:\- | ONLY\_REF\_SPLICE | YES                | 0                    | 0                     | 15        | 15         | GT             | 1\.8323          | AG              | 1\.9899           | 2126                | 0                        |

可以看到这个文件里面少了 `SEPTIN7P14--PSPH`  一行，因为在 STAR-Fusion 的结果中，这个融合的 spanning reads 为 0 ，把他过滤掉了。这个文件中有新的几列：

- Left/RightlocalBreakpoint：断点在构建的 mini-fusion-gene contig 中的位置

- NumCounterFusionLeft/Right：在断点处支持非融合的等位基因的 RNA-seq fragment 的数量

- FAR_left/right：fusion allelic ratio，用来作为相对该基因的非融合等位位点的融合转录本表达量的一种定量，可用下图展示计算方法：

  ![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220125214733245.png)

在这个图中，共有两个支持 fusion 的 fragment，对于基因 A 有 3 个支持非 fusion 的fragment，对于基因 B 有 1 个这样的 fragment，因此可以分别对基因 A 和 B 计算 FAR，作为融合 VS 非融合转录本比例的粗糙估计。

## 融合效应的检测

有些时候融合转录本会产生新的蛋白，改变原蛋白的功能，可以使用 `examine_coding_effect` 参数来检测融合事件对编码区域造成的影响：

```shell
STAR-Fusion --genome_lib_dir /home/data/t040201/data/STAR_data/GRCh38_gencode_v37_CTAT_lib_Mar012021.plug-n-play/ctat_genome_lib_build_dir --left_fq HEC108_1.fastq.gz --right_fq HEC108_2.fastq.gz  --output_dir /home/data/t040201/cell_lines/fusion --FusionInspector inspect --examine_coding_effect
```

会产生一个新的文件 `star-fusion.fusion_predictions.abridged.coding_effect.tsv` ，这个文件比 STAR-Fusion 的直接输出会多出几列：

|    \#FusionName     | JunctionReadCount | SpanningFragCount | est\_J | est\_S | SpliceType        | LeftGene                      | LeftBreakpoint    | RightGene                    | RightBreakpoint   | LargeAnchorSupport | FFPM    | LeftBreakDinuc | LeftBreakEntropy | RightBreakDinuc | RightBreakEntropy | annots                                 | CDS\_LEFT\_ID                         | CDS\_LEFT\_RANGE                   | CDS\_RIGHT\_ID         | CDS\_RIGHT\_RANGE  | PROT\_FUSION\_TYPE | FUSION\_MODEL | FUSION\_CDS                                                  | FUSION\_TRANSL                                               | PFAM\_LEFT                                                   | PFAM\_RIGHT                                                  |                                       |      |      |
| :-----------------: | ----------------- | ----------------- | ------ | ------ | ----------------- | ----------------------------- | ----------------- | ---------------------------- | ----------------- | ------------------ | ------- | -------------- | ---------------- | --------------- | ----------------- | -------------------------------------- | ------------------------------------- | ---------------------------------- | ---------------------- | ------------------ | ------------------ | ------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------- | ---- | ---- |
|   PHACTR4\-\-RCC1   | 128               | 48                | 128    | 48     | ONLY\_REF\_SPLICE | PHACTR4^ENSG00000204138\.13   | chr1:28407463:\+  | RCC1^ENSG00000180198\.16     | chr1:28529858:\+  | YES\_LDAS          | 1\.801  | GT             | 1\.9329          | AG              | 1\.8323           | \["CCLE\_StarF2019"                    | "INTRACHROMOSOMAL\[chr1:0\.01Mb\]"    | "NEIGHBORS\[5579\]"\]              | ENST00000493669\.2     | 1\-16              | ENST00000486790\.2 | 298\-446      | FRAMESHIFT                                                   | chr1\|\+\|\[0\]28407448\-28407463\[0\]<==>chr1\|\+\|\[0\]28529858\-28529939\[0\]\|\[1\]28530523\-28530573\[0\]\|\[1\]28531803\-28531818\[1\] | atggaagatccatttgGACAGGAAGATGTCACCCAAGCGCATAGCTAAAAGAAGGTCCCCCCCAGCAGATGCCATCCCCAAAAGCAAGAAGGTGAAGGACACGAGGGCCGCTGCCTCCCGCCGCGTTCCTGGCGCCCGCTCCTGCCAAGTCTCACACAGGTCCCA | MEDPFGQEDVTQAHS\*KKVPPSRCHPQKQEGEGHEGRCLPPRSWRPLLPSLTQVP     | TGF\_beta\-PARTIAL\|6\-16~\|4\.4e\-06 | \.   |      |
|   LEPROT\-\-LEPR    | 51                | 4                 | 51     | 4      | ONLY\_REF\_SPLICE | LEPROT^ENSG00000213625\.9     | chr1:65425378:\+  | LEPR^ENSG00000116678\.20     | chr1:65565546:\+  | YES\_LDAS          | 0\.5628 | GT             | 1\.8256          | AG              | 1\.9219           | \["GTEx\_recurrent\_StarF2019"         | "ChimerSeq"                           | "INTRACHROMOSOMAL\[chr1:0\.09Mb\]" | "NEIGHBORS\[89682\]"\] | \.                 | \.                 | \.            | \.                                                           | \.                                                           | \.                                                           | \.                                                           | \.                                    | \.   | \.   |
| SHISA9\-\-U91319\.1 | 35                | 13                | 35     | 11\.67 | ONLY\_REF\_SPLICE | SHISA9^ENSG00000237515\.9     | chr16:13203549:\+ | U91319\.1^ENSG00000262801\.6 | chr16:13350180:\+ | YES\_LDAS          | 0\.4775 | GT             | 1\.9899          | AG              | 1\.9656           | \["GTEx\_recurrent\_StarF2019"         | "INTRACHROMOSOMAL\[chr16:0\.01Mb\]"   | "NEIGHBORS\[5816\]"\]              | \.                     | \.                 | \.                 | \.            | \.                                                           | \.                                                           | \.                                                           | \.                                                           | \.                                    | \.   |      |
|  CRISPLD2\-\-CDH13  | 27                | 9                 | 27     | 9      | ONLY\_REF\_SPLICE | CRISPLD2^ENSG00000103196\.12  | chr16:84854829:\+ | CDH13^ENSG00000140945\.17    | chr16:83486477:\+ | YES\_LDAS          | 0\.3684 | GT             | 1\.9656          | AG              | 1\.9329           | \["CCLE\_StarF2019"                    | "INTRACHROMOSOMAL\[chr16:1\.02Mb\]"\] | ENST00000567845\.5                 | 1\-709                 | ENST00000428848\.7 | 665\-2025          | INFRAME       | chr16\|\+\|\[0\]84838496\-84838735\[2\]\|\[0\]84845786\-84845904\[1\]\|\[2\]84849385\-84849517\[2\]\|\[0\]84850568\-84850683\[1\]\|\[2\]84854729\-84854829\[0\]<==>chr16\|\+\|\[1\]83486477\-83486655\[2\]\|\[0\]83602454\-83602594\[2\]\|\[0\]83670790\-83670972\[2\]\|\[0\]83678208\-83678461\[1\]\|\[2\]83748108\-83748250\[0\]\|\[1\]83779968\-83780201\[0\]\|\[1\]83783254\-83783472\[0\]\|\[1\]83795023\-83795030\[2\] | atgagctgcgtcctgggtggtgtcatccccttggggctgctgttcctggtctgcggatcccaaggctacctcctgcccaacgtcactctcttagaggagctgctcagcaaataccagcacaacgagtctcactcccgggtccgcagagccatccccagggaggacaaggaggagatcctcatgctgcacaacaagcttcggggccaggtgcagcctcaggcctccaacatggagtacatgacctgggatgacgaactggagaagtctgctgcagcgtgggccagtcagtgcatctgggagcacgggcccaccagtctgctggtgtccatcgggcagaacctgggcgctcactggggcaggtatcgctctccggggttccatgtgcagtcctggtatgacgaggtgaaggactacacctacccctacccgagcgagtgcaacccctggtgtccagagaggtgctcggggcccatgtgcacgcactacacacagatagtttgggccaccaccaacaagatcggttgtgctgtgaacacctgccggaagatgactgtctggggagaagtttgggagaacgcggtctactttgtctgcaattattctccaaaggggaactggattggagaagccccctacaagaatggccggccctgctctgagtgcccacccagctatggaggcagctgcaggaacaacttgtgttaccgagGCACCACAGTGATGCGGATGACAGCCTTTGATGCAGATGACCCAGCCACCGATAATGCCCTCCTGCGGTATAATATCCGTCAGCAGACGCCTGACAAGCCATCTCCCAACATGTTCTACATCGATCCTGAGAAAGGAGACATTGTCACTGTTGTGTCACCTGCGCTGCTGGACCGAGAGACTCTGGAAAATCCCAAGTATGAACTGATCATCGAGGCTCAAGATATGGCTGGACTGGATGTTGGATTAACAGGCACGGCCACAGCCACGATCATGATCGATGACAAAAATGATCACTCACCAAAATTCACCAAGAAAGAGTTTCAAGCCACAGTCGAGGAAGGAGCTGTGGGAGTTATTGTCAATTTGACAGTTGAAGATAAGGATGACCCCACCACAGGTGCATGGAGGGCTGCCTACACCATCATCAACGGAAACCCCGGGCAGAGCTTTGAAATCCACACCAACCCTCAAACCAACGAAGGGATGCTTTCTGTTGTCAAACCATTGGACTATGAAATTTCTGCCTTCCACACCCTGCTGATCAAAGTGGAAAATGAAGACCCACTCGTACCCGACGTCTCCTACGGCCCCAGCTCCACAGCCACCGTCCACATCACTGTCCTGGATGTCAACGAGGGCCCAGTCTTCTACCCAGACCCCATGATGGTGACCAGGCAGGAGGACCTCTCTGTGGGCAGCGTGCTGCTGACAGTGAATGCCACGGACCCCGACTCCCTGCAGCATCAAACCATCAGGTATTCTGTTTACAAGGACCCAGCAGGTTGGCTGAATATTAACCCCATCAATGGGACTGTTGACACCACAGCTGTGCTGGACCGTGAGTCCCCATTTGTCGACAACAGCGTGTACACTGCTCTCTTCCTGGCAATTGACAGTGGCAACCCTCCCGCTACGGGCACTGGGACTTTGCTGATAACCCTGGAGGACGTGAATGACAATGCCCCGTTCATTTACCCCACAGTAGCTGAAGTCTGTGATGATGCCAAAAACCTCAGTGTAGTCATTTTGGGAGCATCAGATAAGGATCTTCACCCGAATACAGATCCTTTCAAATTTGAAATCCACAAACAAGCTGTTCCTGATAAAGTCTGGAAGATCTCCAAGATCAACAATACACACGCCCTGGTAAGCCTTCTTCAAAATCTGAACAAAGCAAACTACAACCTGCCCATCATGGTGACAGATTCAGGGAAACCACCCATGACGAATATCACAGATCTCAGGGTACAAGTGTGCTCCTGCAGGAATTCCAAAGTGGACTGCAACGCGGCAGGGGCCCTGCGCTTCAGCCTGCCCTCAGTCCTGCTCCTCAGCCTCTTCAGCTTAGCTTGTCTGTGA | MSCVLGGVIPLGLLFLVCGSQGYLLPNVTLLEELLSKYQHNESHSRVRRAIPREDKEEILMLHNKLRGQVQPQASNMEYMTWDDELEKSAAAWASQCIWEHGPTSLLVSIGQNLGAHWGRYRSPGFHVQSWYDEVKDYTYPYPSECNPWCPERCSGPMCTHYTQIVWATTNKIGCAVNTCRKMTVWGEVWENAVYFVCNYSPKGNWIGEAPYKNGRPCSECPPSYGGSCRNNLCYRGTTVMRMTAFDADDPATDNALLRYNIRQQTPDKPSPNMFYIDPEKGDIVTVVSPALLDRETLENPKYELIIEAQDMAGLDVGLTGTATATIMIDDKNDHSPKFTKKEFQATVEEGAVGVIVNLTVEDKDDPTTGAWRAAYTIINGNPGQSFEIHTNPQTNEGMLSVVKPLDYEISAFHTLLIKVENEDPLVPDVSYGPSSTATVHITVLDVNEGPVFYPDPMMVTRQEDLSVGSVLLTVNATDPDSLQHQTIRYSVYKDPAGWLNINPINGTVDTTAVLDRESPFVDNSVYTALFLAIDSGNPPATGTGTLLITLEDVNDNAPFIYPTVAEVCDDAKNLSVVILGASDKDLHPNTDPFKFEIHKQAVPDKVWKISKINNTHALVSLLQNLNKANYNLPIMVTDSGKPPMTNITDLRVQVCSCRNSKVDCNAAGALRFSLPSVLLLSLFSLACL\* | CAP\|62\-200\|1\.6e\-26^LCCL\|150\-164\|1^DUF5607\|170\-196\|1\.8e\-05^LCCL\|287\-378\|2\.2e\-28^Rxt3\|315\-340\|0\.0053^LCCL\|388\-483\|2\.6e\-31^Rxt3\|425\-466\|2\.8e\-07 | \.                                    |      |      |
| PFKFB3\-\-LINC02649 | 25                | 11                | 25     | 11     | ONLY\_REF\_SPLICE | PFKFB3^ENSG00000170525\.21    | chr10:6226365:\+  | LINC02649^ENSG00000215244\.3 | chr10:6326546:\+  | YES\_LDAS          | 0\.3684 | GT             | 1\.9329          | AG              | 1\.9329           | \["INTRACHROMOSOMAL\[chr10:0\.02Mb\]"  | "NEIGHBORS\[16525\]"\]                | \.                                 | \.                     | \.                 | \.                 | \.            | \.                                                           | \.                                                           | \.                                                           | \.                                                           | \.                                    |      |      |
| SEPTIN7P14\-\-PSPH  | 24                | 0                 | 12     | 0      | ONLY\_REF\_SPLICE | SEPTIN7P14^ENSG00000245958\.6 | chr4:119455133:\+ | PSPH^ENSG00000146733\.14     | chr7:56021231:\-  | YES\_LDAS          | 0\.1228 | GT             | 1\.8892          | AG              | 1\.9656           | \["INTERCHROMOSOMAL\[chr4\-\-chr7\]"\] | \.                                    | \.                                 | \.                     | \.                 | \.                 | \.            | \.                                                           | \.                                                           | \.                                                           | \.                                                           |                                       |      |      |
| NRIP1\-\-LINC02246  | 13                | 5                 | 13     | 2\.32  | ONLY\_REF\_SPLICE | NRIP1^ENSG00000180530\.11     | chr21:15064745:\- | LINC02246^ENSG00000281903\.2 | chr21:14857708:\- | YES\_LDAS          | 0\.1567 | GT             | 1\.2729          | AG              | 1\.5546           | \["GTEx\_recurrent\_StarF2019"         | "INTRACHROMOSOMAL\[chr21:0\.04Mb\]"   | "NEIGHBORS\[42683\]"\]             | \.                     | \.                 | \.                 | \.            | \.                                                           | \.                                                           | \.                                                           | \.                                                           | \.                                    | \.   |      |
|  LINC02643\-\-NEBL  | 8                 | 5                 | 8      | 2\.5   | ONLY\_REF\_SPLICE | LINC02643^ENSG00000230109\.1  | chr10:21369967:\- | NEBL^ENSG00000078114\.19     | chr10:21020201:\- | YES\_LDAS          | 0\.1075 | GT             | 1\.9656          | AG              | 1\.7056           | \["INTRACHROMOSOMAL\[chr10:0\.05Mb\]"  | "NEIGHBORS\[47222\]"\]                | \.                                 | \.                     | \.                 | \.                 | \.            | \.                                                           | \.                                                           | \.                                                           | \.                                                           | \.                                    |      |      |
|   TVP23C\-\-CDRT4   | 6                 | 9                 | 5\.88  | 5\.87  | ONLY\_REF\_SPLICE | TVP23C^ENSG00000175106\.17    | chr17:15540433:\- | CDRT4^ENSG00000239704\.11    | chr17:15440285:\- | YES\_LDAS          | 0\.1203 | GT             | 1\.8323          | AG              | 1\.9899           | \["INTRACHROMOSOMAL\[chr17:0\.03Mb\]"  | "NEIGHBORS\[26536\]"\]                | \.                                 | \.                     | \.                 | \.                 | \.            | \.                                                           | \.                                                           | \.                                                           | \.                                                           | \.                                    |      |      |
| NRIP1\-\-LINC02246  | 6                 | 10                | 6      | 5\.79  | ONLY\_REF\_SPLICE | NRIP1^ENSG00000180530\.11     | chr21:15014344:\- | LINC02246^ENSG00000281903\.2 | chr21:14857708:\- | YES\_LDAS          | 0\.1206 | GT             | 1\.8892          | AG              | 1\.5546           | \["GTEx\_recurrent\_StarF2019"         | "INTRACHROMOSOMAL\[chr21:0\.04Mb\]"   | "NEIGHBORS\[42683\]"\]             | \.                     | \.                 | \.                 | \.            | \.                                                           | \.                                                           | \.                                                           | \.                                                           | \.                                    | \.   |      |

主要关注的有 FUSION_MODEL 和 PFAM 列，FUSION MODEL 是基于断点和在多个 isoform 中选择的编码框内的编码结构重构的融合转录本的基因结构，格式为：`chr|strand|[exon_start_codon_phase]|lend-rend[exon_end_codon_phase] ...<==> 另一侧的基因结构（相同的格式）`；PFAM 表示推定的融合蛋白结构域的位置（基于原始蛋白结构域的注释），`~` 符号表示断点破坏了原始蛋白结构域注释。

## 使用 Trinity 重构融合转录本

上面得出的 FUSION_MODEL, FUSION_CDS, FUSION_TRANSL 是基于参考注释和参考基因组序列，如果想要基于实际支持融合事件的 RNA-seq 的 Read 进行从头的（de novo）融合转录本重构，从而获取变异信息或者新的序列特征，就要加上 `--denovo_reconstruct` 参数，通过使用 Trinity 组装比对到 FusionInspector 构建的 fusion contig 上的 reads 来进行转录本重构：

```shell
STAR-Fusion --genome_lib_dir /home/data/t040201/data/STAR_data/GRCh38_gencode_v37_CTAT_lib_Mar012021.plug-n-play/ctat_genome_lib_build_dir --left_fq HEC108_1.fastq.gz --right_fq HEC108_2.fastq.gz  --output_dir /home/data/t040201/cell_lines/fusion --FusionInspector inspect --examine_coding_effect --denovo_reconstruct
```

运行之后就会多出这些文件：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220126094937488.png)

其中 `fusions.fasta ` 是 de novo 装配的转录组序列，`gff3` 是 gff3 格式的转录本比对，其他的文件可以用来进行 IGV 可视化。

该细胞系重构的融合转录本如下：

```shell
>TRINITY_GG_1_c0_g2_i2 PHACTR4--RCC1:2298-34901
CCGGCCCCTGCTGGGGACTACAAGTCCCGTAAGCCTCCGCGGCGGCACGTCCTACCCTACACTGTCCAGCCGGCTCCCTTTTTCCCCCTCCCCGGGGGCCAAGGGCTCCGGCTGCTGCCTGGCGGCCAACGGGCCAGGTAGGATTTCCGGGAGAGGCTGCTGTGGAGGCTGAGGAGGCGGCGGCGGAGATCTGGAAACAGTATCTCACCTCCCTAAACTGGTTAATAGTGGCATGGAAGATCCATTTGGACAGGAAGATGTCACCCAAGCGCATAGCTAAAAGAAGGTCCCCCCCAGCAGATGCCATCCCCAAAAGCAAGAAGGTGAAGGTCTCACACAGGTCCCACAGCACAGAACCCGGCTTGGTGCTGACACTAGGCCAGGGCGACGTGGGCCAGCTGGGGCTGGGTGAGAATGTGATGGAGAGGAAGAAGCCGGCCCTGGTATCCATTCCGGAGGATGTTGTGCAGGCTGAGGCTGGGGGCATGCACACCGTGTGTCTAAGCAAAAGTGGCCAGGTCTATTCCTTCGGCTGCAATGATGAGGGTGCCCTGGGAAGGGACACATCAGTGGAGGGCTCGGAGATGGTCCCTGGGAAAGTGGAGCTGCAAGAGAAGGTGGTACAGGTGTCAGCAGGAGACAGTCACACAGCAGCCCTCACCGATGATGGCCGTGTCTTCCTCTGGGGCTCCTTCCGGGACAATAACGGTGTGATTTGACTGTTGGAGCCCATGAAGAAGAGCATGGTGCCTG
>TRINITY_GG_1_c0_g2_i4 PHACTR4--RCC1:2298-34901
CCGGCCCCTGCTGGGGACTACAAGTCCCGTAAGCCTCCGCGGCGGCACGTCCTACCCTACACTGTCCAGCCGGCTCCCTTTTTCCCCCTCCCCGGGGGCCAAGGGCTCCGGCTGCTGCCTGGCGGCCAACGGGCCAGGTAGGATTTCCGGGAGAGGCTGCTGTGGAGGCTGAGGAGGCGGCGGCGGAGATCTGGAAACAGTATCTCACCTCCCTAAACTGGTTAATAGTGGCATGGAAGATCCATTTGGACAGGAAGATGTCACCCAAGCGCATAGCTAAAAGAAGGTCCCCCCCAGCAGATGCCATCCCCAAAAGCAAGAAGGTGAAGGACACGAGGGCCGCTGCCTCCCGCCGCGTTCCTGGCGCCCGCTCCTGCCAAGTCTCACACAGGTCCCACAGCACAGAACCCGGCTTGGTGCTGACACTAGGCCAGGGCGACGTGGGCCAGCTGGGGCTGGGTGAGAATGTGATGGAGAGGAAGAAGCCGGCCCTGGTATCCATTCCGGAGGATGTTGTGCAGGCTGAGGCTGGGGGCATGCACACCGTGTGTCTAAGCAAAAGTGGCCAGGTCTATTCCTTCGGCTGCAATGATGAGGGTGCCCTGGGAAGGGACACATCAGTGGAGGGCTCGGAGATGGTCCCTGGGAAAGTGGAGCTGCAAGAGAAGGTGGTACAGGTGTCAGCAGGAGACAGTCACACAGCAGCCCTCACCGATGATGGCCGTGTCTTCCTCTGGGGCTCCTTCCGGGACAATAACGGTGTGATTTGACTGTTGGAGCCCATGAAGAAGAGCATGGTGCCTG
>TRINITY_GG_2_c0_g2_i1 SEPTIN7P14--PSPH:1343-30726
CGCAGAAGCTTCTCAATGGCCAGCGCCAGCTGCAGCCCCGGCGGCGCACTCGCCTCACCTGAGCCTGGGAGGAAAATTCTTCCAAGGATGATCTCCCACTCAGAGCTGAGGAAGCTTTTCTACTCAGCAGATGCTGTGTGTTTTGATGTTGACAGCACGGTCATCAGTGAAGAAGGAATCGGAC
>TRINITY_GG_3_c0_g1_i1 SHISA9--U91319.1:9996-25888
CTTGCGGATGTCATGAGACCACAGGGCCACTGCAACACTGATCACATGGAGAGAGACCTAAACATCGTTGTCCACGTCCAGCATTATGAGAACATGGACACGAGAACCCCCATAAATAATCTTCATGCCACCCAGATGAACAACGCAGTGCCCACCTCTCCTCTGCTCCAGCAGATGGGCCATCCACATTCGTACCCGAACCTGGGCCAGATCTCCAACCCCTATGAACAGCAGCCACCAGGAAAAGAGCTCAACAAGTACGCCTCCTTAAAGGCAGTCGAGCTGGAACACCCTTCTTCTCCTGCCTTTGGACATCAGAACTTCAGATTCTCTGGCCTTCAGACTTCAAGACTTGCACTAGTGGCCCCCTGGGTTCTCAAGGTTTTGGCCGCCTCGGTTGAGAGTTACACCATCGGCTTCTTTGGTTCTGAGGCCGTTGGAGTTGGACTGAGCCATGCTACCAGCTTCCCTGGGTCTCCAGCCTGCAGATTGCCTACTGTGAGATTTAGCCTCCATAATCACGTAACAGAATATTGGCATGTAGCACTCCTCAAAACATAGAAAGCAGAAACAAATCAATTCTGCCTGGAGG
>TRINITY_GG_4_c0_g1_i1 PFKFB3--LINC02649:16805-31340
TGAACGTGGAGTCCGTCTGCACACACCGAGAGAGGTCAGAGGATGCAAAGAAGGGACCTAACCCGCTCATGAGACGCAATAGTGTCACCCCGCTAGCCAGCCCCGAACCCACCAAAAAGCCTCGCATCAACAGCTTTGAGGAGCATGTGGCCTCCACCTCGGCCGCCCTGCCCAGCTGCCTGCCCCCGGAGGTGCCCACGCAGCTGCCTGGACAAATATGGAGTTACCAGTAAGGAGCTCCACCGTGACTCTCCTCCCTGCTCCGTTGCCCCGACGAGGAAGTGTGAAAACGTTTCTGGCTCCATCCAAGAGTTACTTCCCTGAAGAAGAGAGGGCTTTGTTGAAGTCTTCCCATGCTTTCTGCACAGGGCTCTGGCCTTGGAGAAGGGATTTCCAGTTACCGCAGTGTCACTTGGCCCTGGGTCTCCTCCCGGGAGAGAGAAGTGTACGGCTCCCAAGGTTCCTGGCAGTTTTGAAAGAGCTCTCAGCCACAGCCAGCTTTACTTTGATCACG
>TRINITY_GG_5_c0_g1_i1 CRISPLD2--CDH13:11156-50276
TGAAGGACTACACCTACCCCTACCCGAGCGAGTGCAACCCCTGGTGTCCAGAGAGGTGCTCGGGGCCCATGTGCACGCACTACACACAGATAGTTTGGGCCACCACCAACAAGATCGGTTGTGCTGTGAACACCTGCCGGAAGATGACTGTCTGGGGAGAAGTTTGGGAGAACGCGGTCTACTTTGTCTGCAATTATTCTCCAAAGGGGAACTGGATTGGAGAAGCCCCCTACAAGAATGGCCGGCCCTGCTCTGAGTGCCCACCCAGCTATGGAGGCAGCTGCAGGAACAACTTGTGTTACCGAGGCACCACAGTGATGCGGATGACAGCCTTTGATGCAGATGACCCAGCCACCGATAATGCCCTCCTGCGGTATAATATCCGTCAGCAGACGCCTGACAAGCCATCTCCCAACATGTTCTACATCGATCCTGAGAAAGGAGACATTGTCACTGTTGTGTCACCTGCGCTGCTGGACCGAGAGACTCTGGAAAATCCCAAGTATGAACTGATCATCGAGGCTCAAGATATGGCTGGACTGGATGTTGGATTAACAGGCACGGCCACGG
>TRINITY_GG_6_c0_g1_i2 NRIP1--LINC02246:2192-24982
CTGGCTCCCTCTTTGCCTTCCACCATGACTGTAAGCTTCCTGAGGCCTCACCACAAGCCAAACAGATGCACGTGCCATGCTTGCACAACCTGCTCTCAGCTGGGCTCACTCATGCATCTGCTATCAGCTGGCTGGTTAACTGTAGTTAGTTTATCTTGATGGCATCATTGGGGAAACTCAGCTCTCTTTCACTGGACTTCTCTTATATTTCTCCAGCAAACTGGAAAGGGTGTGTTCTCGTGGCAGGGGCAGGAGTCCCAGGCCGCCGCGGCTCCCAGCCTCCGGCTCCGTCAGGCTCGGTCCGCGAAGGCGCCTGCCGCCCCGTCCTGGCCCGGCGCCCCGGCGAGCTCTTCCCTCCGACCAGCGGCGCTCAC
>TRINITY_GG_6_c0_g1_i3 NRIP1--LINC02246:4395-24982
CTGGCTCCCTCTTTGCCTTCCACCATGACTGTAAGCTTCCTGAGGCCTCACCACAAGCCAAACAGATGCACGTGCCATGCTTGCACAACCTGCTCTCAGCTGGGCTCACTCATGCATCTGCTATCAGCTGGCTGGTTAACTGTAGTTAGTTTATCTTGATGGCATCATTGGGGAAACTCAGCTCTCTTTCACTGGACTTCTCTTATATTTCTCCAGCAAACTGGAAAGGGTGTGTTCTCGTGGCAGGGGCAGGAGTCGTCTGTCTCCAAGCTCTGAGCCTCTGCTTTCTGAGAAAGAAAATTGAGAAGGCTGTTGAAAAGTAGCTCTGATGTCATCCGGAGTCTTCAGATTCCCTGTCCTCCTTCAGTCAAGTGTGCATCCCAGGCCGCCGCGG
>TRINITY_GG_7_c0_g1_i1 TVP23C--CDRT4:7982-22624
GACTAATGGTTGGCCTACGTTGGTGGAATCACATTGATGAAGATGGAAAGAGCCATTGGGTGTTTGAATCTAGAAAGGAGTCCTCTCAAGAGAATAAAACTGTGTCAGAGGCTGAATCAAGAATCTTTTGGTTGGGACTTATTGCCTGTTCAGTACTGTGGGTGATATTTGCCTTTAGTGCACTCTTCTCCTTCACAGTAAAGTGGCTGGCGGTGGTTATTATGGGTGTGGTGCTACAAGGTGCCAACCTGTATGGTTACATCAGGTGTAAGGTGCGCAGCAGAAAGCATTTAACCAGCATGGCTACTTCATATTTTGGAAAGCAGTTTTTAAGACAAGAATCTGAACCTGTGATGTTAAGAAATCAGTAAATATTAAAAAGAAGATGGATGCAAGAAGGATGAAGAAAGAAGAAGGACTCACAGAAAACACTGGACTTCCCCGGAAGCTACTTGAAAAACATGACCCCTGGCCGGCCTATGTCACCTATACCTCTCAGACAGTGAAAAGACTCATTGAGAAAAGCAAAACTAGAGAACTGGAATGCATGCGTGCCCTCGAGGAAAGACCCTGGGCATCAAGGCAGAATAAACCTTCCAGCGTC
>TRINITY_GG_8_c0_g2_i1 LEPROT--LEPR:2949-14955
CGCGGGGCGACTCCCGGTCTGGCTTGGGCAGGCTGCCCGGGCCGTGGCAGGAAGCCGGAAGCAGCCGCGGCCCCAGTTCGGGAGACATGGCGGGCGTTAAAGCTCTCGTGGCATTATCCTTCAGTGGGGCTATTGGACTGACTTTTCTTATGCTGGGATGTGCCTTAGAGGATTATGGGTGTACTTCTCTGAAGTAAGATGATTTGTCAAAAATTCTGTGTGGTTTTGTTACATTGGGAATTTATTTATGTGATAACTGCGTTTAACTTGTCATATCCAATTACTCCTTGGAGATTTAAGTTGTCTTGCATGCCACCAAATTCAACCTATGACTACTTCCTTTTGCCTGCTGGACTCTCAAAGAATACTTCAAATTCGAATGGACATTATGAGACAGCTGTTGAACCTAAGTTTAATTCAAG
>TRINITY_GG_9_c0_g1_i1 LINC02643--NEBL:2205-23713
AAACAATCAGGAAAGCTCAGCAGCATAAAACTATGAGAAAACTGGGTCTGAGAATCCAGATGTCCTGAGTGAGTAAAGGGAAAATTTTCACTTCTTCCTATGGAAGCCTGTCACCCCCCAACTCCATCCTAGGCTGAGGAAGTCTGCGTTCTTCTCTCACGGAGCTGGAGAACCTTTCAAGACACTACCCGAAGCAGTCCTTCACCACGGTGGCAGATACACCTGAAAATCTTCGCCTGAAGCAGCAAAGTGAATTGCAGAGTCAGGTCAAGTACAAAAGAGATTTTGAAGAAAGCAAAGGGAGGGGCTTCAGCATCGTCACGGACACTCCTGAGCTACAGAGACTGAAGAGGACTCAGGAGCAAATCAGTAATGTAGGTGCCTGTTTATTCAATAGCATGATGGCTCTGATGTGCATTTCTGGATGAGAAAGGGAGGCTGGATGACTT
```

## 总结

如果想要一步完成融合转录本的检测，融合事件的效应，以及重构融合转录本，就可以直接运行：

```shell
STAR-Fusion --genome_lib_dir /home/data/t040201/data/STAR_data/GRCh38_gencode_v37_CTAT_lib_Mar012021.plug-n-play/ctat_genome_lib_build_dir --left_fq HEC108_1.fastq.gz --right_fq HEC108_2.fastq.gz  --output_dir /home/data/t040201/cell_lines/fusion --FusionInspector inspect --examine_coding_effect --denovo_reconstruct
```

主要查看的文件包括 `star-fusion.fusion_predictions.abridged.tsv` （融合事件），`star-fusion.fusion_predictions.abridged.coding_effect.tsv` (融合事件对编码区域的影响) , FusionInspector-inspect 目录下的 `finspector.FusionInspector.fusions.abridged.tsv` （校验后的融合以及 FAR 值）和 `finspector.gmap_trinity_GG.fusions.fasta` （重构的融合转录本序列）文件
