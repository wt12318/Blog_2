---
title: 使用Snakemake搭建流程 
author: wutao
date: 2021-03-19 10:00:00
slug: snakemake
categories:
  - python
  - skills
tags:
  - bioinformatics
index_img: img/snakemake.png
---







学习`snakemake` 官方文档

<!-- more -->

创建一个`snakemake_tutorial`目录,并下载示例数据：

```bash
wget https://github.com/snakemake/snakemake-tutorial-data/archive/v5.24.1.tar.gz

tar --wildcards -xf snakemake-tutorial-data-5.24.1.tar.gz --strip 1 "*/data" "*/environment.yaml"
##--wildcards 根据通配符来提取压缩文件,这里是只提取data文件夹和environment.yaml文件
```

```
(snakemake-tutorial) -bash-4.2$ tree .
.
├── data
│   ├── genome.fa
│   ├── genome.fa.amb
│   ├── genome.fa.ann
│   ├── genome.fa.bwt
│   ├── genome.fa.fai
│   ├── genome.fa.pac
│   ├── genome.fa.sa
│   └── samples
│       ├── A.fastq
│       ├── B.fastq
│       └── C.fastq
├── environment.yaml
└── snakemake-tutorial-data-5.24.1.tar.gz

2 directories, 12 files
```

其中`environment.yaml`文件是用来创建所需的`conda`环境:

```bash
cat environment.yaml

channels:
  - bioconda
  - conda-forge
dependencies:
  - snakemake-minimal >=5.24.1
  - jinja2 =2.11
  - networkx =2.5
  - matplotlib =3.3
  - graphviz =2.42
  - bcftools =1.9
  - samtools =1.9
  - bwa =0.7
  - pysam =0.15
```

使用该配置文件创建`snakemake-tutorial`的环境(使用mamba代替conda来加速下载)：

```bash
mamba env create --name snakemake-tutorial --file environment.yaml

conda activate snakemake-tutorial
```

## 基础：以一个生物信息学流程为例

一个`Snakemake`流程由`Snakefile`文件中的一系列规则(rules)来创建；这些规则通过说明**如何从输入文件得到输出文件**来将流程分解成多个小的步骤,`Snakemake`会通过匹配文件名来自动的决定规则间的依赖关系

接下来以一个生物信息学的流程为例来学习`Snakemake`流程的搭建

这个流程做的工作为：将测序的reads匹配到参考基因组上,并且检测匹配上的reads的变异

### 第一步：Mapping reads

第一个`Snakemake`规则将给定样本的测序reads回帖到给定的参考基因组上去,使用的工具为[bwa的mem算法](http://bio-bwa.sourceforge.net/)

创建一个`Snakefile`文件,写上下面的规则：

```python
rule bwa_map:
    input:
        "data/genome.fa",
        "data/samples/A.fastq"
    output:
        "mapped_reads/A.bam"
    shell:
        "bwa mem {input} | samtools view -Sb - > {output}"
```

一个`Snakemake`规则有一个名字,这里是`bwa_map`;还有一些指令,上面的例子里是`input`, `output`和`shell`;`input`和`output`指令中是一系列的文件名(python 字符串),指定了输入和输出文件(如果有多个文件,用逗号分割);`shell`指令也是一个字符串,表示需要执行的shell命令,在shell命令字符串中可以使用花括号来指代规则中的其他部分,比如这里使用`{input}`来指代`input`指令中的内容,使用`{output}`指代`output`指令中的内容;上面的`input`里面有两个字符串,这时`snakemake`替代`{input}`时会用空格分隔开两个输入文件 

接下来可以执行这个流程：

```bash
snakemake --cores 1 

Building DAG of jobs...
Using shell: /usr/bin/bash
Provided cores: 1 (use --cores to define parallelism)
Rules claiming more threads will be scaled down.
Job counts:
        count   jobs
        1       bwa_map
        1
Select jobs to execute...

[Sat Mar 20 18:17:43 2021]
rule bwa_map:
    input: data/genome.fa, data/samples/A.fastq
    output: mapped_reads/A.bam
    jobid: 0

[M::bwa_idx_load_from_disk] read 0 ALT contigs
[M::process] read 25000 sequences (2525000 bp)...
[M::mem_process_seqs] Processed 25000 reads in 1.267 CPU sec, 1.267 real sec
[main] Version: 0.7.17-r1188
[main] CMD: bwa mem data/genome.fa data/samples/A.fastq
[main] Real time: 1.757 sec; CPU: 1.318 sec
[Sat Mar 20 18:17:44 2021]
Finished job 0.
1 of 1 steps (100%) done
Complete log: /slst/home/wutao2/snakemake_tutorial/.snakemake/log/2021-03-20T181738.569944.snakemake.log

tree .
.
├── data
│   ├── genome.fa
│   ├── genome.fa.amb
│   ├── genome.fa.ann
│   ├── genome.fa.bwt
│   ├── genome.fa.fai
│   ├── genome.fa.pac
│   ├── genome.fa.sa
│   └── samples
│       ├── A.fastq
│       ├── B.fastq
│       └── C.fastq
├── environment.yaml
├── mapped_reads
│   └── A.bam
├── Snakefile
└── snakemake-tutorial-data-5.24.1.tar.gz
```

也可以使用`-n`或者`--dry-run`参数使snakemake显示执行的”计划“(没有真正的执行流程);使用`-p`参数来打印需要执行的命令：

```bash
snakemake -np

Building DAG of jobs...
Job counts:
        count   jobs
        1       bwa_map
        1

[Sat Mar 20 18:25:11 2021]
rule bwa_map:
    input: data/genome.fa, data/samples/A.fastq
    output: mapped_reads/A.bam
    jobid: 0

bwa mem data/genome.fa data/samples/A.fastq | samtools view -Sb - > mapped_reads/A.bam
Job counts:
        count   jobs
        1       bwa_map
        1
This was a dry-run (flag -n). The order of jobs does not reflect the order of execution.
```

### 第二步：使规则适用的范围更广

上面的规则只能对单个样本`data/samples/A.fastq`适用,在snakemake中可以使用通配符(wildcard)来扩展规则的适用范围：

```bash
rule bwa_map:
    input:
        "data/genome.fa",
        "data/samples/{sample}.fastq"
    output:
        "mapped_reads/{sample}.bam"
    shell:
        "bwa mem {input} | samtools view -Sb - > {output}"
```

`Snakemake`会将`output`中的`{sample}`替换成一个合适的值,并且将`input`中的`{sample}`也替换成同样的值,我们在运行流程就需要指定输出文件的名称(这样snakemake才知道如何替换通配符)：

```bash
snakemake -np mapped_reads/B.bam

Building DAG of jobs...
Job counts:
        count   jobs
        1       bwa_map
        1

[Sat Mar 20 18:50:12 2021]
rule bwa_map:
    input: data/genome.fa, data/samples/B.fastq
    output: mapped_reads/B.bam
    jobid: 0
    wildcards: sample=B

bwa mem data/genome.fa data/samples/B.fastq | samtools view -Sb - > mapped_reads/B.bam
Job counts:
        count   jobs
        1       bwa_map
        1
This was a dry-run (flag -n). The order of jobs does not reflect the order of execution.
```
这个时候snakemake就将`{sample}`替换成`B`了       
也可以同时生成多个文件：

```bash
snakemake -np mapped_reads/A.bam mapped_reads/B.bam
##或snakemake -np mapped_reads/{A,B}.bam

Building DAG of jobs...
Job counts:
        count   jobs
        2       bwa_map
        2

[Sat Mar 20 18:52:00 2021]
rule bwa_map:
    input: data/genome.fa, data/samples/B.fastq
    output: mapped_reads/B.bam
    jobid: 0
    wildcards: sample=B

bwa mem data/genome.fa data/samples/B.fastq | samtools view -Sb - > mapped_reads/B.bam

[Sat Mar 20 18:52:00 2021]
rule bwa_map:
    input: data/genome.fa, data/samples/A.fastq
    output: mapped_reads/A.bam
    jobid: 1
    wildcards: sample=A

bwa mem data/genome.fa data/samples/A.fastq | samtools view -Sb - > mapped_reads/A.bam
Job counts:
        count   jobs
        2       bwa_map
        2
This was a dry-run (flag -n). The order of jobs does not reflect the order of execution.
```

### 第三步：Sorting read alignments

接下来需要使用`samtools`中的`sort`命令来对BAM文件进行排序,将下面的规则写到刚才的`bwa_map`规则的下面：

```bash
rule samtools_sort:
    input:
        "mapped_reads/{sample}.bam"
    output:
        "sorted_reads/{sample}_sorted.bam"
    shell:
        "samtools sort -T sorted_reads/{wildcards.sample} -O bam {input} > {output}"
```

这个规则的输入文件是刚才`bwa_map`规则的输出文件;这里面需要注意的是在shell命令中可以通过`wildcards`对象来获取不同通配符的值(即wildcards对象的属性)

```bash
snakemake -np sorted_reads/B_sorted.bam
Building DAG of jobs...
Job counts:
        count   jobs
        1       bwa_map
        1       samtools_sort
        2

[Sun Mar 21 16:06:07 2021]
rule bwa_map:
    input: data/genome.fa, data/samples/B.fastq
    output: mapped_reads/B.bam
    jobid: 1
    wildcards: sample=B

bwa mem data/genome.fa data/samples/B.fastq | samtools view -Sb - > mapped_reads/B.bam

[Sun Mar 21 16:06:07 2021]
rule samtools_sort:
    input: mapped_reads/B.bam
    output: sorted_reads/B_sorted.bam
    jobid: 0
    wildcards: sample=B

samtools sort -T sorted_reads/B -O bam mapped_reads/B.bam > sorted_reads/B_sorted.bam
Job counts:
        count   jobs
        1       bwa_map
        1       samtools_sort
        2
This was a dry-run (flag -n). The order of jobs does not reflect the order of execution.
```
可以看到当指定输出为`B_sorted.bam`的时候,会先运行bwa得到`B.bam`然后再运行samtools得到`B_sorted.bam`

### 第四步：Indexing read alignments

接下来我们需要使用`samtools`对排序的read alignments建立索引,将下面的规则补充到之前的规则下面:

```python
rule samtools_index:
    input:
        "sorted_reads/{sample}_sorted.bam"
    output:
        "sorted_reads/{sample}.bam.bai"
    shell:
        "samtools index {input} {output}"
```

snakemake会将不同的任务串成有向无环图(DAG),可以使用下面的命令来可视化流程：

```bash
snakemake --dag sorted_reads/{A,B}.bam.bai | dot -Tsvg > dag.svg
```
<center>

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/dag.svg)

</center>

DAG图的节点表示任务,边表示任务间的依赖关系,在节点中还会展示通配符的值(如`sample:B`);另外不需要运行的规则就用虚线边框表示：

```bash
##运行第一个规则
snakemake --cores 1 mapped_reads/A.bam

##再创建DAG
snakemake --dag sorted_reads/{A,B}.bam.bai | dot -Tsvg > dag1.svg
```
![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/dag1.svg)

### 第五步：Calling genomic variants

接下来需要检测reads中的变异信息,用的工具为：`samtools`和`bcftools`

需要注意的是`snakemake`是通过目标文件(输出)来反推输入文件的,上面几个规则中的输出和输入共用一个通配符(sample);而这个步骤的输入是所有的bam及其索引,输出是一个文件(vcf)，所以需要在input指令下将所有的输入文件都写出来(因为此时snakemake无法通过输出推断输入)

snakemake提供了一个`expand`函数,可以方便的将文件名收集起来,就不需要一个一个写了：

```python
SAMPLES = ["A", "B"]

expand("sorted_reads/{sample}.bam", sample=SAMPLES)

```
将`SAMPLE`列表中的内容取代前面的通配符,也可以提供多个通配符,得到的结果是多个通配符的乘积：

```python
expand("sorted_reads/{sample}.{replicate}.bam", sample=SAMPLES, replicate=[0, 1])

###结果是：
["sorted_reads/A.0.bam", "sorted_reads/A.1.bam", "sorted_reads/B.0.bam", "sorted_reads/B.1.bam"]
```

因此需要在`Snakefile`的最前面定义`SAMPLES`,然后将下面的规则放到之前的规则下面：

```python
SAMPLES = ["A", "B"]

...


rule bcftools_call:
    input:
        fa="data/genome.fa",
        bam=expand("sorted_reads/{sample}_sorted.bam", sample=SAMPLES),
        bai=expand("sorted_reads/{sample}.bam.bai", sample=SAMPLES)
    output:
        "calls/all.vcf"
    shell:
        "samtools mpileup -g -f {input.fa} {input.bam} | "
        "bcftools call -mv - > {output}"
```

在shell命令中可以通过名称或者位置来指定输入或输出文件：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210321171336544.png)

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210321171348704.png)

另外一个需要注意的是：如果命令太长,可以分多行写,但是在每行的命令的末尾要留一个空格:

```bash
##如果不留空格,拼起来会出错
"samtools mpileup"
"-g -f {input.fa} {input.bam}"

"samtools mpileup-g -f {input.fa} {input.bam}"
```

现在再来看一下DAG图：

```bash
snakemake --dag calls/all.vcf | dot -Tsvg > dag2.svg
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/dag2.svg)

### 第六步：Using custom scripts

在Snakemake中还可以引用外部的脚本(python/R)来完成一系列的工作;需要在`script`指令中指定脚本的路径:

```python
rule plot_quals:
    input:
        "calls/all.vcf"
    output:
        "plots/quals.svg"
    script:
        "scripts/plot-quals.py"
```

```python
##plot-quals.py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pysam import VariantFile

quals = [record.qual for record in VariantFile(snakemake.input[0])]
plt.hist(quals)

plt.savefig(snakemake.output[0])
```
上面的规则和python脚本最终产生一个变异检测质量值的直方图

规则的不同部分(input,output,wildcards等)在外部脚本中都是`snakemake`对象的属性,比如`snakemake.input`就是含有输入文件名的列表

同样在R代码中,`snakemake`是作为S4对象存在的,S4类的属性是R列表,因此我们可以通过类似`snakemake@input[[1]]`的形式来获取第一个输入文件

### 第七步：Adding a target rule

前面都是通过命令行来指定目标文件(snakemake是通过目标文件逐步反推),如果没有指定目标文件则认为第一个规则的output是目标文件,因此我们可以在第一个规则中不加output,只包含input,并且这个input是整个流程的最终输出文件,这样我们就可以无需指定输出文件了(如果输出有很多,这样就不方便)：

```python
##将下面的规则放到第一个
rule all:
    input:
        "plots/quals.svg"
```

可以看看整个流程的DAG图：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/dag3.svg)

接下来运行整个流程：

```bash
snakemake --cores 1
```
得到最终的结果为变异检测质量值的直方图：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/quals.svg)

注意：snakemake只在以下几种情况下才会执行任务(jobs)
![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210322103940316.png)

但是`Snakemake`也提供了强制运行的选项`--forcerun`,后面接输出文件或者规则(`--forceall`可以强制运行整个流程)：

```bash
snakemake --cores 1 --forcerun samtools_sort
Building DAG of jobs...
Using shell: /usr/bin/bash
Provided cores: 1 (use --cores to define parallelism)
Rules claiming more threads will be scaled down.
Job counts:
        count   jobs
        1       att
        1       bcftools_call
        1       plot_quals
        2       samtools_index
        2       samtools_sort
        7
Select jobs to execute...

[Mon Mar 22 10:29:47 2021]
rule samtools_sort:
    input: mapped_reads/A.bam
    output: sorted_reads/A_sorted.bam
    jobid: 3
    wildcards: sample=A

[Mon Mar 22 10:29:48 2021]
Finished job 3.
1 of 7 steps (14%) done
Select jobs to execute...

[Mon Mar 22 10:29:48 2021]
rule samtools_index:
    input: sorted_reads/A_sorted.bam
    output: sorted_reads/A.bam.bai
    jobid: 7
    wildcards: sample=A

[Mon Mar 22 10:29:48 2021]
Finished job 7.
2 of 7 steps (29%) done
Select jobs to execute...

[Mon Mar 22 10:29:48 2021]
rule samtools_sort:
    input: mapped_reads/B.bam
    output: sorted_reads/B_sorted.bam
    jobid: 5
    wildcards: sample=B

[Mon Mar 22 10:29:48 2021]
Finished job 5.
3 of 7 steps (43%) done
Select jobs to execute...

[Mon Mar 22 10:29:48 2021]
rule samtools_index:
    input: sorted_reads/B_sorted.bam
    output: sorted_reads/B.bam.bai
    jobid: 8
    wildcards: sample=B

[Mon Mar 22 10:29:48 2021]
Finished job 8.
4 of 7 steps (57%) done
Select jobs to execute...

[Mon Mar 22 10:29:48 2021]
rule bcftools_call:
    input: data/genome.fa, sorted_reads/A_sorted.bam, sorted_reads/B_sorted.bam, sorted_reads/A.bam.bai, sorted_reads/B.bam.bai
    output: calls/all.vcf
    jobid: 2

[warning] samtools mpileup option `g` is functional, but deprecated. Please switch to using bcftools mpileup in future.
[mpileup] 2 samples in 2 input files
Note: none of --samples-file, --ploidy or --ploidy-file given, assuming all sites are diploid
[Mon Mar 22 10:29:49 2021]
Finished job 2.
5 of 7 steps (71%) done
Select jobs to execute...

[Mon Mar 22 10:29:49 2021]
rule plot_quals:
    input: calls/all.vcf
    output: plots/quals.svg
    jobid: 1

[Mon Mar 22 10:30:00 2021]
Finished job 1.
6 of 7 steps (86%) done
Select jobs to execute...

[Mon Mar 22 10:30:00 2021]
localrule att:
    input: plots/quals.svg
    jobid: 0

[Mon Mar 22 10:30:00 2021]
Finished job 0.
7 of 7 steps (100%) done
Complete log: /slst/home/wutao2/snakemake_tutorial/.snakemake/log/2021-03-22T102940.208336.snakemake.log
```

## 进阶用法

### 第一步：指定使用的线程数

可以在规则中使用`threads`指令来指定需要的线程数(实际用到的可以小于等于指定的线程数),比如可以把bwa的线程指定为8个(如果不指定,默认是1)：

```python
rule bwa_map:
    input:
        "data/genome.fa",
        "data/samples/{sample}.fastq"
    output:
        "mapped_reads/{sample}.bam"
    threads: 8
    shell:
        "bwa mem -t {threads} {input} | samtools view -Sb - > {output}"
```

在实际运行中用到的线程数由Snakemake来控制,保证同时运行的所有任务的总线程数不超过给定的总线程数,可以通过snakemake的参数`--cores`来指定给定的线程数(也就是说使用的线程数不超过`--cores`指定的数量),如果`--cores`后面没有数字则使用所有可用的核

### 第二步：配置文件

在前面的步骤中是通过一个python列表来指定需要考虑的样本(`SAMPLES = ["A", "B"]`),但是如果想要流程能够更好的适应新的数据,我们可以使用配置文件,配置文件的格式可以是`JSON`或者`YAML`;在流程中使用`configfile`指令来指定配置文件

`snakemake`会将配置文件读入,并将其内容存到名称为`config`的**字典变量**中

现在可以将之前的`SAMPLES`移除,加上配置文件：

```python
###配置文件config.yaml
samples:
    A: data/samples/A.fastq
    B: data/samples/B.fastq
```
```python
###将下面的指令放到Snakefile的开头
configfile: "config.yaml"

###将expand函数改写
rule bcftools_call:
    input:
        fa="data/genome.fa",
        bam=expand("sorted_reads/{sample}_sorted.bam", sample=config["samples"]),##config是一个字典
        bai=expand("sorted_reads/{sample}.bam.bai", sample=config["samples"])
    output:
        "calls/all.vcf"
    shell:
        "samtools mpileup -g -f {input.fa} {input.bam} | "
        "bcftools call -mv - > {output}"
```

### 第三步：输入函数

上面通过`expand`和配置文件改写了`bcftools_call`规则,注意到配置文件中也有`fastq`文件的路径,我们可不可以也将`bwa_map`进行类似的改写呢？

首先需要了解Snakemake流程执行的步骤：

- 初始化(initialization)：在流程内定义的文件被解析,所有的规则被实例化(`expand`函数就是在此时被执行)
- DAG：通过填充通配符和根据输出文件匹配输入文件来构建任务的有向无环图
- scheduling：根据可获得的资源来执行任务

这两个规则的区别在于：`bcftools_call`不需要根据输出文件来推断输入文件(因为该步骤输入的是所有文件),而`bwa_map`规则需要根据输出来推断输入文件(比如,如果job的输出是B.bam,那么输入必须是B.fastq),因此无法在`bwa_map`中使用`expand`函数

但是我们可以使用**input function**来完成根据输出匹配输入的任务：

```python
rule bwa_map:
    input:
        "data/genome.fa",
        lambda wildcards: config["samples"][wildcards.sample]
    output:
        "mapped_reads/{sample}.bam"
    threads: 8
    shell:
        "bwa mem -t {threads} {input} | samtools view -Sb - > {output}"
```

上面所展示的`input function`是lambda函数(为了方便所以使用lambda,一般的函数都可以的),参数是`wildcards`对象,通过`sample`属性获取output中匹配的通配符(比如output中是A.bam,wildcards.sample得到的就是"A",`config["samples"][wildcards.sample]`得到的就是配置文件中A的路径)

### 第四步：规则参数

有时候shell命令中的参数并不是静态的,比如要根据输入的样本名调整某些参数的值;snakemake提供了`params`指令













