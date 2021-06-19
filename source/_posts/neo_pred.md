---
title: 使用NeoPredPipe预测新抗原
author: wutao
date: 2021-03-25 17:03:30
slug: NeoPredPipe
categories:
  - skills
tags:
  - bioinformatics
index_img: img/neo.png
---



使用NeoPredPipe预测新抗原

<!-- more -->

[NeoPredPipe](https://github.com/MathOnco/NeoPredPipe)是一个可以从单区域或多区域测序得到的VCF文件来预测新抗原的流程工具,使用的注释软件为`ANNOVAR`;另外该软件在预测新抗原之后还有一个筛选步骤,这个筛选步骤是依据[2017 Nature](https://www.nature.com/articles/nature24473)提出的`Neoantigen recognition potential`来进行的

## 安装ANNOVAR

在[官网](https://annovar.openbioinformatics.org/en/latest/user-guide/download/) 下载ANNOVAR软件(需要填申请表,下载链接会发到邮箱), ANNOVAR是用perl写的,所以在安装之前需要先下载安装perl

下载解压后将路径添加到PATH中,向`~/.bashrc`中添加：

```bash
export PATH="/public/slst/home/wutao2/software/annovar:$PATH"
```

使用`source ~/.bashrc `激活PATH

下载注释所需的参考文件,这里下载的是`hg38的refgene`：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210325085447536.png)

```bash
perl annotate_variation.pl --downdb --webfrom annovar --buildver hg38 refGene ~/software/annovar/humandb/
# -downdb表明该命令的用途是下载数据库
# -buildver指定基因组版本
# -webform annovar 从annovar提供的镜像下载
# refGene代表的是下载的数据库的名字
#~/software/annovar/humandb/表示数据库存储的路径
```

下载的文件为：`hg38_refGeneMrna.fa`,`hg38_refGene.txt`和`hg38_refGeneVersion.txt`

## 下载PeptideMatch和参考多肽序列

 `PeptideMatch`可以用来将得到的新抗原肽与参考肽序列进行比对,进而检查预测的`neoantigen`是不是"新"的

`PeptideMatch`的下载地址在[here](https://research.bioinformatics.udel.edu/peptidematch/commandlinetool.jsp)(需要安装`java`)

另外还需要下载`fasta `格式的参考蛋白序列,数据在[here](https://www.ebi.ac.uk/reference_proteomes/),可以看到人的数据为[UP000005640 9606 HUMAN Homo sapiens](ftp://ftp.ebi.ac.uk/pub/databases/reference_proteomes/QfO/Eukaryota/UP000005640_9606.fasta.gz) 从FTP下载得到`UP000005640_9606.fasta`文件,然后利用`PeptideMatch`处理得到`index`:

```shell
java -jar ~/software/PeptideMatchCMD_1.0.jar -a index -d UP000005640_9606.fasta -i UP000005640_9606_index

Command line options: -a index -d UP000005640_9606.fasta -i UP000005640_9606_index 
Indexing to directory "/slst/home/wutao2/protein_database/UP000005640_9606_index" ...
Indexing "UP000005640_9606.fasta" ...
Indexing "UP000005640_9606.fasta" finished
Time used: 00 hours, 00 mins, 24.869 seconds
```

## 安装和配置NeoPredPipe

该工具使用的是`python2.7`,所以需要先创建一个`python2`的`conda`环境：

```bash
mamba create -n python27 python=2.7.13

###安装依赖
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple##设置镜像
python -m pip install biopython==1.70
```

然后下载安装该软件

首先克隆`github`仓库：

```bash
git clone https://github.com/MathOnco/NeoPredPipe.git
```

然后需要配置`usr_path.ini`文件,将原始的文件内容进行修改为：

```bash
[annovar]
convert2annovar = /public/slst/home/wutao2/software/annovar/convert2annovar.pl
annotatevariation = /public/slst/home/wutao2/software/annovar/annotate_variation.pl
coding_change =/public/slst/home/wutao2/software/annovar/coding_change.pl
gene_table = /public/slst/home/wutao2/software/annovar/humandb/hg38_refGene.txt
gene_fasta =/public/slst/home/wutao2/software/annovar/humandb/hg38_refGeneMrna.fa
humandb =/public/slst/home/wutao2/software/annovar/humandb/
[netMHCpan]
netMHCpan = /public/slst/home/wutao2/software/netMHCpan-4.1/netMHCpan
[PeptideMatch]
peptidematch_jar = /public/slst/home/wutao2/software/PeptideMatchCMD_1.0.jar
reference_index = /public/slst/home/wutao2/protein_database/UP000005640_9606_index/
[blast]
blastp =/public/slst/home/wutao2/software/ncbi-blast-2.11.0+/bin/blastp
```

测试安装是否成功:

```bash
python ./NeoPredPipe.py --help
usage: NeoPredPipe.py [-h] [-E EPITOPES [EPITOPES ...]] [-l] [-d] [-r] [-p]
                      [--manualproc] [--EL] [-I VCFDIR] [-H HLAFILE]
                      [-o OUTPUTDIR] [-n OUTNAME] [-pp]
                      [-c COLREGIONS [COLREGIONS ...]] [-a] [-m]
                      [-x EXPRESSION] [--expmulti] [-t]

optional arguments:
  -h, --help            show this help message and exit
  -E EPITOPES [EPITOPES ...], --epitopes EPITOPES [EPITOPES ...]
                        Epitope lengths for predictions. Default: 8 9 10
  -l                    Specifies whether to delete the ANNOVAR log file.
                        Default: True. Note: Use for debugging.
  -d                    Specifies whether to delete intermediate files created
                        by program. Default: True. Note: Set flag to resume
                        job.
  -r, --cleanrun        Specify this alone with no other options to clean-up a
                        run. Be careful that you mean to do this!!
  -p, --preponly        Prep files only without running neoantigen
                        predictions. The prediction step takes the most time.
  --manualproc          Process vcf files into annovar-input format manually,
                        to avoid issues from non 'genotype-calling' formats.
  --EL                  Flag to perform netMHCpan predictions with Eluted
                        Ligand option (without the -BA flag). Please note that
                        the output will NOT be compatible with downstream
                        Recognition Potential analysis. Default=False (BA
                        predictions)

Required arguments:
  -I VCFDIR             Input vcf file directory location. Example: -I
                        ./Example/input_vcfs/
  -H HLAFILE            HLA file for vcf patient samples OR directory with
                        patient-specific directories from running POLYSOLVER
                        (see Readme).
  -o OUTPUTDIR          Output Directory Path
  -n OUTNAME            Name of the output file for neoantigen predictions

Post Processing Options:
  -pp                   Flag to perform post processing. Default=True.
  -c COLREGIONS [COLREGIONS ...]
                        Columns of regions within vcf that are not normal
                        within a multiregion vcf file after the format field.
                        Example: 0 is normal in test samples, tumor are the
                        other columns. Program can handle different number of
                        regions per vcf file.
  -a                    Flag to not filter neoantigen predictions and keep all
                        regardless of prediction value.
  -m                    Specifies whether to perform check if predicted
                        epitopes match any normal peptide. If set to True,
                        output is added as a column to neoantigens file.
                        Requires PeptideMatch specified in usr_paths.ini.
                        Default=False
  -x EXPRESSION, --expression EXPRESSION
                        RNAseq expression quantification file(s), if
                        specified, expression information is added to output
                        tables.
  --expmulti            Flag to specify if expression file(s) has information
                        on multiple regions in multiple columns.
                        Default=False.
  -t                    Flag to turn off a neoantigen burden summary table.
                        Default=True.
```

该软件的输入文件有：

- VCF文件：可以是单区域测序也可以是多区域测序
- hla文件：hla文件的格式如下

  
  | Patient | HLA-A_1 | HLA-A_2 | HLA-B_1 | HLA-B_2 | HLA-C_1 | HLA-C_2 |
  |  --- |  --- |  --- |  --- |  --- |  --- |  ---  |
  | test1 | hla_a_31_01_02 | hla_a_02_01_80 | hla_b_40_01_02 | hla_b_50_01_01 | hla_c_03_04_20 | hla_c_06_02_01_02 |
  | test2 | hla_a_01_01_01_01 | NA | hla_b_07_02_01 | NA | hla_c_01_02_01 | NA |

    Patient名称要和vcf文件的名称相匹配；制表符分割；可以不要列名,但是顺序要匹配；当两个位点预测的HLA是一样的时候(A1和A2,B1和B2,C1和C2),需要用NA代替
  
- 表达文件：在-x参数后指定,制表符分割,第一列是gene id第二列是表达值;支持的id有：Ensembl gene ID, Ensembl transcript ID, RefSeq transcript ID, UCSC transcript ID


使用测试数据进行测试(该软件提供的测试数据的参考基因组是`hg19`,因此将上面配置文件中的`gene_table`和`gene_fasta`改成`hg19`的)：


```sh
NeoPredPipe.py --preponly -I ~/software/NeoPredPipe/Example/input_vcfs -H   ~/software/NeoPredPipe/Example/HLAtypes/hlatypes.txt -o ./test/ -n TestRun -c 1 2 -E 8 9 10

INFO: Annovar reference files of build hg19 were given, using this build for all analysis.
INFO: Begin.
INFO: Running convert2annovar.py on /public/slst/home/wutao2/software/NeoPredPipe/Example/input_vcfs/test1.vcf
INFO: ANNOVAR VCF Conversion Process complete /public/slst/home/wutao2/software/NeoPredPipe/Example/input_vcfs/test1.vcf
INFO: Running annotate_variation.pl on ./test/avready/test1.avinput
INFO: ANNOVAR annotation Process complete for ./test/avready/test1.avinput
INFO: Running coding_change.pl on ./test/avannotated/test1.avannotated.exonic_variant_function
INFO: Coding predictions complete for ./test/avannotated/test1.avannotated.exonic_variant_function
INFO: Input files prepared and completed for test1
INFO: Running convert2annovar.py on /public/slst/home/wutao2/software/NeoPredPipe/Example/input_vcfs/test2.vcf
INFO: ANNOVAR VCF Conversion Process complete /public/slst/home/wutao2/software/NeoPredPipe/Example/input_vcfs/test2.vcf
INFO: Running annotate_variation.pl on ./test/avready/test2.avinput
INFO: ANNOVAR annotation Process complete for ./test/avready/test2.avinput
INFO: Running coding_change.pl on ./test/avannotated/test2.avannotated.exonic_variant_function
INFO: Coding predictions complete for ./test/avannotated/test2.avannotated.exonic_variant_function
INFO: Input files prepared and completed for test2
INFO: Complete.
INFO: Preprocessed intermediary files are in avready, avannotated and fastaFiles. If you wish to perform epitope prediction, run the pipeline again without the --preponly flag, intermediary files will be automatically detected.
```

这一步是准备输入文件的,也就是运行`ANNOVAR`将变异进行注释得到多肽序列(`convert2annovar.py`,`annotate_variation.pl`,和`coding_change.pl `)

也可以直接预测得到结果：

```bash
NeoPredPipe.py -I ~/software/NeoPredPipe/Example/input_vcfs -H ~/software/NeoPredPipe/Example/HLAtypes/hlatypes.txt -o ./test_results/ -n TestRun -c 1 2 -E 8 9 10

INFO: Annovar reference files of build hg19 were given, using this build for all analysis.
INFO: Begin.
INFO: Running convert2annovar.py on /public/slst/home/wutao2/software/NeoPredPipe/Example/input_vcfs/test1.vcf
INFO: ANNOVAR VCF Conversion Process complete /public/slst/home/wutao2/software/NeoPredPipe/Example/input_vcfs/test1.vcf
INFO: Running annotate_variation.pl on ./test_results/avready/test1.avinput
INFO: ANNOVAR annotation Process complete for ./test_results/avready/test1.avinput
INFO: Running coding_change.pl on ./test_results/avannotated/test1.avannotated.exonic_variant_function
INFO: Coding predictions complete for ./test_results/avannotated/test1.avannotated.exonic_variant_function
INFO: Predicting neoantigens for test1
INFO: Running Epitope Predictions for test1 on epitopes of length 9.Indels
INFO: Running Epitope Predictions for test1 on epitopes of length 9
INFO: Running Epitope Predictions for test1 on epitopes of length 8
INFO: Running Epitope Predictions for test1 on epitopes of length 8.Indels
INFO: Running Epitope Predictions for test1 on epitopes of length 10
INFO: Running Epitope Predictions for test1 on epitopes of length 10.Indels
INFO: Predictions complete for test1 on epitopes of length 10.Indels
INFO: Digesting neoantigens for test1
INFO: Digesting neoantigens for test1
INFO: Digesting neoantigens for test1
INFO: Object size of neoantigens: 48472 Kb
Processing genotype information according to A (list of alleles) field.
INFO: Digesting neoantigens for test1
INFO: Digesting neoantigens for test1
INFO: Digesting neoantigens for test1
INFO: Object size of neoantigens: 3768 Kb
Processing genotype information according to A (list of alleles) field.
INFO: Running convert2annovar.py on /public/slst/home/wutao2/software/NeoPredPipe/Example/input_vcfs/test2.vcf
INFO: ANNOVAR VCF Conversion Process complete /public/slst/home/wutao2/software/NeoPredPipe/Example/input_vcfs/test2.vcf
INFO: Running annotate_variation.pl on ./test_results/avready/test2.avinput
INFO: ANNOVAR annotation Process complete for ./test_results/avready/test2.avinput
INFO: Running coding_change.pl on ./test_results/avannotated/test2.avannotated.exonic_variant_function
INFO: Coding predictions complete for ./test_results/avannotated/test2.avannotated.exonic_variant_function
INFO: Predicting neoantigens for test2
INFO: Skipping Sample! No peptides to predict for test2
INFO: Running Epitope Predictions for test2 on epitopes of length 9
INFO: Running Epitope Predictions for test2 on epitopes of length 8
INFO: Skipping Sample! No peptides to predict for test2
INFO: Running Epitope Predictions for test2 on epitopes of length 10
INFO: Skipping Sample! No peptides to predict for test2
INFO: Predictions complete for test2 on epitopes of length 10.Indels
INFO: Digesting neoantigens for test2
INFO: Digesting neoantigens for test2
INFO: Digesting neoantigens for test2
INFO: Object size of neoantigens: 26744 Kb
Processing genotype information according to A (list of alleles) field.
INFO: Summary Tables Complete.
INFO: Summary Tables Complete.
INFO: Complete
```

输出文件是没有表头的,每列的信息为：

> The primary output file of neoantigens has the following format, separated by tabulators (columns 12-26 are taken from [here](http://www.cbs.dtu.dk/services/NetMHCpan/output.php)):
>
> - **Sample**: vcf filename/patient identifier
> - **R1**: Region 1 of a multiregion sample, binary for presence (1) or absence (0), regions above the number of regions in the sample (for varying number of biopsies) are indicated by -1. Can be *n* numbers of regions. *Only present in multiregion samples*.
> - **R2**: Region 2 of a multiregion sample, binary for presence (1) or absence (0), regions above the number of regions in the sample (for varying number of biopsies) are indicated by -1. Can be *n* numbers of regions. *Only present in multiregion samples*.
> - **R3**: Region 3 of a multiregion sample, binary for presence (1) or absence (0), regions above the number of regions in the sample (for varying number of biopsies) are indicated by -1. Can be *n* numbers of regions. *Only present in multiregion samples*.
> - **Line**: Line number from the *.avready file (same as the vcf) to identify mutation yielding corresponding neoantigen.
> - **chr**: Chromosome of mutation
> - **allelepos**: Position of the mutation
> - **ref**: Reference base at the position
> - **alt**: Alternative base at the location
> - **GeneName:RefID**: Gene name and RefSeq ID separated by a colon. Multiple genes/RefSeq IDs separated by a comma.
> - **Expression**: Expression value of the gene. Expression values for multiple regions (*if using the -expmulti flag*) are comma-separated. NA for genes that are not found in the corresponding expression file, or for samples without expression information. *Only present if the -x flag is used*.
> - **pos**: Residue number (starting from 0)
> - **hla**: Molecule/allele name
> - **peptide**: Amino acid sequence of the potential ligand
> - **core**: The minimal 9 amino acid binding core directly in contact with the MHC
> - **Of**: The starting position of the Core within the Peptide (if > 0, the method predicts a N-terminal protrusion)
> - **Gp**: Position of the deletion, if any.
> - **Gl**: Length of the deletion.
> - **Ip**: Position of the insertions, if any.
> - **Il**: Length of the insertion.
> - **Icore**: Interaction core. This is the sequence of the binding core including eventual insertions of deletions.
> - **Identity**: Protein identifier, i.e. the name of the Fasta entry.
> - **Score**: The raw prediction score
> - **Binding Affinity**: Predicted binding affinity in nanoMolar units.
> - **Rank**: Rank of the predicted affinity compared to a set of random natural peptides. This measure is not affected by inherent bias of certain molecules towards higher or lower mean predicted affinities. Strong binders are defined as having %rank<0.5, and weak binders with %rank<2. We advise to select candidate binders based on %Rank rather than nM Affinity
> - **Candidate**: Symbol (<=) used to denote a Strong or Week Binder in BindLevel
> - **BindLevel**: (SB: strong binder, WB: weak binder). The peptide will be identified as a strong binder if the % Rank is below the specified threshold for the strong binders, by default 0.5%. The peptide will be identified as a weak binder if the % Rank is above the threshold of the strong binders but below the specified threshold for the weak binders, by default 2%.
> - **Novelty**: Binary value for indicating if the epitope is novel (1) or exists in the reference proteome (0). *Only present if -m flag is set to perform peptide matching in postprocessing*.

## Recognition Potential

在预测新抗原之后,该工具还可以根据2017 nature文章的方法计算neoantigen recognition potential

## 注意事项

- `-c`参数的含义：

  ![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210324145740028.png)`-c`参数表示从`FORMAT`列往后第几列是需要关注的(一般是tumor列);在单区域测序的样本中可以不指定

- 如果有表达数据,表达数据的`gene id`如果是`Ensembl gene ID`(也就是`ENSG`开头),要去掉版本号(小数点后的数字),因为该软件进行ID转化时使用的文件[mart_table_hg38_unique.txt](https://github.com/MathOnco/NeoPredPipe/blob/master/mart_table_hg38_unique.txt) 时不带版本号的

- HLA的格式需要是像HLA-A03:01这样的
- 输出的列：注意netMHCpan的版本
  ![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210326124059938.png)

