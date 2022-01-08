---
title: 使用R包GenomicDataCommons下载和处理TCGA数据
author: wutao
date: 2022-01-07 10:00:00
categories:
  - 生物信息
index_img: img/GenomicDataCommons.png

---



GenomicDataCommons R 包学习，并使用该包计算 TCGA 样本的测序深度
<!-- more -->

GDC (Genomic Data Commons) 是美国国家癌症研究所建立的在癌症精准医疗数据方面的数据共享平台，目前已经包含几个大型的癌症基因组数据集，比如 TCGA 和 TARGET。GDC 的数据模型非常复杂，可以用下图来简单展示：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/%E4%B8%8B%E8%BD%BD.png)

节点表示实体，比如项目，样本，诊断，文件等，实体之间的关系用边来表示，节点和边都有相应的属性。

# 快速开始

这一部分展示一些基础功能

## 安装

安装和一般的 `Bioconductor` 包一样：

``` r
if (!require("BiocManager"))
    install.packages("BiocManager")
BiocManager::install('GenomicDataCommons')
```

``` r
library(GenomicDataCommons)
```

## 检查连接和状态

`GenomicDataCommons` 包需要网络连接，并且在使用时 NCI 的 GDC API 处于可操作和非维护状态，使用 `status` 来检查连接和状态：

``` r
GenomicDataCommons::status()
>> $commit
>> [1] "b49b90e1318040f447906940f3dff145809d9ea0"
>> 
>> $data_release
>> [1] "Data Release 31.0 - October 29, 2021"
>> 
>> $status
>> [1] "OK"
>> 
>> $tag
>> [1] "3.0.0"
>> 
>> $version
>> [1] 1
```

如果我们需要在脚本或者开发的包中判断连接是否正常，可以使用 `stopifnot` 函数：

``` r
stopifnot(GenomicDataCommons::status()$status=="OK")
```

## 寻找数据

在下载数据之前，我们需要先制作原始数据的 `manifest` 文件，这个文件中有数据的 UUID ，可以被 GDC 的 API
用来定位下载的文件。比如下面的代码获取了卵巢癌 RNA-seq数据的原始 counts：

``` r
ge_manifest = files() %>%
    filter( cases.project.project_id == 'TCGA-OV') %>% 
    filter( type == 'gene_expression' ) %>%
    filter( analysis.workflow_type == 'HTSeq - Counts')  %>%
    manifest()
>> Rows: 379 Columns: 5
>> ── Column specification ────────────────────────────────────────────────────────────────────────────────────────────────────────
>> Delimiter: "\t"
>> chr (4): id, filename, md5, state
>> dbl (1): size
>> 
>> ℹ Use `spec()` to retrieve the full column specification for this data.
>> ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
head(ge_manifest)
>> # A tibble: 6 × 5
>>   id                                   filename                                             md5                      size state 
>>   <chr>                                <chr>                                                <chr>                   <dbl> <chr> 
>> 1 451c9c6d-f062-4ca2-8459-efed929bd305 9c6f2d76-7701-434e-a876-c04ab14cccea.htseq.counts.gz d296b0cb99d14b4b91383… 263970 relea…
>> 2 5bc23348-57f9-4929-a90b-a557696ca955 af6e5654-e755-4c15-b3e5-807da2642e25.htseq.counts.gz 615e3868d8943e94859a2… 253867 relea…
>> 3 7c8ea118-fbce-4dc9-803d-1b6e30d06704 d73c0f69-ab9b-4408-bf89-aa34bf829351.htseq.counts.gz ef4a8c5d45c6de49b590b… 263239 relea…
>> 4 4b83b0f5-4fc7-4a3a-a090-3ce497fb1af4 c9689d9f-6138-42a8-a58e-1b44dc4b193f.htseq.counts.gz f1ffd9f86b0f6c97cbb66… 257423 relea…
>> 5 749e9e26-eb75-4681-b039-6966e911ae7a 43622957-7bed-4d24-a31e-5fadf41216e1.htseq.counts.gz 50e84291a89a72fe3d43e… 255649 relea…
>> 6 a2d8e28d-0371-4421-8ca4-4629a8cc7b72 9a420e2d-15c8-41f3-859f-25872765f75e.htseq.counts.gz 3e2e347efa85410ecbd8c… 253306 relea…
nrow(ge_manifest)
>> [1] 379
```

## 下载数据

通过上面步骤的筛选，获得了 379 个样本的基因表达数据的 UUID，我们可以通过这些 UUID 来下载相应的基因表达数据：

``` r
##演示只下载了前20个文件，已经指定了 cache dir
fnames = lapply(ge_manifest$id[1:20],gdcdata)
>> Rows: 1 Columns: 5
>> ── Column specification ────────────────────────────────────────────────────────────────────────────────────────────────────────
>> Delimiter: "\t"
>> chr (4): id, filename, md5, state
>> dbl (1): size
>> 
>> ℹ Use `spec()` to retrieve the full column specification for this data.
>> ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
>> Rows: 1 Columns: 5
>> ── Column specification ────────────────────────────────────────────────────────────────────────────────────────────────────────
>> Delimiter: "\t"
>> chr (4): id, filename, md5, state
>> dbl (1): size
>> 
>> ℹ Use `spec()` to retrieve the full column specification for this data.
>> ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
>> Rows: 1 Columns: 5
>> ── Column specification ────────────────────────────────────────────────────────────────────────────────────────────────────────
>> Delimiter: "\t"
>> chr (4): id, filename, md5, state
>> dbl (1): size
>> 
>> ℹ Use `spec()` to retrieve the full column specification for this data.
>> ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
>> Rows: 1 Columns: 5
>> ── Column specification ────────────────────────────────────────────────────────────────────────────────────────────────────────
>> Delimiter: "\t"
>> chr (4): id, filename, md5, state
>> dbl (1): size
>> 
>> ℹ Use `spec()` to retrieve the full column specification for this data.
>> ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
>> Rows: 1 Columns: 5
>> ── Column specification ────────────────────────────────────────────────────────────────────────────────────────────────────────
>> Delimiter: "\t"
>> chr (4): id, filename, md5, state
>> dbl (1): size
>> 
>> ℹ Use `spec()` to retrieve the full column specification for this data.
>> ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
>> Rows: 1 Columns: 5
>> ── Column specification ────────────────────────────────────────────────────────────────────────────────────────────────────────
>> Delimiter: "\t"
>> chr (4): id, filename, md5, state
>> dbl (1): size
>> 
>> ℹ Use `spec()` to retrieve the full column specification for this data.
>> ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
>> Rows: 1 Columns: 5
>> ── Column specification ────────────────────────────────────────────────────────────────────────────────────────────────────────
>> Delimiter: "\t"
>> chr (4): id, filename, md5, state
>> dbl (1): size
>> 
>> ℹ Use `spec()` to retrieve the full column specification for this data.
>> ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
>> Rows: 1 Columns: 5
>> ── Column specification ────────────────────────────────────────────────────────────────────────────────────────────────────────
>> Delimiter: "\t"
>> chr (4): id, filename, md5, state
>> dbl (1): size
>> 
>> ℹ Use `spec()` to retrieve the full column specification for this data.
>> ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
>> Rows: 1 Columns: 5
>> ── Column specification ────────────────────────────────────────────────────────────────────────────────────────────────────────
>> Delimiter: "\t"
>> chr (4): id, filename, md5, state
>> dbl (1): size
>> 
>> ℹ Use `spec()` to retrieve the full column specification for this data.
>> ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
>> Rows: 1 Columns: 5
>> ── Column specification ────────────────────────────────────────────────────────────────────────────────────────────────────────
>> Delimiter: "\t"
>> chr (4): id, filename, md5, state
>> dbl (1): size
>> 
>> ℹ Use `spec()` to retrieve the full column specification for this data.
>> ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
>> Rows: 1 Columns: 5
>> ── Column specification ────────────────────────────────────────────────────────────────────────────────────────────────────────
>> Delimiter: "\t"
>> chr (4): id, filename, md5, state
>> dbl (1): size
>> 
>> ℹ Use `spec()` to retrieve the full column specification for this data.
>> ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
>> Rows: 1 Columns: 5
>> ── Column specification ────────────────────────────────────────────────────────────────────────────────────────────────────────
>> Delimiter: "\t"
>> chr (4): id, filename, md5, state
>> dbl (1): size
>> 
>> ℹ Use `spec()` to retrieve the full column specification for this data.
>> ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
>> Rows: 1 Columns: 5
>> ── Column specification ────────────────────────────────────────────────────────────────────────────────────────────────────────
>> Delimiter: "\t"
>> chr (4): id, filename, md5, state
>> dbl (1): size
>> 
>> ℹ Use `spec()` to retrieve the full column specification for this data.
>> ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
>> Rows: 1 Columns: 5
>> ── Column specification ────────────────────────────────────────────────────────────────────────────────────────────────────────
>> Delimiter: "\t"
>> chr (4): id, filename, md5, state
>> dbl (1): size
>> 
>> ℹ Use `spec()` to retrieve the full column specification for this data.
>> ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
>> Rows: 1 Columns: 5
>> ── Column specification ────────────────────────────────────────────────────────────────────────────────────────────────────────
>> Delimiter: "\t"
>> chr (4): id, filename, md5, state
>> dbl (1): size
>> 
>> ℹ Use `spec()` to retrieve the full column specification for this data.
>> ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
>> Rows: 1 Columns: 5
>> ── Column specification ────────────────────────────────────────────────────────────────────────────────────────────────────────
>> Delimiter: "\t"
>> chr (4): id, filename, md5, state
>> dbl (1): size
>> 
>> ℹ Use `spec()` to retrieve the full column specification for this data.
>> ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
>> Rows: 1 Columns: 5
>> ── Column specification ────────────────────────────────────────────────────────────────────────────────────────────────────────
>> Delimiter: "\t"
>> chr (4): id, filename, md5, state
>> dbl (1): size
>> 
>> ℹ Use `spec()` to retrieve the full column specification for this data.
>> ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
>> Rows: 1 Columns: 5
>> ── Column specification ────────────────────────────────────────────────────────────────────────────────────────────────────────
>> Delimiter: "\t"
>> chr (4): id, filename, md5, state
>> dbl (1): size
>> 
>> ℹ Use `spec()` to retrieve the full column specification for this data.
>> ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
>> Rows: 1 Columns: 5
>> ── Column specification ────────────────────────────────────────────────────────────────────────────────────────────────────────
>> Delimiter: "\t"
>> chr (4): id, filename, md5, state
>> dbl (1): size
>> 
>> ℹ Use `spec()` to retrieve the full column specification for this data.
>> ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
>> Rows: 1 Columns: 5
>> ── Column specification ────────────────────────────────────────────────────────────────────────────────────────────────────────
>> Delimiter: "\t"
>> chr (4): id, filename, md5, state
>> dbl (1): size
>> 
>> ℹ Use `spec()` to retrieve the full column specification for this data.
>> ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
```

下载需要用到的函数是 `gdcdata`，该函数接受文件的 UUID 作为输入，返回下载的文件并以 `Filename` 重命名文件，需要注意的是我们最好指定`gdc_cache` 作为 下载文件的存放目录（cache），避免多次重复下载（当然默认的 cache 目录也可以）。

可以查看一下下载文件的结构：

``` bash
tree ~/.cache/GenomicDataCommons
>> [01;34m/home/data/t040201/.cache/GenomicDataCommons[00m
>> ├── [01;34m02ab2f95-023b-492e-8dc9-20b4f52f209f[00m
>> │   └── [01;31mb1d5c444-da0d-4360-bd45-31c94217adfc.FPKM-UQ.txt.gz[00m
>> ├── [01;34m0f2ef2fa-cf6e-4d03-a648-ccd9fdf8b8c5[00m
>> │   └── [01;31mb64ef80a-d41c-4f92-a3ed-e43d55abb2c2.htseq.counts.gz[00m
>> ├── [01;34m1480554d-8579-4146-93ba-a7b23b8c9a5b[00m
>> │   └── [01;31ma2082ad4-279e-422f-b5a7-cb7fbeb7a6df.htseq.counts.gz[00m
>> ├── [01;34m1c099182-703f-4aaf-b1d6-0f9be5094b9a[00m
>> │   └── [01;31mfea3c4d0-7b9f-4279-989e-535aaefbdfeb.FPKM.txt.gz[00m
>> ├── [01;34m3b82236e-b055-4d10-acb4-b5a5992a1261[00m
>> │   └── [01;31m6aa7225b-3d10-4b19-a472-0adeb21c26cf.htseq.counts.gz[00m
>> ├── [01;34m451c9c6d-f062-4ca2-8459-efed929bd305[00m
>> │   └── [01;31m9c6f2d76-7701-434e-a876-c04ab14cccea.htseq.counts.gz[00m
>> ├── [01;34m4958b9d4-59fa-49e4-a849-1b777452f6d2[00m
>> │   └── [01;31m3e00e8aa-31bc-454e-b558-7bcbad5f47ad.htseq.counts.gz[00m
>> ├── [01;34m4b83b0f5-4fc7-4a3a-a090-3ce497fb1af4[00m
>> │   └── [01;31mc9689d9f-6138-42a8-a58e-1b44dc4b193f.htseq.counts.gz[00m
>> ├── [01;34m581b0bcf-a7e5-44ec-818a-9193a0565095[00m
>> │   └── [01;31m347a8eeb-204f-41df-98d3-61394d2b7cd7.htseq.counts.gz[00m
>> ├── [01;34m5bc23348-57f9-4929-a90b-a557696ca955[00m
>> │   └── [01;31maf6e5654-e755-4c15-b3e5-807da2642e25.htseq.counts.gz[00m
>> ├── [01;34m5e57bd1d-f38d-447d-9d7c-1fa1c5a2ecaa[00m
>> │   └── [01;31mb269c35d-7f91-4c66-8bef-59906ec87745.htseq.counts.gz[00m
>> ├── [01;34m717ca3c0-0b0e-4cf3-af34-15d0fbfd7f68[00m
>> │   └── [01;31mc255d022-a659-42bd-9099-6853b41b64c7.htseq.counts.gz[00m
>> ├── [01;34m749e9e26-eb75-4681-b039-6966e911ae7a[00m
>> │   └── [01;31m43622957-7bed-4d24-a31e-5fadf41216e1.htseq.counts.gz[00m
>> ├── [01;34m750f8926-4361-4a92-8c72-59c82baad867[00m
>> │   └── [01;31m195a13be-31a4-47ce-bc3d-6aba8451e304.htseq.counts.gz[00m
>> ├── [01;34m769c5f60-fdf1-49b1-bba3-7f4a3ef1f9a8[00m
>> │   └── [01;31m9e198dfe-9fc6-48e2-ba06-90c49ddf48aa.htseq.counts.gz[00m
>> ├── [01;34m7c8ea118-fbce-4dc9-803d-1b6e30d06704[00m
>> │   └── [01;31md73c0f69-ab9b-4408-bf89-aa34bf829351.htseq.counts.gz[00m
>> ├── [01;34ma0765b46-5fbb-49d0-b8aa-682486927d0f[00m
>> │   └── [01;31m5a928267-356d-47e9-b8b2-f477eaa261fa.htseq.counts.gz[00m
>> ├── [01;34ma2d8e28d-0371-4421-8ca4-4629a8cc7b72[00m
>> │   └── [01;31m9a420e2d-15c8-41f3-859f-25872765f75e.htseq.counts.gz[00m
>> ├── [01;34mb44b0d52-562a-476a-a607-a7148d128359[00m
>> │   └── [01;31m062b6d0c-06b0-4d27-8702-32501278cd9c.htseq.counts.gz[00m
>> ├── [01;34mc44d4c5c-b855-4e82-bd90-0f4b6c0e0015[00m
>> │   └── [01;31m75303476-cdec-4ae4-aaf5-01abdc3213ab.htseq.counts.gz[00m
>> ├── [01;34mc9cd12a4-b4b3-416e-9046-9c06dd884547[00m
>> │   └── [01;31mac6e38b7-143c-491e-9892-4c28f51ddce5.htseq.counts.gz[00m
>> └── [01;34mcd32019f-e6e5-4cf9-a4b4-d38dfdfc0f0a[00m
>>     └── [01;31mfedd52be-18a8-423f-ba8a-4f9416f11ff5.htseq.counts.gz[00m
>> 
>> 22 directories, 22 files
```

<p class="note note-primary">
如果下载的是 `controlled-access` 数据，需要提供 `token`
</p>


## 元数据获取

### 临床数据

获取临床数据是一个常见的任务，`gdc_clinical` 函数接受 `case_ids`，返回一个有个四个数据框的列表：

-   人口学统计（demographic），包括性别，种族，年龄等
-   诊断（diagnoses），包括临床分期，生存时间，回访信息等
-   暴露（exposures），包括吸烟，饮酒记录等
-   main，包括疾病类型，诊断时间等

``` r
case_ids = cases() %>% results(size=10) %>% ids()
clindat = gdc_clinical(case_ids)
names(clindat)
>> [1] "demographic" "diagnoses"   "exposures"   "main"

head(clindat[["demographic"]])
>> # A tibble: 6 × 15
>>   vital_status gender race    ethnicity  age_at_index submitter_id  days_to_birth created_datetime year_of_birth demographic_id 
>>   <chr>        <chr>  <chr>   <chr>             <int> <chr>                 <int> <lgl>                    <int> <chr>          
>> 1 Alive        female not re… not repor…           65 TCGA-A8-A07G…        -23926 NA                        1944 38caac77-d856-…
>> 2 Dead         female black … not hispa…           49 TCGA-A2-A3XY…        -18059 NA                        1961 ac6eba06-6e54-…
>> 3 Alive        female white   not hispa…           56 TCGA-E2-A152…        -20705 NA                        1953 680b1fdd-143c-…
>> 4 Alive        female white   not hispa…           40 TCGA-E2-A15E…        -14894 NA                        1969 1b53b23d-b98c-…
>> 5 Alive        female white   not repor…           73 TCGA-AR-A0U0…        -26993 NA                        1931 2f82f157-cce2-…
>> 6 Alive        female white   not repor…           52 TCGA-BH-A0E1…        -19192 NA                        1957 7c593975-bc80-…
>> # … with 5 more variables: updated_datetime <chr>, state <chr>, year_of_death <lgl>, days_to_death <int>, case_id <chr>
head(clindat[["diagnoses"]])
>> # A tibble: 6 × 29
>>   case_id synchronous_mal… ajcc_pathologic… days_to_diagnos… created_datetime last_known_dise… tissue_or_organ… days_to_last_fo…
>>   <chr>   <chr>            <chr>                       <int> <lgl>            <chr>            <chr>                       <int>
>> 1 8cf8b6… No               Stage IIA                       0 NA               not reported     Breast, NOS                   577
>> 2 deba32… No               Stage IIB                       0 NA               not reported     Breast, NOS                  1064
>> 3 a80154… No               Stage I                         0 NA               not reported     Breast, NOS                  2128
>> 4 0a2a35… No               Stage IIA                       0 NA               not reported     Breast, NOS                   630
>> 5 e3c336… No               Stage IIB                       0 NA               not reported     Breast, NOS                  1988
>> 6 606fbc… No               Stage IIB                       0 NA               not reported     Breast, NOS                   477
>> # … with 21 more variables: primary_diagnosis <chr>, age_at_diagnosis <int>, updated_datetime <chr>, year_of_diagnosis <int>,
>> #   prior_malignancy <chr>, state <chr>, prior_treatment <chr>, days_to_last_known_disease_status <lgl>,
>> #   ajcc_staging_system_edition <chr>, ajcc_pathologic_t <chr>, days_to_recurrence <lgl>, morphology <chr>,
>> #   ajcc_pathologic_n <chr>, ajcc_pathologic_m <chr>, submitter_id <chr>, classification_of_tumor <chr>, diagnosis_id <chr>,
>> #   icd_10_code <chr>, site_of_resection_or_biopsy <chr>, tumor_grade <chr>, progression_or_recurrence <chr>
head(clindat[["exposures"]])
>> # A tibble: 6 × 10
>>   case_id    cigarettes_per_d… updated_datetime   alcohol_history exposure_id   submitter_id years_smoked state created_datetime
>>   <chr>      <lgl>             <chr>              <chr>           <chr>         <chr>        <lgl>        <chr> <lgl>           
>> 1 8cf8b620-… NA                2019-07-31T21:48:… Not Reported    a8be57c5-620… TCGA-A8-A07… NA           rele… NA              
>> 2 deba32e4-… NA                2019-07-31T21:29:… Not Reported    9aa5cc63-cea… TCGA-A2-A3X… NA           rele… NA              
>> 3 a8015490-… NA                2019-07-31T21:52:… Not Reported    134cee43-117… TCGA-E2-A15… NA           rele… NA              
>> 4 0a2a3529-… NA                2019-07-31T21:31:… Not Reported    fdf22b2e-0ba… TCGA-E2-A15… NA           rele… NA              
>> 5 e3c336f5-… NA                2019-07-31T15:38:… Not Reported    3d1c43ad-b1d… TCGA-AR-A0U… NA           rele… NA              
>> 6 606fbc6a-… NA                2019-07-31T21:42:… Not Reported    1d844394-425… TCGA-BH-A0E… NA           rele… NA              
>> # … with 1 more variable: alcohol_intensity <lgl>
head(clindat[["main"]])
>> # A tibble: 6 × 8
>>   id                                   disease_type  submitter_id created_datetime primary_site updated_datetime  case_id  state
>>   <chr>                                <chr>         <chr>        <lgl>            <chr>        <chr>             <chr>    <chr>
>> 1 8cf8b620-7ab6-4b6e-84bc-ff5a83f381fa Ductal and L… TCGA-A8-A07G NA               Breast       2019-08-06T14:14… 8cf8b62… rele…
>> 2 deba32e4-0e68-4711-941b-3b63bd965afb Ductal and L… TCGA-A2-A3XY NA               Breast       2019-08-06T14:14… deba32e… rele…
>> 3 a8015490-9740-45c9-8bd2-eb6d1beefc2e Ductal and L… TCGA-E2-A152 NA               Breast       2019-08-06T14:16… a801549… rele…
>> 4 0a2a3529-f645-4967-9a58-89ee20b8bb62 Ductal and L… TCGA-E2-A15E NA               Breast       2019-08-06T14:16… 0a2a352… rele…
>> 5 e3c336f5-c32f-4c5d-81fb-e2408ae145b2 Ductal and L… TCGA-AR-A0U0 NA               Breast       2019-08-06T14:15… e3c336f… rele…
>> 6 606fbc6a-b41b-441d-9401-51e54912bf5e Ductal and L… TCGA-BH-A0E1 NA               Breast       2019-08-06T14:15… 606fbc6… rele…
```

### 广义的元数据获取

我们可以通过 `GenomicDataCommons` 结合各种参数（比如 `filter`, `select`, `expand`等）来灵活的获取想要的元数据：

``` r
expands = c("diagnoses","annotations",
             "demographic","exposures")
clinResults = cases() %>%
    GenomicDataCommons::select(NULL) %>%
    GenomicDataCommons::expand(expands) %>%
    results(size=50)
str(clinResults[[1]],list.len=6)
>>  chr [1:50] "8cf8b620-7ab6-4b6e-84bc-ff5a83f381fa" "deba32e4-0e68-4711-941b-3b63bd965afb" ...
```

# 基本设计

从上面的例子中可以看出这个包的设计行为和 `dplyr` 是非常类似的，一些动词也是直接和 `dplyr` 的函数同名(filter, select 等)。简单来说，该包检索，获取元数据和文件分为三步：

-   请求构造函数（query constructors，如上面的 `cases()`, `files()`）
-   一系列的动词用来过滤样本，选择字段，聚合并生成最终的请求对象（query object，如 `filter`, `select`等）
-   使用一系列动词基于上面得到的请求对象获取文件，返回结果（如 `gdcdata`）

完成上述过程的基本函数如下：

-   创建请求：
    -   `projects()`
    -   `cases()`
    -   `files()`
    -   `annotations()`
-   操作请求：
    -   `filter()`
    -   `facet()`
    -   `select()`
    -   `expand()`
-   GDC API 字段的内省（introspection, 即支持哪些查询，有哪些字段及字段类型等信息）：
    -   `mapping()`
    -   `available_fields()`
    -   `default_fields()`
    -   `grep_fields()`
    -   `field_picker()`
    -   `available_values()`
    -   `available_expand()`
-   执行 API 调用，获取请求的结果：
    -   `results()`
    -   `count()`
    -   `response()`
-   原始数据下载：
    -   `gdcdata()`
    -   `transfer()`
    -   `gdc_client()`
-   汇总，聚合字段值（也叫 faceting）：
    -   `aggregations()`
-   Control 数据的授权（token）：
    -   `gdc_token()`
-   BAM 文件切片：
    -   `slicing()`

# 用法

上面那些基本函数构成了两大类操作：

-   检索元数据和寻找数据文件
-   传输原始数据或者处理后的数据

## 检索元数据

### 创建检索

有四种方便的函数可以创建 `GDCQuery` 对象来获取不同类型的数据：

-   `project()`
-   `cases()`
-   `files()`
-   `annotations()`

这些对象都含有下列的一些元素：

-   字段（fields）：需要下载的字段，如果没有指定字段，就会使用默认字段来取回数据（默认字段可以通过`default_fields()` 来查看）
-   过滤器（filters）：含有调用 `filter()` 方法后获得的结果，并用来筛选取回的数据
-   facets：当调用 `aggregations()` 时对数据汇总所需的字段名
-   存档（archive）：可以是 “default” 或者 `legacy` (legacy 是比较老的数据)
-   token：下载 control 数据的凭证，对于获取元数据可以不需要，只需在下载真正数据时提供即可

``` r
pquery = projects()
str(pquery)
>> List of 5
>>  $ fields : chr [1:10] "dbgap_accession_number" "disease_type" "intended_release_date" "name" ...
>>  $ filters: NULL
>>  $ facets : NULL
>>  $ legacy : logi FALSE
>>  $ expand : NULL
>>  - attr(*, "class")= chr [1:3] "gdc_projects" "GDCQuery" "list"

default_fields("projects")
>>  [1] "dbgap_accession_number" "disease_type"           "intended_release_date"  "name"                  
>>  [5] "primary_site"           "project_autocomplete"   "project_id"             "releasable"            
>>  [9] "released"               "state"
```

可以看到初始状态下大部分元素是空的（NULL）

### 取回结果

当有了一个请求对象后，我们就可以从 GDC 取回结果了。可以使用 `count()` 来得到最基本的结果类型，这个函数返回满足 `filter` 标准的记录数量，由于我们目前没有定义任何的过滤条件，所以这里的 `count()` 返回的是所有的 `project` 记录（在 default 存档中）：

``` r
pcount = pquery %>% count()
pcount
>> [1] 70
```

`results()` 可以直接取回结果：

``` r
presults = pquery %>% results()
```

返回的结果从 GDC 的 Json 格式被转换成 R 里面的 List，可以使用 `str()` 来简单的查看数据结构：

``` r
str(presults)
>> List of 9
>>  $ id                    : chr [1:10] "TCGA-BRCA" "GENIE-MSK" "GENIE-VICC" "GENIE-UHN" ...
>>  $ primary_site          :List of 10
>>   ..$ TCGA-BRCA            : chr "Breast"
>>   ..$ GENIE-MSK            : chr [1:49] "Connective, subcutaneous and other soft tissues" "Kidney" "Prostate gland" "Other and unspecified major salivary glands" ...
>>   ..$ GENIE-VICC           : chr [1:46] "Connective, subcutaneous and other soft tissues" "Kidney" "Prostate gland" "Other and unspecified major salivary glands" ...
>>   ..$ GENIE-UHN            : chr [1:42] "Connective, subcutaneous and other soft tissues" "Kidney" "Prostate gland" "Other and unspecified major salivary glands" ...
>>   ..$ CPTAC-2              : chr [1:6] "Ovary" "Rectum" "Other and unspecified female genital organs" "Breast" ...
>>   ..$ CMI-ASC              : chr [1:9] "Bladder" "Other and ill-defined sites" "Other and ill-defined digestive organs" "Heart, mediastinum, and pleura" ...
>>   ..$ BEATAML1.0-COHORT    : chr "Hematopoietic and reticuloendothelial systems"
>>   ..$ CGCI-BLGSP           : chr "Hematopoietic and reticuloendothelial systems"
>>   ..$ BEATAML1.0-CRENOLANIB: chr "Hematopoietic and reticuloendothelial systems"
>>   ..$ CMI-MPC              : chr [1:2] "Prostate gland" "Lymph nodes"
>>  $ dbgap_accession_number: chr [1:10] NA NA NA NA ...
>>  $ project_id            : chr [1:10] "TCGA-BRCA" "GENIE-MSK" "GENIE-VICC" "GENIE-UHN" ...
>>  $ disease_type          :List of 10
>>   ..$ TCGA-BRCA            : chr [1:9] "Complex Epithelial Neoplasms" "Fibroepithelial Neoplasms" "Adnexal and Skin Appendage Neoplasms" "Adenomas and Adenocarcinomas" ...
>>   ..$ GENIE-MSK            : chr [1:49] "Miscellaneous Bone Tumors" "Myeloid Leukemias" "Gliomas" "Lipomatous Neoplasms" ...
>>   ..$ GENIE-VICC           : chr [1:43] "Leukemias, NOS" "Chronic Myeloproliferative Disorders" "Myeloid Leukemias" "Gliomas" ...
>>   ..$ GENIE-UHN            : chr [1:39] "Leukemias, NOS" "Chronic Myeloproliferative Disorders" "Miscellaneous Bone Tumors" "Myeloid Leukemias" ...
>>   ..$ CPTAC-2              : chr [1:5] "Not Reported" "Cystic, Mucinous and Serous Neoplasms" "Adenomas and Adenocarcinomas" "Ductal and Lobular Neoplasms" ...
>>   ..$ CMI-ASC              : chr "Soft Tissue Tumors and Sarcomas, NOS"
>>   ..$ BEATAML1.0-COHORT    : chr [1:6] "Leukemias, NOS" "Chronic Myeloproliferative Disorders" "Myeloid Leukemias" "Plasma Cell Tumors" ...
>>   ..$ CGCI-BLGSP           : chr "Mature B-Cell Lymphomas"
>>   ..$ BEATAML1.0-CRENOLANIB: chr "Myeloid Leukemias"
>>   ..$ CMI-MPC              : chr "Adenomas and Adenocarcinomas"
>>  $ name                  : chr [1:10] "Breast Invasive Carcinoma" "AACR Project GENIE - Contributed by Memorial Sloan Kettering Cancer Center" "AACR Project GENIE - Contributed by Vanderbilt-Ingram Cancer Center" "AACR Project GENIE - Contributed by Princess Margaret Cancer Centre" ...
>>  $ releasable            : logi [1:10] TRUE FALSE FALSE FALSE TRUE TRUE ...
>>  $ state                 : chr [1:10] "open" "open" "open" "open" ...
>>  $ released              : logi [1:10] TRUE TRUE TRUE TRUE TRUE TRUE ...
>>  - attr(*, "row.names")= int [1:10] 1 2 3 4 5 6 7 8 9 10
>>  - attr(*, "class")= chr [1:3] "GDCprojectsResults" "GDCResults" "list"
```

可以看到默认只返回 10 条记录，我们可以使用 `results()` 的 `size` 和 `from` 参数来改变需要返回的数量（size 表示返回的记录数，from 表示从哪个索引开始返回数据）；也有一个简便的函数 `results_all()` 返回所有的请求结果，可想而知这种方法可能会导致下载的数据非常大，花费的时间比较久，因此尽量使用 `counts()` 和 `results()` 来获取数据。

``` r
length(ids(presults))
>> [1] 10

presults = pquery %>% results_all()
length(ids(presults))
>> [1] 70

length(ids(presults)) == count(pquery)
>> [1] TRUE
```

结果是以列表的形式存储的，可以使用 `purrr`，`rlist`，`data.tree` 等 R 包来操作复杂嵌套的列表，另外也可以使用 `listviewer` 包来交互式的查看列表的结构（其实和在 Rstudio 中直接打开差别不大）：

``` r
listviewer::jsonedit(presults)
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220108121517890.png)

### 字段和值

前面也已经看到从 GDC 检索和获取数据的中心就是指定返回哪些字段，根据字段和值进行筛选并进行聚合统计。GenomicDataCommons 有两个简单的函数：`available_fields()` 和 `default_fields()`，这两个函数的参数可以是代表访问点（endpoint）名称的字符（“cases”, “files”, “annotations”, 或 “projects”）或者是 `GDCQuery`
对象，返回相应的字段名称：

``` r
default_fields('files')
>>  [1] "access"                         "acl"                            "average_base_quality"          
>>  [4] "average_insert_size"            "average_read_length"            "channel"                       
>>  [7] "chip_id"                        "chip_position"                  "contamination"                 
>> [10] "contamination_error"            "created_datetime"               "data_category"                 
>> [13] "data_format"                    "data_type"                      "error_type"                    
>> [16] "experimental_strategy"          "file_autocomplete"              "file_id"                       
>> [19] "file_name"                      "file_size"                      "imaging_date"                  
>> [22] "magnification"                  "md5sum"                         "mean_coverage"                 
>> [25] "msi_score"                      "msi_status"                     "pairs_on_diff_chr"             
>> [28] "plate_name"                     "plate_well"                     "platform"                      
>> [31] "proportion_base_mismatch"       "proportion_coverage_10x"        "proportion_coverage_10X"       
>> [34] "proportion_coverage_30x"        "proportion_coverage_30X"        "proportion_reads_duplicated"   
>> [37] "proportion_reads_mapped"        "proportion_targets_no_coverage" "read_pair_number"              
>> [40] "revision"                       "stain_type"                     "state"                         
>> [43] "state_comment"                  "submitter_id"                   "tags"                          
>> [46] "total_reads"                    "tumor_ploidy"                   "tumor_purity"                  
>> [49] "type"                           "updated_datetime"
length(available_fields('files'))
>> [1] 981
head(available_fields('files'))
>> [1] "access"                      "acl"                         "analysis.analysis_id"        "analysis.analysis_type"     
>> [5] "analysis.created_datetime"   "analysis.input_files.access"
```

知道有哪些字段后，我们就可以通过 `select` 动词来选择需要的字段；注意，这里的 `select` 和 `dplyr` 的`select` 并不是完全一样的，后者的 `select` 只能选择已经展示的字段，而这里的 `select` 可以所有的可选字段（也就是 default 和 available 的区别）：

``` r
qcases = cases()
qcases$fields##Default fields 
>>  [1] "aliquot_ids"              "analyte_ids"              "case_autocomplete"        "case_id"                 
>>  [5] "consent_type"             "created_datetime"         "days_to_consent"          "days_to_lost_to_followup"
>>  [9] "diagnosis_ids"            "disease_type"             "index_date"               "lost_to_followup"        
>> [13] "portion_ids"              "primary_site"             "sample_ids"               "slide_ids"               
>> [17] "state"                    "submitter_aliquot_ids"    "submitter_analyte_ids"    "submitter_diagnosis_ids" 
>> [21] "submitter_id"             "submitter_portion_ids"    "submitter_sample_ids"     "submitter_slide_ids"     
>> [25] "updated_datetime"

default_fields(qcases)
>>  [1] "aliquot_ids"              "analyte_ids"              "case_autocomplete"        "case_id"                 
>>  [5] "consent_type"             "created_datetime"         "days_to_consent"          "days_to_lost_to_followup"
>>  [9] "diagnosis_ids"            "disease_type"             "index_date"               "lost_to_followup"        
>> [13] "portion_ids"              "primary_site"             "sample_ids"               "slide_ids"               
>> [17] "state"                    "submitter_aliquot_ids"    "submitter_analyte_ids"    "submitter_diagnosis_ids" 
>> [21] "submitter_id"             "submitter_portion_ids"    "submitter_sample_ids"     "submitter_slide_ids"     
>> [25] "updated_datetime"

##可以选择default field中没有的字段
qcases = cases() %>% GenomicDataCommons::select(available_fields('cases'))
head(qcases$fields)
>> [1] "case_id"                       "aliquot_ids"                   "analyte_ids"                  
>> [4] "annotations.annotation_id"     "annotations.case_id"           "annotations.case_submitter_id"
```

由于检索字段在下载数据过程中是一个常用的操作，因此该包提供了一些函数来快速的找到想要的字段，如
`grep_fields()` 和 `field_picker()` （貌似现在的版本已经把 `field_picker()` 函数给删了）。

### 聚合统计

结合 `facet` 和 `aggregations()` 可以对一个或多个字段中的值进行统计，类似 base R 中的 `table` 操作（但是一次只能对一个字段操作，不能进行交叉操作），返回一个数据框（tibbles）:

``` r
res = files() %>% facet(c('type','data_type')) %>% aggregations()
res$type
>>    doc_count                           key
>> 1     151568    annotated_somatic_mutation
>> 2      89689       simple_somatic_mutation
>> 3      87849                 aligned_reads
>> 4      66555               gene_expression
>> 5      58540           copy_number_segment
>> 6      45843          copy_number_estimate
>> 7      32268              mirna_expression
>> 8      30075                   slide_image
>> 9      25837        biospecimen_supplement
>> 10     14750          structural_variation
>> 11     13732        methylation_beta_value
>> 12     12962           clinical_supplement
>> 13      7906            protein_expression
>> 14      4410   aggregated_somatic_mutation
>> 15      4368       masked_somatic_mutation
>> 16      2746      masked_methylation_array
>> 17        54 secondary_expression_analysis
```

使用 `aggregations()` 可以方便的统计一个字段中有哪些值，有利于我们后续的针对字段的筛选。

### 筛选

GenomicDataCommons 提供了和 dplyr 的 filter 同名且类似的函数 `filter()` 用来对特定的字段的值进行筛选，比如只想要上面的 `type` 中的 `gene_expression` 数据，就可以这样来筛选：

``` r
qfiles = files() %>% filter( type == 'gene_expression')
str(get_filter(qfiles))
>> List of 2
>>  $ op     : 'scalar' chr "="
>>  $ content:List of 2
>>   ..$ field: chr "type"
>>   ..$ value: chr "gene_expression"
```

如果我们现在想要基于某个 TCGA 的测序项目来构建过滤器（比如 TCGA-OV），但是不知道具体的字段名称是什么，这时候就可以使用上面提到的 `grep_fields()` 来查找可能的字段：

``` r
grep_fields("files","project")
>>  [1] "cases.project.dbgap_accession_number"         "cases.project.disease_type"                  
>>  [3] "cases.project.intended_release_date"          "cases.project.name"                          
>>  [5] "cases.project.primary_site"                   "cases.project.program.dbgap_accession_number"
>>  [7] "cases.project.program.name"                   "cases.project.program.program_id"            
>>  [9] "cases.project.project_id"                     "cases.project.releasable"                    
>> [11] "cases.project.released"                       "cases.project.state"                         
>> [13] "cases.tissue_source_site.project"
```

看起来 `cases.project.project_id` 有可能符合我们的要求，接着再用 `facet` 和 `aggregations` 来检查该字段中有没有我们想要的值：

``` r
files() %>% 
    facet('cases.project.project_id') %>% 
    aggregations() %>% 
    head()
>> $cases.project.project_id
>>    doc_count                   key
>> 1      36134                 FM-AD
>> 2      34686             TCGA-BRCA
>> 3      45952               CPTAC-3
>> 4      18723             TCGA-LUAD
>> 5      36470             GENIE-MSK
>> 6      17717             TCGA-UCEC
>> 7      16694             TCGA-HNSC
>> 8      16776               TCGA-OV
>> 9      15826             TCGA-THCA
>> 10     29433         MMRF-COMMPASS
>> 11     16696             TCGA-LUSC
>> 12     16230              TCGA-LGG
>> 13     16733             TCGA-KIRC
>> 14     28464            GENIE-DFCI
>> 15     15648             TCGA-PRAD
>> 16     15701             TCGA-COAD
>> 17     13332              TCGA-GBM
>> 18     20772         TARGET-ALL-P2
>> 19     14025             TCGA-SKCM
>> 20     14096             TCGA-STAD
>> 21     12856             TCGA-BLCA
>> 22     11762             TCGA-LIHC
>> 23      9373             TCGA-CESC
>> 24      9353             TCGA-KIRP
>> 25      8228             TCGA-SARC
>> 26     14037             REBC-THYR
>> 27      8142            TARGET-AML
>> 28      5791             TCGA-PAAD
>> 29      5783             TCGA-ESCA
>> 30      5460             TCGA-PCPG
>> 31      5401             TCGA-READ
>> 32      9888               CPTAC-2
>> 33      5978             TCGA-TGCT
>> 34      8981     BEATAML1.0-COHORT
>> 35      5792            TARGET-NBL
>> 36      8167             HCMI-CMDC
>> 37      4814             TCGA-LAML
>> 38      5771         CGCI-HTMCP-CC
>> 39      3781             TCGA-THYM
>> 40      5941               CMI-MBC
>> 41      2782              TCGA-ACC
>> 42      2520             TCGA-KICH
>> 43      2677             TARGET-WT
>> 44      4805          NCICCR-DLBCL
>> 45      2580             TCGA-MESO
>> 46      2352              TCGA-UVM
>> 47      3113             TARGET-OS
>> 48      3982         TARGET-ALL-P3
>> 49      3857             GENIE-MDA
>> 50      3833            GENIE-VICC
>> 51      1813              TCGA-UCS
>> 52      3320             GENIE-JHU
>> 53      1456             TCGA-CHOL
>> 54      2632             GENIE-UHN
>> 55      1358             TCGA-DLBC
>> 56      2477            CGCI-BLGSP
>> 57      1049             TARGET-RT
>> 58      1038            GENIE-GRCC
>> 59       994            WCDT-MCRPC
>> 60       934               CMI-ASC
>> 61       801             GENIE-NKI
>> 62       798              OHSU-CNL
>> 63       703   ORGANOID-PANCREATIC
>> 64       570               CMI-MPC
>> 65       417           CTSP-DLBCL1
>> 66       339              TRIO-CRU
>> 67       222 BEATAML1.0-CRENOLANIB
>> 68       169           TARGET-CCSK
>> 69       133         TARGET-ALL-P1
>> 70        21        VAREPOP-APOLLO
```

这个字段确实是有 `TCGA-OV` 的，然后就可以基于这个字段的值使用 `filter` 获取想要的数据：

``` r
qfiles = files() %>%
    filter( cases.project.project_id == 'TCGA-OV' & type == 'gene_expression')
str(get_filter(qfiles))
>> List of 2
>>  $ op     : 'scalar' chr "and"
>>  $ content:List of 2
>>   ..$ :List of 2
>>   .. ..$ op     : 'scalar' chr "="
>>   .. ..$ content:List of 2
>>   .. .. ..$ field: chr "cases.project.project_id"
>>   .. .. ..$ value: chr "TCGA-OV"
>>   ..$ :List of 2
>>   .. ..$ op     : 'scalar' chr "="
>>   .. ..$ content:List of 2
>>   .. .. ..$ field: chr "type"
>>   .. .. ..$ value: chr "gene_expression"

qfiles %>% count()
>> [1] 1137

##使用多次filter也是可以的
qfiles2 = files() %>%
    filter( cases.project.project_id == 'TCGA-OV') %>% 
    filter( type == 'gene_expression') 
qfiles2 %>% count()
>> [1] 1137
```

检索到数据后就可以使用 `manidfest()`来基于当前的请求构建需要下载数据的元数据：

``` r
manifest_df = qfiles %>% manifest()
>> Rows: 1137 Columns: 5
>> ── Column specification ────────────────────────────────────────────────────────────────────────────────────────────────────────
>> Delimiter: "\t"
>> chr (4): id, filename, md5, state
>> dbl (1): size
>> 
>> ℹ Use `spec()` to retrieve the full column specification for this data.
>> ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
head(manifest_df)
>> # A tibble: 6 × 5
>>   id                                   filename                                             md5                      size state 
>>   <chr>                                <chr>                                                <chr>                   <dbl> <chr> 
>> 1 1c099182-703f-4aaf-b1d6-0f9be5094b9a fea3c4d0-7b9f-4279-989e-535aaefbdfeb.FPKM.txt.gz     a344c91e930a0217efa2b… 506691 relea…
>> 2 02ab2f95-023b-492e-8dc9-20b4f52f209f b1d5c444-da0d-4360-bd45-31c94217adfc.FPKM-UQ.txt.gz  45d6b0fcc30607948a53b… 547204 relea…
>> 3 30c9def0-c3b5-45b7-a427-020b735c836a ab8603dd-2f94-4c83-9927-455958be0007.FPKM.txt.gz     c82b63def1fcc6b9d3b52… 503463 relea…
>> 4 451c9c6d-f062-4ca2-8459-efed929bd305 9c6f2d76-7701-434e-a876-c04ab14cccea.htseq.counts.gz d296b0cb99d14b4b91383… 263970 relea…
>> 5 75c0968c-51aa-4f09-812f-7f85727d1e09 5b23f813-9970-485b-8577-ab2f3c029c26.FPKM-UQ.txt.gz  736a8c6706423a7c1ff40… 559501 relea…
>> 6 57d791ef-c2cc-4963-a618-05f92d5a4436 6aa7225b-3d10-4b19-a472-0adeb21c26cf.FPKM-UQ.txt.gz  8b44a1f944fc78b39e0f9… 536734 relea…
```

可以看到返回的结果中既有 FPKM 又有 FPKM-UQ 和 counts，比较混乱，可以进行进一步的筛选：

``` r
grep_fields("files","workflow")
>>  [1] "analysis.metadata.read_groups.read_group_qcs.workflow_end_datetime"  
>>  [2] "analysis.metadata.read_groups.read_group_qcs.workflow_link"          
>>  [3] "analysis.metadata.read_groups.read_group_qcs.workflow_start_datetime"
>>  [4] "analysis.metadata.read_groups.read_group_qcs.workflow_type"          
>>  [5] "analysis.metadata.read_groups.read_group_qcs.workflow_version"       
>>  [6] "analysis.workflow_end_datetime"                                      
>>  [7] "analysis.workflow_link"                                              
>>  [8] "analysis.workflow_start_datetime"                                    
>>  [9] "analysis.workflow_type"                                              
>> [10] "analysis.workflow_version"                                           
>> [11] "downstream_analyses.workflow_end_datetime"                           
>> [12] "downstream_analyses.workflow_link"                                   
>> [13] "downstream_analyses.workflow_start_datetime"                         
>> [14] "downstream_analyses.workflow_type"                                   
>> [15] "downstream_analyses.workflow_version"

##analysis.workflow_type
files() %>% 
    filter( cases.project.project_id == 'TCGA-OV') %>% 
    filter( type == 'gene_expression') %>% 
    facet('analysis.workflow_type') %>% 
    aggregations() %>% 
    head()
>> $analysis.workflow_type
>>    doc_count                                           key
>> 1       2400                                       DNAcopy
>> 2       1178                                        ASCAT2
>> 3       1051       BWA with Mark Duplicates and Cocleaning
>> 4        998                         BCGSC miRNA Profiling
>> 5        623                                      Liftover
>> 6        610                                       MuTect2
>> 7        610                            MuTect2 Annotation
>> 8        610                                 SomaticSniper
>> 9        610                      SomaticSniper Annotation
>> 10       610                                      VarScan2
>> 11       610                           VarScan2 Annotation
>> 12       606                                          MuSE
>> 13       606                               MuSE Annotation
>> 14       499                                       BWA-aln
>> 15       379                                HTSeq - Counts
>> 16       379                                  HTSeq - FPKM
>> 17       379                               HTSeq - FPKM-UQ
>> 18       379                                   STAR 2-Pass
>> 19         2          MuSE Variant Aggregation and Masking
>> 20         2       MuTect2 Variant Aggregation and Masking
>> 21         2 SomaticSniper Variant Aggregation and Masking
>> 22         2      VarScan2 Variant Aggregation and Masking
>> 23         1                    GISTIC - Copy Number Score
>> 24      3630                                      _missing

##HTSeq - Counts
qfiles = files() %>% filter( ~ cases.project.project_id == 'TCGA-OV' &
                            type == 'gene_expression' &
                            analysis.workflow_type == 'HTSeq - Counts')
manifest_df = qfiles %>% manifest()
>> Rows: 379 Columns: 5
>> ── Column specification ────────────────────────────────────────────────────────────────────────────────────────────────────────
>> Delimiter: "\t"
>> chr (4): id, filename, md5, state
>> dbl (1): size
>> 
>> ℹ Use `spec()` to retrieve the full column specification for this data.
>> ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
nrow(manifest_df)
>> [1] 379
```

## 设置 Token

GDC 的数据分为 `controlled-access` 和 `open` 两种类型；要下载 `controlled-access` 的数据，需要一个 Token 文件，该文件可以从 GDC 官网下载（登录账号之后）：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE%202022-01-08%20134754.png)

然后可以用 `gdc_token()` 函数来导入 Token，该函数以下面三种方式依次来找 token，如果没有找到则报错：

-   以环境变量 `GDC_TOKEN` 保存字符形式的 token
-   以环境变量 `GDC_TOKEN_FILE` 保存的 token 文件名来找 token
-   在在家目录下找以文件名 `.gdc_token` 保存的 token 文件

以第三种为例，将下载的 token 文件重命名为 `.gdc_token`：

``` bash
cp data/gdc-user-token.2022-01-08T06_55_22.799Z.txt .gdc_token
```

``` r
token = gdc_token()
```

## 数据文件下载

### 通过 GDC API 下载数据

`gdcdata` 可以接受含有一个或多个文件 UUID 的字符向量进行文件的下载，生成该字符向量的简单方式就是把上面获得的元数据的第一列拿出来就行：

``` r
fnames = gdcdata(manifest_df$id[1:2],progress=FALSE)
>> Rows: 2 Columns: 5
>> ── Column specification ────────────────────────────────────────────────────────────────────────────────────────────────────────
>> Delimiter: "\t"
>> chr (4): id, filename, md5, state
>> dbl (1): size
>> 
>> ℹ Use `spec()` to retrieve the full column specification for this data.
>> ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
```

如果要下载 control 的数据，则需要提供 token（上面通过 `gdc_token` 得到的字符串）；另外可以使用`BioCParallel` 进行多线程下载来缩短下载的时间。

### 大数据的下载

当需要下载体积巨大的数据，比如测序的 BAM 文件或者 较大的 VCF 文件时，可以使用 `client` 的下载方法：

``` r
fnames = gdcdata(manifest_df$id[3:10], access_method = 'client')
>> Rows: 8 Columns: 5
>> ── Column specification ────────────────────────────────────────────────────────────────────────────────────────────────────────
>> Delimiter: "\t"
>> chr (4): id, filename, md5, state
>> dbl (1): size
>> 
>> ℹ Use `spec()` to retrieve the full column specification for this data.
>> ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.
```

# 总结

这个包下载数据首先是要构建一个 query（请求），比如 `files()` 和 `cases()`，这些函数返回一个 GDCQuery 的 S3 对象，包含 filter, facets，和其他的参数；然后进行筛选，有几个重要的函数：filter 是根据 field 的值来选取记录；select 选择 field；expand 是 field 的聚合，提供了一些相关的 field 组成的 group，便于检索；facts 是统计 common vlaue，类似 table；接着就可以通过 `manifest()` 下载元数据，通过元数据下载真正的数据文件（`gdcdata`）

# 案例：计算 TCGA 样本的测序深度

测序深度的计算公式为：

$$
D=\frac{L *N}{G}
$$
L 是 reads 的长度，N 是 reads 的数目，G 是测序目标区域的长度（比如全外显子，全基因组等）；因此对于 TCGA 的全外显子测序，我们需要知道样本测序读长的长度，reads 的总数目，目标区域就是外显子的长度。

首先需要得到样本测序读长的长度，也就是 reads length，这个信息可以从测序的元数据获取（代码参考[这里](https://support.bioconductor.org/p/107984/)）：

``` r
library(GenomicDataCommons)

##筛选数据，方便演示只选了样本量比较少的ACC,前20个样本
q = files() %>% 
    filter( cases.project.project_id == 'TCGA-ACC') %>%
  filter(data_type == 'Aligned Reads' 
         & experimental_strategy == 'WXS' 
         & data_format == 'BAM') %>% 
  GenomicDataCommons::select('file_id') %>% 
  expand('analysis.metadata.read_groups') 

z = results(q,size=20)

library(dplyr)
>> 
>> Attaching package: 'dplyr'
>> The following objects are masked from 'package:GenomicDataCommons':
>> 
>>     count, filter, select
>> The following objects are masked from 'package:stats':
>> 
>>     filter, lag
>> The following objects are masked from 'package:base':
>> 
>>     intersect, setdiff, setequal, union
t <- z$analysis$metadata$read_groups
names(t) <- z$file_id
a <- t %>% bind_rows(.id = "file_ids") %>% as_tibble()
head(a)
>> # A tibble: 6 × 19
>>   file_ids   sequencing_date experiment_name  target_capture_… submitter_id  target_capture_kit_… is_paired_end target_capture_…
>>   <chr>      <chr>           <chr>            <chr>            <chr>         <chr>                <lgl>         <chr>           
>> 1 99818fa6-… 2013-09-12T08   TCGA-OR-A5KO-01… NimbleGen        efc0de29-4b5… http://www.nimblege… TRUE          06 465 668 001  
>> 2 a9f11831-… 2013-08-05T02   TCGA-OR-A5LD-10… NimbleGen        afc22b50-442… http://www.nimblege… TRUE          06 465 668 001  
>> 3 1a14bcc9-… 2013-08-12T02   TCGA-OR-A5JJ-10… NimbleGen        17fa8960-7a4… http://www.nimblege… TRUE          06 465 668 001  
>> 4 f3af7cc5-… 2013-08-12T03   TCGA-OR-A5JV-10… NimbleGen        b7c4c670-624… http://www.nimblege… TRUE          06 465 668 001  
>> 5 870c0429-… 2013-07-29T11   TCGA-OR-A5LP-01… NimbleGen        b99240a5-473… http://www.nimblege… TRUE          06 465 668 001  
>> 6 b1da0e48-… 2013-08-06T11   TCGA-OR-A5L2-01… NimbleGen        b99bd79e-aa3… http://www.nimblege… TRUE          06 465 668 001  
>> # … with 11 more variables: library_strategy <chr>, platform <chr>, created_datetime <chr>, updated_datetime <chr>,
>> #   read_group_name <chr>, library_name <chr>, target_capture_kit_name <chr>, sequencing_center <chr>, state <chr>,
>> #   read_length <int>, read_group_id <chr>

##整理结果
re <- a %>% 
  dplyr::select(experiment_name,read_length) %>% 
  mutate(sample=substr(experiment_name,1,16)) %>% 
  distinct(sample,read_length,.keep_all = T) %>% 
  dplyr::filter(grepl("TCGA",sample)) %>% 
  dplyr::filter(as.numeric(substr(sample,14,15)) <= 9)

table(re$read_length) ##read length都是101
>> 
>> 101 
>>  13
```

然后需要根据 BAI 文件得到 reads 总数目（源代码来自[Coverage data for TCGA BAM files (biostars.org)](https://www.biostars.org/p/9472253/)）：

``` r
#get BAM file manifest
manifest = GenomicDataCommons::files() %>%  
  GenomicDataCommons::filter( cases.project.project_id == 'TCGA-ACC') %>% 
  GenomicDataCommons::filter(experimental_strategy == "WXS" &
                               data_format == "BAM") %>%   
  GenomicDataCommons::manifest()
>> Rows: 184 Columns: 5
>> ── Column specification ────────────────────────────────────────────────────────────────────────────────────────────────────────
>> Delimiter: "\t"
>> chr (4): id, filename, md5, state
>> dbl (1): size
>> 
>> ℹ Use `spec()` to retrieve the full column specification for this data.
>> ℹ Specify the column types or set `show_col_types = FALSE` to quiet this message.

dt <- manifest %>% 
  mutate(tmp=gsub(".+[.]TCGA","TCGA",filename)) %>% 
  mutate(sample=substr(tmp,1,16)) %>% 
  dplyr::filter(!grepl("hg19",filename)) %>% 
  dplyr::filter(sample %in% re$sample)
```

由于 BAM 文件较大，而计算 mapped 的总 reads 数只需要 BAM 的索引文件（也就是 BAI 文件），因此我们下载 BAI 文件就行，但是 TCGA 并没有直接提供 BAI 文件的 UUID，因此我们需要进行进一步的处理。根据 GDC
官方的[文档](https://gdc.cancer.gov/about-gdc/gdc-faqs)显示，在用 api 下载 BAM 文件时在末尾加上 `?pretty=true&expand=index_files`就可以得到一个含有 BAI 的 UUID 的 JSON 文件：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220108143545648.png)

因此我们可以利用这个特征来得到 BAI 的 UUID：

``` r
res <- vector("list",length(unique(dt$id)))
for (i in 1:length(res)){
  con = curl::curl(paste0("https://api.gdc.cancer.gov/files/", dt$id[i], "?pretty=true&expand=index_files"))
  tbl = jsonlite::fromJSON(con)
  bai = data.frame(id = tbl$data$index_files$file_id,
                   filename = tbl$data$index_files$file_name,
                   md5 = tbl$data$index_files$md5sum,
                   size = tbl$data$index_files$file_size,
                   state = tbl$data$index_files$state)
  res[[i]] <- bai
  cat("complete",i,"\n")
}
>> complete 1 
>> complete 2 
>> complete 3 
>> complete 4 
>> complete 5 
>> complete 6 
>> complete 7 
>> complete 8 
>> complete 9 
>> complete 10 
>> complete 11 
>> complete 12 
>> complete 13
re <- bind_rows(res)
re
>>                                      id                                            filename                              md5
>> 1  a16a54a0-cd4a-43b7-8272-2565b0930bda TCGA-OR-A5KO-01A-11D-A29I-10_Illumina_gdc_realn.bai 8c6d16ebde6d101530ea3dcb7111fd9d
>> 2  327a4450-1c36-4677-9288-4a54a7d0d842 TCGA-OR-A5LP-01A-11D-A29I-10_Illumina_gdc_realn.bai 0214ad9219ec0df5d324682b291427a7
>> 3  4278fd34-4f28-4768-bb8e-c9388a702d39 TCGA-OR-A5L2-01A-11D-A30A-10_Illumina_gdc_realn.bai 271381422b85433078fc95b9f5018dbd
>> 4  bf2a9bea-c1d6-41c4-a11c-acfb24916c4c TCGA-PK-A5HC-01A-11D-A30A-10_Illumina_gdc_realn.bai 70c691fc38702fad1df7f10c97e0ce3c
>> 5  8047e7a5-4eaf-4518-9b8a-76c3c5bbd0c3 TCGA-OR-A5JR-01A-11D-A29I-10_Illumina_gdc_realn.bai 332d7af2eb8c669670f55ab688e27f45
>> 6  8e4e4082-9856-49c8-8d69-24ac7351014a TCGA-OR-A5LS-01A-11D-A29I-10_Illumina_gdc_realn.bai be312d99dfcd1efe4a98814c0a1a48ed
>> 7  907e80da-c646-414f-8d98-d5e3e7b4686f TCGA-OR-A5KZ-01A-11D-A29I-10_Illumina_gdc_realn.bai 574fec9fed318557d84435b043d947bf
>> 8  29daa003-4169-41cf-8452-83c06b6730e0 TCGA-OR-A5LA-01A-11D-A29I-10_Illumina_gdc_realn.bai c89f2e68dbf5bbf4cb1071ec7dbacfee
>> 9  7f59ebcb-a57d-458d-ba7b-9b77abb1f3f7 TCGA-OR-A5LK-01A-11D-A29I-10_Illumina_gdc_realn.bai 9330ad033f2eabb6909d3e834d80d14e
>> 10 00fb3422-6608-44ec-9afe-39e7bb0b52d4 TCGA-OR-A5L5-01A-11D-A29I-10_Illumina_gdc_realn.bai 67feea2ef796f4dc872f3d344b59880e
>> 11 c4595133-679a-4967-beb3-df0d9867ffa6 TCGA-OR-A5JJ-01A-11D-A29I-10_Illumina_gdc_realn.bai 303acd0b62ac94d46ce8ca7b46b0b874
>> 12 8fab8841-35f4-4c1e-a3f2-83a8d3fb4ab6 TCGA-OU-A5PI-01A-12D-A29I-10_Illumina_gdc_realn.bai e39845ba41c2ebe18f9bf9639a6962af
>> 13 950a115d-8102-47bc-aec2-5c3425a929fd TCGA-OR-A5JC-01A-11D-A29I-10_Illumina_gdc_realn.bai 88a3e25b655dcb9e3da954fd2b264192
>>       size    state
>> 1  6810304 released
>> 2  6797480 released
>> 3  7075944 released
>> 4  6882408 released
>> 5  6929096 released
>> 6  6898896 released
>> 7  8387160 released
>> 8  6714080 released
>> 9  6785888 released
>> 10 6834032 released
>> 11 6675856 released
>> 12 6776432 released
>> 13 6707696 released
# write.table(re,file = "~/data/TCGA_bai_manifest.txt",sep = "\t",row.names = F,quote = F)
```

然后基于 BAI 的 UUID 来下载 BAI 文件（这里还是使用了 gdc-client 来下载）：

``` bash
../gdc-client download -m ../../TCGA_ACC_bai_manifest.txt -t ../gdc-user-token.2022-01-08T06_55_22.799Z.txt

ls > files
mkdir BAI
cat files | while read i;do cp ./$i/*.bai ./BAI/;done
ls BAI/
#TCGA-OR-A5JC-01A-11D-A29I-10_Illumina_gdc_realn.bai  TCGA-OR-A5LA-01A-11D-A29I-10_Illumina_gdc_realn.bai
#TCGA-OR-A5JJ-01A-11D-A29I-10_Illumina_gdc_realn.bai  TCGA-OR-A5LK-01A-11D-A29I-10_Illumina_gdc_realn.bai
#TCGA-OR-A5JR-01A-11D-A29I-10_Illumina_gdc_realn.bai  TCGA-OR-A5LP-01A-11D-A29I-10_Illumina_gdc_realn.bai
#TCGA-OR-A5KO-01A-11D-A29I-10_Illumina_gdc_realn.bai  TCGA-OR-A5LS-01A-11D-A29I-10_Illumina_gdc_realn.bai
#TCGA-OR-A5KZ-01A-11D-A29I-10_Illumina_gdc_realn.bai  TCGA-OU-A5PI-01A-12D-A29I-10_Illumina_gdc_realn.bai
#TCGA-OR-A5L2-01A-11D-A30A-10_Illumina_gdc_realn.bai  TCGA-PK-A5HC-01A-11D-A30A-10_Illumina_gdc_realn.bai
#TCGA-OR-A5L5-01A-11D-A29I-10_Illumina_gdc_realn.bai
```

再利用 `samtools` 的 idxstats 功能统计 reads（这里随便拿一个 BAM 文件就行）：

``` bash
ls *.bai > bai_files

#!/bin/bash
cat bai_files | while read i
do
  newname=`basename $i .bai`
  mv ../BAI/dummy.bam ../BAI/$newname.bam
  samtools idxstats ../BAI/$newname.bam > ./stat_files/$newname.txt
  mv ../BAI/$newname.bam ../BAI/dummy.bam
done
```

``` bash
head ~/data/TCGA_bai/test/stat_files/TCGA-OR-A5JC-01A-11D-A29I-10_Illumina_gdc_realn.txt
>> chr1 248956422   8655782 9484
>> chr10    133797422   6282801 6836
>> chr11    135086622   4863250 5439
>> chr11_KI270721v1_random  100316  3985679 4321
>> chr12    133275309   7341751 8237
>> chr13    114364328   5048809 5472
>> chr14    107043718   4231568 4745
>> chr14_GL000009v2_random  201709  3673268 4058
>> chr14_GL000225v1_random  211173  3236746 3564
>> chr14_KI270722v1_random  194050  4261146 4737
```

得到的文件的第三列就是各个染色体 mapped 的总 reads，然后就可以利用上面的公式进行计算测序深度了：

``` r
files <- list.files("~/data/TCGA_bai/test/stat_files/")
dt <- data.frame(file=files)
dt$samples <- gsub(".+[.]TCGA","TCGA",files) %>% 
  gsub("[.][0-9].+[_gdc_realn.txt]","",.) %>% 
  substr(.,1,16)
dt$depth <- NA
for (i in 1:nrow(dt)){
  a <- data.table::fread(paste0("~/data/TCGA_bai/test/stat_files/",dt$file[i]),data.table = F)
  depth <- ((sum(a$V3))/(38000000)) * 101
  dt$depth[i] <- depth
}
dt
>>                                                   file          samples    depth
>> 1  TCGA-OR-A5JC-01A-11D-A29I-10_Illumina_gdc_realn.txt TCGA-OR-A5JC-01A 253.7207
>> 2  TCGA-OR-A5JJ-01A-11D-A29I-10_Illumina_gdc_realn.txt TCGA-OR-A5JJ-01A 242.3565
>> 3  TCGA-OR-A5JR-01A-11D-A29I-10_Illumina_gdc_realn.txt TCGA-OR-A5JR-01A 290.4985
>> 4  TCGA-OR-A5KO-01A-11D-A29I-10_Illumina_gdc_realn.txt TCGA-OR-A5KO-01A 304.1783
>> 5  TCGA-OR-A5KZ-01A-11D-A29I-10_Illumina_gdc_realn.txt TCGA-OR-A5KZ-01A 463.0472
>> 6  TCGA-OR-A5L2-01A-11D-A30A-10_Illumina_gdc_realn.txt TCGA-OR-A5L2-01A 301.6314
>> 7  TCGA-OR-A5L5-01A-11D-A29I-10_Illumina_gdc_realn.txt TCGA-OR-A5L5-01A 283.6341
>> 8  TCGA-OR-A5LA-01A-11D-A29I-10_Illumina_gdc_realn.txt TCGA-OR-A5LA-01A 195.8037
>> 9  TCGA-OR-A5LK-01A-11D-A29I-10_Illumina_gdc_realn.txt TCGA-OR-A5LK-01A 267.7285
>> 10 TCGA-OR-A5LP-01A-11D-A29I-10_Illumina_gdc_realn.txt TCGA-OR-A5LP-01A 272.8540
>> 11 TCGA-OR-A5LS-01A-11D-A29I-10_Illumina_gdc_realn.txt TCGA-OR-A5LS-01A 313.4699
>> 12 TCGA-OU-A5PI-01A-12D-A29I-10_Illumina_gdc_realn.txt TCGA-OU-A5PI-01A 260.9008
>> 13 TCGA-PK-A5HC-01A-11D-A30A-10_Illumina_gdc_realn.txt TCGA-PK-A5HC-01A 329.6680
```
