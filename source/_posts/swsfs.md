---
title: 滑窗顺序前向特征选择SWSFS
date: 2022-06-12 19:14:18
tags: paper
index_img: img/swsfs.png
categories:
  - 机器学习
---



滑窗顺序前向特征选择（sliding window sequential forward feature selection，SWSFS）

<!-- more -->

SWSFS 来自文献 [A random forest approach to the detection of epistatic interactions in case-control studies ](A random forest approach to the detection of epistatic interactions in case-control studies)。这篇文献研究的是 SNP 之间的上位相互作用

Epistatic 指的是上位性，不同突变的表型效应之间相互作用，合成致死也是一种上位性。

> Life would have been much simpler, and perhaps even boring, if epistasis were completely absent. In reality, however, epistasis abounds, rendering biology full of surprises and complexity. For instance, a commonly encountered type of epistasis is synthetic lethality, where simultaneously deleting two genes from the genome of a normal organism is lethal despite the fact that deleting each of them separately is viable

> Epistasis, a term coined by William Bateson in 1909 [[1](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5313148/#pgen.1006558.ref001)], refers to the interdependence of mutations in their phenotypic effects. Let the phenotypic value of a trait relative to that of the wild type be *f*~A~ and *f*~B~ for mutants A and B, respectively, and let the phenotypic value of the corresponding double mutant be *f* ~AB~ . Although variation exists, epistasis is usually defined by ε = *f*~AB~ − *f*~A~*f*~B~ and is said to be positive when ε > 0 and negative when ε < 0. -- **Epistasis Analysis Goes Genome-Wide**

从二分类问题去研究 case-control 数据，将 case 认为是 positive 样本，将 control 认为是 negative 样本，SNP marker 当作是分类变量（3个可能的值代表 3 个基因型），使用随机森林模型进行分类。首先使用所有的 snp 进行训练得到每个 SNP 的 gini  importance，然后使用 SWSFS（sliding window sequential forward feature selection）选择能够最小化分类误差的 SNP 子集，最后对于这一较小的子集可以使用穷举的方法研究所有可能的 SNP 相关作用。

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220611150645-tgo03fx.png" style="zoom:50%;" />



关于决策树和随机森林可以看 [1](https://wutaoblog.com.cn/2021/08/24/hands_on_ml_ch7/#%E9%9A%8F%E6%9C%BA%E6%A3%AE%E6%9E%97) 和 [2](https://wutaoblog.com.cn/2021/03/04/hands_on_ml_ch6/) ，随机森林的特点在于

* 对于每个树的训练集都是通过对总的样本有放回的随机抽样得到的（[bagging](https://wutaoblog.com.cn/2021/08/24/hands_on_ml_ch7/#Bagging-and-Pasting) 方法）
* 在使用 bagging 时，有些实例可能会被抽到多次而另一些实例可能根本不会被抽到，平均来说，对于每个预测器大概只有 63% 的实例被抽到，剩下的 37% 的实例就叫做 out-of-bag (oob) 实例，在这些样本上预测就可以得到 OBB 误差
* 随机森林中的决策树是没有修枝的
* 在每个节点的数据集分割中是随机选择特征来构建决策树

随机森林的可调的超参数主要有 3 个：`mtry` 表示在每个节点处随机选择的特征数量，`trees` 表示森林中的树数量，`min_n` 表示一个节点最小样本量（大于这个样本量才能继续分割）。随机森林使用 gini 重要性来衡量特征的重要性：

$$
GI(v)=\sum_{T\in \bold T}\sum_{\eta\in N_T}\Delta \Phi(\eta)I(\eta,v)
$$

 $v$ 是特征，$\eta$ 是节点，$I$ 是指示函数，当 $v$ 是 $\eta$ 的分割变量时为 1 ，否则为0；$\Delta \Phi(\eta)$ 表示在该节点使用这个变量分割后 gini 不纯度的下降（gini decrease），因此某个特征的 gini 重要性为在森林中以该特征为分割变量的所有节点的 gini decrease 的加和。

设 $M=\{m_1,...,m_L\}$ 为所有的变量，首先对所有的数据训练随机森林模型得到每个变量的 gini 重要性 $G=\{g1,...,g_L\}$ 并对这些变量按照 gini 重要性从大到小排序得到 $O=\{o_1,...,o_L\}$ ，接下来按照 SWSFS 算法选择使得分类误差最小的子集：

SWSFS 算法：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220612152028-1rv05j2.png" style="zoom:50%;" />



这个过程可用下图来表示：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220612154048-gtlt3ri.png" style="zoom:67%;" />



比如 i 等于 25 的时候，计算重要性排名前 25 的变量构建的随机森林模型的 error 并保存到 Error 变量中（Error 变量中已经存储了前面 1 个变量，2 个变量直到 24 个变量构建的模型的 OBB 分类误差），然后查看利用前 5 个变量（25-20=5）构建的模型误差是否是这 20 个误差中最小的（窗口大小），如果是最小的就把这个窗口左边界的变量加入候选变量集合，如果不是则移到窗口进行下一步计算。

下面以 TCGA 内膜癌基因表达数据为例来使用 SWSFS 筛选和 OS 相关的基因集合。首先需要在所有的数据上训练随机森林模型：

```r
library(dplyr)
library(tidymodels)
library(vip)
##读入数据
fpkm <- data.table::fread("~/data/edo_project/TCGA-UCEC.htseq_fpkm-uq.tsv.gz",data.table = F)
which(substr(colnames(fpkm),14,15) < 10) -> a ##癌症样本
##基因 ID 转化
fpkm <- fpkm %>%
  dplyr::select(Ensembl_ID,all_of(a)) %>%
  rename(id=Ensembl_ID)
mapping <- data.table::fread("~/data/edo_project/gencode.v22.annotation.gene.probeMap",data.table = F)
fpkm <- left_join(
  fpkm,mapping %>% select(id,gene)
) %>% select(gene,everything()) %>% select(-id)
fpkm <- fpkm[!duplicated(fpkm$gene),]
rownames(fpkm) <- fpkm$gene
fpkm <- fpkm %>% dplyr::select(-gene)

##根据 MAD 过滤基因，只选MAD 前 1000 的基因作为示例
mad_gene <- apply(fpkm,1,mad)
top_mad <- mad_gene %>% sort(.,decreasing=TRUE)
top_mad <- top_mad[1:1000]
fpkm <- fpkm[names(top_mad),]

##合并OS生存数据
sample_dt <- as.data.frame(t(fpkm))
sample_dt$sample <- substr(rownames(sample_dt),1,12)
sample_dt <- sample_dt[!duplicated(sample_dt$sample),]
surv <- readRDS("~/Immunoediting/data/pancancer_survial.rds")
sample_dt <- left_join(sample_dt,surv %>% select(sample,OS))
sample_dt <- sample_dt %>% select(-sample)
sample_dt$OS <- as.factor(sample_dt$OS)

###构建随机森林模型
tree_rec <- recipe(OS ~ ., data = sample_dt)
tune_spec <- rand_forest(
  mtry = round(sqrt(1000)),
  trees = tune(),
  min_n = tune()
) %>%
  set_mode("classification") %>%
  set_engine("randomForest")

tune_wf <- workflow() %>%
  add_recipe(tree_rec) %>%
  add_model(tune_spec)
##交叉验证
set.seed(2022061102)
trees_folds <- vfold_cv(sample_dt)
 
#对 min_n, trees 进行 grid search
doParallel::registerDoParallel(cores=30)
set.seed(2022061103)
tune_res <- tune_grid(
  tune_wf,
  resamples = trees_folds,
  grid = 20
)
##画个图
tune_res %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  select(mean, min_n, trees) %>%
  pivot_longer(min_n:trees,
               values_to = "value",
               names_to = "parameter"
  ) %>%
  ggplot(aes(value, mean, color = parameter)) +
  geom_point(show.legend = FALSE) +
  facet_wrap(~parameter, scales = "free_x") +
  labs(x = NULL, y = "AUC")

##选择合适的范围进行调参
rf_grid <- grid_regular(
  trees(range = c(750, 1000)),
  min_n(range = c(20, 30)),
  levels = 5
)
set.seed(2022061104)
doParallel::registerDoParallel(cores=30)
regular_res <- tune_grid(
  tune_wf,
  resamples = trees_folds,
  grid = rf_grid
)

##基于 ROAUC 选择最佳模型
best_auc  <- select_best(regular_res, "roc_auc")
final_rf <- finalize_model(
  tune_spec,
  best_auc
)
##变量重要性
tree_prep <- prep(tree_rec)
juiced <- juice(tree_prep)
set.seed(2022061105)
model_fit <- final_rf  %>%
  set_engine("randomForest", importance = T) %>%
  fit(OS ~ .,
      data = juice(tree_prep)
  )
model_fit %>%
  vip(geom = "point")

library(randomForestExplainer)
impt_frame <- measure_importance(model_fit$fit,measures="gini_decrease")
```

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220612170401-wrq5tfx.png" style="zoom:50%;" />



<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220612170423-skaftdf.png" style="zoom:50%;" />



将上面的代码包装成函数以便于在 SWSFS 中使用来计算 OBB 误差：

```r
get_error <- function(data_set,features){
  tmp_sample <- data_set %>% select(OS,all_of(features))
  tree_rec <- recipe(OS ~ ., data = tmp_sample)
  tune_spec <- rand_forest(
    mtry = round(sqrt(length(features))),
    trees = tune(),
    min_n = tune()
  ) %>%
    set_mode("classification") %>%
    set_engine("randomForest")

  tune_wf <- workflow() %>%
    add_recipe(tree_rec) %>%
    add_model(tune_spec)
  trees_folds <- vfold_cv(sample_dt)
  #grid search for trees and min_n
  message("Grid search for trees and min_n ... ")
  doParallel::registerDoParallel(cores=30)
  tune_res <- tune_grid(
    tune_wf,
    resamples = trees_folds,
    grid = 20
  )
  select_par <- tune_res %>%
    collect_metrics() %>%
    filter(.metric == "roc_auc") %>%
    select(mean, min_n, trees) %>%
    arrange(desc(mean))

  ##按照前5个选调参范围
  rf_grid <- grid_regular(
    trees(range = c(min(select_par$trees[1:5]), max(select_par$trees[1:5]))),
    min_n(range = c(min(select_par$min_n[1:5]), max(select_par$min_n[1:5]))),
    levels = 5
  )

  doParallel::registerDoParallel(cores=30)
  regular_res <- tune_grid(
    tune_wf,
    resamples = trees_folds,
    grid = rf_grid
  )
  ##select best model
  best_auc  <- select_best(regular_res, "roc_auc")
  final_rf <- finalize_model(
    tune_spec,
    best_auc
  )
  message("Search done ! The best model is \n")
  print(final_rf)
  ##error
  tree_prep <- prep(tree_rec)
  juiced <- juice(tree_prep)
  model_fit <- final_rf  %>%
    set_engine("randomForest", importance = T) %>%
    fit(OS ~ .,
        data = juice(tree_prep)
    )

  obb_error <- model_fit$fit$err.rate[dim(model_fit$fit$err.rate)[1]]
  return(obb_error)
}
```

接着就可以基于上面的伪代码实现 SWSFS：

```r
##选前50个作为实例
impt_frame <- impt_frame %>% slice_max(order_by = gini_decrease,n=50)
i <- 1
k <- 50
w <- 20
Error <- c(1:50)
var_set <- vector("character")
while (i <= 50) {
  Error[i] <- get_error(data_set = sample_dt,features = impt_frame$variable[1:i])
  if((i > w) && (1 == which.min(Error[(i-w):i]))){
    k <- i - w
    message("Add ",impt_frame$variable[k],"! \n")
    var_set <- append(var_set,impt_frame$variable[k])
  }
  i <- i + 1
  message(paste0(i," Complete! \n"))
}

```

