---
title: 分类模型评估指标
author: wutao
date: 2021-11-06 10:00:00
categories:
  - 机器学习
index_img: img/class.jpg
---







分类模型评估的基本概念

<!-- more -->

机器学习的一个重要步骤是模型性能的评估，特别是分类模型，有一些概念容易混淆。这里做一些记录并用 R/python 进行简单实现。

分类模型评估的指标大都是根据混淆矩阵来计算的，对于二分类问题，混淆矩阵是将模型预测结果和真实结果以 2 × 2 的列联表的形式展示，从而比较分类结果和实例的真实信息。矩阵的行表示预测类别的数量，列表示真实类别的数量：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20211005154111695.png)

从这个混淆矩阵中我们可以直接得到假阳性率，真阳性率，假阴性率和真阴性率：

假阳性率（FPR）：实际是阴性的，但是预测出来是阳性的比率

$$
FPR = \frac{FP}{FP+TN}
$$
真阳性率（TPR）：实际是阳性的，预测也是阳性的比例

$$
TPR = \frac{TP}{TP+FN}
$$
假阴性率（FNR）：实际是阳性的，但是预测是阴性的比例

$$
FNR = \frac{FN}{FN+TP}
$$
真阴性率（TNR）：实际是阴性，预测也是阴性的比例：

$$
TNR = \frac{TN}{TN+FP}
$$

## Accuracy

精度（accuracy）衡量的是所有样本中正确分类（包括正类和负类）的比例：

$$
ACC = \frac{TP + TN }{TP + FP + TN + FN}
$$
但是在类别非平衡的数据集中使用精度会带来问题，因为将数据归类到多数类就会得到高的精度。

`yardstick` 包是 `tidymodel` 系列中用来进行模型评估的包，可以使用这个包中的一系列函数进行分类模型评估指标的计算：

``` r
library(yardstick)
library(dplyr)
dt <- two_class_example
head(dt)
>>    truth      Class1       Class2 predicted
>> 1 Class2 0.003589243 0.9964107574    Class2
>> 2 Class1 0.678621054 0.3213789460    Class1
>> 3 Class2 0.110893522 0.8891064779    Class2
>> 4 Class1 0.735161703 0.2648382969    Class1
>> 5 Class2 0.016239960 0.9837600397    Class2
>> 6 Class1 0.999275071 0.0007249286    Class1


accuracy(dt, truth = truth, estimate = predicted)
>> # A tibble: 1 x 3
>>   .metric  .estimator .estimate
>>   <chr>    <chr>          <dbl>
>> 1 accuracy binary         0.838
```

从上面的公式中也可以看出，精度是根据预测的标签和真实的标签进行计算的，所以如果模型输出的是概率值，那么选择不同的阈值也会得到不同的精度：

``` r
threshold <- seq(0,1,0.01)
df <- data.frame(threshold=threshold,acc=NA)
df$acc <- sapply(df$threshold,
                 function(x){
                   dt %>% 
                     mutate(predicted=ifelse(Class1<x,"Class2","Class1")) %>% 
                     mutate(predicted=factor(predicted,levels = c("Class1","Class2"))) %>% 
                     accuracy(.,truth=truth,estimate=predicted) %>% 
                     select(.estimate) %>% as.numeric()
                 })

library(ggplot2)
library(ggprism)
ggplot(data=df,aes(x=threshold,y=acc))+
  geom_line()+
  theme_prism()
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/unnamed-chunk-2-1.png)

在 python 中使用 `scikit-learn` 也可以方便的计算一系列的分类评估指标：

``` python
from sklearn.metrics import confusion_matrix, accuracy_score

y_pred_class = r.dt["predicted"]
y_true  = r.dt["truth"]
tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()
accuracy = (tp + tn) / (tp + fp + fn + tn)

# or simply

accuracy_score(y_true, y_pred_class)
>> 0.838
```

## Precision、Recall (Sensitivity)、Specificity，F1 score

Precision 表示预测出来的正类中有多少是真实正类，即该分类器预测的正类有多少是准确的；Recall (Sensitivity，也叫真阳性率 TPR) 表示真实正类有多少被分类器预测出来，即真实的正类有多少被该分类器“召回” (比如所有实际患癌的人群中检测出阳性的比例，代表了模型对患者的检出能力水平)：
$$
Precision = \frac{TP}{TP+FP}
$$
$$
Recall = \frac{TP}{TP+FN}
$$
Specificity 关注的是负类样本，表示的是真实的负类中有多少是被预测为负类（比如所有实际未患癌的人群中检测出阴性的比例，代表了模型对健康人群的排除能力水平），也就是真阴性率（TNR）：

$$
Specificity = \frac{TN}{FP+TN}
$$
F1 score 是对 precision 和 recall 的调和，F1 其实是 F-beta 的一种特殊情况 (beta=1):

$$
F\_{beta} = (1+\beta ^2)\frac{precison \times recall }{\beta ^2 \times precision + recall}
$$

`yardstick` 中的 `sens()`，`spec()`，`recall()`，`precision()`，可以用来计算上面的指标：

``` r
###sensitivity
sens(dt,truth = truth,estimate = predicted)
>> # A tibble: 1 x 3
>>   .metric .estimator .estimate
>>   <chr>   <chr>          <dbl>
>> 1 sens    binary         0.880
##和recall一样
recall(dt,truth = truth,estimate = predicted)
>> # A tibble: 1 x 3
>>   .metric .estimator .estimate
>>   <chr>   <chr>          <dbl>
>> 1 recall  binary         0.880
##两个向量形式
sens_vec(dt$truth,dt$predicted)
>> [1] 0.879845

###Specificity
spec(dt,truth = truth,estimate = predicted)
>> # A tibble: 1 x 3
>>   .metric .estimator .estimate
>>   <chr>   <chr>          <dbl>
>> 1 spec    binary         0.793

###precision
precision(dt,truth = truth,estimate = predicted)
>> # A tibble: 1 x 3
>>   .metric   .estimator .estimate
>>   <chr>     <chr>          <dbl>
>> 1 precision binary         0.819
```

`f_meas` 可以用来计算 F-beta，其中的 *β* 参数指定上面公式中的 *β* 值，表示给 recall 的权重是 precision 的 *β* 倍:

``` r
f_meas(dt,truth,predicted)
>> # A tibble: 1 x 3
>>   .metric .estimator .estimate
>>   <chr>   <chr>          <dbl>
>> 1 f_meas  binary         0.849
```

对于 python，`metrics` 模块中的一些方法可以计算上述的指标：

``` python
from sklearn import metrics
y_pred = [0,1,0,0]
y_true = [0,1,0,1]
metrics.precision_score(y_true,y_pred)
>> 1.0
metrics.recall_score(y_true, y_pred)
>> 0.5
metrics.f1_score(y_true, y_pred)
>> 0.6666666666666666
metrics.fbeta_score(y_true, y_pred, beta=1)
>> 0.6666666666666666
```

## ROC, PRC

ROC 曲线的横纵坐标分别是假阳性率（FPR，也叫 1 - Specificity）和真阳性率（TPR，recall 或 sensitivity）；PRC 曲线的的横纵坐标分别是 recall 和 precision；这两种曲线都是在不同的预测概率阈值下计算横纵坐标的值，然后绘制相应的曲线。

R 里面可以使用 `roc_curve` 计算不同阈值处的 TPR 和 FPR，然后再使用 ggplot2 可视化，也可以直接用 `autoplot` 函数一步到位：

``` r
roc_curve(two_class_example, truth, Class1)
>> # A tibble: 502 x 3
>>    .threshold specificity sensitivity
>>         <dbl>       <dbl>       <dbl>
>>  1 -Inf           0                 1
>>  2    1.79e-7     0                 1
>>  3    4.50e-6     0.00413           1
>>  4    5.81e-6     0.00826           1
>>  5    5.92e-6     0.0124            1
>>  6    1.22e-5     0.0165            1
>>  7    1.40e-5     0.0207            1
>>  8    1.43e-5     0.0248            1
>>  9    2.38e-5     0.0289            1
>> 10    3.30e-5     0.0331            1
>> # ... with 492 more rows

library(ggplot2)
roc_curve(two_class_example, truth, Class1) %>%
  ggplot(aes(x = 1 - specificity, y = sensitivity)) +
  geom_path() +
  geom_abline(lty = 3) +
  coord_equal() +
  theme_bw()
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/unnamed-chunk-7-1.png)

``` r
autoplot(roc_curve(two_class_example, truth, Class1))
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/unnamed-chunk-7-2.png)

PR 曲线也是类似，使用函数 `pr_curve` ：

``` r
pr_curve(two_class_example, truth, Class1)
>> # A tibble: 501 x 3
>>    .threshold  recall precision
>>         <dbl>   <dbl>     <dbl>
>>  1     Inf    0               1
>>  2       1.00 0.00388         1
>>  3       1.00 0.00775         1
>>  4       1.00 0.0116          1
>>  5       1.00 0.0155          1
>>  6       1.00 0.0194          1
>>  7       1.00 0.0233          1
>>  8       1.00 0.0271          1
>>  9       1.00 0.0310          1
>> 10       1.00 0.0349          1
>> # ... with 491 more rows
pr_curve(two_class_example, truth, Class1) %>%
  ggplot(aes(x = recall, y = precision)) +
  geom_path() +
  coord_equal() +
  theme_bw()
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/unnamed-chunk-8-1.png)

``` r
autoplot(pr_curve(two_class_example, truth, Class1))
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/unnamed-chunk-8-2.png)

除了画曲线之外，也可以使用曲线下面积来比较两个不同模型的预测性能，使用的函数是 `roc_auc` 和 `pr_auc`：

``` r
roc_auc(two_class_example, truth, Class1)
>> # A tibble: 1 x 3
>>   .metric .estimator .estimate
>>   <chr>   <chr>          <dbl>
>> 1 roc_auc binary         0.939
pr_auc(two_class_example, truth, Class1)
>> # A tibble: 1 x 3
>>   .metric .estimator .estimate
>>   <chr>   <chr>          <dbl>
>> 1 pr_auc  binary         0.946
```

Python 的 sklearn.metrics 模块中提供了计算 ROC 和 PR 曲线以及曲线下面积的方法：

``` python
##roc_curve 会返回FPR, TPR以及相应的阈值
##roc_auc_score 计算ROC 曲线下面积
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
y = np.array([1, 1, 2, 2])
scores = np.array([0.1, 0.4, 0.35, 0.8])##正类的概率或者由classifier.decision_function返回的决策函数的值
fpr, tpr, thresholds = roc_curve(y, scores, pos_label=2)##当类别是{-1, 1} 或者 {0, 1} 时，不需要设定pos_label，默认是 1 类，不然就需要设定

roc_auc = roc_auc_score(y, scores)##scores 是较大类的概率

lw = 2
plt.plot(
    fpr,
    tpr,
    color="darkorange",
    lw=lw,
    label="ROC curve (area = %0.2f)" % roc_auc,
)
>> [<matplotlib.lines.Line2D object at 0x0000013642AC81C0>]
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
>> [<matplotlib.lines.Line2D object at 0x0000013642AC8610>]
plt.xlim([0.0, 1.0])
>> (0.0, 1.0)
plt.ylim([0.0, 1.05])
>> (0.0, 1.05)
plt.xlabel("False Positive Rate")
>> Text(0.5, 0, 'False Positive Rate')
plt.ylabel("True Positive Rate")
>> Text(0, 0.5, 'True Positive Rate')
plt.title("Receiver operating characteristic example")
>> Text(0.5, 1.0, 'Receiver operating characteristic example')
plt.legend(loc="lower right")
>> <matplotlib.legend.Legend object at 0x0000013642412D00>
plt.show()
```

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/unnamed-chunk-3-1.png" width="672" />

``` python
###一个逻辑回归的例子
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
X, y = load_breast_cancer(return_X_y=True)
clf = LogisticRegression(solver="liblinear").fit(X, y)
clf.classes_
>> array([0, 1])
y_score = clf.predict_proba(X)[:, 1]
roc_auc_score(y, y_score)
>> 0.9948073569050262
roc_auc_score(y, clf.decision_function(X))
>> 0.9948073569050262
```

``` python
##precision_recall_curve 会返回precision, recall以及相应的阈值
##average_precision_score 计算的就是PR 曲线的 AUC
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
y = np.array([1, 1, 2, 2])
scores = np.array([0.1, 0.4, 0.35, 0.8])
precision, recall, thresholds = precision_recall_curve(y_true, scores)
precision
>> array([1., 1., 1.])
recall
>> array([1. , 0.5, 0. ])
thresholds
>> array([0.4, 0.8])
average_precision_score(y,scores)
>> 0.5
```

对于绘制 PR 曲线， scikit-learn 提供了两个函数：

-   PrecisionRecallDisplay.from_estimator：可以输入分类器
-   PrecisionRecallDisplay.from_predictions：输入分类器的结果

也可以直接使用 `PrecisionRecallDisplay`，参数是 precision 和 recall：

``` python
from sklearn.datasets import make_classification
from sklearn.metrics import (precision_recall_curve,
                             PrecisionRecallDisplay)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
X, y = make_classification(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y,                       random_state=0)
clf = SVC(random_state=0)
clf.fit(X_train, y_train)
>> SVC(random_state=0)
predictions = clf.predict(X_test)
precision, recall, _ = precision_recall_curve(y_test, predictions)
disp = PrecisionRecallDisplay(precision=precision, recall=recall)
disp.plot()
>> <sklearn.metrics._plot.precision_recall_curve.PrecisionRecallDisplay object at 0x0000013642EAB8E0>
plt.show()
```

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/unnamed-chunk-3-3.png" width="672" />

``` python
PrecisionRecallDisplay.from_estimator(clf, X_test, y_test)
>> <sklearn.metrics._plot.precision_recall_curve.PrecisionRecallDisplay object at 0x00000136441ADC40>
plt.show()
```

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/unnamed-chunk-3-5.png" width="672" />

``` python
PrecisionRecallDisplay.from_predictions(y_test, predictions)
>> <sklearn.metrics._plot.precision_recall_curve.PrecisionRecallDisplay object at 0x00000136444D8040>
plt.show()
```

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/unnamed-chunk-3-7.png" width="672" />

ROC 曲线绘制也有一个类似的函数 `RocCurveDisplay`：

``` python
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import RocCurveDisplay
from sklearn.datasets import load_wine

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
svc = SVC(random_state=42)
svc.fit(X_train, y_train)
>> SVC(random_state=42)
svc_disp = RocCurveDisplay.from_estimator(svc, X_test, y_test)
svc_disp.plot
>> <bound method RocCurveDisplay.plot of <sklearn.metrics._plot.roc_curve.RocCurveDisplay object at 0x000001364441FC40>>
plt.show()
```

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/unnamed-chunk-3-9.png" width="672" />
