---
title: 【hands on ML Ch3】-分类  
date: 2021-02-04 10:00:00    
index_img: img/image1.jpg
categories:
  - 机器学习
---

Hands on ML 第三章笔记，主要是分类相关的知识
<!-- more -->
本章使用的数据集是MNIST数据集，有70000张手写的数字图像(这个数据集也被称为是机器学习的“hello world”)

Scikit-Learn提供了一些函数来下载常用的数据集，下面的代码可以下载MNIST数据集：

``` python
from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1)
mnist.keys()
```

通过Scikit-Learn下载的数据是字典的结构，包含key和value，比如`DESCR`key表示数据集的描述，`data` key表示数据集，`target` key表示数据集的标签。

使用Scikit-Learn下载太慢，所以在openml官网下载了csv格式的[数据](https://www.openml.org/d/554)，再使用numpy读入：

``` python
import os
import numpy as np
data = np.loadtxt("../test/mnist_784.csv",delimiter=",",skiprows=1)

data.shape

>> (70000, 785)
X = data[:,0:784]###data without lable

y = data[:,784]###lable

X.shape
>> (70000, 784)
y.shape
>> (70000,)
```

每个图片都有784个特征，因为每张图片都由28\*28个像素构成，每个特征就代表一个像素的密度(0-255):

``` python
import matplotlib as mpl 
import matplotlib.pyplot as plt

some_digit = X[0] 
some_digit_image = some_digit.reshape(28, 28) 
plt.imshow(some_digit_image,cmap="binary") 
>> <matplotlib.image.AxesImage object at 0x0000015BA034A700>
plt.axis("off") 
>> (-0.5, 27.5, 27.5, -0.5)
plt.show()
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/unnamed-chunk-3-1.png)

``` python
y[0]
>> 5.0
```

首先要做的就是划分训练集和测试集(MNIST数据集已经打乱过了，所以每个交叉验证的fold都是类似的)

``` python
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
```

## 训练二分类器

二分类器的目的是在数据中辨别出两种类别，比如这里我们想要鉴别某个手写的数字是5还是不是5；首先将数据的lable进行重塑：

``` python
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
```

我们首先尝试SGD(Stochastic Gradient Descent)分类器【随机梯度下降是一种算法，Scikit-Learn里面的SGDClassifier类指的是一系列模型，这些模型的优化算法都是SGD，SGDClassifier类默认的是线性SVM模型，参照官网上的[说明](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)】:

``` python
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)##随机梯度下降需要设定种子数 
sgd_clf.fit(X_train, y_train_5)
>> SGDClassifier(random_state=42)
sgd_clf.predict([some_digit])
>> array([ True])
```

## 模型性能评估

这一部分是重点

### 使用交叉验证来评估准确性

和第二章一样，使用cross_val_score函数来进行交叉验证，注意这里使用的评价方法不是第二章里面的RSME了，而是使用精确度(正确预测的比例)：

``` python
from sklearn.model_selection import cross_val_score

cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
>> array([0.95035, 0.96035, 0.9604 ])
```

看起来结果不错，但是如果我们构建一个非常简单的模型：将所有的图片都分到不是5的类中，这个模型的精确度是多少呢？

``` python
from sklearn.base import BaseEstimator

class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None): 
      pass 
    def predict(self, X): 
      return np.zeros((len(X), 1), dtype=bool)##返回False
    
never_5_clf = Never5Classifier()
cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")
>> array([0.91125, 0.90855, 0.90915])
```

这个模型都有0.9以上的准确度，因为只有10%的图片是5，所以总是猜不是5，90%是对的，这说明：仅仅使用准确度来衡量模型是不太好的，特别是对于有偏向性的数据(skewed datasets)。

### 混淆矩阵

评估一个分类器的更好的方法是混淆矩阵(confusion matrix)，混淆矩阵的每一行是真实的类，每一列是预测的类；要计算混淆矩阵，首先要获取预测值，可以使用cross_val_predict函数，这个函数也进行交叉验证，不过返回的不是评估分数而是在每一个验证集上的预测值(因此是“clean”的预测，所谓clean指的是预测使用的是在训练过程中没有看过的数据)：

``` python
from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
```

然后就可以使用confusion_matrix函数来获得混淆矩阵：

``` python
from sklearn.metrics import confusion_matrix

confusion_matrix(y_train_5, y_train_pred)
>> array([[53892,   687],
>>        [ 1891,  3530]], dtype=int64)
```

这个混淆矩阵可以使用下图来表示：
![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210204170319913.png)

一个完美的分类器的混淆矩阵应该只有主对角线上是非零值

关于混淆矩阵，有一些重要的指标：

-   精度(precision)表示 在预测的positive里面真实的也是positive的比例：

    $$
    precision = \frac{TP}{TP+FP}
    $$

-   召回率(recall)(或者叫灵敏度sensitivity; 真阳性率FPR)表示
    在真实的positive里面预测是positive的比例：

    $$
    recall = \frac{TP}{TP+FN}
    $$

### 精度和召回率

Scikit-Learn也提供了函数来计算精度和召回率：

``` python
from sklearn.metrics import precision_score, recall_score

precision_score(y_train_5,y_train_pred)
>> 0.8370879772350012
recall_score(y_train_5, y_train_pred)
>> 0.6511713705958311
```

这些值的意思是：当这个分类器认为某个图片是5，那么有83.7%的机率是对的；并且这个分类器只检测到65%的是5的图片

也可以将精度和召回率结合成一个值：F_1 score (两者的几何平均，几何平均给予小的值更大的权重)：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210204171113298.png)

可以使用 f1_score()函数来计算：

``` python
from sklearn.metrics import f1_score

f1_score(y_train_5, y_train_pred)
>> 0.7325171197343846
```

需要注意的是：**在不同情况下，我们对于precision和recall的关注度是不一样的**：

比如，如果训练的分类器的任务是检测对儿童安全的视频，那么这个分类器的precision就更重要(尽可能保证预测是安全的视频实际上也是安全的，而不是说将所有的安全的视频都给检出)；而如果分类器的任务是根据商场的监控图像来检测小偷，这个时候分类器的recall就更重要(将所有的小偷尽可能全部检测出，虽然有可能发出假的的警报)。

### Precision/Recall 平衡

对于每个观测值，SGDClassifier都会依据决策函数(decision function)来计算一个值，再根据特定的阈值，如果计算的值高于阈值则为positive类，低于阈值则为negative类，所以改变这个阈值就是使得precision和recall有所变化，这个过程可以用下图来表示：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210204173205578.png)

在Scikit-Learn中，我们可以通过decision_function()方法来获取每个观测值的决策函数值：

``` python
y_scores = sgd_clf.decision_function([some_digit])
y_scores
>> array([2164.22030239])
threshold = 0##设置阈值为0
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred
>> array([ True])
threshold = 8000##改变阈值
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred
>> array([False])
```

那么我们怎么选择一个合适的阈值呢？首先可以使用cross_val_predict()得到每个实例的决策函数值(同样是“clean”的)，然后使用 precision_recall_curve()函数来计算所有阈值的precision和recall值：

``` python
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")

from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

###可视化
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.legend(loc="center right", fontsize=16)
    plt.xlabel("Threshold",fontsize=16)
    plt.grid(True)                    
    plt.axis([-50000, 50000, 0, 1])
    
recall_90_precision = recalls[np.argmax(precisions >= 0.90)]
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]


plt.figure(figsize=(8, 4))            
>> <Figure size 800x400 with 0 Axes>
```

``` python
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.plot([threshold_90_precision, threshold_90_precision], [0., 0.9], "r:")      
>> [<matplotlib.lines.Line2D object at 0x0000015B4B54FD90>]
plt.plot([-50000, threshold_90_precision], [0.9, 0.9], "r:")          
>> [<matplotlib.lines.Line2D object at 0x0000015B9C2D3EB0>]
plt.plot([-50000, threshold_90_precision], [recall_90_precision, recall_90_precision], "r:")
>> [<matplotlib.lines.Line2D object at 0x0000015B9C2E0820>]
plt.plot([threshold_90_precision], [0.9], "ro")                           
>> [<matplotlib.lines.Line2D object at 0x0000015B9C2E0400>]
plt.plot([threshold_90_precision], [recall_90_precision], "ro")           
>> [<matplotlib.lines.Line2D object at 0x0000015B9C2E0B50>]
```

``` python
plt.show()
```

![](/img/hands_on_ML_ch3_files/figure-markdown_github/unnamed-chunk-17-1.png)

注意：当提高阈值时，precision不一定总是上升的(以上面那个轴为例，当阈值从中间向右移动一位precision就会下降：4/5→3/4);但是Recall总是下降的

另外，我们也可以直接展示precision和recall的关系：

``` python
def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])
    plt.grid(True)

plt.figure(figsize=(8, 6))
>> <Figure size 800x600 with 0 Axes>
```

``` python
plot_precision_vs_recall(precisions, recalls)
plt.plot([recall_90_precision, recall_90_precision], [0., 0.9], "r:")
>> [<matplotlib.lines.Line2D object at 0x0000015B9C4F4880>]
plt.plot([0.0, recall_90_precision], [0.9, 0.9], "r:")
>> [<matplotlib.lines.Line2D object at 0x0000015B9C4F4C10>]
plt.plot([recall_90_precision], [0.9], "ro")
>> [<matplotlib.lines.Line2D object at 0x0000015B9C4F44F0>]
```

``` python
plt.show()
```

<img src="/img/hands_on_ML_ch3_files/figure-markdown_github/unnamed-chunk-20-1.png" width="672" />

假如我们现在想要分类器达到90%的precision，可以使用numpy的np.argmax函数(返回第一个最大值的index)：

``` python
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]##true是1，false是0，因此返回第一个1，也就是第一个true的位置

y_train_pred_90 = (y_scores >= threshold_90_precision)##预测

precision_score(y_train_5, y_train_pred_90)
>> 0.9000345901072293
recall_score(y_train_5, y_train_pred_90)
>> 0.4799852425751706
```

### ROC曲线

ROC曲线全称为：receiver operating characteristic curve；ROC曲线展示了真阳性率(true positive rate, recall的另一个叫法)和假阳性率(false positive rate, FPR)的关系

$$
FPR = \\frac{FP}{FP+TN}=1-TNR=1-\\frac{TN}{FP+TN}
$$

这里面的TNR又叫做特异性(specificity)，所以**ROC曲线画的是recall/sensitivity(两个是一样的)和1-specificity的关系**

可以使用roc_curve函数来计算FPR和TPR：

``` python
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

##plot
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
    plt.axis([0, 1, 0, 1])            
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)  
    plt.grid(True)

plt.figure(figsize=(8, 6))                  
>> <Figure size 800x600 with 0 Axes>
```

``` python
plot_roc_curve(fpr, tpr)
fpr_90 = fpr[np.argmax(tpr >= recall_90_precision)]          
plt.plot([fpr_90, fpr_90], [0., recall_90_precision], "r:") 
>> [<matplotlib.lines.Line2D object at 0x0000015B9C708A60>]
plt.plot([0.0, fpr_90], [recall_90_precision, recall_90_precision], "r:") 
>> [<matplotlib.lines.Line2D object at 0x0000015B9C708D60>]
plt.plot([fpr_90], [recall_90_precision], "ro")               
>> [<matplotlib.lines.Line2D object at 0x0000015B9C708580>]
```

``` python
plt.show()
```

<img src="/img/hands_on_ML_ch3_files/figure-markdown_github/unnamed-chunk-24-1.png" width="672" />

图中的虚线表示完全随机的分类器的ROC曲线，一个好的分类器要尽可能离这条线远，并且向左上角靠拢(高的recall并且比较低的假阳性)，一种比较不同的分类器的方法就是计算ROC曲线下面积(AUC)，越接近1说明这个模型越好(图中虚线的AUC是0.5)：

``` python
from sklearn.metrics import roc_auc_score

roc_auc_score(y_train_5, y_scores)
>> 0.9604938554008616
```

现在我们可以来比较一下 随机森林分类器(RandomForestClassifier)和SVM分类器(SGDClassifier,默认参数)了。

要注意的是RandomForestClassifier没有decision_function方法而是predict_proba方法，该方法返回的是一个array数组，每一行是一个观测，每一列是该观测属于各类的概率：

``` python
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")

y_probas_forest
>> array([[0.11, 0.89],
>>        [0.99, 0.01],
>>        [0.96, 0.04],
>>        ...,
>>        [0.02, 0.98],
>>        [0.92, 0.08],
>>        [0.94, 0.06]])
```

roc_curve()函数需要的输入是label和score(用来选取不同的阈值)，所以在这里使用是5类(positive类)的概率作为score：

``` python
y_scores_forest = y_probas_forest[:, 1]

fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5,y_scores_forest)

##plot
plt.plot(fpr, tpr, "b:", label="SGD")
>> [<matplotlib.lines.Line2D object at 0x0000015B9CD23580>]
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest") 
plt.legend(loc="lower right") 
>> <matplotlib.legend.Legend object at 0x0000015B9CD23EB0>
plt.show()
```

<img src="/img/hands_on_ML_ch3_files/figure-markdown_github/unnamed-chunk-27-1.png" width="672" />

``` python
roc_auc_score(y_train_5, y_scores_forest)
>> 0.9983436731328145
```

## 多类别分类

有一些算法能够处理多分类问题(比如SGD 分类器，随随机森林分类器和朴素贝叶斯分类器)而一些算法只能处理二分类问题(比如逻辑斯蒂回归，支持向量机等)，但是我们可以使用一些方法来使这些算法可以用来处理多分类问题。

主要有两种方法：

-   一对多策略(one-versus-the-rest (OvR)):
    比如要将手写图片分为0-9一共10个类别，那么我们就可以训练10个分类器，每个分类器处理的是一个二分类问题(属于这一类还是不属于这一类)，都可以得到一个score，对于每个图片就选择10个分类器中score最高的分类器所对应的类作为该图片的预测类

-   一对一策略(one-versus-one
    (OvO)):对所有的类两两组合训练二分类的分类器，如果有N类，那么就需要训练N\*(N-1)/2个分类器，对于一个图片就需要运行所有的分类器(10类别是45个)，在这些结果中预测次数最多的类就是该图片的预测类，这个方法的好处是在训练时只需要对一部分训练数据进行训练(只涉及要识别的类的数据，比如0-1分类器只需要对所有的0/1图片进行训练)

对于一些算法(比如支持向量机)对大的训练集处理比较困难(scale poorly with the size of the training set),对于这些算法OvO策略比较适合，因为训练的时候不需要全部的训练集；对于大部分的二分类算法，OvR比较适合。

Scikit-Learn会依据算法的不同来选择OvO或者OvR：

``` python
##支持向量机SVM算法，默认是使用OvO
from sklearn.svm import SVC
svm_clf = SVC()
svm_clf.fit(X_train, y_train)##多分类
>> SVC()
svm_clf.predict([some_digit])
>> array([5.])
```

如果想要指定OvO或者OvR，可以使用OneVsOneClassifier或者OneVsRestClassifier类：

``` python
from sklearn.multiclass import OneVsRestClassifier

ovr_clf = OneVsRestClassifier(SVC())##SVC的OVR策略
ovr_clf.fit(X_train, y_train)
>> OneVsRestClassifier(estimator=SVC())
ovr_clf.predict([some_digit])
>> array([5.])
```

对SGD分类器进行多分类任务的训练也是类似的，不过SGD分类器本身就可以进行多分类任务，所以不会运行OVO或者OVR：

``` python
sgd_clf.fit(X_train, y_train)
>> SGDClassifier(random_state=42)
sgd_clf.predict([some_digit])
>> array([3.])
cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")##检测预测精度
>> array([0.87365, 0.85835, 0.8689 ])
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))##将变量进行缩放
cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")##精度有所提升
>> array([0.8983, 0.891 , 0.9018])
```

## 错误分析

当我们通过一系列的步骤找到了一个不错的模型并想要进一步提升其性能，一种方法就是分析这个模型犯的错误。

首先需要查看混淆矩阵：

``` python
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)

conf_mx = confusion_matrix(y_train, y_train_pred)
conf_mx
>> array([[5577,    0,   22,    5,    8,   43,   36,    6,  225,    1],
>>        [   0, 6400,   37,   24,    4,   44,    4,    7,  212,   10],
>>        [  27,   27, 5220,   92,   73,   27,   67,   36,  378,   11],
>>        [  22,   17,  117, 5227,    2,  203,   27,   40,  403,   73],
>>        [  12,   14,   41,    9, 5182,   12,   34,   27,  347,  164],
>>        [  27,   15,   30,  168,   53, 4444,   75,   14,  535,   60],
>>        [  30,   15,   42,    3,   44,   97, 5552,    3,  131,    1],
>>        [  21,   10,   51,   30,   49,   12,    3, 5684,  195,  210],
>>        [  17,   63,   48,   86,    3,  126,   25,   10, 5429,   44],
>>        [  25,   18,   30,   64,  118,   36,    1,  179,  371, 5107]],
>>       dtype=int64)
```

可以用热图的形式将混淆矩阵可视化：

``` python
plt.matshow(conf_mx, cmap=plt.cm.gray)
>> <matplotlib.image.AxesImage object at 0x0000015B9C0DC1C0>
plt.show()
```

<img src="/img/hands_on_ML_ch3_files/figure-markdown_github/unnamed-chunk-33-1.png" width="480" />

从这个图来看，结果是比较好的，因为大部分都集中在对角线上，但是这里面查看的是绝对数值，可能某个类的总数就比较小，比如5类，因此我们将混淆矩阵中的每个值除以相应类的图片总数(行和)得到相对值：

``` python
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
```

再将对角线上的值归为0，因此处理后的混淆矩阵中的值就是错误率：

``` python
np.fill_diagonal(norm_conf_mx, 0)##对角线归0

plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
>> <matplotlib.image.AxesImage object at 0x0000015B9C508E50>
plt.show()
```

<img src="/img/hands_on_ML_ch3_files/figure-markdown_github/unnamed-chunk-35-1.png" width="480" />

可以看到8类的列最亮，也就是说很多图片都被错误地分成8了；另外3和5也是经常被相互错分的。

我们可以针对这种错误来想办法提升模型，例如：可以收集更多的图片，这些图片长得像8但又不是8，用这些数据作为训练集；还可以编码一些新的特征，比如图像中闭环的数目(8有2个，6有1个，5没有)；也可以对图像进行预处理(使图像居中，突出某些特征等)。

## 多标签分类

多标签分类指的是：对于一个观测值可以输出多个类别；比如一个人像识别系统被训练可以识别3张脸A,B,C，当来了一张A和C的照片，这个分类器就会输出\[1,0,1\],也就是对这一张照片可以有3个类别。

这里，我们可以将每个图片都赋予两个类的属性，图片上的数值是否大于7和数字是否为偶数(这里使用的是K近邻分类算法)：

``` python
from sklearn.neighbors import KNeighborsClassifier

y_train_large = (y_train >= 7) 
y_train_odd = (y_train % 2 == 1) 
y_multilabel = np.c_[y_train_large, y_train_odd] 
knn_clf = KNeighborsClassifier() 
knn_clf.fit(X_train,y_multilabel)
>> KNeighborsClassifier()
knn_clf.predict([some_digit])
>> array([[False,  True]])
```

评估多标签分类器的方法有很多，取决于不同的项目;比如可以使用每个标签的F1 score的均值：

``` python
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)

f1_score(y_multilabel, y_train_knn_pred, average="macro")##有不同的平均方法，具体可以看文档
>> 0.976410265560605
```

## 多输出分类

全称为多输出-多标签分类，意思是：对于每个观测值有多个标签(像上面的多标签分类一样)，并且对于每个标签有多个值(上面只有T/F两个值)，举个例子：我们现在有一个系统，输入是有噪声的图片，输出是降噪后的图片；那么对于每个图片，输出有多个标签(每个像素都是一个标签)并且每个标签有多个值(像素密度从0-255)。
