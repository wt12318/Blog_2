---
title: 【hands on ML Ch7】- 集成学习
date: 2021-08-24 22:00:00    
index_img: img/image1.jpg
categories:
  - 机器学习
---

hands on ML 第七章，集成学习和随机森林

<!-- more -->

集成学习就是将一系列的预测器按照某种方式聚合在一起，从而期望结果比单个预测器的效果要好；比如我们可以在训练集中随机选取的子集上训练一系列的决策树分类器，然后获取所有树的预测，对于某一个实例，将其多数分类器预测的结果作为最终的预测结果(这种集成学习方法也叫做随机森林)，这一章将讲解常用的集成方法，包括：bagging，boosting 和 stacking。

## Voting Classifiers

上面那个例子就是一个多数表决分类器( majority-vote classifier)，更一般的说就是在训练集上训练多个不同的模型，最后对某个实例的预测是基于所有模型预测值的 “投票结果” 来决定，多数表决分类器又叫硬投票分类器(hard voting classifier)：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210309170505986.png)

为什么集成学习比单个的(弱)学习器的效果要好？可以从下面这个例子来理解：

假设我们有一个硬币,每次抛硬币有51%的可能是正面朝上,有49%的可能是反面朝上;扔1000次,大部分朝上(朝上的硬币数目大于500)的概率为：

$$
P(X>500) = 1-binom(500,1000,0.51)
$$
在R里面计算得到概率为：

``` r
1-pbinom(500,1000,0.51)
>> [1] 0.7260986
```

同理在扔10000次后，得到硬币大部分朝上的概率为：

``` r
1-pbinom(5000,10000,0.51)
>> [1] 0.9767183
```

进行的实验次数越多，这个概率就越大(大数定理)

现在我们想像：有1000个分类器,每个分类器正确预测的概率为51%,如果我们按照多数投票规则来整合1000个分类器，那么准确率可以达到72%(有超过一半预测的是正确的概率);但是这个前提是这些分类器是相互独立的,但在实际情况中很难保证这一点(因为我们的模型都是在同一个数据上训练的，一种增强独立性的策略就是使用多个非常不同的分类器)。

下面以 moons 数据集为例，训练 3 个不同的分类器（随机森林，逻辑回归和支持向量机），然后再使用硬投票的方式得到最终的预测结果：

``` python
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import VotingClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


log_clf = LogisticRegression(random_state=42) 
rnd_clf = RandomForestClassifier(random_state=42) 
svm_clf = SVC(random_state=42) 
voting_clf = VotingClassifier(estimators=[('lr', log_clf),
                                          ('rf', rnd_clf), 
                                          ('svc', svm_clf)], 
                              voting='hard') 

voting_clf.fit(X_train, y_train)
>> VotingClassifier(estimators=[('lr', LogisticRegression(random_state=42)),
>>                              ('rf', RandomForestClassifier(random_state=42)),
>>                              ('svc', SVC(random_state=42))])
```

``` python
from sklearn.metrics import accuracy_score
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
>> LogisticRegression(random_state=42)
>> LogisticRegression 0.864
>> RandomForestClassifier(random_state=42)
>> RandomForestClassifier 0.896
>> SVC(random_state=42)
>> SVC 0.896
>> VotingClassifier(estimators=[('lr', LogisticRegression(random_state=42)),
>>                              ('rf', RandomForestClassifier(random_state=42)),
>>                              ('svc', SVC(random_state=42))])
>> VotingClassifier 0.912
```

可以看到投票分类器的精度比 3 个单独的模型都有提升。

如果单独的模型都可以估计每个类的概率（也就是有 `predict_proba()`方法），那么就可以对每个类取所有分类器的概率的均值，然后取均值最大的那一类（也可以通过 `weights` 来指定每个分类器的权重），比如有 3 个分类器进行 3 分类的问题，`weight` 使用默认的（每个分类器权重一样）：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210803182813097.png)

最终加权平均后的有最大概率的类为 2，所以分类的结果是 2，这种投票方式叫做软投票。

将上面的例子改为软投票，需要将 `voting` 参数改为 `soft`，并且注意支持向量机默认是不计算概率的，需要设置 `probability` 参数为 True（用交叉验证来估计类的概率）：

``` python
svm_clf = SVC(probability=True,random_state=42) 


voting_clf = VotingClassifier(estimators=[('lr', log_clf),          ('rf', rnd_clf), ('svc', svm_clf)], 
         voting='soft') 

voting_clf.fit(X_train, y_train)
>> VotingClassifier(estimators=[('lr', LogisticRegression(random_state=42)),
>>                              ('rf', RandomForestClassifier(random_state=42)),
>>                              ('svc', SVC(probability=True, random_state=42))],
>>                  voting='soft')
```

``` python
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)
  print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
>> LogisticRegression(random_state=42)
>> LogisticRegression 0.864
>> RandomForestClassifier(random_state=42)
>> RandomForestClassifier 0.896
>> SVC(probability=True, random_state=42)
>> SVC 0.896
>> VotingClassifier(estimators=[('lr', LogisticRegression(random_state=42)),
>>                              ('rf', RandomForestClassifier(random_state=42)),
>>                              ('svc', SVC(probability=True, random_state=42))],
>>                  voting='soft')
>> VotingClassifier 0.92
```

可以看到精度进一步提升了。

## Bagging and Pasting

前面讲过可以通过选择不同的预测器来增加模型间的独立性，另一种方法是使用同一种预测器，但是在训练集的随机选择的不同子集上进行训练；当这种随机选择为有放回的抽样时，这种方法叫做 **bagging**（bootstrap aggregating 缩写），当为不放回抽样时，这种方法叫做 **pasting**（也就是说对于 bagging ，一个预测器的训练样本中可能会有相同的样本点）。

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210803185335885.png)

当所有的预测器都被训练之后，新的实例进入时就可以聚合所有的预测器的结果做出最终的预测。对于分类问题，这个聚合函数就是和前面类似的硬投票方法（如果可以预测概率，还是使用软投票），对于回归问题，聚合函数就是所有预测值的平均。通常来说，聚合后的模型和单个模型有个相似的偏差（bias），但是 variance 更小，另外从上面图也可以看出：预测器可以并行地训练（在不同的 CPU，甚至不同的服务器上）。

在 Scikit-Learn 中可以使用 `BaggingClassifier` 类（或者 `BaggingRegressor` 回归）来进行 bagging 和 pasting；下面的代码训练了 500 个决策树模型，每一个在随机抽取的 100 个样本（有放回，如果将 `bootstrap`
设为 False，那么进行的就是无放回的 pasting）中训练，`n_jobs` 表示使用的 CPU 核数（-1 表示使用所有的 CPU，在windows 上好像不行）：

``` python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier( DecisionTreeClassifier(random_state=42),
                             n_estimators=500, 
                             max_samples=100, 
                             bootstrap=True,random_state=42)
                             
bag_clf.fit(X_train, y_train)
>> BaggingClassifier(base_estimator=DecisionTreeClassifier(random_state=42),
>>                   max_samples=100, n_estimators=500, random_state=42)
y_pred = bag_clf.predict(X_test)
accuracy_score(y_test, y_pred)
>> 0.904
```

在使用 bagging 时，有些实例可能会被抽到多次而另一些实例可能根本不会被抽到；在 `BaggingClassifier` 中默认是有放回地抽取 n 个训练实例（n 也是训练集的大小），因此平均来说，对于每个预测器大概只有 63% 的实例被抽到，剩下的 37% 的实例就叫做 out-of-bag (oob) 实例；由于这些实例没有被抽取作为训练样本，所以可以在这些实例上进行模型的评估。

> 为什么是 63%？

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/473522c600135107452b432799ecabd.jpg)

可以设置 `obb_score` 为 True 来让 `BaggingClassifier` 进行自动的 oob 评估，计算的结果保存在 `oob_score_` 中：

``` python
bag_clf = BaggingClassifier(
  DecisionTreeClassifier(random_state=42),
  n_estimators=500, 
  bootstrap=True,random_state=42,oob_score=True
)

bag_clf.fit(X_train,y_train)
>> BaggingClassifier(base_estimator=DecisionTreeClassifier(random_state=42),
>>                   n_estimators=500, oob_score=True, random_state=42)
bag_clf.oob_score_
>> 0.896
```

``` python
##验证一下
y_pred = bag_clf.predict(X_test)
accuracy_score(y_test,y_pred)
>> 0.92
```

`BaggingClassifier` 除了可以对实例进行抽样外，还支持对特征进行抽样，需要将 `max_samples` 和`bootstrap` 替换成 `max_features` 和 `bootstrap_features` ，这种方法对处理高维数据时是比较有用的。同时对实例和特征进行抽样的方法叫做 Random Patches，只对特征抽样的方法叫做 Random Subspaces。

## 随机森林

随机森林是决策树的集成，通常是通过 bagging 方法进行训练，max_sample 设置为全部的训练集大小。可以使用 `RandomForestClassifier` 类来训练随机森林模型：

``` python
from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16)
rnd_clf.fit(X_train, y_train)
>> RandomForestClassifier(max_leaf_nodes=16, n_estimators=500)
y_pred_rf = rnd_clf.predict(X_test)
```

`RandomForestClassifier` 类除了有着 `DecisionTreeClassifier` 类的控制树生长的超参数外，还有`BaggingClassifier` 类中控制集成的超参数，但是在随机森林中有一些超参数被固定下来：`splitter` 被固定为 `random`（在随机选取的部分特征中寻找最优分割点），`max_samples` 设定为 1，`base_estimator` 设定为 `DecisionTreeClassifier`。因此下面的 `BaggingClassifier` 和上面的随机森林几乎是一样的：

``` python
bag_clf = BaggingClassifier( DecisionTreeClassifier(splitter="random", max_leaf_nodes=16), n_estimators=500, max_samples=1.0, bootstrap=True)
```

随机森林是在随机选取的一部分特征中寻找最优分割点构建决策树，还有一种更随机的情况：在选取的特征中使用随机的分割点来构建树（两次随机）；这种方法叫做 Extremely Randomized Trees （Extra-Trees），具有更高的 bias 和 更低的 variance（随机性越大，越容易 “脱靶” ，也就是偏离要找的真实的函数空间，但是由于各个树模型之间的相关性降低，整个模型的 variance 也会降低），另外这种方法的训练时间也比较短（不需要花费大量时间来寻找最优分割点）。在 `Scikit-learn` 中可以使用 `ExtraTreesClassifier` 或 `ExtraTreesRegressor` 类进行训练 `Extra-Trees` 模型。

随机森林另一个比较好的特性是可以比较容易的衡量特征的相对重要性。Scikit-learn 通过计算**在森林的所有的树中使用某个特征的树节点平均减少的不纯度来衡量特征的重要性** （使用该特征来分割的所有节点的基尼指数变化量的平均值）。Scikit-learn 在训练之后会自动计算这个值，然后再进行归一化使得所有的重要性加和为 1，可以通过 `feature_importances_` 来获取特征重要性的值，下面是在鸢尾花数据集上使用随机森林计算特征重要性的例子：

``` python
from sklearn.datasets import load_iris
iris = load_iris()
rnd_clf = RandomForestClassifier(n_estimators=500)
rnd_clf.fit(iris["data"], iris["target"])
>> RandomForestClassifier(n_estimators=500)
for name, score in zip(iris["feature_names"], rnd_clf.feature_importances_):
  print(name, score)
>> sepal length (cm) 0.10814410934379921
>> sepal width (cm) 0.023458740913181834
>> petal length (cm) 0.4285690228014531
>> petal width (cm) 0.43982812694156587
```

## Boosting

Boosting 指的是能够将多个弱学习器聚合成一个强的学习器的任何集成方法；大部分 Boosting 方法的思想是：依次训练预测器，每一个预测器都力求在前一个基础上有所提升。目前有很多 boosting 方法，最流行的是 **AdaBoost** 和 梯度提升  **Gradient Boosting**）。

### AdaBoost

对于一个新的预测器要去纠正前面的预测器的一种方法就是更关注那些前一个预测器欠拟合的实例，因此新的预测器就会越来越关注那些难分类的实例，这种技术就叫做 AdaBoost。

简单来说：当训练一个 AdaBoost 分类器的时候，首先训练一个基分类器，用这个基分类器在训练集上做预测，然后上调这个分类器误分类样本的权重，接着基于这些更新的权重训练第二个分类器，重复这个步骤直到错误率不再降低；当所有的预测器都训练完了，就可以组合这些预测器得到最终的集成模型（每个预测器的权重由该预测器的错误率计算得到）如下图：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210822103038087.png)

用数学表示：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210822111723135.png)

在 Scikit-learn 中 AdaBoost 的多分类版本为 SAMME （Stagewise Additive Modeling using a Multiclass Exponential loss function），当需要估计类别概率时使用的是 SAMME 的变体 SAMME.R。下面的代码使用了 `AdaBoostClassifier` 类，基分类器为深度为 1 的决策树（只有一个节点）：

``` python
from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(
  DecisionTreeClassifier(max_depth=1),
  n_estimators=200, 
  algorithm="SAMME.R",
  learning_rate=0.5) 

ada_clf.fit(X_train, y_train)
>> AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1),
>>                    learning_rate=0.5, n_estimators=200)
```

### Gradient Boosting

Gradient Boosting 和 AdaBoost 类似，也是通过逐步加入新的预测器，每一个预测器对前一个进行矫正使结果变好，但是和 AdaBoost 不同的是 Gradient Boosting 并不是通过更新实例的权重而是通过**拟合前一个预测器的残差（residual errors）**。

下面以一个回归的任务为例，使用决策树作为基预测器（这样的模型叫做**Gradient Tree Boosting** 或 **Gradient Boosted Regression Trees (GBRT)**，XGBoost 也是这种模型），首先对训练数据应用决策树回归，然后计算残差，接着再对残差应用第二个决策树回归，计算第二个模型的残差，再用第三个决策树回归：

``` python
##创建数据
import numpy as np
np.random.seed(42)
X = np.random.rand(100, 1) - 0.5
y = 3*X[:, 0]**2 + 0.05 * np.random.randn(100)

##对训练集应用决策树回归
from sklearn.tree import DecisionTreeRegressor

tree_reg1 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg1.fit(X, y)

##计算残差，应用决策树到残差
>> DecisionTreeRegressor(max_depth=2, random_state=42)
y2 = y - tree_reg1.predict(X)
tree_reg2 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg2.fit(X, y2)

##继续
>> DecisionTreeRegressor(max_depth=2, random_state=42)
y3 = y2 - tree_reg2.predict(X)
tree_reg3 = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg3.fit(X, y3)
>> DecisionTreeRegressor(max_depth=2, random_state=42)
```

现在我们有 3 个决策树集成的模型，在预测时只需要将 3 个树的预测相加就行了：

``` python
X_new = np.array([[0.8]])
y_pred = sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))
y_pred
>> array([0.75026781])
```

下图是各个树以及其集成的结果：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210823084836674.png)

训练 GBRT 的更简单的方式是调用 Scikit-learn 中的 `GradientBoostingRegressor` 类，和随机森林一样，这个类也有控制决策树生长的参数（深度，最小叶子节点大小等）和控制集成训练的参数（预测器的数量等）：

``` python
from sklearn.ensemble import GradientBoostingRegressor

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3, learning_rate=1.0, random_state=42)
gbrt.fit(X, y)
>> GradientBoostingRegressor(learning_rate=1.0, max_depth=2, n_estimators=3,
>>                           random_state=42)
```

注意这个 `learning_rate` 超参数，这个超参数会缩放每个预测器的贡献，它是和 `n_estimators` 超参数相对应的，当学习率设的较小时，就需要更多的预测器来拟合数据，这种正则化的技术叫做收缩（shrinkage）。
![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210823090136193.png)

`GradientBoostingRegressor` 也有 `subsample` 超参数，来指定训练每个树时随机抽取的实例数量，使用随机抽取的样本来训练 Gradient Boosting 的方法叫做 **Stochastic Gradient Boosting**。

#### Early stopping

为了找到较优的树数量，我们可以使用 early stop 的方法，对于 GBRT 有两种策略：

1.  使用 `staged_predict()` 方法，该方法返回一个在集成模型训练的每个步骤（逐步增加树的数量）做出的预测的迭代器，然后在这些步骤中挑出最好的一步，接着使用这个最佳的参数来训练模型：

``` python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=49)

gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120, random_state=42)
gbrt.fit(X_train, y_train)
>> GradientBoostingRegressor(max_depth=2, n_estimators=120, random_state=42)
errors = [mean_squared_error(y_val, y_pred)
          for y_pred in gbrt.staged_predict(X_val)]
bst_n_estimators = np.argmin(errors) + 1

gbrt_best = GradientBoostingRegressor(max_depth=2, n_estimators=bst_n_estimators, random_state=42)
gbrt_best.fit(X_train, y_train)
>> GradientBoostingRegressor(max_depth=2, n_estimators=56, random_state=42)
```

``` python
bst_n_estimators
>> 56
```

下面是每一个步骤的验证误差以及使用最佳树数量训练的模型：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210823091010369.png)

2. 上面讲到的方法实际上不是真正的 early stop，因为它是训练完所有的模型后再选出来的；我们可以使用
   `warm_start=True` 来进行**增量学习**，也就是在增加树的数量时是在之前已经训练好的结果上增加树继续训练，下面的例子展示了当验证误差在 5 轮迭代中（每一次迭代增加树的数量）没有减少时就停止训练：

``` python
gbrt = GradientBoostingRegressor(max_depth=2, warm_start=True, random_state=42)

min_val_error = float("inf")
error_going_up = 0
for n_estimators in range(1, 120):
    gbrt.n_estimators = n_estimators
    gbrt.fit(X_train, y_train)
    y_pred = gbrt.predict(X_val)
    val_error = mean_squared_error(y_val, y_pred)
    if val_error < min_val_error:
        min_val_error = val_error
        error_going_up = 0
    else:
        error_going_up += 1
        if error_going_up == 5:
            break  # early 
          
>> GradientBoostingRegressor(max_depth=2, n_estimators=1, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=2, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=3, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=4, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=5, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=6, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=7, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=8, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=9, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=10, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=11, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=12, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=13, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=14, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=15, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=16, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=17, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=18, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=19, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=20, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=21, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=22, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=23, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=24, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=25, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=26, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=27, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=28, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=29, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=30, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=31, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=32, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=33, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=34, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=35, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=36, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=37, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=38, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=39, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=40, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=41, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=42, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=43, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=44, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=45, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=46, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=47, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=48, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=49, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=50, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=51, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=52, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=53, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=54, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=55, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=56, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=57, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=58, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=59, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=60, random_state=42,
>>                           warm_start=True)
>> GradientBoostingRegressor(max_depth=2, n_estimators=61, random_state=42,
>>                           warm_start=True)
print(gbrt.n_estimators)
>> 61
print("Minimum validation MSE:", min_val_error)
>> Minimum validation MSE: 0.002712853325235463
```

除了 Scikit-learn 中提供的接口之外，我们还可以使用 `XGBoost` 提供的优化的梯度提升实现：

``` python
import xgboost
xgb_reg = xgboost.XGBRegressor(random_state=42)
xgb_reg.fit(X_train, y_train)
>> XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
>>              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
>>              importance_type='gain', interaction_constraints='',
>>              learning_rate=0.300000012, max_delta_step=0, max_depth=6,
>>              min_child_weight=1, missing=nan, monotone_constraints='()',
>>              n_estimators=100, n_jobs=12, num_parallel_tree=1, random_state=42,
>>              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
>>              tree_method='exact', validate_parameters=1, verbosity=None)
y_pred = xgb_reg.predict(X_val)
val_error = mean_squared_error(y_val, y_pred) 
print("Validation MSE:", val_error) 
>> Validation MSE: 0.004000408205406276
```

## Stacking

之前讲到的集成方法都是基于某些预定的规则，比如硬投票等，另一个策略是另外训练一个模型来聚合所有预测器的预测结果，这种方法就叫做 stacking （stacked generalization），进行聚合的模型叫做 `blending` 预测器或者元学习器（meta learner）：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210823105022753.png)

通常是使用一个 **hold-out set** 的方法来训练 blender：首先将训练集分成两个子集，第一个子集用来训练上面那个图中第一层的预测器，如下图：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210823183715857.png)

当第一层的预测器训练好之后，使用这些预测器在刚刚分出来的第二个子集上（hold out set）做预测，因此对于这个 hold-out 集上的实例都有 3 个预测值，我们可以把这些预测值当成新的特征构建一个新的训练集，然后使用这个新的训练集训练出 blender：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210823184105821.png)

这个过程总结如下：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210823190812406.png)

实际上也可以使用不同的模型训练不同的 blender 构成一个 blender 层，最后再训练一个 blender 来聚合这些 blenders：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210823191304050.png)

这个时候就需要将数据集分成 3 个部分，第一部分用来训练第一层，然后用第一层的预测器在第二部分数据中做预测，基于这些预测值构建新的训练集来训练第二层（可以使用不同的模型来训练），接着用第二层的预测器在第三部分数据中做预测，基于这些预测值构建新的训练集来训练第三层，得到最终的 blender。

在 Scikit-learn 0.22 的版本中已经支持通过 `StackingClassifier`和 `StackingRegressor` 类来 直接使用stacking 了，需要注意的是 scikit-learn在训练基预测器时是使用了交叉验证的方式计算数据的预测值，然后利用不同基预测器的预测值构成新的训练集训练最后的 blender，下面是一个回归的例子：

``` python
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.neighbors import KNeighborsRegressor
estimators = [('ridge', RidgeCV()),
              ('lasso', LassoCV(random_state=42)),
              ('knr', KNeighborsRegressor(n_neighbors=20,
              metric='euclidean'))]
              
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import StackingRegressor

final_estimator = GradientBoostingRegressor(
  n_estimators=25, subsample=0.5, min_samples_leaf=25,
  max_features=1,random_state=42)
  
reg = StackingRegressor(
  estimators=estimators,
  final_estimator=final_estimator)
  
from sklearn.datasets import load_diabetes
X, y = load_diabetes(return_X_y=True)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

reg.fit(X_train, y_train)
>> StackingRegressor(estimators=[('ridge',
>>                                RidgeCV(alphas=array([ 0.1,  1. , 10. ]))),
>>                               ('lasso', LassoCV(random_state=42)),
>>                               ('knr',
>>                                KNeighborsRegressor(metric='euclidean',
>>                                                    n_neighbors=20))],
>>                   final_estimator=GradientBoostingRegressor(max_features=1,
>>                                                             min_samples_leaf=25,
>>                                                             n_estimators=25,
>>                                                             random_state=42,
>>                                                             subsample=0.5))
y_pred = reg.predict(X_test)
from sklearn.metrics import r2_score
print('R2 score: {:.2f}'.format(r2_score(y_test, y_pred)))
>> R2 score: 0.53
```
