---
title: 【hands on ML Ch6】-决策树模型
date: 2021-03-04 10:00:00    
index_img: img/image1.jpg
---



hands on ML 第六章，决策树模型

<!-- more -->

决策树是一种多能的机器学习算法，可以处理分类，回归，甚至多输出问题(见第二章)

## 训练和可视化决策树

首先在iris数据集上训练一个决策树模型并可视化：

``` python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from matplotlib import pyplot as plt

iris = load_iris()
x = iris.data[:,2:]##取petal length和width变量
y = iris.target

tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(x,y)
>> DecisionTreeClassifier(max_depth=2)
```

Graphviz是一个开源的图（Graph）可视化软件，采用抽象的图和网络来表示结构化的信息。在数据科学领域，Graphviz的一个用途就是实现决策树可视化,因此我们需要使用`export_graphviz()`将树结构导出为一个`.dot`文件

``` python
from sklearn.tree import export_graphviz
from graphviz import Source

export_graphviz(
  tree_clf,
  out_file="../test/iris_tree.dot",
  feature_names=iris.feature_names[2:],
  class_names=iris.target_names,
  rounded=True,
  filled=True,
  special_characters=True
)
```

然后需要下载[Graphviz](https://www.graphviz.org/download/),打开powershell：

``` r
dot -Tpng iris_tree.dot -o iris_tree.png
```

<center>

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/iris_tree.png)

</center>

## 理论

决策树可以用来处理分类和回归任务，主要思想就是：根据特征对数据集进行划分，决策树的学习分成3个步骤：

-   特征选择
-   生成决策树
-   决策树的修剪(正则化)

### 特征选择

特征选择的就是选择对训练数据有较好分类能力的特征，也就是说通过某个特征将数据集分成若干子集，这些子集中数据的一致性(纯度)应该比原来的数据集要高；在决策树中使用熵来表示这个纯度

对离散型随机变量X，其概率分布为：

*P*(*X* = *x*<sub>*i*</sub>) = *p*<sub>*i*</sub>, *i* = 1, 2, ..., *n*
则X的熵定义为：

$$
H(X) = - \\sum\_{i=1}^np\_ilog\_2p\_i 
$$
设有随机变量X,Y,其联合概率分布为：

*P*(*X* = *x*<sub>*i*</sub>, *Y* = *y*<sub>*j*</sub>) = *p*<sub>*i**j*</sub>, *i* = 1, 2, ..., *n*; *j* = 1, 2, ..., *m*

条件熵为在X给定的条件下Y的条件概率分布的熵对X的数学期望:

$$
H(Y|X) = \\sum\_{i=1}^np\_iH(Y|X=x\_i)\\\\
p\_i=P(X=x\_i),i=1,2,...,n
$$
由实际数据计算得到的熵和条件熵叫做经验熵和经验条件熵；设数据集为D,根据特征A将数据集分成若干个子集*D*<sub>*i*</sub>,那么D的经验熵(*H*(*D*))和给定A的条件下D的经验条件熵(*H*(*D*|*A*))为:

$$
H(D)=-\\sum\_{k=1}^K\\frac{|D\_k|}{|D|}log\_2\\frac{|D\_k|}{|D|},\\\\
H(D|A)=\\sum\_{i=1}^n\\frac{|D\_i|}{|D|}H(D\_i)=\\sum\_{i=1}^n\\frac{|D\_i|}{|D|}\\sum\_{k=1}^K\\frac{|D\_{ik}|}{|D\_i|}log\_2\\frac{|D\_{ik}|}{|D\_i|}\\\\
$$
|*D*<sub>*k*</sub>|表示k类样本的数目,|*D*|是总的样本数,|*D*<sub>*i**k*</sub>|表示在第i个子集中k类样本的数目,|*D*<sub>*i*</sub>|表示第i个子集的样本数

一个好的分类特征应该是：根据这个特征划分的数据集后的熵应该比原来数据集的熵要低,因此定义信息增益*g*(*D*, *A*)为：

*g*(*D*, *A*) = *H*(*D*) − *H*(*D*|*A*)

所以根据信息增益来选择特征：**对训练集(或子集)计算每个特征的信息增益，选择信息增益最大的特征来划分数据集**

信息增益计算的是绝对值，因此对取值较多的特征有倾向性(取值越多,加和也越大),所以将信息增益除以该特征的经验熵来标准化信息增益，得到信息增益比:

$$
g\_k(D,A)=\\frac{g(D,A)}{H\_A(D)},H\_A(D)=-\\sum\_{i=1}^n\\frac{|D\_i|}{|D|}log\_2\\frac{|D\_i|}{|D|}
$$
n表示特征A可以取值的个数(A的水平)

### 生成决策树

生成决策树的算法有3种：ID3,C4.5和CRAT，CART算法比较特殊，后面单独讲；前两种算法都只可以用来分类，CART既可以分类也可以回归

ID3算法在决策树的各个节点上应用信息增益法则选择特征，递归构建决策树：从根节点开始，对节点计算所有可能的特征的信息增益，选择信息增益最大的特征作为节点的特征，由该特征的不同取值建立子节点，再对子节点递归地调用以上方法构建决策树，直到所有特征的信息增益都很小或者没有特征可以选择为止

C4.5算法和ID3的区别在于使用信息增益比来选择特征

### 决策树的剪枝

在生成决策树的过程中是以尽可能的准确分类为标准，但是这样往往会出现过拟合的情况，为了避免过拟合，需要限制模型的自由度，即对模型进行正则化约束，在决策树模型里面就是剪枝

决策树的剪枝是通过最小化损失函数来实现；决策树学习的损失函数为：

$$
C\_{\\alpha}(T)=\\sum\_{t=1}^{|T|}N\_tH\_t(T)+\\alpha|T|\\\\
H\_t(T)=-\\sum\_k^K\\frac{N\_{tk}}{N\_t}log2\\frac{N\_{tk}}{N\_t}
$$
其中t表示叶节点,|T|是叶节点个数,*N*<sub>*t*</sub>是t叶节点的样本数,*N*<sub>*t**k*</sub>是t叶节点中k类样本的个数

将损失函数的第一项记作*C*(*T*),

$$
C(T)=\\sum\_{t=1}^{|T|}N\_tH\_t(T)=-\\sum\_{t=1}^{|T|}N\_t\\sum\_k^K\\frac{N\_{tk}}{N\_t}log2\\frac{N\_{tk}}{N\_t}=-\\sum\_{t=1}^{|T|}\\sum\_k^KN\_{tk}log2\\frac{N\_{tk}}{N\_t}
$$

损失函数可以写成：

*C*<sub>*α*</sub>(*T*) = *C*(*T*) + *α*|*T*|

*C*(*T*)表示模型对数据的拟合程度(如果完全拟合，那么经验熵就为为0)，|T|表示模型的复杂度(叶子节点的多少)，*α*的作用就是在两者间平衡(对模型复杂度有个惩罚)

决策树剪枝的过程为：从下往上进行回缩，如果回缩前的模型为*T*<sub>*A*</sub>,回缩后的模型为*T*<sub>*B*</sub>:

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210307192547332.png)

如果有：

*C*<sub>*α*</sub>(*T*<sub>*B*</sub>) ≤ *C*<sub>*α*</sub>(*T*<sub>*A*</sub>)

那么就进行回缩剪枝，将父节点变为叶节点

### CART算法

CART的全称为classification and regression
tree,可以用来处理**分类和回归**任务，得到的决策树是二叉树，内部节点的取值只有是和否,左分支为“是”的分支,右分支为“否”的分支

#### 分类

CART算法使用**基尼指数**作为最优特征的选择依据，而不是信息增益

在分类问题中，假设有K个类，样本点属于第k类的概率为*p*<sub>*k*</sub>,那么概率分布的基尼指数为：

$$
Gini(p)=\\sum\_{k=1}^Kp\_k(1-p\_k)=1-\\sum\_{k=1}^Kp\_k^2
$$
对于给定的样本集合D，基尼指数为：
$$
Gini(D)=1-\\sum\_{k=1}^K(\\frac{|C\_k|}{|D|})^2
$$

如果数据集D可以根据特征A的某个值分割成D1和D2两个部分，则在特征A的条件下，集合D的基尼指数为：

$$
Gini(D,A)=\\frac{|C\_1|}{|D|}Gini(D\_1)+\\frac{|C\_2|}{|D|}Gini(D\_2)
$$
因此CART算法构建决策树的过程为：在所有可能的特征A和其切分点a的组合中选择使上式最小的A和a将数据分成两个子集，生成两个子节点，再在子节点上重复这个过程，直到满足停止条件

以最开始的鸢尾花决策树为例：决策树做预测比较简单：就是从根节点(最上面)往下进行判断；如果现在有一个iris花,从根节点开始(深度为0)，花瓣长度是否小于2.45,如果小于2.45就是往左走，此时左边的节点没有子节点，这样的节点叫做叶子节点，然后就可以判断该花是setosa类

从上图可以看到每个节点都有一些属性(gini,samples,value,class)：

-   samples属性：该节点所应用的样本数量，比如在深度为1的右侧节点中有100个训练实例的花瓣长度大于2.45，在这100个里面又有54个实例的花瓣宽度小于1.75(深度为2的左节点)
-   value属性：该节点中每个类型有多少训练实例；比如最底部的右侧节点的value表示46个实例中有0个Iris
    setosa,1个 Iris versicolor,和45个Iris virginica
-   gini属性：该节点的不纯度，如果该节点所有的实例都是一个类，那么gini就是0，表示纯的；比如深度为1的左节点，全部是setosa

该决策树的决策边界可以用下图来表示：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210306162213679.png)

##### 估计类的概率

决策树也可以估计一个实例属于特定类的概率  
首先找到这个实例所属的叶子节点，然后返回该节点中各类的训练实例所占的比例作为这个实例属于各个类的概率；比如现在有一个鸢尾花花瓣长5cm宽1.5cm，那么它所属的叶子节点为深度为2的左节点，所以决策树输出概率为:0%是setosa,90.7%(49/54)是versicolor,9.3%(5/54)是virginica，如果让决策树来预测这个花的类别，会输出class
1 (versicolor):

``` python
tree_clf.predict_proba([[5,1.5]])
>> array([[0.        , 0.90740741, 0.09259259]])
tree_clf.predict([[5,1.5]])
>> array([1])
```

需要注意的是：落在某个叶子节点中的所有实例的输出概率都是一样的(上面决策边界图里面同一个长方形里面的点)

#### 回归

决策树的回归也是根据某个特征来划分数据集，但是和分类不同，在划分的子集上并不是对应着一个类，而是对应着一个输出，可以用下图来理解：

<center>

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210307202506758.png)

</center>

图中黑色的竖线代表划分(上图只有一个特征)，有颜色的横线表示每次划分后在相应的子集中的输出

假设已将输入空间(数据集)划分成M个单元(子集)：*R*<sub>1</sub>, *R*<sub>2</sub>, ..., *R*<sub>*M*</sub>,在*R*<sub>*m*</sub>单元上有一个固定的输出值*C*<sub>*m*</sub>,所以回归树模型可以表示为：
$$
f(x)=\\sum\_{m=1}^MC\_mI(x\\in R\_m)
$$
I函数表示x在*R*<sub>*m*</sub>里面的时候为1，否则为0  
在每个单元上可以使用平方误差来表示回归树的预测误差，通过最小化平方误差，我们就可以求解出每个单元上的最优输出值*Ĉ*<sub>*m*</sub>:

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/2021-03-07_20-40-32.jpg)

*Ĉ*<sub>*m*</sub> = *a**v**e*(*y*<sub>*i*</sub>|*x*<sub>*i*</sub> ∈ *R*<sub>*m*</sub>)
每个子集上的最优输出有了，那么现在的问题就是怎样进行划分？
对于特征j和其分割点s，(j,s)对输入空间进行划分得到两个子空间*R*<sub>1</sub>, *R*<sub>2</sub>：

*R*<sub>1</sub>(*j*, *s*) = {*x*|*x*<sup>*j*</sup> ≤ *s*}; *R*<sub>2</sub>(*j*, *s*) = {*x*|*x*<sup>*j*</sup> &gt; *s*}
目的就是找到最优的(j,s)使得：

*m**i**n*<sub>*j*, *s*</sub>\[*m**i**n*<sub>*c*<sub>1</sub></sub>∑<sub>*x*<sub>*i*</sub> ∈ *R*<sub>1</sub>(*j*, *s*)</sub>(*y*<sub>*i*</sub> − *c*<sub>1</sub>)<sup>2</sup> + *m**i**n*<sub>*c*<sub>2</sub></sub>∑<sub>*x*<sub>*i*</sub> ∈ *R*<sub>2</sub>(*j*, *s*)</sub>(*y*<sub>*i*</sub> − *c*<sub>2</sub>)<sup>2</sup>\]
通常的做法为：遍历特征j，对固定的切分特征j扫描切分点s(如果是连续的需要离散化)，然后选择使上式最小的(j,s)组合，按照(j,s)组合对数据集进行划分，接着继续对子集重复该步骤，直到满足停止条件

在Scikit-Learn里面可以使用`DecisionTreeRegressor`类进行回归树的构建：

``` python
import numpy as np
# Quadratic training set + noise
np.random.seed(42)
m = 200
X = np.random.rand(m, 1)
y = 4 * (X - 0.5) ** 2
y = y + np.random.randn(m, 1) / 10

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg.fit(X, y)
>> DecisionTreeRegressor(max_depth=2, random_state=42)
export_graphviz(
        tree_reg,
        out_file="../test/iris_tree1.dot",
        feature_names=["x1"],
        rounded=True,
        filled=True
    )
```

``` r
dot -Tpng iris_tree1.dot -o iris_tree2.png
```

<center>

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/iris_tree2.png)

</center>

#### 剪枝

CART算法的剪枝和一般的决策树剪枝不同

CART算法对决策树的每一个内部节点都进行剪枝，生成一个子决策树的序列；假设树的结构如下：

<center>

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210308223511075.png)

</center>

设整体树为*T*<sub>0</sub>,对*T*<sub>0</sub>的任意内部节点t,可以计算以t为单节点的树的损失函数*C*<sub>*α*</sub>(*t*)和以t为根节点的子树的损失函数*C*<sub>*α*</sub>(*T*<sub>*t*</sub>);当*α*充分小的时候*C*<sub>*α*</sub>(*T*<sub>*t*</sub>) &lt; *C*<sub>*α*</sub>(*t*)(对树的复杂度惩罚较小,较复杂的树能够较好的拟合数据，因此损失函数较低)，当*α*增大到某一个值的时候,两者相等，也就是单节点的树和子树的损失函数值相等,但是单节点的树比较简单,因此取单节点树,即对树进行剪枝

因此对*T*<sub>0</sub>中的每个内部节点都可以计算一个两者相等时的*α*值：

$$
g(t) = \\frac{C(t)-C\_\\alpha(T\_t)}{|T\_t|-1}
$$
表示剪枝后整体损失函数减少的程度

剪枝过程就为：对*T*<sub>0</sub>的每个内部节点计算*g*(*t*<sub>*i*</sub>),剪去有最小*g*(*t*<sub>*i*</sub>)的内部节点的子节点，得到子树*T*<sub>*i*</sub>,然后继续对*T*<sub>*i*</sub>进行剪枝,直到根节点；对于得到的子树序列*T*<sub>1</sub>, *T*<sub>2</sub>, ..., *T*<sub>*n*</sub>通过交叉验证的方法选择最优的子树*T*<sub>*α*</sub>,此时也可以确定相应的*α*了

#### Sci-kit learn中的剪枝参数

上面所讲的剪枝方法称为后剪枝(post
pruning),即在树构建好了之后再去进行修剪;与之对应的是预剪枝,也就是在构建树的过程中限制树的生长来减少过拟合

Sci-kit learn提供了一些**预剪枝**的参数：

-   `max_depth` int, default=None;树的最大深度
-   `min_samples_split`和`min_samples_leaf` int/float
    如果是整数,则表示绝对数量;如果是浮点数,则表示占样本总数的比例;`min_samples_split`为内部节点进行切割所需的最小样本数,`min_samples_leaf`为切割后形成的叶节点内所含的最小样本数
-   `min_weight_fraction_leaf`：该参数一般和`class_weight`参数一起使用,主要解决不平衡的样本问题(某一类或几类比其他的类占比要大得多);对于不平衡的样本可以使用`class_weight`指定权重(使用字典指定类的权重{class\_label:
    weight}或者直接用`balance`表示自动平衡各类),然后使用`min_weight_fraction_leaf`来指定在每个叶节点所必须的最小权重比例(占总权重)
-   `max_feature`:随机选择max\_feature数量的特征进行最优化,有多种选择，具体可以参考官网
-   `min_impurity_decrease`:
    设定不纯度下降的最小值，只有大于设定阈值的分割才会发生

Sci-kit
learn使用的**后剪枝策略**就是上面讲的CART的剪枝算法,提供的参数为`ccp_alpha`;上面提到剪枝过程是逐次选择最小*g*(*t*<sub>*i*</sub>)的内部节点进行剪枝,因此我们所选择的*g*(*t*<sub>*i*</sub>)是逐渐增大的，**当*g*(*t*<sub>*i*</sub>)大于`ccp_alpha`的时候就停止剪枝**

## 其他参数

上面已经用过了`DecisionTreeRegressor`和`DecisionTreeClassifier`类的中的一些参数,现在来看一下其他的参数

-   `criterion`:可选gini或者entropy；表示不纯度的衡量指标
-   `random_state`:随机种子数，Sci-kit
    learn在选择最优的split的时候，并不是选择所有的特征，而是随机选择一部分特征(数量由`max_features`来控制)，从中选择不纯度指标最优的特征进行分割，因此具有“随机化”
-   `splitter`:
    有两个选项：`best`和`random`;两者在对每个feature选择阈值来分割时有区别：`best`是使用不纯度指标来评估每个可能的阈值，从而找到最优的切分点，而`random`是利用一个均匀随机抽样的函数(函数输入是特征的最小值,最大值和上面提到的random\_state，也就是说依据均匀分布在相应特征的取值范围内随机选一个值作为分割点)；因此**使用`random`参数带来的随机化可以在一定程度上减少过拟合**

## 实例

### 参考资料

<https://www.bilibili.com/video/BV1ut41197F6?from=search&seid=9344266940719140153>

<https://www.bilibili.com/video/BV1ZK4y1b7Xt>

<https://www.bilibili.com/video/BV1MA411J7wm>

<https://towardsdatascience.com/how-to-tune-a-decision-tree-f03721801680>

李航统计学习

Sci-Kit learn 官网教程
