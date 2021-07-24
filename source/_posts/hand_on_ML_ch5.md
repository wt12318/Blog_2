---
title: 【Hands on ML ch5】- 支持向量机（SVM）
author: wutao
date: 2021-06-20 23:00:00
categories:
  - ML
tags:
  - python
index_img : /img/image1.jpg
---



Hands on ML 第五章

<!-- more -->

## 线性SVM分类器

线性SVM包含数据线性可分的线性支持向量机模型,又称为**硬间隔SVM**和数据线性不可分的线性支持向量机模型,又称为**软间隔SVM**;两者的区别在于：硬间隔不允许间隔内有样本点存在,而软间隔可以

在`Scikit-Learn`中可以指定超参数`C`来决定对间隔中的点“惩罚”强度(见[SVM理论部分](https://wutaoblog.com.cn/2021/06/19/svm_theory/)软间隔最大化的优化问题);如果C比较小,那么惩罚比较小,所以在间隔中的点也比较多,相反,如果C比较大,在间隔中的点就比较少:

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210403220433035.png)

因此,在SVM模型过拟合的时候可以考虑减少C；下面是一个在鸢尾花数据集上训练SVM模型的例子:

``` python
import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

iris = datasets.load_iris()
X = iris["data"][:,(2,3)]##只选长度和宽度两个特征
y = (iris["target"]==2).astype(np.float64) ##virginica

svm_clf = Pipeline([
  ("scaler",StandardScaler()),
  ("linear_svc",LinearSVC(C=1,loss="hinge")),##合页损失函数
])

svm_clf.fit(X,y)
>> Pipeline(steps=[('scaler', StandardScaler()),
>>                 ('linear_svc', LinearSVC(C=1, loss='hinge'))])
```

``` python
##predict
svm_clf.predict([[5.5,1.7]])
>> array([1.])
```

除了使用 `linearSVC` 类外，也可以用 `SVC`类（线性核）：`SVC(kernel="linear", C=1)` 或者使用 `SGDClassifier` 类，用随机梯度下降的方式来优化合页损失函数 `SGDClassifier(loss="hinge", alpha=1/(m*C))`（m 是 batch 的大小）

## 非线性 SVM 分类器

通过核技巧将线性不可分的数据映射到高维空间，使其更有可能线性可分，通常使用的核函数有：多项式核函数和高斯核函数，可以通过 SVC 来使用各种核函数，核函数的表达式和参数如下表：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210620214739030.png)

其中 coef0 表示式子中的常数项 r。 下面以 moon 数据集为例：

``` python
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=100, noise=0.15, random_state=42)

def plot_dataset(X, y, axes):
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^")
    plt.axis(axes)
    plt.grid(True, which='both')
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)

plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
plt.show()
```

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/unnamed-chunk-1-1.png" width="672" />

首先来看多项式核函数，分别使用 d=1,r=1,C=5 以及 d=10,r=100,C=5：

``` python
from sklearn.svm import SVC

poly_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
    ])
poly_kernel_svm_clf.fit(X, y)
>> Pipeline(steps=[('scaler', StandardScaler()),
>>                 ('svm_clf', SVC(C=5, coef0=1, kernel='poly'))])
poly100_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=10, coef0=100, C=5))
    ])
poly100_kernel_svm_clf.fit(X, y)

##画图
>> Pipeline(steps=[('scaler', StandardScaler()),
>>                 ('svm_clf', SVC(C=5, coef0=100, degree=10, kernel='poly'))])
def plot_predictions(clf, axes):
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    y_decision = clf.decision_function(X).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
    plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)
    

fig, axes = plt.subplots(ncols=2, figsize=(10.5, 4), sharey=True)

plt.sca(axes[0])
plot_predictions(poly_kernel_svm_clf, [-1.5, 2.45, -1, 1.5])
plot_dataset(X, y, [-1.5, 2.4, -1, 1.5])
plt.title(r"$d=3, r=1, C=5$", fontsize=18)
>> Text(0.5, 1.0, '$d=3, r=1, C=5$')
plt.sca(axes[1])
plot_predictions(poly100_kernel_svm_clf, [-1.5, 2.45, -1, 1.5])
plot_dataset(X, y, [-1.5, 2.4, -1, 1.5])
plt.title(r"$d=10, r=100, C=5$", fontsize=18)
>> Text(0.5, 1.0, '$d=10, r=100, C=5$')
plt.ylabel("")
>> Text(0, 0.5, '')
plt.show()
```

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/unnamed-chunk-1-3.png" width="1008" />

可以看到自由度越高（d越大），决策边阶越不规则，模型就有过拟合的倾向；接下来看一下高斯核函数：

``` python
from sklearn.svm import SVC

gamma1, gamma2 = 0.1, 5
C1, C2 = 0.001, 1000
hyperparams = (gamma1, C1), (gamma1, C2), (gamma2, C1), (gamma2, C2)

svm_clfs = []
for gamma, C in hyperparams:
    rbf_kernel_svm_clf = Pipeline([
            ("scaler", StandardScaler()),
            ("svm_clf", SVC(kernel="rbf", gamma=gamma, C=C))
        ])
    rbf_kernel_svm_clf.fit(X, y)
    svm_clfs.append(rbf_kernel_svm_clf)
>> Pipeline(steps=[('scaler', StandardScaler()),
>>                 ('svm_clf', SVC(C=0.001, gamma=0.1))])
>> Pipeline(steps=[('scaler', StandardScaler()),
>>                 ('svm_clf', SVC(C=1000, gamma=0.1))])
>> Pipeline(steps=[('scaler', StandardScaler()),
>>                 ('svm_clf', SVC(C=0.001, gamma=5))])
>> Pipeline(steps=[('scaler', StandardScaler()),
>>                 ('svm_clf', SVC(C=1000, gamma=5))])
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10.5, 7), sharex=True, sharey=True)

for i, svm_clf in enumerate(svm_clfs):
    plt.sca(axes[i // 2, i % 2])
    plot_predictions(svm_clf, [-1.5, 2.45, -1, 1.5])
    plot_dataset(X, y, [-1.5, 2.45, -1, 1.5])
    gamma, C = hyperparams[i]
    plt.title(r"$\gamma = {}, C = {}$".format(gamma, C), fontsize=16)
    if i in (0, 1):
        plt.xlabel("")
    if i in (1, 3):
        plt.ylabel("")
>> Text(0.5, 1.0, '$\\gamma = 0.1, C = 0.001$')
>> Text(0.5, 0, '')
>> Text(0.5, 1.0, '$\\gamma = 0.1, C = 1000$')
>> Text(0.5, 0, '')
>> Text(0, 0.5, '')
>> Text(0.5, 1.0, '$\\gamma = 5, C = 0.001$')
>> Text(0.5, 1.0, '$\\gamma = 5, C = 1000$')
>> Text(0, 0.5, '')
plt.show()
```

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/unnamed-chunk-1-5.png" width="1008" />

gamma 相当于一个正则化的超参数，当 gamma 增加时，决策边界就会变窄，变得不规则，实例的影响范围就会变小（对噪音的容忍度），因此如果模型过拟合可以考虑减小 gamma。

上面也提到在 scikit-learn 中有 3 种方法来调用 SVM，这三种方法的优化算法以及是否支持核函数方面有所不同，下表是三者的比较：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210620230916381.png)

## SVM 回归

SVM 做分类的时候是想找到一个两类之间的最大的“街道”，而且限制实例对“街道”边界的越界；在做回归任务的时候则相反，想要找到一条街道，使得尽可能多的实例点在这个街道上，并且对那些不在街道上的点进行惩罚。使用 `LinearSVR` 或者 `SVR`（核函数）可以进行 SVM 回归；街道的宽度由超参数 epsilon 控制，下面是对一些随机数据的例子：

``` python
np.random.seed(42)
m = 50
X = 2 * np.random.rand(m, 1)
y = (4 + 3 * X + np.random.randn(m, 1)).ravel()

from sklearn.svm import LinearSVR

svm_reg = LinearSVR(epsilon=1.5, random_state=42)
svm_reg.fit(X, y)
>> LinearSVR(epsilon=1.5, random_state=42)
svm_reg1 = LinearSVR(epsilon=1.5, random_state=42)
svm_reg2 = LinearSVR(epsilon=0.5, random_state=42)
svm_reg1.fit(X, y)
>> LinearSVR(epsilon=1.5, random_state=42)
svm_reg2.fit(X, y)
>> LinearSVR(epsilon=0.5, random_state=42)
def find_support_vectors(svm_reg, X, y):
    y_pred = svm_reg.predict(X)
    off_margin = (np.abs(y - y_pred) >= svm_reg.epsilon)
    return np.argwhere(off_margin)

svm_reg1.support_ = find_support_vectors(svm_reg1, X, y)
svm_reg2.support_ = find_support_vectors(svm_reg2, X, y)

eps_x1 = 1
eps_y_pred = svm_reg1.predict([[eps_x1]])

def plot_svm_regression(svm_reg, X, y, axes):
    x1s = np.linspace(axes[0], axes[1], 100).reshape(100, 1)
    y_pred = svm_reg.predict(x1s)
    plt.plot(x1s, y_pred, "k-", linewidth=2, label=r"$\hat{y}$")
    plt.plot(x1s, y_pred + svm_reg.epsilon, "k--")
    plt.plot(x1s, y_pred - svm_reg.epsilon, "k--")
    plt.scatter(X[svm_reg.support_], y[svm_reg.support_], s=180, facecolors='#FFAAAA')
    plt.plot(X, y, "bo")
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.legend(loc="upper left", fontsize=18)
    plt.axis(axes)

fig, axes = plt.subplots(ncols=2, figsize=(9, 4), sharey=True)
plt.sca(axes[0])
plot_svm_regression(svm_reg1, X, y, [0, 2, 3, 11])
plt.title(r"$\epsilon = {}$".format(svm_reg1.epsilon), fontsize=18)
>> Text(0.5, 1.0, '$\\epsilon = 1.5$')
plt.ylabel(r"$y$", fontsize=18, rotation=0)
>> Text(0, 0.5, '$y$')
plt.annotate(
        '', xy=(eps_x1, eps_y_pred), xycoords='data',
        xytext=(eps_x1, eps_y_pred - svm_reg1.epsilon),
        textcoords='data', arrowprops={'arrowstyle': '<->', 'linewidth': 1.5}
    )
>> Text(1, [5.02640746], '')
plt.text(0.91, 5.6, r"$\epsilon$", fontsize=20)
>> Text(0.91, 5.6, '$\\epsilon$')
plt.sca(axes[1])
plot_svm_regression(svm_reg2, X, y, [0, 2, 3, 11])
plt.title(r"$\epsilon = {}$".format(svm_reg2.epsilon), fontsize=18)
>> Text(0.5, 1.0, '$\\epsilon = 0.5$')
plt.show()
```

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/unnamed-chunk-1-7.png" width="864" />

对于非线性的回归任务，就可以使用核技巧，下面使用了 2 个自由度的多项式核：

``` python
np.random.seed(42)
m = 100
X = 2 * np.random.rand(m, 1) - 1
y = (0.2 + 0.1 * X + 0.5 * X**2 + np.random.randn(m, 1)/10).ravel()
from sklearn.svm import SVR

svm_poly_reg1 = SVR(kernel="poly", degree=2, C=100, epsilon=0.1, gamma="scale")
svm_poly_reg2 = SVR(kernel="poly", degree=2, C=0.01, epsilon=0.1, gamma="scale")
svm_poly_reg1.fit(X, y)
>> SVR(C=100, degree=2, kernel='poly')
svm_poly_reg2.fit(X, y)
>> SVR(C=0.01, degree=2, kernel='poly')
fig, axes = plt.subplots(ncols=2, figsize=(9, 4), sharey=True)
plt.sca(axes[0])
plot_svm_regression(svm_poly_reg1, X, y, [-1, 1, 0, 1])
plt.title(r"$degree={}, C={}, \epsilon = {}$".format(svm_poly_reg1.degree, svm_poly_reg1.C, svm_poly_reg1.epsilon), fontsize=18)
>> Text(0.5, 1.0, '$degree=2, C=100, \\epsilon = 0.1$')
plt.ylabel(r"$y$", fontsize=18, rotation=0)
>> Text(0, 0.5, '$y$')
plt.sca(axes[1])
plot_svm_regression(svm_poly_reg2, X, y, [-1, 1, 0, 1])
plt.title(r"$degree={}, C={}, \epsilon = {}$".format(svm_poly_reg2.degree, svm_poly_reg2.C, svm_poly_reg2.epsilon), fontsize=18)
>> Text(0.5, 1.0, '$degree=2, C=0.01, \\epsilon = 0.1$')
plt.show()
```

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/unnamed-chunk-1-9.png" width="864" />
