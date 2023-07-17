---
title: 【hands on ML Ch4】-训练模型
date: 2021-02-08 10:00:00    
index_img: img/image1.jpg
categories:
  - 机器学习
---

hands on ML 主要包括线性回归，多项式回归，逻辑回归，softmax回归，梯度下降和正则化方法
<!-- more -->
本章主要包括：

-   线性回归模型
-   多项式回归模型
-   逻辑回归模型
-   Softmax回归模型
-   一些正则化的技术
-   梯度下降

## 线性回归

一般线性回归的表示行形式为：输入特征的加权求和再加上截距项(或者叫做bias term) *ŷ* = *θ*<sub>0</sub> + *θ*<sub>1</sub>*x*<sub>1</sub> + *θ*<sub>2</sub>*x*<sub>2</sub> + ... + *θ*<sub>*n*</sub>*x*<sub>*n*</sub> (*ŷ*是预测值，n是特征数量，*x*<sub>*i*</sub>是特征值，*θ*<sub>*j*</sub>是模型参数)，也可以写成向量形式：*ŷ* = *h*<sub>*θ*</sub>(*x*) = *θ*<sup>*T*</sup>*X* (*θ*是参数向量，X是输入特征向量)

在第二章中已经讲过衡量一个线性回归模型常用的指标是RMSE，因此我们可以通过最小化RMSE来找到参数*θ*,为了简化计算，在实际操作中我们是最小化MSE的(MSE最小化，平方根自然也就是最小的)：

$$
MSE(X,h_{\theta})=\frac{1}{m}\sum_{i=1}^{m}(\theta^TX^{(i)}-y^{(i)})^2
$$

求使损失函数最小的*θ*最直接的方法就是进行数学求解(解析解，也叫normal equation)，MSE的Normal Equation为：*θ̂* = (*X*<sup>*T*</sup>*X*)<sup> − 1</sup>*X*<sup>*T*</sup>*y*，我们可以来验证一下：

``` python
##生成数据
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

x = 2 * np.random.rand(100,1)
y = 4 + 3 * x + np.random.randn(100,1)
plt.plot(x, y, "b.")
>> [<matplotlib.lines.Line2D object at 0x0000021801228460>]
plt.xlabel("$x_1$", fontsize=18)
>> Text(0.5, 0, '$x_1$')
plt.ylabel("$y$", rotation=0, fontsize=18)
>> Text(0, 0.5, '$y$')
plt.axis([0, 2, 0, 15])
>> (0.0, 2.0, 0.0, 15.0)
plt.show()
```

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/unnamed-chunk-1-1.png" width="672" />

计算*θ*的Normal equation:

``` python
x_b = np.c_[np.ones((100,1)),x]##x_0 = 1
theta_best = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y)

theta_best
>> array([[4.12634538],
>>        [2.81427971]])
```

`np.c_`进行的是增加列的操作(R里面的cbind);`np.ones((100,1))`产 100行1列的矩阵，元素都是1;`np.linalg`是numpy中线性代数模块;`inv`是矩阵求逆方法;`T`是矩阵转置方法;`dot`是矩阵乘法。

现在我们使用计算出的*θ̂*来预测：

``` python
x_new = np.array([[0],[2]])
x_new_b = np.c_[np.ones((2,1)),x_new]
y_pre = x_new_b.dot(theta_best)
y_pre
>> array([[4.12634538],
>>        [9.7549048 ]])
```

``` python
plt.plot(x_new,y_pre,"r-")
>> [<matplotlib.lines.Line2D object at 0x000002180339EFA0>]
plt.plot(x,y,"b.")
>> [<matplotlib.lines.Line2D object at 0x00000218011C2EB0>]
plt.axis([0,2,0,15])
>> (0.0, 2.0, 0.0, 15.0)
plt.show()
```

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/unnamed-chunk-4-1.png" width="672" />

在Scikit-Learn中可以使用`LinearRegression`来方便的进行线性回归的计算：

``` python
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(x,y)
>> LinearRegression()
lin_reg.intercept_, lin_reg.coef_
>> (array([4.12634538]), array([[2.81427971]]))
lin_reg.predict(x_new)
>> array([[4.12634538],
>>        [9.7549048 ]])
```

`LinearRegression`类是基于`scipy.linalg.lstsq`函数的，该函数是通过SVD进行计算pseudoinverse(*X*<sup>+</sup>)然后再计算*θ̂* = *X*<sup>+</sup>*y*,这样计算有两个好处：pseudoinverse的计算比直接计算矩阵的逆效率更高(why?)；当*X*<sup>*T*</sup>*X*不可逆的时候NormalEquation是无法计算的，而pseudoinverse是可以计算的

计算Normal Equation的计算复杂度是比较大的(求矩阵的逆的计算复杂度为 *O*(*n*<sup>2.4</sup>)\~*O*(*n*<sup>3</sup>),使用SVD方法的计算复杂度为*O*(*n*<sup>2</sup>))

## 梯度下降

### 数学理论

这一部分参考李宏毅老师的机器学习课程

现在的问题是：找到*θ*<sup>\*</sup>：  
$$
\theta^* = argmin_{\theta}L(\theta)
$$
*L*(*θ*)是损失函数，

现在假设*θ*由两个参数构成：{*θ*<sub>1</sub>,*θ*<sub>2</sub>}, *L*(*θ*)的等高线如下图：  
![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210227170016311.png)

给定一个点，我们是否可以在其邻域内找到一个使*L*(*θ*)最小的点然后向这个点移动最终到达全局最小点(如上图)；那么怎样找到这个点呢？

这里需要引入**[泰勒级数](https://www.bilibili.com/video/BV1Gx411Y7cz?from=search&seid=4438787146009065334)**的概念：**泰勒级数利用函数在某个点的导数来近似在这个点附近的函数值**,数学表示为：
在*x* = *x*<sub>0</sub>附近有：
$$
h(x) = h(x_0)+h^{'}(x_0)(x-x_0)+\frac{h^{''}(x_0)}{2!}(x-x_0)^2+...
$$
当x接近*x*<sub>0</sub>的时候可以将高次式忽略：
$$
h(x) \approx  h(x_0)+h^{'}(x_0)(x-x_0)
$$
对于多个变量也是类似的：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210227180156655.png)

回到上面的问题:

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210227180518842.png)

**如果红色的圆圈足够小**，我们就可以使用泰勒级数来近似损失函数：
$$
L(\theta) \approx  L(a,b)+\frac{\partial L(a,b)}{\partial \theta_1}(\theta_1-a)+\frac{\partial L(a,b)}{\partial \theta_2}(\theta_2-b)
$$
令$s=L(a,b)$,$u=\frac{\partial L(a,b)}{\partial \theta_1}$,$v=\frac{\partial L(a,b)}{\partial \theta_2}$, 将上式简化:
$$
L(\theta) \approx s + u(\theta_1-a)+v(\theta_2-b)
$$
我们现在的问题就是：在红色的圆圈内找到*θ*<sub>1</sub>和*θ*<sub>2</sub>使得*L*(*θ*)最小，

如果使$\theta_1-a=\Delta \theta_1$, $\theta_2-b=\Delta \theta_2$，那么*L*(*θ*)就可以表示为两个向量的乘积：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210227182845081.png)

要使*L*(*θ*)最小，那么就要使这两个向量反向(并且$(\Delta \theta_1,\Delta \theta_2)$在圆上)：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210227183501439.png)

这个就是梯度下降的形式：
$$
\theta^i = \theta^{i-1} - \eta \bigtriangledown L(\theta^{i-1})
$$


### 梯度下降的注意事项

#### 学习率的调整

学习率(*η*)是一个重要的超参数，决定了梯度下降的步伐有多大;如果学习率比较小,那么收敛到最小值需要迭代的次数就比较多，如果学习率比较大,那么就可能跳过了最小值，甚至有可能比起始值还要大：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210227185426216.png)

除了手动设定学习率之外，我们还可以使学习率随着训练的进行逐渐减少(在每次迭代时，决定学习率的函数叫做*learning schedule*)。

#### 随机梯度下降和 mini-batch 梯度下降

上面提到的损失函数都是对所有的训练数据来计算的(所有预测值和真实值的误差和)，而随机梯度下降所使用的计算梯度的函数是随机选取的观测值的预测值和真实值的误差(只看一个点)，而 mini-batch 则是取一小部分数据进行梯度下降，下面是不同方法的比较：

![image-20210704111451263](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210704111451263.png)

#### 特征的归一化

下面的图比较形象的表示了归一化对学习的影响：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210227210417449.png)

如果两个特征的范围不一样，那么在更新参数时对损失函数的下降的贡献就会不一样。

在Scikit learn中可以使用`SGDRegressor`来进行随机梯度下降求解线性回归模型：

``` python
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(max_iter=1000,tol=1e-3,penalty=None,eta0=0.1)
sgd_reg.fit(x,y.ravel())##ravel将列向量转为一维向量
>> SGDRegressor(eta0=0.1, penalty=None)
```

`max_iter`表示epoch的数目(epoch指全部训练数据都被模型“看了”一遍)；`tol`表示如果在某一个epoch上损失函数下降小于tol的数值，则训练停止；`penalty`表示正则化(后面讲);`eta0`表示初始的学习率大小，默认的学习率是:$eta0/pow(t,power\_t)$, power_t的默认值是0.25

## 多项式回归

可以使用线性模型来拟合非线性的数据，一个简单的做法就是将每个特征加上幂次作为新的特征，然后对这些拓展的特征进行训练线性模型，这个技术叫做**多项式回归(polynomial regression)**

``` python
##模拟数据
m = 100
np.random.seed(123)
x = 6 * np.random.rand(m,1) - 3 ##均匀分布
y = 0.5 * x**2 + x + 2 + np.random.randn(m,1)##正态分布

plt.plot(x, y, "b.")
>> [<matplotlib.lines.Line2D object at 0x000002180E9B1910>]
plt.xlabel("$x_1$", fontsize=18)
>> Text(0.5, 0, '$x_1$')
plt.ylabel("$y$", rotation=0, fontsize=18)
>> Text(0, 0.5, '$y$')
plt.axis([-3, 3, 0, 10])
>> (-3.0, 3.0, 0.0, 10.0)
plt.show()
```

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/unnamed-chunk-7-1.png" width="672" />

使用`PolynomialFeatures`类将特征加上平方后作为新的特征：

``` python
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2,include_bias=False)
x_poly = poly_features.fit_transform(x)

x[0]
>> array([1.17881511])
x_poly[0]
>> array([1.17881511, 1.38960507])
```

然后重新训练模型：

``` python
lin_reg = LinearRegression()
lin_reg.fit(x_poly,y)
>> LinearRegression()
lin_reg.intercept_, lin_reg.coef_
>> (array([2.03146145]), array([[0.95505451, 0.50182851]]))
```

预测：

``` python
x_new=np.linspace(-3, 3, 100).reshape(100, 1)
x_new_poly = poly_features.transform(x_new)
y_new = lin_reg.predict(x_new_poly)
plt.plot(x, y, "b.")
>> [<matplotlib.lines.Line2D object at 0x000002180ECF2C40>]
plt.plot(x_new, y_new, "r-", linewidth=2, label="Predictions")
>> [<matplotlib.lines.Line2D object at 0x000002180ECF2E80>]
plt.xlabel("$x_1$", fontsize=18)
>> Text(0.5, 0, '$x_1$')
plt.ylabel("$y$", rotation=0, fontsize=18)
>> Text(0, 0.5, '$y$')
plt.legend(loc="upper left", fontsize=14)
>> <matplotlib.legend.Legend object at 0x000002180EB14490>
plt.axis([-3, 3, 0, 10])
>> (-3.0, 3.0, 0.0, 10.0)
plt.show()
```

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/unnamed-chunk-10-1.png" width="672" />

需要注意的是：`PolynomialFeatures(degree=d)`会将原来的n个特征变成$\\frac{(n+d)!}{d!n!}$个特征；比如有两个特征a,b,经过自由度为3的PolynomialFeatures转化后就有10个特征(包括1),要注意特征爆炸的问题

## 学习曲线

使用高自由度的多项式回归模型可能会在训练集上过拟合，然而简单的线性模型可能是欠拟合的，那么我们该怎样决定模型的复杂程度或者说判断模型是过拟合还是欠拟合呢？

在第二章中，使用了交叉验证的方法来估计模型的泛化能力；如果一个模型在训练集上表现的比较好但是依据交叉验证的指标，其泛化能力比较差(在验证集上表现不好)，那么这个模型就是过拟合；如果一个模型在训练集和验证集上表现都不好，那么这个模型是欠拟合的。

另外一个方法就是检查**学习曲线**(learning curves),**学习曲线展示了模型在训练集和验证集上的表现和训练集大小或者训练的迭代次数之间的关系**;要画这个图，需要在不同大小的训练集的子集上训练模型，得到模型的表现指标。

我们先来画一个简单线性回归的学习曲线：

``` python
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def plot_learning_curves(model,x,y):
  x_train,x_val,y_train,y_val = train_test_split(x,y,test_size=0.2)
  train_errors,val_errors = [],[]
  for m in range(1,len(x_train)):
    model.fit(x_train[:m],y_train[:m])
    y_train_predict = model.predict(x_train[:m])
    y_val_predict = model.predict(x_val)
    train_errors.append(mean_squared_error(y_train[:m],y_train_predict))
    val_errors.append(mean_squared_error(y_val,y_val_predict))
  
  plt.plot(np.sqrt(train_errors),"r-+",linewidth=2,label="train")
  plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
  plt.legend(loc="upper right", fontsize=14)  
  plt.xlabel("Training set size", fontsize=14)
  plt.ylabel("RMSE", fontsize=14) 
    
    
lin_reg = LinearRegression()
plot_learning_curves(lin_reg,x,y)
plt.axis([0, 80, 0, 3]) 
>> (0.0, 80.0, 0.0, 3.0)
plt.show()        
```

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/unnamed-chunk-11-1.png" width="672" />

当只有一两个训练数据的时候，模型拟合的非常好，同时由于训练集较少，泛化能力较弱所以在验证集中表现不好；当训练集逐渐增大，一方面由于数据的噪音，另一方面因为模型是线性的，而数据不是线性的，所以模型在训练集上的误差上升，但是由于训练集增多，泛化能力会一定程度的上升，所以在验证集上的误差降低，最终两者都到达一个平台。

这个学习曲线是一个典型的欠拟合的模型的特征：**两个曲线都到达一个平台；并且两者比较接近，都比较高**。

接下来看一下有10个自由度的多项式回归模型的学习曲线：

``` python
from sklearn.pipeline import Pipeline

polynomial_regression = Pipeline([
        ("poly_features", PolynomialFeatures(degree=20, include_bias=False)),
        ("lin_reg", LinearRegression()),
    ])

plot_learning_curves(polynomial_regression, x, y)
plt.axis([0, 80, 0, 3])
>> (0.0, 80.0, 0.0, 3.0)
plt.show()           
```

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/unnamed-chunk-12-1.png" width="672" />

这个学习曲线也有两个特征：

-   在训练集上的误差比上面的线性回归模型要低
-   在两个曲线间有一个gap，这意味着模型在训练集上比在验证集上的表现要好得多，而这是**过拟合**的特征(可能需要收集更多的数据)

## BIAS/VARIANCE TRADE-OFF

## 正则化线性模型

在第一章和第二章已经讲过了减少过拟合风险的方法之一就是正则化模型(也就是约束模型)；对于多项式模型最简单的正则化方法就是减少模型的自由度；对于线性模型，正则化一般是通过约束模型的权重来实现，常用的有3种方法：岭回归(Ridge Regression),Lasso回归,弹性网络(Elastic Net)。

### 岭回归

岭回归就是在线性回归的损失函数后面加上了一个正则化的项:

$$
J(\theta) = MSE(\theta) + \alpha \frac{1}{2}\sum_{i=1}^{n}\theta_i^2
$$
加上这一项之后就会使得模型在训练的过程中尽量保持特征权重(*θ*)比较小。  

> 注意：在岭回归等正则化的模型中，训练时使用的损失函数与计算模型性能时用的指标不一定相同(在分类模型中更是如此)；另外在训练正则化的模型时，对特征一定要归一化

下图，左边是线性回归使用岭正则化，右图是多项式回归使用岭正则化，展示了不同*α*值时的情况：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210301230247251.png)

可以看到增加*α*会是曲线更加平缓(减少了variance但是增加了bias)。

对于岭回归，和线性回归一样，可以使用normal equation的方法或者梯度下降的方法求解：

``` python
##Normal equation
from sklearn.linear_model import Ridge

ridge_reg = Ridge(alpha=1,solver="cholesky")
ridge_reg.fit(x,y)
>> Ridge(alpha=1, solver='cholesky')
ridge_reg.predict([[1.5]])
>> array([[4.58785445]])
```

``` python
##梯度下降
sgd_reg = SGDRegressor(penalty="l2")
sgd_reg.fit(x,y.ravel())
>> SGDRegressor()
sgd_reg.predict([[1.5]])
>> array([4.56218836])
```

“l2”指的是L2范数(norm);*L*<sub>*p*</sub>范数的定义为：
$$
||x||_p = \sqrt[p]{\sum_i |x_i|^p}
$$
因此L2范数为：
$$
||x||_2 = \sqrt[2]{\sum_i |x_i|^2}
$$

所以岭回归的正则化项就是*α* 1/2(\|\|*w*\|\|<sub>2</sub>)<sup>2</sup>,w是*θ*<sub>1</sub>到*θ*<sub>*n*</sub>的参数向量(特征权重)

### Lasso回归

Lasso的全称为Least Absolute Shrinkage and Selection Operator ，和岭回归类似也是在损失函数后面加上一个正则化项，只不过Lasso加的是L1范数：
$$
J(\theta) = MSE(\theta) + \alpha\sum_{i=1}^n|\theta_i|
$$
lasso回归可以用来进行特征选择(why)

上面那个损失函数在*θ*<sub>*i*</sub> = 0的地方是不可微分的，但是可以通过将梯度向量替换成次梯度向量(subgradient vector)来解决这个问题：
$$
g(\theta,J)=\triangledown MSE(\theta)+ \alpha 	\left(               
\begin{array}{cccc}
 sign(\theta_1)\\
 sign(\theta_2)\\
 \vdots  \\
 sign(\theta_n)
\end{array}
\right ) where \ sign(\theta_n)= \begin{cases}
-1\ if\ \theta_i <0 \\
0\ \ if\ \theta_i =0 \\
+1 \ if\ \theta_i >0 
\end{cases}
$$

在Scikit-Learn中可以使用`Lasso`或者`SGDRegressor`(指定l1范数的惩罚项)：

``` python
from sklearn.linear_model import Lasso

lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(x,y)
>> Lasso(alpha=0.1)
lasso_reg.predict([[1.5]])
>> array([4.52578706])
sgd_lasso = SGDRegressor(penalty="l1")
sgd_lasso.fit(x,y.ravel())
>> SGDRegressor(penalty='l1')
sgd_lasso.predict([[1.5]])
>> array([4.57060493])
```

### 弹性网络

弹性网络(Elastic Net)是岭回归和lasso回归中间的“调和”，其正则化项是岭回归和lasso回归的正则化项的混合，可以通过*r*来控制混合的比例:
$$
J(\theta)=MSE(\theta)+r\alpha\sum_{i=1}^n|\theta_i|+\frac{1-r}{2}\alpha\sum_{i=1}^n\theta_i^2
$$

什么时候使用单独的线性回归，什么时候使用正则化的模型，这些正则化方法应该选哪个；一般来说要避免使用单独的线性回归，所以更多的情况下是使用正则化的模型，当我们知道特征中只有一部分是有用的，可以使用lasso或者弹性网络来选择变量；另外尽可能的使用弹性网络，因为**当特征的数量比训练样本的数量要多或者几个特征间相关性比较强时，lasso表现不稳定**。

sklearn中的`ElasticNet`可以用来建立弹性网络模型：

``` python
from sklearn.linear_model import ElasticNet
elastic_net = ElasticNet(alpha=0.1,l1_ratio=0.5)##l1_ratio指的是r
elastic_net.fit(x,y)
>> ElasticNet(alpha=0.1)
elastic_net.predict([[1.5]])

##也可以使用SGDRgressor
>> array([4.52788619])
sgd_elastic = SGDRegressor(penalty="elasticnet",alpha=0.1,l1_ratio=0.5)
sgd_elastic.fit(x,y.ravel())
>> SGDRegressor(alpha=0.1, l1_ratio=0.5, penalty='elasticnet')
sgd_elastic.predict([[1.5]])
>> array([4.51766322])
```

### Early Stopping

另一个方法去正则化迭代的学习算法(如梯度下降)是：当验证集误差达到最小值的时候就停止训练；这种方法叫做**early stopping**,如下图所示：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210304083008485.png)

当学习算法学习的时候，在训练集和验证集上的误差都会降低，但是一段时间之后会出现在验证集上的误差上升的情况，这意味着模型开始过拟合，因此最好在未过拟合之前就停止训练模型(验证集误差最低)。

注意：在随机梯度下降或者小批次梯度下降中，曲线不会像上图那样平滑，因此很难判定是否达到最小值；一个解决方法就是：当训练一段时间之后，验证集的误差一直比最小值要高(每一个epoch之后就把验证集误差和之前所有的误差比较，看看是不是最小值，进行迭代更新)，就停止训练，记录下验证集误差最小时的模型参数：

``` python
##data
np.random.seed(42)
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 2 + X + 0.5 * X**2 + np.random.randn(m, 1)

X_train, X_val, y_train, y_val = train_test_split(X[:50], y[:50].ravel(), test_size=0.5, random_state=10)
```

``` python
from copy import deepcopy
from sklearn.preprocessing import StandardScaler

##数据预处理
poly_scaler = Pipeline([
        ("poly_features", PolynomialFeatures(degree=90, include_bias=False)),
        ("std_scaler", StandardScaler())
    ])

X_train_poly_scaled = poly_scaler.fit_transform(X_train)
X_val_poly_scaled = poly_scaler.transform(X_val)

sgd_reg = SGDRegressor(max_iter=1, tol=-np.infty, warm_start=True,
                       penalty=None, learning_rate="constant", eta0=0.0005, random_state=42)

minimum_val_error = float("inf")
best_epoch = None
best_model = None
for epoch in range(1000):
    sgd_reg.fit(X_train_poly_scaled, y_train)  # continues where it left off
    y_val_predict = sgd_reg.predict(X_val_poly_scaled)
    val_error = mean_squared_error(y_val, y_val_predict)
    if val_error < minimum_val_error:
        minimum_val_error = val_error
        best_epoch = epoch
        best_model = deepcopy(sgd_reg)
```

``` python
best_epoch
>> 239
minimum_val_error
>> 1.3513110512453865
```

首先预处理步骤对数据进行多项式转化，然后进行归一化；SGDRegressor参数中设置max_iter=1意思是每次训练只进行一个epoch(因为后面显式地进行epoch的迭代),tol前面讲过(如果在某一个epoch上损失函数下降小于tol的数值，则训练停止),warm_start=T表示调用fit时会使用上次训练得到的模型参数作为初始值继续进行训练(热启动),random_state表示当对每个新的epoch都会进行shuffle(默认)时取的随机种子数，来保证结果可重复

## 逻辑回归

逻辑回归可以用来估计某个实例属于某一类别的概率，如果概率大于50%，则认为该实例属于该类(1),否则不属于该类(0),因此是二分类的分类器

### 估计概率
