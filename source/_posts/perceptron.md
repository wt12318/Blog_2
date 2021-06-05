---
title: 感知机模型
date: 2021-03-14 16:47:27
tags: 机器学习
index_img: img/per.png
---







感知机模型

<!-- more -->

机器学习方法都是由3个要素构成的：

-   模型：包含输入空间,输出空间和假设空间(包含所有可能的决策函数)
-   策略：按照什么样的准则选择最优的模型(损失函数)
-   算法：如何找到最优模型(最优化问题)

## 感知机模型

输入空间：$X \in R^n$ (n维实数)  
输出空间：$Y = {+1,-1}$  
假设空间：

$$
f(x)=sign(w\cdot x+b)=\left\{ 
\begin{matrix}
+1, w\cdot x+b\ge0\\
-1, w\cdot x+b<0 \\
\end{matrix}
\right.
$$
注意：$w,x,b$都是向量,$w\cdot b$也就是向量的内积,比如在二维空间中：

<center>

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/演示文稿1_01.png)

</center>

就是要找一个直线$w_1x_1+w_2x_2+b=0$将点分成两类(这条直线更一般的名称叫做超平面)；另外感知机模型对数据的假设是:数据是线性可分的;比如下图所示的数据所对应的就不是一个线性可分的输入空间

<center>

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/Rd9f79181b6f2972e0795a5815e8dc3a0.png)

</center>

## 学习策略

感知机的损失函数为：**误分类点到超平面S的总距离**,通过最小化这个距离得到最优的超平面(超平面的参数就是w和b)

首先我们需要一些基础知识：

### 超平面的法向量

对于一个超平面S ($w\cdot x+b$),其法向量为$w$:

设超平面S上有两个点：A点$(x_A)$和B点$(x_B)$有：

$$
\left\{ \begin{matrix}         
wx_A+b=0\\
wx_B+b=0 \\
\end{matrix}\right. \\
\Rightarrow w(x_A-x_B)=0
$$
因为$x_A-x_B$是超平面S上的一个向量,两个向量的乘积为0,所以$w$垂直于S,即$w$为超平面S的法向量

### 点到超平面的距离

输入空间中任一点$x_0$到超平面S ($w\cdot x+b$)的距离d为：

$$
d = \frac{1}{||w||}|w\cdot x_0+b|
$$
<center>

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210314141051985.png)

</center>

设点$x_0$在S上投影为$x_1$,则$w\cdot x_1+b=0$;由于向量$\vec {x_1x_0}$与S的法向量$w$平行,所以：

$$
|\vec w\cdot \vec{x_1x_0}|=||\vec w||×||\vec{x_1x_0}||cos<\vec w,\vec{x_1x_0}>=||\vec w||×||\vec{x_1x_0}||=||\vec w||d
$$
对于$\vec w\cdot \vec{x_1x_0}$又有(假设$w$和$x$都是N维的向量,上面的图只是一个3维的例子)：

$$
\vec w\cdot \vec{x_1x_0}=w^1 (x_1^1-x_0^1)+w^2(x_1^2-x_0^2)+...+w^N(x_1^N-x_0^N) \\ =w^1x_1^1+w^2x_1^2+...+w^Nx_1^N-(w^1x_0^1+w^2x_0^2+...+w^Nx_0^N) \\ =-b-(w^1x_0^1+w^2x_0^2+...+w^Nx_0^N)
$$
因此由上面两个式子,可以得出：

$$
||w||d=|-b-(w^1x_0^1+w^2x_0^2+...+w^Nx_0^N)|=|w\cdot x_0 +b|\\
\Rightarrow d=\frac{|w\cdot x_0 +b|}{||w||} 
$$

回到感知机模型中,因为误分类点$w\cdot x+b$和类标签的符号是相反的(当$w\cdot x+b$大于0时,误分类的类标签是-1;当$w\cdot x+b$小于0时,误分类的类标签是+1),所以误分类点到超平面S的距离也可以表示为:

$$
d_i = \frac{-y_i(w\cdot x_i+b)}{||w||}
$$
误分类点的总距离为：

$$
-\frac{1}{||w||}\sum_{x_i\in M}y_i(w\cdot x_i+b),M为误分类点的集合
$$
所以感知机的损失函数为：

$$
L(w,b)=-\sum_{x_i\in M}y_i(w\cdot x_i+b)
$$

## 学习算法

可以使用梯度下降或者随机梯度下降的方法来求解使损失函数最小化时的参数$w,b$

损失函数$L(w,b)$的梯度为：

$$
\nabla_{w}L(w,b)=\frac{\partial L(w,b)}{\partial w}=-\sum_{x_i\in M}y_ix_i \\
\nabla_{b}L(w,b)=\frac{\partial L(w,b)}{\partial b}=-\sum_{x_i\in M}y_i
$$

所以按照梯度下降法,对每个误分类点更新w,b:

$$
\left\{ \begin{matrix}         
w := w+\eta\sum_iy_ix_i\\
b := b+\eta\sum_iy_i\\
\end{matrix}\right. \\
$$
$\eta$是学习率;在实际应用中一般选择使用随机梯度下降:

$$
\left\{ \begin{matrix}         
w := w+\eta y_ix_i\\
b := b+\eta y_i\\
\end{matrix}\right. \\
$$
感知机的学习算法(随机梯度下降法)的步骤为:

- 选取初值$w_0,b_0$
- 在训练集中选取数据$(x_i,y_i)$
- 如果选取的点是误分类点,也就是说$y_i(w\cdot x_i+b)\le0$,按照上式对参数进行更新
- 转至第二步,直到训练集中没有误分类点

## 算法收敛性

证明如下的定理：

设训练数据集$T={(x_1,y_1),(x_2,y_2),...,(x_N,y_N)}$是线性可分的：

1. 存在满足条件$||\hat w_{opt}||$=1的超平面$\hat w_{opt} \cdot \hat x=w_{opt}\cdot x+b_{opt}=0$将数据集完全正确分开,且存在$r>0$,对所有的$i=1,2,..,N$有：
$$
y_i(\hat w_{opt} \cdot \hat x)=y_i(w_{opt}\cdot x+b_{opt})\ge r
$$
2. 令$R=\max||\hat x_i||$,则感知机在训练集上的误分类次数k满足不等式:
$$
k \le (\frac{R}{r})^2
$$

首先为了方便,将b放进了w和x中,也就是:
$$
\hat w=(w^T,b)^T \ ,\hat x = (x^T,1)^T
$$
先证明1：     
由于数据集是线性可分的,肯定存在一个超平面将数据集完全分开,即对$i=1,2,...,N$,都有：
$$
y_i(\hat w_{opt} \cdot \hat x)>0
$$

因此只需要r为$y_i(\hat w_{opt} \cdot \hat x)$的最小值,就会有：
$$
y_i(\hat w_{opt} \cdot \hat x) \ge r
$$

再来看2：      

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/2021-03-14_16-08-55_00.png)

也就是说误分类的次数是有上界的,经过有限次搜索肯定是可以找到将训练集完全分开的超平面

## Sci-kit learn

scikit learn 中的Perceptron类和SGDClassifier类都可以进行感知机模型的计算：

```{python}
from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron

X, y = load_digits(return_X_y=True)
clf = Perceptron(random_state=0)
##也可以使用SGDClassifier(loss="perceptron", eta0=1, learning_rate="constant", penalty=None)
clf.fit(X, y)
clf.score(X, y)
```
