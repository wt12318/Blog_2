---
title: 【hands on ML Ch2】-端到端机器学习
date: 2021-01-23 10:00:00    
index_img: img/image1.jpg
categories:
  - 机器学习
---



hands on ML 第二章，展示了一个实例项目的完整流程

<!-- more -->

本章展示了一个实例项目的完整流程，主要步骤包括：

-   组织项目(look at the big picture)

-   获取数据

-   对数据进行探索和可视化

-   对数据进行预处理

-   选择模型进行训练

-   微调模型

-   展示结果

-   启动，监控并维护系统

本章使用的数据为加州房屋价格数据集，来自1990年的人口普查数据,包括每个地区(人口普查单位)的中位数收入，人口，中位数住房价格等信息，需要建立一个模型来预测住房价格

## Look at the Big Picture

第一步就是**确定问题** ：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210123145823410.png)

这个项目是处于一个数据管线(pipeline)上的一环，目的是预测出地区的住房价格以供后续的投资决策参考

有关机器学习系统的问题包括：这个系统是监督的还是非监督的还是增强学习类型；是分类任务还是回归任务还是其他；需要使用批量学习还是在线学习

这个任务是典型的监督学习，回归任务(单变量回归)；由于没有连续的数据流进入系统，所以采用批量学习(batch
learning)

第二步是选择一个**性能衡量指标**，对于回归问题最常用的是RMSE(Root Mean Square Error,均方根误差):

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210123152509580.png)

也可以使用其他的函数，比如，如果数据中离群点比较多，可以使用MAE(mean absolute error,平均绝对误差)，这个衡量相较RMSE对离群点更不敏感

第三步是 **再次检查假设** 帮助我们较早的发现可能的问题，比如如果系统的下游需要的不是数值而是价格的分类(低中高)，那么这个问题就变成分类问题而不是回归问题了；所以需要在项目开始前将这些问题考虑到，避免时间精力的浪费

## Get the Data

编写函数来自动下载数据并解压：

``` python
import os
```

``` python
import tarfile
import urllib
import urllib.request

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("../test/datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

fetch_housing_data()
```

然后使用pandas来读入数据，返回一个pandas的DataFrame 对象：

``` python
HOUSING_PATH = os.path.join("../test/datasets", "housing")

import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()
```

可以看一下数据的结构：
![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210123161251034.png)

也可以使用`info` 方法来查看数据的描述,可以展示数据的行数，每列的类型以及非空值的数量

``` python
housing.info()
>> <class 'pandas.core.frame.DataFrame'>
>> RangeIndex: 20640 entries, 0 to 20639
>> Data columns (total 10 columns):
>>  #   Column              Non-Null Count  Dtype  
>> ---  ------              --------------  -----  
>>  0   longitude           20640 non-null  float64
>>  1   latitude            20640 non-null  float64
>>  2   housing_median_age  20640 non-null  float64
>>  3   total_rooms         20640 non-null  float64
>>  4   total_bedrooms      20433 non-null  float64
>>  5   population          20640 non-null  float64
>>  6   households          20640 non-null  float64
>>  7   median_income       20640 non-null  float64
>>  8   median_house_value  20640 non-null  float64
>>  9   ocean_proximity     20640 non-null  object 
>> dtypes: float64(9), object(1)
>> memory usage: 1.6+ MB
```

注意到`total_bedrooms` 变量只有20433个非空值，因此后续可能要对该变量进行缺失值的处理

对于`ocean_proximity`这个变量，可以使用`value_counts()` 方法来看其具体的分类情况：

``` python
housing["ocean_proximity"].value_counts()
>> <1H OCEAN     9136
>> INLAND        6551
>> NEAR OCEAN    2658
>> NEAR BAY      2290
>> ISLAND           5
>> Name: ocean_proximity, dtype: int64
```

使用`describe` 方法可以得到数据的汇总统计信息：

``` python
housing.describe()
>>           longitude      latitude  ...  median_income  median_house_value
>> count  20640.000000  20640.000000  ...   20640.000000        20640.000000
>> mean    -119.569704     35.631861  ...       3.870671       206855.816909
>> std        2.003532      2.135952  ...       1.899822       115395.615874
>> min     -124.350000     32.540000  ...       0.499900        14999.000000
>> 25%     -121.800000     33.930000  ...       2.563400       119600.000000
>> 50%     -118.490000     34.260000  ...       3.534800       179700.000000
>> 75%     -118.010000     37.710000  ...       4.743250       264725.000000
>> max     -114.310000     41.950000  ...      15.000100       500001.000000
>> 
>> [8 rows x 9 columns]
```

``` python
housing.describe()["median_house_value"]
>> count     20640.000000
>> mean     206855.816909
>> std      115395.615874
>> min       14999.000000
>> 25%      119600.000000
>> 50%      179700.000000
>> 75%      264725.000000
>> max      500001.000000
>> Name: median_house_value, dtype: float64
```

除了得到一些数值信息之外，对数据的探索更直接的方式是通过可视化来得到数据的一些特征,最简单的就是画直方图来反映数据的分布

``` python
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
>> array([[<AxesSubplot:title={'center':'longitude'}>,
>>         <AxesSubplot:title={'center':'latitude'}>,
>>         <AxesSubplot:title={'center':'housing_median_age'}>],
>>        [<AxesSubplot:title={'center':'total_rooms'}>,
>>         <AxesSubplot:title={'center':'total_bedrooms'}>,
>>         <AxesSubplot:title={'center':'population'}>],
>>        [<AxesSubplot:title={'center':'households'}>,
>>         <AxesSubplot:title={'center':'median_income'}>,
>>         <AxesSubplot:title={'center':'median_house_value'}>]],
>>       dtype=object)
plt.show()
```

![unnamed-chunk-8-1](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/unnamed-chunk-8-1.png)

观察数据的分布可以得到一些可能注意不到的信息：比如这里的`median income` 变量，看横坐标范围是0.5-15，所以不可能是以美元作为单位，这个时候我们就要尽量弄清楚这些已经经过处理的数值是怎么得到的(这里经过了转化，单位变成了$10000，并且下限是0.5，上限是15);另外我们看到这些*变量的尺度差异比较大*，后续需要进行缩放处理(scaling); 还有就是这些变量看起来都是偏向分布的(tailed distribution),这对于某些机器学习算法的学习可能比较困难，所以后续可能要进行转化，使其分布趋向于钟形分布

### 创建测试集

为什么要在选择模型之前就要创建测试集呢？

因为人的大脑是一种惊人的模式检测系统，可能我们在观察了测试数据之后可能会偶然发现有意思的模式从而就会有偏向性的选择某个模型，在测试集上估计误差的时候就会过于乐观(data snooping bias)

因此我们需要提前将测试集划分好，并且在模型训练过程中不触及测试集

在划分训练集和测试集的时候主要有两种方法：

-   完全随机抽样

-   分层抽样

Scikit-Learn 提供了一些函数来划分训练集和测试集

``` python
###完全随机抽样
from sklearn.model_selection import train_test_split

##random_states是随机种子数
train_set, test_set = train_test_split(housing, test_size=0.2,random_state=42)
```

假设这个项目中中位数收入(median income)对预测median housing prices是比较重要的变量，因此我们在创建测试集的时候希望能够代表不同类别的收入群体；由于median income是一个连续性的变量，所以我们需要将其转化成分类变量：

``` python
import numpy as np
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
                               
housing["income_cat"].value_counts()
>> 3    7236
>> 2    6581
>> 4    3639
>> 5    2362
>> 1     822
>> Name: income_cat, dtype: int64
housing["income_cat"].hist()
>> <AxesSubplot:>
```

然后可以使用Scikit-Learn的**StratifiedShuffleSplit类**来进行分层抽样：

``` python
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
```

最后需要将我们创建的用于分层抽样的变量`income_cat` 删除：

``` python
for set_ in (strat_train_set,strat_test_set):
  set_.drop("income_cat",axis=1,inplace=True)
```

## 对数据进行探索和可视化

首先我们要确保对数据的探索和可视化只对训练集进行，另外如果数据集比较大，这一步骤也可以选择一部分数据集作为“exploration set”

对数据的可视化要选取合适的形式，比如这个项目是不同地区的房价，因此可以以经纬度来展示不同的变量(住房价格，人口密度等)；有些时候可视化需要调整一些参数使得模式更加清晰(比如点的透明度)，有时候可以将一些变量进行合并

对数据的探索是一个迭代的过程，当我们建立起一个原型系统之后，在运行的过程中可以分析其输出然后返回来再次进行这个探索步骤，从而获得更深的理解

## 数据预处理

将数据(预)处理的过程包装成函数是非常有用的：

-   在任何数据集上都可以便捷地重复数据转化步骤

-   可以将经常用到的函数打包成库，以便未来的项目进行复用

-   如果我们的项目是部署在动态的系统上，就可以使用这些函数对新输入的数据进行转化

-   更重要的是：通过函数，我们可以尝试不同的转化参数或不同的转化步骤的组合对最终模型性能的影响

首先获取训练集的拷贝并将数据的labels去掉：

``` python
housing = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set
housing_labels = strat_train_set["median_house_value"].copy()
```

### 数据清洗

大部分机器学习算法是不能够处理缺失值的，而我们之前看到`total_bedrooms` 变量是有一些缺失值的，

``` python
sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
sample_incomplete_rows
>>        longitude  latitude  ...  median_income  ocean_proximity
>> 4629     -118.30     34.07  ...         2.2708        <1H OCEAN
>> 6068     -117.86     34.01  ...         5.1762        <1H OCEAN
>> 17923    -121.97     37.35  ...         4.6328        <1H OCEAN
>> 13656    -117.30     34.05  ...         1.6675           INLAND
>> 19252    -122.79     38.48  ...         3.1662        <1H OCEAN
>> 
>> [5 rows x 9 columns]
```

对于缺失值的处理可以有3种选择：

-   将相应的地区删除(删除观测值，也就是行)

-   将有缺失值的变量删除(删除列)

-   将缺失值填补为某个值(比如0,平均值,中位数等)

需要注意的是：如果采取用某个值填补缺失值，需要将这个值存储下来，不只是训练集，之后还要用这个值来填充测试集中的缺失值，新的数据中的缺失值

使用pandas DataFrame中的`dropna()` `drop` 和`fillna()`方法可以实现：

``` python
# option 1 
housing.dropna(subset=["total_bedrooms"]) 
>>        longitude  latitude  ...  median_income  ocean_proximity
>> 17606    -121.89     37.29  ...         2.7042        <1H OCEAN
>> 18632    -121.93     37.05  ...         6.4214        <1H OCEAN
>> 14650    -117.20     32.77  ...         2.8621       NEAR OCEAN
>> 3230     -119.61     36.31  ...         1.8839           INLAND
>> 3555     -118.59     34.23  ...         3.0347        <1H OCEAN
>> ...          ...       ...  ...            ...              ...
>> 6563     -118.13     34.20  ...         4.9312           INLAND
>> 12053    -117.56     33.88  ...         2.0682           INLAND
>> 13908    -116.40     34.09  ...         3.2723           INLAND
>> 11159    -118.01     33.82  ...         4.0625        <1H OCEAN
>> 15775    -122.45     37.77  ...         3.5750         NEAR BAY
>> 
>> [16354 rows x 9 columns]
```

``` python
# option 2 
housing.drop("total_bedrooms", axis=1) 
>>        longitude  latitude  ...  median_income  ocean_proximity
>> 17606    -121.89     37.29  ...         2.7042        <1H OCEAN
>> 18632    -121.93     37.05  ...         6.4214        <1H OCEAN
>> 14650    -117.20     32.77  ...         2.8621       NEAR OCEAN
>> 3230     -119.61     36.31  ...         1.8839           INLAND
>> 3555     -118.59     34.23  ...         3.0347        <1H OCEAN
>> ...          ...       ...  ...            ...              ...
>> 6563     -118.13     34.20  ...         4.9312           INLAND
>> 12053    -117.56     33.88  ...         2.0682           INLAND
>> 13908    -116.40     34.09  ...         3.2723           INLAND
>> 11159    -118.01     33.82  ...         4.0625        <1H OCEAN
>> 15775    -122.45     37.77  ...         3.5750         NEAR BAY
>> 
>> [16512 rows x 8 columns]
```

``` python
# option 3
median = housing["total_bedrooms"].median() 

housing["total_bedrooms"].fillna(median, inplace=True)
```

Scikit-Learn提供了一个方便的类`SimpleImputer` 来处理缺失值：

``` python
###首先需要创建一个SimpleImputer实例
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")

###由于只能对数值变量计算meidan，所以将字符变量删除
housing_num = housing.drop("ocean_proximity",axis=1)

###使用fit方法计算median
imputer.fit(housing_num)

##计算的结果存储在statistics_实例变量中
>> SimpleImputer(strategy='median')
imputer.statistics_
>> array([-118.51  ,   34.26  ,   29.    , 2119.5   ,  433.    , 1164.    ,
>>         408.    ,    3.5409])
housing_num.median().values

>> array([-118.51  ,   34.26  ,   29.    , 2119.5   ,  433.    , 1164.    ,
>>         408.    ,    3.5409])
X = imputer.transform(housing_num)####现在就相当于在训练集上"trained" 这个imputer，再使用他去对数据集进行transform(填充缺失值)
X

###将Numpy array转化成数据框
>> array([[-121.89  ,   37.29  ,   38.    , ...,  710.    ,  339.    ,
>>            2.7042],
>>        [-121.93  ,   37.05  ,   14.    , ...,  306.    ,  113.    ,
>>            6.4214],
>>        [-117.2   ,   32.77  ,   31.    , ...,  936.    ,  462.    ,
>>            2.8621],
>>        ...,
>>        [-116.4   ,   34.09  ,    9.    , ..., 2098.    ,  765.    ,
>>            3.2723],
>>        [-118.01  ,   33.82  ,   31.    , ..., 1356.    ,  356.    ,
>>            4.0625],
>>        [-122.45  ,   37.77  ,   52.    , ..., 1269.    ,  639.    ,
>>            3.575 ]])
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)
```

### 处理分类变量

这个例子中只有一个变量是分类变量`ocean_proximity` ,

``` python
housing_cat = housing[["ocean_proximity"]]
housing_cat.head(10)
>>       ocean_proximity
>> 17606       <1H OCEAN
>> 18632       <1H OCEAN
>> 14650      NEAR OCEAN
>> 3230           INLAND
>> 3555        <1H OCEAN
>> 19480          INLAND
>> 8879        <1H OCEAN
>> 13685          INLAND
>> 4937        <1H OCEAN
>> 4861        <1H OCEAN
```

对于分类变量一般有两种处理方法：

-   用多个数值去编码不同的类别

-   使用dummy变量，也就是one-hot编码(该类别为1，其他类别为0)

使用Scikit-Learn的OrdinalEncoder类和OneHotEncoder类可以分别处理上述两种情况：

``` python
from sklearn.preprocessing import OrdinalEncoder

###fit_transform相当于上面的先fit再transform
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10]

###结果存储在categories_变量中
>> array([[0.],
>>        [0.],
>>        [4.],
>>        [1.],
>>        [0.],
>>        [1.],
>>        [0.],
>>        [1.],
>>        [0.],
>>        [0.]])
ordinal_encoder.categories_
>> [array(['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'],
>>       dtype=object)]
```

``` python
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot
>> <16512x5 sparse matrix of type '<class 'numpy.float64'>'
>>  with 16512 stored elements in Compressed Sparse Row format>
cat_encoder.categories_

###为了便于储存，结果是稀疏矩阵，可以转化为正常的矩阵形式
>> [array(['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'],
>>       dtype=object)]
housing_cat_1hot.toarray()
###也可以使用参数OneHotEncoder(sparse=False)
>> array([[1., 0., 0., 0., 0.],
>>        [1., 0., 0., 0., 0.],
>>        [0., 0., 0., 0., 1.],
>>        ...,
>>        [0., 1., 0., 0., 0.],
>>        [1., 0., 0., 0., 0.],
>>        [0., 0., 0., 1., 0.]])
```

需要注意的点是：在机器学习算法中通常会认为差值较小的值比差值较大的值更相似，如果使用不同的数值来编码分类变量(第一种方法)，需要注意其含义(在这里0和4比0和1更相似)

### **Custom Transformers**

我们也可以定义自己的转化器，需要做的就是：创建一个类，并实现3个方法(fit, transfrom,fit\_transform),可以通过添加Scikit Learn的`TransformerMixin`类作为一个基础类来自动添加最后一个类(fit\_ftansform),除此之外，还可以添加`BaseEstimator`作为基础类，从而可以获得两个额外的方法(get\_params() 和set\_params())，来更方便的**进行超参数的调试**，下面是一个合并变量的例子：

``` python
from sklearn.base import BaseEstimator, TransformerMixin

# column index
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
```

### 特征缩放

当输入数据的范围(scale)相差较大，机器学习算法一般不会表现很好

一般有两种方法可以使所有的变量的尺度一致：

-   min-max 缩放(也叫normalization) 将数据缩放到0-1的范围(也可以选择其他的范围)，计算方法是：减去最小值然后除以最大值与最小值的差值；Scikit-learn 提供了`MinMaxScaler` 转化器(feature\_range超参数来修改范围)
    
-   Standardization 减去均值然后除以标准差，这种方法并不会将数值绑定到某个范围并且受离群值影响比较小；Scikit-learn提供了`StandardScaler` 转化器

### **Transformation Pipelines**

Scikit-Learn提供了`Pipeline`类可以用来组合一系列的数据转化过程，下面是将之前对数值变量的处理组合成pipeline：

``` python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)
```

`Pipeline` 的输入是一个列表，每个元素都是name/estimator对，最后一个estimator必须是转化器(也就是说最后一个必须有fit\_transform方法)

为了可以**同时处理数值变量和分类变量**，我们可以使用Scikit-Learn的**ColumnTransformer**(0.20版本)：

``` python
from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)
```

ColumnTransformer的输入是一个列表，列表的元素是元组，元组包含：名称+转化器+需要转化的列名

## 选择并训练模型

### 在测试集上训练并评估模型

首先尝试线性回归模型并在训练集上计算误差(RMSE):

``` python
from sklearn.linear_model import LinearRegression

###构建线性回归模型
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

###计算RMSE
>> LinearRegression()
from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse
>> 68628.19819848923
```

需要预测的变量`median_housing_values`的范围在$14999~$500001之间，所以这个误差有点大，也就是**underfitting**,上一章讲到解决欠拟合可以从3个方面考虑：选择一个更复杂的模型；选择更好的变量；减少模型的约束(现在这个线性回归没有正则化，所以不用考虑这一点)，这里首先尝试一个更复杂的模型：决策树回归，模型的使用类似：

``` python
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor() 
tree_reg.fit(housing_prepared,housing_labels)
>> DecisionTreeRegressor()
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse
>> 0.0
```

现在这个模型的误差是0！很大可能是过拟合了，但是要注意：**我们不能够在测试集上测试我们的模型看看是不是过拟合，然后再来调整**，所以这里我们需要应用上章讲到的**将训练集再划分成训练集和验证集**，在训练集中训练模型，在验证集中检测模型然后再进行模型的调整，得到一个较好的结果后再去测试集上检测

### 使用交叉验证来更好的评估模型

交叉验证指的是：随机将训练集分成几份(一般是10份)，每一份称为fold；然后进行训练和评估10次，每次选择一个不同的fold进行评估，在剩余9份中进行训练：

``` python
from sklearn.model_selection import cross_val_score 
scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10) 
tree_rmse_scores = np.sqrt(-scores)
```

注意：在Scikit-Learn中得到的score是功效函数(越大越好)而不是损失函数，所以是负数；得到的结果是评估分数的数组：

``` python
def display_scores(scores):
  print("Scores:",scores)
  print("Mean:",scores.mean())
  print("SD:",scores.std())
  
display_scores(tree_rmse_scores)
>> Scores: [68524.35504919 66981.1355597  70797.83977591 69247.66817087
>>  69998.59463448 74424.9303865  71390.885174   71908.43423181
>>  77419.30345977 68252.98473013]
>> Mean: 70894.61311723632
>> SD: 2962.180203662978
```

可以看到这个决策树模型在验证集上的误差比线性回归模型还要差;另外使用交叉验证除了可以估计模型的性能之外还可以衡量这个估计的精确度(标准差)

我们再算一下线性回归模型的交叉验证结果：

``` python
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
scoring="neg_mean_squared_error", cv=10)

lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)
>> Scores: [66782.73843989 66960.118071   70347.95244419 74739.57052552
>>  68031.13388938 71193.84183426 64969.63056405 68281.61137997
>>  71552.91566558 67665.10082067]
>> Mean: 69052.46136345083
>> SD: 2731.674001798342
```

最后再尝试另一种集成学习模型：随机森林回归(通过随机选特征的子集来训练多个决策树然后对预测进行平均)

``` python
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()

##看一下再训练集上的误差
forest_reg.fit(housing_prepared,housing_labels)
>> RandomForestRegressor()
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse


##交叉验证
>> 18695.54976048593
forest_reg_score = cross_val_score(forest_reg,housing_prepared, housing_labels,scoring="neg_mean_squared_error", cv=10)

forest_reg_rmse_scores = np.sqrt(-forest_reg_score)
display_scores(forest_reg_rmse_scores)
>> Scores: [49766.07996985 47777.52088347 49930.25427399 52365.21532167
>>  49565.91367179 53400.89852447 48887.63710491 47989.04820215
>>  52920.62118389 50075.38104365]
>> Mean: 50267.857017983544
>> SD: 1882.7452463475934
```

随机森林的误差已经要小很多了，但是在训练集上的误差仍然要比在验证集上的误差小很多，说明还是有过拟合的(回顾：可以通过简化模型，加上正则项或者收集更多的数据)

当我们实验了多个模型，应该将每个模型都保存起来，包括模型的超参数，训练参数，交叉验证的值，预测值等以便于模型间的比较，在Python中可以通过`pickle`模块或者`joblib`库来存储scikit-learn模型(jonlib在存储大的Numpy数组上更有效率)：

``` python
import joblib
joblib.dump(my_model,"my_model.pkl)

my_model_loaded = joblib.load("my_model.pkl")
```

## 微调模型

调整模型的超参数，有以下几种常用方法

### 网格搜索

网格搜索就是类似于穷举法，尝试所有的可能；Scikit-learn提供了`GridSearchCV`来进行网格搜索并使用交叉验证来评估所有的超参数的组合：

``` python
from sklearn.model_selection import GridSearchCV

##提供需要实验的超参数值
param_grid = [ {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]}, {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},]

forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)

grid_search.fit(housing_prepared, housing_labels)
>> GridSearchCV(cv=5, estimator=RandomForestRegressor(),
>>              param_grid=[{'max_features': [2, 4, 6, 8],
>>                           'n_estimators': [3, 10, 30]},
>>                          {'bootstrap': [False], 'max_features': [2, 3, 4],
>>                           'n_estimators': [3, 10]}],
>>              return_train_score=True, scoring='neg_mean_squared_error')
```

param\_grid 提供需要实验的超参数值，是一个列表，列表的元素是字典，每个字典里面是需要尝试的超参数的值，所以这里面就是：首先评估第一个字典中的`3*4`个超参数的组合，一共12个模型，再评估第二个字典中的`2*3`个超参数的组合，一个6个模型，所以总的需要评估12+6=18个模型，对每个模型训练5次(交叉验证中CV=5)

得到的最好的结果存储在`best_params_`中：

``` python
grid_search.best_params_
>> {'max_features': 8, 'n_estimators': 30}
```

整个模型在`best_estimator_`中

``` python
grid_search.best_estimator_
>> RandomForestRegressor(max_features=8, n_estimators=30)
```

我们也可以得到每个超参数组合的交叉验证的score：

``` python
cvres = grid_search.cv_results_

##平均误差
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
  print(np.sqrt(-mean_score), params)
>> 64530.62351934246 {'max_features': 2, 'n_estimators': 3}
>> 55357.890909127585 {'max_features': 2, 'n_estimators': 10}
>> 52935.050118540785 {'max_features': 2, 'n_estimators': 30}
>> 60838.564876061646 {'max_features': 4, 'n_estimators': 3}
>> 53294.075996366715 {'max_features': 4, 'n_estimators': 10}
>> 50664.777897014326 {'max_features': 4, 'n_estimators': 30}
>> 58678.16606697331 {'max_features': 6, 'n_estimators': 3}
>> 51832.23262797085 {'max_features': 6, 'n_estimators': 10}
>> 50022.944964854985 {'max_features': 6, 'n_estimators': 30}
>> 58836.35998556703 {'max_features': 8, 'n_estimators': 3}
>> 51798.661030790616 {'max_features': 8, 'n_estimators': 10}
>> 49981.14999745153 {'max_features': 8, 'n_estimators': 30}
>> 62437.81349999718 {'bootstrap': False, 'max_features': 2, 'n_estimators': 3}
>> 54286.58589119645 {'bootstrap': False, 'max_features': 2, 'n_estimators': 10}
>> 59249.56753707383 {'bootstrap': False, 'max_features': 3, 'n_estimators': 3}
>> 52761.26326802062 {'bootstrap': False, 'max_features': 3, 'n_estimators': 10}
>> 59203.193281533706 {'bootstrap': False, 'max_features': 4, 'n_estimators': 3}
>> 52169.3677107009 {'bootstrap': False, 'max_features': 4, 'n_estimators': 10}
```

使用max\_feature=8和n\_estimators=30得到的误差为49682，比之前要好，说明微调模型是有效果的

注意：一些数据处理的步骤也可以认为是超参数，比如前面的是否添加一些合并的变量(add\_bedrooms\_per\_room)，缺失值的处理，特征选择等；也可以使用类似的方法进行调整

### 随机搜索

当超参数的搜索空间比较大的时候，使用随机搜索的方法比较好
随机搜索不是尝试所有可能的组合，而是在每次迭代中对每个超参数随机选取一个值，然后对这些随机选取的超参数组合进行评估，这种方法有两个主要的优势：

-   如果我们设置循环数目为1000，那么这种方法对每个超参数都会尝试1000个不同的值(网格法只会尝试给定的值)
-   通过设定循环的次数就可以控制进行超参数搜寻的成本

### 集成方法

另一个调整模型的方法就是将表现最好的不同模型结合起来(就像决策树回归模型一样)

### 分析最好的模型和其误差

我们可以通过对模型的检查来获得对问题的更好的理解，比如我们可以查看在随机森林回归模型中不同变量对模型预测的重要性：

``` python
feature_importances = grid_search.best_estimator_.feature_importances_

feature_importances
>> array([6.77924618e-02, 6.07623085e-02, 4.33993956e-02, 1.47232036e-02,
>>        1.49909386e-02, 1.44308483e-02, 1.44214759e-02, 3.75628068e-01,
>>        4.07637376e-02, 1.13639737e-01, 6.10155827e-02, 6.75873135e-03,
>>        1.66811158e-01, 1.30669013e-04, 1.75998132e-03, 2.97170208e-03])
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)
>> [(0.3756280684089266, 'median_income'), (0.16681115793474138, 'INLAND'), (0.11363973729374365, 'pop_per_hhold'), (0.06779246179070629, 'longitude'), (0.06101558267732972, 'bedrooms_per_room'), (0.06076230849002859, 'latitude'), (0.04339939564469106, 'housing_median_age'), (0.04076373758593783, 'rooms_per_hhold'), (0.014990938598037237, 'total_bedrooms'), (0.0147232036472, 'total_rooms'), (0.01443084830180091, 'population'), (0.014421475858182577, 'households'), (0.006758731352675241, '<1H OCEAN'), (0.002971702084100027, 'NEAR OCEAN'), (0.001759981318404895, 'NEAR BAY'), (0.0001306690134939975, 'ISLAND')]
```

通过这个信息，我们就可以将那些不重要的变量丢弃(比如这里的ocean\_proximity分类变量中除了INLAND外的其他类别)

### 在测试集上评估模型系统

在测试集上的计算和前面的流程类似：

``` python
final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1) 
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)##注意，仅仅是transform,使用在训练集上已经"train"的参数来transfrom测试集
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
```

有些时候，这样的点估计不太够，我们可以使用`scipy.stats.t.interval()`来计算置信区间：

``` python
from scipy import stats
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2

np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,loc=squared_errors.mean(),scale=stats.sem(squared_errors)))
>> array([46303.36875963, 50242.61230504])
```

最后就是部署模型了：
![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210203174058586.png)
