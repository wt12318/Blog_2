---
title: Python 基础03
date: 2021-03-01 16:58:48
tags: 编程
index_img: img/python.jpg
---

天池python task3

<!-- more -->

Task3 包含：

-   函数
-   类与对象
-   魔法方法

## 函数

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/函数%20-%20坚果云_00.png)

函数是带名称的代码块，用于完成具体的工作，结构为：

``` python
def functionname(parameters):
    """函数文档字符串"""
    functionsuite
    return [expression]
```

### 函数文档

函数文档字符串(DocStrings)是对函数的描述;在函数体的第一行使用3个单引号或者双引号来定义文档字符串；使用惯例是：**首行描述函数功能，第二行空行，第三行为函数的具体描述**

可以使用`__doc__`来获取函数的文档字符串

``` python
def div(x,y):
  '''除法计算
  
  y不能为0'''
  return(x/y)

div(1,2)
>> 0.5
print(div.__doc__)##__doc__获取文档
>> 除法计算
>>   
>>   y不能为0
```

### 函数参数

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210301214319502.png)

形参(parameter)是函数工作所需要的信息，实参(argument)是函数调用时传递的信息；函数调用时实参会被传递给形参

**传递实参的方式**有：位置实参和关键字实参：

``` python
def printinfo(name,age):
  print('Name:{0},Age:{1}'.format(name, age))
  
printinfo("ada",12)##按照位置传递实参
>> Name:ada,Age:12
printinfo(age=12,name="ada")##按照关键字传递实参
>> Name:ada,Age:12
```

对于形参，我们可以给其指定默认值，如果给这样的形参提供了实参则使用实参的值，如果没有对应的实参则使用默认值：

``` python
def printinfo1(name,age=10):
  print('Name:{0},Age:{1}'.format(name, age))
  
printinfo1("ada",12)##提供了位置实参
>> Name:ada,Age:12
printinfo1("ada")##没有提供实参，使用默认值
>> Name:ada,Age:10
```

有时候不知道函数要接受的实参的个数，这个时候可以使用加星号的形参名，将多余的实参放到以形参名命名的**元组**中：

``` python
def printinfo3(num1, *num2):
    print(num1)
    for var in num2:
        print(var)
    print(type(num2),len(num2))
        
printinfo3(10)
>> 10
>> <class 'tuple'> 0
printinfo3(10,20,30)
>> 10
>> 20
>> 30
>> <class 'tuple'> 2
```

也可以使用在形参名前加两个星号，将多余的实参(参数名和值构成的键值对)放到以形参名命名的**字典**中

``` python
def printinfo4(num1, *num2, **others):
    print(num1)
    print(num2,type(num2),len(num2))
    print(others,type(others),len(others))
    
printinfo4(10,20,30)
>> 10
>> (20, 30) <class 'tuple'> 2
>> {} <class 'dict'> 0
printinfo4(10,20,30,a=1,b=2)
>> 10
>> (20, 30) <class 'tuple'> 2
>> {'a': 1, 'b': 2} <class 'dict'> 2
```

如果在传递实参时对某个实参想要强制使用关键字来传递，可以使用\*将其与前面的参数分开：

``` python
def printinfo15(name,*,age):
  print('Name:{0},Age:{1}'.format(name, age))

printinfo15("ada",10)
>> Error in py_call_impl(callable, dots$args, dots$keywords): TypeError: printinfo15() takes 1 positional argument but 2 were given
printinfo15("ada",age=10)##必须使用关键字
>> Name:ada,Age:10
```

### 变量作用域

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210301220956029.png)

定义在函数内部的变量只有在函数内部也可以使用，具有局部作用域，称为局部变量；定义在函数外部的变量在全局都可以访问，称为全局变量

``` python
a = 4##全局变量
def printinfo6(num1):
  num2 = 2
  print(num1+a+num2)

printinfo6(1)
>> 7
num2##局部变量
>> Error in py_call_impl(callable, dots$args, dots$keywords): NameError: name 'num2' is not defined
```

在内部作用域中想要改变外部作用域的变量时需要使用`global`(外部全局变量)和`nonlocal`(外部非全局变量)关键字

``` python
def printinfo7(num1):  
    global a  
    a = 2  
    num2 = 2  
    print(num1+a+num2)

printinfo7(1)
>> 5
print(a) ##a发生了改变
>> 2
```

当一个函数包含在另一个函数内部，这种函数叫做内嵌函数，内嵌函数只能在函数内部进行调用

``` python
def outer():  
    print('outer函数在这被调用')    
    def inner():    
        print('inner函数在这被调用')    
    inner()  # 该函数只能在outer函数内部被调用

outer()
>> outer函数在这被调用
>> inner函数在这被调用

inner()##不能在外部访问
>> Error in py_call_impl(callable, dots$args, dots$keywords): NameError: name 'inner' is not defined
```

当一个内嵌函数对外层的非全局作用域的变量进行引用，那么这个内嵌函数就是**闭包**：

``` python
def funx(x):  
    def funy(y):    
        return(x * y)    
    return funy

new_f = funx(8)
print(new_f,type(new_f))
>> <function funx.<locals>.funy at 0x00000209F628AE50> <class 'function'>
new_f(2)>> 16
```

从上面的例子可以看出，我们可以使用闭包来创建函数，作为函数工厂来使用(和R里面的闭包类似)。

上面也提到了可以使用`nonlocal`来改变外层非全局变量：

``` python
def funx(x):
  num1 = 4
  print("original num1 is ",num1)
  def funy(y):
    nonlocal num1
    num1 = 2
    print("current num1 is ",num1)
    return(x * y + num1)
  return funy

new_f = funx(8) 
>> original num1 is  4
new_f(2)
>> current num1 is  2
>> 18
```

如果一个函数在内部调用自己，那么这个函数就是递归函数,下面以计算n的阶乘为例：

``` python
def n_fac(n):
  if n == 1:
    return 1
  return n * n_fac(n-1)

n_fac(100)
>> 93326215443944152681699238856266700490715968264381621468592963895217599993229915608941463976156518286253697920827223758251185210916864000000000000000000000000
```

### Lambda 表达式

上面讲的都是以 `def` 关键字定义的 “正规” 的函数，除了这种函数之外，python 中还有一种用 `lambda` 关键字定义的匿名函数，结构为：

```python
lambda argument_list: expression
```

- argument_list 是参数，可以是位置参数和关键字参数，和上面的一样
- expression 对传入函数的实参进行的操作

lambda 不需要 `return` 语句，表达式 expression 的结果就是返回值；匿名函数不能访问到 argument_list 之外的参数，下面是一些例子：

```python

```

