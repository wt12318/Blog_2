---
title: Python 基础 03
date: 2021-03-01 16:58:48
tags: 编程
index_img: img/python.jpg
---

函数，类与对象

<!-- more -->

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

函数文档字符串(DocStrings)是对函数的描述;在函数体的第一行使用3个单引号或者双引号来定义文档字符串；使用惯例是：**首行描述函数功能，第二行空行，第三行为函数的具体描述**，可以使用`__doc__`来获取函数的文档字符串

``` python
def div(x,y):
  '''除法计算
  
  y不能为0'''
  return(x/y)

div(1,2)
>> 0.5
print(div.__doc__)
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
>> <function funx.<locals>.funy at 0x000001CF87C3EE50> <class 'function'>
new_f(2)
>> 16
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

``` python
lambda argument_list: expression
```

-   argument_list 是参数，可以是位置参数和关键字参数，和上面的一样
-   expression 对传入函数的实参进行的操作

lambda 不需要 `return` 语句，表达式 expression 的结果就是返回值；匿名函数不能访问到 argument_list
之外的参数，下面是一些例子：

``` python
def sqr(x):
    return x ** 2


print(sqr)
>> <function sqr at 0x000001CF87C3EF70>
y = [sqr(x) for x in range(10)]
print(y)
>> [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
lbd_sqr = lambda x: x ** 2
print(lbd_sqr)
>> <function <lambda> at 0x000001CF87C6B040>
y = [lbd_sqr(x) for x in range(10)]
print(y)
>> [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
sumary = lambda arg1, arg2: arg1 + arg2
print(sumary(10, 20)) 
>> 30
func = lambda *args: sum(args)##元组
print(func(1, 2, 3, 4, 5))  
>> 15
func = lambda **args: sum(args.values())##列表
print(func(a=1,b=2))
>> 3
```

匿名函数常常用在函数式编程的高阶函数中，高阶函数指的是函数的参数也是一个函数，在 python 比较常用的现成高阶函数有 `filter` 和 `map` ：

``` python
##filter(function, iterable) 过滤序列，过滤掉不符合条件的元素，返回一个迭代器对象

odd = lambda x: x % 2 == 1
templist = filter(odd, [1, 2, 3, 4, 5, 6, 7, 8, 9])

##map(function, *iterables) 根据提供的函数对指定序列做映射

m1 = map(lambda x: x ** 2, [1, 2, 3, 4, 5])
print(list(m1))  
>> [1, 4, 9, 16, 25]
m2 = map(lambda x, y: x + y, [1, 3, 5, 7, 9], [2, 4, 6, 8, 10])
print(list(m2)) 
>> [3, 7, 11, 15, 19]
```

我们也可以自己定义高阶函数：

``` python
def apply_to_list(fun, some_list):
    return fun(some_list)

lst = [1, 2, 3, 4, 5]
print(apply_to_list(sum, lst))
>> 15
print(apply_to_list(len, lst))
>> 5
print(apply_to_list(lambda x: sum(x) / len(x), lst))
>> 3.0
```

## 类与方法

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/%E7%B1%BB%E5%92%8C%E6%96%B9%E6%B3%95.svg)

``` python
#定义类
class test:
  def __init__(fuck,name):
    fuck.name = name

##创建实例
d1 = test("aa")
d1.name
>> 'aa'
```

``` python
class Car:
  length = 5#类属性
  def __init__(self,name,brand,max_people):##构造方法
    self.name = name##实例属性
    self.brand = brand
    self.number_of_people = 0
    self.max_people = max_people
  def show(self):##实例方法
    print(f"the name of this car is {self.name}")
    print(f"the brand of this car is {self.brand}")
  def run(self):
    print(f"the {self.name} is running")
    
  def set_people(self,num_of_people):
    if num_of_people > self.max_people:
      print(f"不能超过最大人数：{self.max_people}")
    else:
      self.number_of_people = num_of_people
      
  def increase_people(self):
    if self.number_of_people + 1 > self.max_people:
      print(f"不能超过最大人数：{self.max_people}")
    else:
      self.number_of_people += 1
  
  def reduce_people(self):
    if self.number_of_people - 1 < 0:
      print("车中已经没有人了")
    else:
      self.number_of_people -= 1

a = Car("a","b",5)
a.name
>> 'a'
a.brand
>> 'b'
a.max_people
>> 5
a.set_people(3)
a.number_of_people
>> 3
a.increase_people()
a.number_of_people
>> 4
a.increase_people()
a.increase_people()
>> 不能超过最大人数：5
a.number_of_people
>> 5
a.reduce_people()
a.reduce_people()
a.reduce_people()
a.reduce_people()
a.reduce_people()
a.reduce_people()
>> 车中已经没有人了
a.number_of_people
>> 0
```

``` python
##继承
class small_car(Car):
  def __init__(self,name,brand,max_people,height):
    super().__init__(name,brand,max_people)
    self.height = height
  ##重写父类方法
  def show(self):
    print(f"the name of this car is {self.name} and it is small!")
    
a = small_car("a","b",5,10)
a.number_of_people
>> 0
a.show()
>> the name of this car is a and it is small!
```

`issubclass(class, classinfo)` 方法用于判断参数 class 是否是类型参数 classinfo 的子类:

``` python
issubclass(small_car,Car)
>> True
issubclass(Car,Car)##类是自身的子类
>> True
issubclass(Car,object)##类都是对象的子类
>> True
```

`hasattr(object,name)` 用于判断对象是否包含对应的属性:

``` python
##类是判断其类属性，实例则可以判断类和实例属性
hasattr(Car, "length")
>> True
hasattr(Car, "name")
>> False
hasattr(a, "name")
>> True
hasattr(a, "length")
>> True
```

`getattr(object, name[, default])` 用于返回一个对象属性值，如果不存在该属性，则返回第三个默认参数的值：

``` python
getattr(a,"name")
>> 'a'
getattr(a,"nu")
>> Error in py_call_impl(callable, dots$args, dots$keywords): AttributeError: 'small_car' object has no attribute 'nu'
getattr(a,"nu",1)
>> 1
```

`setattr(object, name, value)` 用于设置属性值，该属性不一定是存在的：

``` python
setattr(a,"nu",1)
a.nu
>> 1
```

`delattr(object, name)` 用于删除属性:

``` python
delattr(a,nu)
>> Error in py_call_impl(callable, dots$args, dots$keywords): NameError: name 'nu' is not defined
getattr(a,"nu")
>> 1
```
