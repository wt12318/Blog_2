---
title: Python 基础02
date: 2021-02-24 16:57:07
tags: 编程
index_img: img/python.jpg
categories:
  - python
---

python主要数据结构

<!-- more -->

-   列表
-   元组
-   字符串
-   字典
-   集合

## 列表

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/列表%20-%20坚果云_00.png)

### 创建列表

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210224222711937.png)

-   直接创建

``` python
x = [1,2,3]

print(x,type(x))
>> [1, 2, 3] <class 'list'>
```

-   使用`range`函数创建

``` python
x = list(range(10))##默认是stop，从0开始
print(x,type(x))
>> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] <class 'list'>
x = list(range(1,11,2))
print(x,type(x))
>> [1, 3, 5, 7, 9] <class 'list'>
x = list(range(10,1,-2))
print(x,type(x))
>> [10, 8, 6, 4, 2] <class 'list'>
```

-   利用推导式创建列表

``` python
x = [0] * 5 
a = [x] * 4
b = x * 4
print(a,type(a))
>> [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]] <class 'list'>
print(b,type(b))
>> [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] <class 'list'>
a[0][0] = 1
print(a)
>> [[1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0]]
```

``` python
x = [i ** 2 for i in range(10,1,-2)]
print(x,type(x))
>> [100, 64, 36, 16, 4] <class 'list'>
```

### 添加和删除元素

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210225211530039.png)

#### 添加元素

`append`在列表末尾添加元素(作为一个整体添加)：

``` python
x = ["a","b","c"]
x.append("d")
print(x)
>> ['a', 'b', 'c', 'd']
```

`extend`也是在列表末尾添加元素(添加的是元素的元素)，注意区分两者：

``` python
x.append([1,2,3])
print(x)
>> ['a', 'b', 'c', 'd', [1, 2, 3]]
x.extend([1,2,3])
print(x)
>> ['a', 'b', 'c', 'd', [1, 2, 3], 1, 2, 3]
```

`insert`在指定位置插入元素：

``` python
x.insert(2,"f")##在第三个位置插入f
print(x)
>> ['a', 'b', 'f', 'c', 'd', [1, 2, 3], 1, 2, 3]
```

#### 删除元素

根据元素的位置删除元素可以使用`pop`,`del`；`pop`方法移除指定位置的元素并且返回该元素(”弹出“)：

``` python
x.pop()
>> 3
x.pop(1)
>> 'b'
del x[0:2]
```

根据元素的值删除元素可以使用`remove`方法

``` python
x.remove(1)
print(x)
>> ['c', 'd', [1, 2, 3], 2]
```

### 获取列表元素

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210225212857353.png)

切片的操作为\[start,stop,step\],注意是左闭右开：

``` python
print(x)
>> ['c', 'd', [1, 2, 3], 2]
x[1::2]
>> ['d', 2]
x[:4:1]
>> ['c', 'd', [1, 2, 3], 2]
x[::-1]
>> [2, [1, 2, 3], 'd', 'c']
```

**浅拷贝与深拷贝**

对于不可变数据类型，深浅拷贝是一样的(内存地址不变)；对于可变数据类型，浅拷贝只拷贝最外层的可变数据结构(内存地址发生改变)，而深拷贝则拷贝每层的可变数据类型，[参考](https://mp.weixin.qq.com/s/e8N-s2w4gYQPKETVH62EAg)

``` python
a = [1,2]
b = 1
x = [a,2,3]
y = [b,2]

z = x[:] ##浅拷贝
k = y[:] ##浅拷贝

import copy
z_d = copy.deepcopy(x) ##深拷贝
k_d = copy.deepcopy(y) ##深拷贝

##可变数据类型
print(id(x),id(z))##不一样
>> 2241811121856 2241811122112
print(id(x[0]),id(z[0]))##一样
>> 2241817618048 2241817618048
print(id(x),id(z_d))##不一样
>> 2241811121856 2241817628928
print(id(x[0]),id(z_d[0])) ##不一样

##不可变数据类型
>> 2241817618048 2241817628992
print(id(y),id(k))##不一样
>> 2241811157952 2241811156032
print(id(y[0]),id(k[0]))##一样
>> 2240632088880 2240632088880
print(id(y),id(k_d)) ##不一样
>> 2241811157952 2241817629056
print(id(y[0]),id(k_d[0])) ##一样
>> 2240632088880 2240632088880
```

### 常用操作符

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210228094615853.png)

``` python
l1 = [1,2,3]
l2 = [4,5,6]
l3 = [2,1,3]

print(l1 == l2)
>> False
print(l1 == l3)
>> False
print(l1 + l2)
>> [1, 2, 3, 4, 5, 6]
print(l1 * 3)
>> [1, 2, 3, 1, 2, 3, 1, 2, 3]
print(1 in l1)
>> True
print(0 not in l1)
>> True
```

### 其他方法

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210228095235247.png)

``` python
l1.count(1)
>> 1
l4 = l1 * 3
print(l4)
>> [1, 2, 3, 1, 2, 3, 1, 2, 3]
print(l4.index(1))
>> 0
print(l4.index(1,2))##从第三个元素开始找
>> 3
print(l4.index(1,1,3)) ##在第二到第四个元素范围内找(左闭右开), 但是里面没有，报错
>> Error in py_call_impl(callable, dots$args, dots$keywords): ValueError: 1 is not in list
l4.reverse()
print(l4)
>> [3, 2, 1, 3, 2, 1, 3, 2, 1]
l4.sort()
print(l4)
>> [1, 1, 1, 2, 2, 2, 3, 3, 3]
l4 = l1 * 3
sorted(l4)
>> [1, 1, 1, 2, 2, 2, 3, 3, 3]
print(l4)
>> [1, 2, 3, 1, 2, 3, 1, 2, 3]
```

## 元组

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/元组%20-%20坚果云_00.png)

元组和列表类似，不同的是元组是不可变数据类型(但是可以嵌套可变数据类型，可以直接更改其元素)

``` python
t1 = (1,2,[1,2,3])
t1[0] = 2
>> Error in py_call_impl(callable, dots$args, dots$keywords): TypeError: 'tuple' object does not support item assignment
t1[2][0] = 0
print(t1)
>> (1, 2, [0, 2, 3])
```

元组相关的操作符和方法也和列表类似：

``` python
t1 = (1,2,3)
t2 = (4,5,6)

t1 == t2
>> False
print(t1 + t2)
>> (1, 2, 3, 4, 5, 6)
print(t1 * 3)
>> (1, 2, 3, 1, 2, 3, 1, 2, 3)
print(1 in t1)
>> True
print(0 not in t1)
>> True
print(t1.count(1))
>> 1
print(t1.index(1))
>> 0
```

### 元组拆包

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210228103946589.png)

元组拆包(解压)就是将元组拆成单个的元素(结构要和原来的元组相同)

``` python
t = (1,2,3,("a","b",["j",9]))

(a,b,c,(d,e,f)) = t
print(a,b,c,d,e,f,end="\n")
>> 1 2 3 a b ['j', 9]
(a,b,c,(d,e,[f,g])) = t
print(a,b,c,d,e,f,g,end="\n")
>> 1 2 3 a b j 9
```

如果我们只想要其中几个元素，可以将其他元素赋给`*rest`(通配符)或者`*_`：

``` python
(a,*rest,(b,c,[d,e])) = t
print(a,b,c,d,e,end="\n")
>> 1 a b j 9
print(*rest)
>> 2 3
(a,*_,(b,c,[d,e])) = t
```

## 字符串

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210228105301884.png)

如果字符串中出现了一些有特殊含义的字符需要使用`\`对其进行转义，也可以在字符串前面加上`r`来表示原始字符串：

``` python
print("a \n b")
>> a 
>>  b
print(r"a \n b")
>> a \n b
```

字符串的切片和拼接和列表，元组类似：

``` python
s1 = "abcdef"

s1[0:4:2]
>> 'ac'
print(s1 + "abc")
>> abcdefabc
print(s1 * 3)
>> abcdefabcdefabcdef
```

### 常用内置方法

-   大小写的转换：`capitalize`将字符串的第一个字符转换成大写；`lower`将所有字符转换为小写；`upper`将所有字符转换成大写；`swapcase`将大写字符转换成小写，将小写字符转换成大写；`title`将所有单词的首字母转换成大写

``` python
s2 = "An apple"

s2.lower()
>> 'an apple'
s2.upper()
>> 'AN APPLE'
s2.swapcase()
>> 'aN APPLE'
s2.title()
>> 'An Apple'
s2.lower().capitalize()
>> 'An apple'
```

-   `count(str, beg= 0,end=len(string))`返回`str`在字符串中出现的次数，可以使用`beg`和`end`参数指定范围,**大小写敏感**

``` python
s2.count("A")
>> 1
s2.count("A",1,5)
>> 0
```

-   检查子字符串：`endswith(str, beg=0,end=len(string))`检查字符串是否以`str`结束；`startswith(substr, beg=0,end=len(string))`检查字符串是否以`str`开头；`find(str, beg=0, end=len(string))`检查`str`是否在字符串中，如果在，返回第一个值的索引，如果不在，返回-1；`rfind(str, beg=0,end=len(string))`类似`find`,不过是从右边开始查找

``` python
s3 = "this is a string"

s3.endswith("str")
>> False
s3.endswith("ing")
>> True
s3.startswith("this")
>> True
s3.find("str")
>> 10
s3.find("stre")
>> -1
s3.rfind("str")
>> 10
s3.find("is")
>> 2
s3.rfind("is")##从右边
>> 5
```

-   `isnumeric`检查字符串是不是只包含数字字符

``` python
s4 = "123"
s4.isnumeric()
>> True
s4 = s4 + "a"
s4.isnumeric()
>> False
```

-   对齐并填充：`ljust(width,fillchar)`
    将字符串左对齐,并使用fillchar填充到指定的宽度(width)；与之对应的是`rjust(width,fillchar)`右对齐

``` python
s4.ljust(8,"*")
>> '123a****'
s4.rjust(8,"*")
>> '****123a'
```

-   截断字符串：`lstrip(char)`去掉字符串左边空格(默认)或者指定字符(char);与之对应的是`rstrip`(右边)和`strip`(左边加右边)

``` python
s4 = "  " + s4 + "123  "
s4
>> '  123a123  '
s4.lstrip()
>> '123a123  '
s4.rstrip()
>> '  123a123'
s4.strip()
>> '123a123'
s4 = s4.strip()

s4.strip("123")
>> 'a'
```

-   字符串切割：`partition(sub)`在字符串中找sub字符串，找到之后将原字符串以子字符串分成3部分，如果找不到返回原字符串加上`,`;`rpartition(sub)`和`partition`类似，不过是从右边开始寻找，**注意：这里的寻找只找第一个，所以两者的结果可能不同**;
    `split(str=" ",num)`以str为分隔符切割字符串，可以指定分割产生的子字符串的个数，返回子字符串构成的列表

``` python
s5 = 'abc123abc'
print(s5.partition('b'))
>> ('a', 'b', 'c123abc')
print(s5.rpartition('b'))
>> ('abc123a', 'b', 'c')
```

``` python
s5.split("b")##分隔符是b
>> ['a', 'c123a', 'c']
s5.split("b",1)#num指定的是"切割"的次数
>> ['a', 'c123abc']
```

-   `replace(old,new,max)`将字符串中old子字符串替换成new新字符串，可以通过max指定替换的最大次数

``` python
s5.replace("bc","**",1)
>> 'a**123abc'
s5.replace("bc","**")
>> 'a**123a**'
```

-   `splitlines(keepends)`
    按行分割字符串，返回各行构成的列表(分割符可以为’, ‘’,
    ’)；可以通过keepends来指定是否保留分隔符

``` python
s6 = "abc\n123\rbcd\r\n000"
print(s6)
>> abc
>> 123bcd
>> 000
s6.splitlines()
>> ['abc', '123', 'bcd', '000']
s6.splitlines(keepends=True)
>> ['abc\n', '123\r', 'bcd\r\n', '000']
```

-   字符串的转化：`maketrans(intab,outtab)`创建intab到outtab的映射；`translate(table,deletechars="")`根据table来进行转化，可以使用`deletechars`来指定删除的字符

``` python
s7 = "abc has three characters"

intab = "abc"
outtab = "123"

transtab = s7.maketrans(intab,outtab)
transtab ###ASCII对应
>> {97: 49, 98: 50, 99: 51}
s7.translate(transtab)
>> '123 h1s three 3h1r13ters'
```

### 字符串格式化

-   format 方法
    在字符串中使用括号表示format的参数，进行替换；括号内的数字表示位置，字母表示参数名称

``` python
"{1} is {0}".format("num","1")
>> '1 is num'
"{second} is {first}".format(first = "num", second = "1")
>> '1 is num'
```

-   字符串格式化符号

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210228162350177.png)

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210228162434625.png)

-   f字符串：在需要格式化的字符串前面加上f或F，里面用大括号来代替变量

``` python
a = "'123'"
b = "'cdf'"

f"{a} has the same length with {b}"
>> "'123' has the same length with 'cdf'"
```

## 字典

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/字典%20-%20坚果云_00.png)

### 创建字典

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210228162934647.png)

``` python
d1 = {"a":1,"b":2,"c":3}
print(d1,type(d1))
>> {'a': 1, 'b': 2, 'c': 3} <class 'dict'>
```

``` python
##先创建空字典再填充
d2 = dict()
d2["a"] = 1
d2["b"] = 2
print(d2,type(d2))

##根据可映射对象来创建
>> {'a': 1, 'b': 2} <class 'dict'>
o1 = [("a",1),("b",2)]
o2 = (("c",1),("d",2))

d3 = dict(o1)
d4 = dict(o2)

print(d3,type(d3))
>> {'a': 1, 'b': 2} <class 'dict'>
print(d4,type(d4))

##根据关键字参数来创建
>> {'c': 1, 'd': 2} <class 'dict'>
d5 = dict(a=1,b=2)
print(d5,type(d5))
>> {'a': 1, 'b': 2} <class 'dict'>
```

还可以使用`fromkeys(seq,value)`方法来创建；以seq中的元素做键，value作为值(所有的键的值都是value，如果没有则为None)

``` python
seq = ("a","b","c")
dict.fromkeys(seq)
>> {'a': None, 'b': None, 'c': None}
dict.fromkeys(seq,10)
>> {'a': 10, 'b': 10, 'c': 10}
dict.fromkeys(seq,(1,2,3)) ##不会分开匹配
>> {'a': (1, 2, 3), 'b': (1, 2, 3), 'c': (1, 2, 3)}
```

### 访问字典

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210228163808865.png)
`keys`方法返回一个可迭代对象(键)，可以使用`list()`来转化为列表

``` python
d5
>> {'a': 1, 'b': 2}
d5.keys()
>> dict_keys(['a', 'b'])
type(d5.keys())
>> <class 'dict_keys'>
list(d5.keys())
>> ['a', 'b']
```

同样的，`values`方法返回的是包含值的可迭代对象：

``` python
d5
>> {'a': 1, 'b': 2}
d5.values()
>> dict_values([1, 2])
type(d5.values())
>> <class 'dict_values'>
list(d5.values())
>> [1, 2]
```

`items`方法返回的是键值对元组构成的可迭代对象：

``` python
d5.items()
>> dict_items([('a', 1), ('b', 2)])
type(d5.items())
>> <class 'dict_items'>
list(d5.items())
>> [('a', 1), ('b', 2)]
```

`get(key,default=None)`方法返回指定键(key)的值，如果没有找到则返回默认值(default)

``` python
d5.get("a")
>> 1
d5.get("c")##返回None 什么都没有

d5.get("c","not in dict")
>> 'not in dict'
```

### 修改,添加,删除元素

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210228165518914.png)
`update(dict2)`方法将dict2中的键值对更新到字典中

``` python
d5
>> {'a': 1, 'b': 2}
d5.update({"a":4})
d5
>> {'a': 4, 'b': 2}
```

`setdefault(key,default=None)`方法和get类似，不过他如果没有找到键的话会添加键，并将值设为default(和get一样也会返回default值)

``` python
d5.setdefault("c",5)
>> 5
d5
>> {'a': 4, 'b': 2, 'c': 5}
```

删除元素有3种方法：`pop(key,default)`方法删除键(key)所对应的值并返回该值，如果key不存在则返回default；`del dict[key]`语句删除key对应的值；`clear`方法删除所有元素

``` python
d5.pop("a")
>> 4
d5
>> {'b': 2, 'c': 5}
d5.pop("a","not in dict")
>> 'not in dict'
del d5["b"]
d5
>> {'c': 5}
d5.clear()
d5
>> {}
```

## 集合

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/集合.nbmx%20-%20坚果云_00.png)

### 集合的创建

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210228185240284.png)

注意：在创建空集合的时候只能使用`set()`而不能使用`{}`，因为`{}`创建的是空字典

``` python
set1 = set()

set1.add("a")
set1
>> {'a'}
set1.add(("a","b"))
set1

###直接创建
>> {'a', ('a', 'b')}
set2 = {"a","b","c","a"}
set2

###将列表/元组/字符串转化成集合
>> {'a', 'c', 'b'}
set("abc")
>> {'c', 'a', 'b'}
set(("a","b","c"))
>> {'c', 'a', 'b'}
set(["a",1,2])
>> {'a', 2, 1}
```

### 添加,删除,修改元素

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210228185911041.png)

``` python
set3 = {"a","b","c"}
set3
>> {'a', 'c', 'b'}
set3.add("d")
set3
>> {'a', 'c', 'd', 'b'}
set3.add("a")##相同元素 不执行操作
set3

###update 更新集合 和前面讲的类似，不同的是集合中的元素唯一
>> {'a', 'c', 'd', 'b'}
set3.update("b","e")
set3

###remove移除指定元素
>> {'d', 'a', 'b', 'c', 'e'}
set3.remove("a")
set3

###discard也是移除元素，但是元素不存在不会报错
>> {'d', 'b', 'c', 'e'}
set3.remove("a")
>> Error in py_call_impl(callable, dots$args, dots$keywords): KeyError: 'a'
set3.discard("a")

set3

###pop随机移除
>> {'d', 'b', 'c', 'e'}
set3.pop()
>> 'd'
set3
>> {'b', 'c', 'e'}
```

### 集合操作

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210228190454050.png)
`intersection(set1, set2)`
返回两个集合的交集,也可以使用`&`,`intersection_update(set1, set2)`取交集并更新原来的集合(相当于将原来集合中不重叠的元素删除)

``` python
a = set("abcd")
b = set("cdef")

print(a,b,end="\n")
>> {'c', 'a', 'd', 'b'} {'c', 'd', 'f', 'e'}
a.intersection(b)
>> {'c', 'd'}
print(a,b,end="\n")##原来的集合没有改变
>> {'c', 'a', 'd', 'b'} {'c', 'd', 'f', 'e'}
a.intersection_update(b)
print(a,b,end="\n")##原来的集合改变
>> {'c', 'd'} {'c', 'd', 'f', 'e'}
```

union(set1, set2) 返回两个集合的并集,也可以使用`|`

``` python
a.union(b)
>> {'c', 'd', 'f', 'e'}
a | b
>> {'c', 'd', 'f', 'e'}
```

difference(set)
返回集合的差集,也可以使用`-`,difference\_update(set)更改原来的集合

``` python
a.difference(b)
>> set()
b.difference(a)
>> {'f', 'e'}
a-b
>> set()
b-a
>> {'f', 'e'}
b.difference_update(a)
print(a,b)
>> {'c', 'd'} {'f', 'e'}
```

symmetric\_difference(set)返回集合的异或，或者使用`^`

集合的异或指的是：(参考https://www.cnblogs.com/organic/p/5023038.html)下图绿色的部分

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/818872-20151206095708736-923889139.png)

``` python
a = set("abcd")
b = set("cdef")

print(a,b,end="\n")
>> {'c', 'a', 'd', 'b'} {'c', 'd', 'f', 'e'}
a.symmetric_difference(b)
>> {'f', 'a', 'b', 'e'}
a.symmetric_difference_update(b)
print(a,b,end="\n")
>> {'f', 'e', 'a', 'b'} {'c', 'd', 'f', 'e'}
```

issubset(set)判断集合是否被set包含，也可以使用`<=`;issuperset(set)判断集合是否包含set，也可以使用`>=`

``` python
c = set("ab")

c.issubset(a)
>> True
c <= a
>> True
a.issuperset(c)
>> True
a >= c
>> True
```

isdisjoint(set) 用于判断两个集合是不是不相交

``` python
a.isdisjoint(b)
>> False
```
