---
title: python 基础01
date: 2021-02-13 16:54:19
tags: 编程
index_img: img/python.jpg
---





天池python task1

<!-- more -->

Task1 主要包含以下内容：

-   变量，运算符和基本数据类型

-   位运算

-   条件语句

-   循环语句

-   异常处理

## 变量，运算符和基本数据类型

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210213221003385.png)

``` python
abc = "hello"
b1_ = 7
_b = 1
第一个变量 = "hello world" ##可以使用中文

print(第一个变量)
>> hello world
```

``` python
1a = "hello" ##不能以数字开头

File "<ipython-input-1-ef2f4120639d>", line 1
    1a = "hello"
     ^
SyntaxError: invalid syntax
```

### 运算符

运算符有以下几类：

-   算术运算符
-   比较运算符
-   逻辑运算符
-   位运算符
-   三元运算符
-   其他运算符

算术运算符：

<table>
<thead>
<tr class="header">
<th style="text-align: center;">操作符</th>
<th style="text-align: center;">名称</th>
<th style="text-align: center;">示例</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td style="text-align: center;"><code>+</code></td>
<td style="text-align: center;">加</td>
<td style="text-align: center;"><code>1 + 1</code></td>
</tr>
<tr class="even">
<td style="text-align: center;"><code>-</code></td>
<td style="text-align: center;">减</td>
<td style="text-align: center;"><code>2 - 1</code></td>
</tr>
<tr class="odd">
<td style="text-align: center;"><code>*</code></td>
<td style="text-align: center;">乘</td>
<td style="text-align: center;"><code>3 * 4</code></td>
</tr>
<tr class="even">
<td style="text-align: center;"><code>/</code></td>
<td style="text-align: center;">除</td>
<td style="text-align: center;"><code>3 / 4</code></td>
</tr>
<tr class="odd">
<td style="text-align: center;"><code>//</code></td>
<td style="text-align: center;">整除</td>
<td style="text-align: center;"><code>3 // 4</code></td>
</tr>
<tr class="even">
<td style="text-align: center;"><code>%</code></td>
<td style="text-align: center;">取余</td>
<td style="text-align: center;"><code>3 % 4</code></td>
</tr>
<tr class="odd">
<td style="text-align: center;"><code>**</code></td>
<td style="text-align: center;">幂</td>
<td style="text-align: center;"><code>2 ** 3</code></td>
</tr>
</tbody>
</table>


``` python
3 // 4
>> 0
4 // 3
>> 1
4 % 3
>> 1
```

比较运算符(结果是布尔值True/False)：

<table>
<thead>
<tr class="header">
<th style="text-align: center;">操作符</th>
<th style="text-align: center;">名称</th>
<th style="text-align: center;">示例</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td style="text-align: center;"><code>&gt;</code></td>
<td style="text-align: center;">大于</td>
<td style="text-align: center;"><code>2 &gt; 1</code></td>
</tr>
<tr class="even">
<td style="text-align: center;"><code>&gt;=</code></td>
<td style="text-align: center;">大于等于</td>
<td style="text-align: center;"><code>2 &gt;= 4</code></td>
</tr>
<tr class="odd">
<td style="text-align: center;"><code>&lt;</code></td>
<td style="text-align: center;">小于</td>
<td style="text-align: center;"><code>1 &lt; 2</code></td>
</tr>
<tr class="even">
<td style="text-align: center;"><code>&lt;=</code></td>
<td style="text-align: center;">小于等于</td>
<td style="text-align: center;"><code>5 &lt;= 2</code></td>
</tr>
<tr class="odd">
<td style="text-align: center;"><code>==</code></td>
<td style="text-align: center;">等于</td>
<td style="text-align: center;"><code>3 == 4</code></td>
</tr>
<tr class="even">
<td style="text-align: center;"><code>!=</code></td>
<td style="text-align: center;">不等于</td>
<td style="text-align: center;"><code>3 != 5</code></td>
</tr>
</tbody>
</table>


``` python
3 >= 4 
>> False
3 != 5
>> True
```

逻辑运算符(结果也是布尔值True/False):

<table>
<thead>
<tr class="header">
<th style="text-align: center;">操作符</th>
<th style="text-align: center;">名称</th>
<th style="text-align: center;">示例</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td style="text-align: center;"><code>and</code></td>
<td style="text-align: center;">与</td>
<td style="text-align: center;"><code>(3 &gt; 2) and (3 &lt; 5)</code></td>
</tr>
<tr class="even">
<td style="text-align: center;"><code>or</code></td>
<td style="text-align: center;">或</td>
<td style="text-align: center;"><code>(1 &gt; 3) or (9 &lt; 2)</code></td>
</tr>
<tr class="odd">
<td style="text-align: center;"><code>not</code></td>
<td style="text-align: center;">非</td>
<td style="text-align: center;"><code>not (2 &gt; 1)</code></td>
</tr>
</tbody>
</table>


``` python
( 3 >= 4 ) and ( 3 !=5 )
>> False
( 3 >= 4 ) or ( 3 !=5 )
>> True
not( 3 !=5 )
>> False
```

位运算符:

<table>
<thead>
<tr class="header">
<th style="text-align: center;">操作符</th>
<th style="text-align: center;">名称</th>
<th style="text-align: center;">示例</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td style="text-align: center;"><code>~</code></td>
<td style="text-align: center;">按位取反</td>
<td style="text-align: center;"><code>~4</code></td>
</tr>
<tr class="even">
<td style="text-align: center;"><code>&amp;</code></td>
<td style="text-align: center;">按位与</td>
<td style="text-align: center;"><code>4 &amp; 5</code></td>
</tr>
<tr class="odd">
<td style="text-align: center;"><code>|</code></td>
<td style="text-align: center;">按位或</td>
<td style="text-align: center;"><code>4 | 5</code></td>
</tr>
<tr class="even">
<td style="text-align: center;"><code>^</code></td>
<td style="text-align: center;">按位异或</td>
<td style="text-align: center;"><code>4 ^ 5</code></td>
</tr>
<tr class="odd">
<td style="text-align: center;"><code>&lt;&lt;</code></td>
<td style="text-align: center;">左移</td>
<td style="text-align: center;"><code>4 &lt;&lt; 2</code></td>
</tr>
<tr class="even">
<td style="text-align: center;"><code>&gt;&gt;</code></td>
<td style="text-align: center;">右移</td>
<td style="text-align: center;"><code>4 &gt;&gt; 2</code></td>
</tr>
</tbody>
</table>


三元运算符,也叫条件表达式，可以简化条件判断和赋值操作：

``` python
#condition_is_true if condition else condition_is_false
##如果条件为真，则返回if前面的结果，如果为假则返回else后面的结果

x = "hello"

print("x 是一个字符") if isinstance(x,str) else print("x 不是字符")
>> x 是一个字符
```

其他运算符：

<table>
<thead>
<tr class="header">
<th style="text-align: center;">操作符</th>
<th style="text-align: center;">名称</th>
<th style="text-align: center;">示例</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td style="text-align: center;"><code>in</code></td>
<td style="text-align: center;">存在</td>
<td style="text-align: center;"><code>'A' in ['A', 'B', 'C']</code></td>
</tr>
<tr class="even">
<td style="text-align: center;"><code>not in</code></td>
<td style="text-align: center;">不存在</td>
<td style="text-align: center;"><code>'h' not in ['A', 'B', 'C']</code></td>
</tr>
<tr class="odd">
<td style="text-align: center;"><code>is</code></td>
<td style="text-align: center;">是</td>
<td style="text-align: center;"><code>"hello" is "hello"</code></td>
</tr>
<tr class="even">
<td style="text-align: center;"><code>not is</code></td>
<td style="text-align: center;">不是</td>
<td style="text-align: center;"><code>"hello" is not "hello"</code></td>
</tr>
</tbody>
</table>


需要注意的是`is`/`not is`
和`==`/`!=`的区别，`is`/`not is`比较的是内存地址，而`==`/`!=`比较的是变量的值

``` python
###不可变类型，两者是一样的；因为对于不可变类型，值一样内存地址就一样
x = "hello"
id(x) ##使用id查看内存地址
>> 2241811015152
y = "hello"
id(y)
>> 2241811015152
x == y
>> True
x is y
>> True
id(x) == id(y)
>> True
```

``` python
###对于可变类型，两者是有区别的
a = [1,2,3]
id(a)
>> 2241811025408
b = [1,2,3]
id(b)
>> 2241810930752
id(a) == id(b)
>> False
a == b
>> True
a is b
>> False
```

运算符的优先级：

<table>
<thead>
<tr class="header">
<th>运算符</th>
<th>描述</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td>**</td>
<td>指数（最高优先级）</td>
</tr>
<tr class="even">
<td>~+-</td>
<td>按位翻转，一元加号和减号</td>
</tr>
<tr class="odd">
<td>* / % //</td>
<td>乘，除，取模和取整除）</td>
</tr>
<tr class="even">
<td>+ -</td>
<td>加法减法</td>
</tr>
<tr class="odd">
<td>&gt;&gt; &lt;&lt;</td>
<td>右移，左移运算符</td>
</tr>
<tr class="even">
<td>&amp;</td>
<td>位‘AND’</td>
</tr>
<tr class="odd">
<td>^|</td>
<td>位运算符</td>
</tr>
<tr class="even">
<td>&lt;=&lt;&gt;&gt;=</td>
<td>比较运算符</td>
</tr>
<tr class="odd">
<td>&lt;&gt;==!=</td>
<td>等于运算符</td>
</tr>
<tr class="even">
<td>=%=/=//=-=+=*=**=</td>
<td>赋值运算符</td>
</tr>
<tr class="odd">
<td>is is not</td>
<td>身份运算符</td>
</tr>
<tr class="even">
<td>in not in</td>
<td>成员运算符</td>
</tr>
<tr class="odd">
<td>not and or</td>
<td>逻辑运算符</td>
</tr>
</tbody>
</table>


### 基本数据类型

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210214114944454.png)

基本数据类型包括：整型，浮点型和布尔型

<table>
<thead>
<tr class="header">
<th style="text-align: center;">类型</th>
<th style="text-align: center;">名称</th>
<th style="text-align: center;">示例</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td style="text-align: center;">int</td>
<td style="text-align: center;">整型 <code>&lt;class 'int'&gt;</code></td>
<td style="text-align: center;"><code>-876, 10</code></td>
</tr>
<tr class="even">
<td style="text-align: center;">float</td>
<td style="text-align: center;">浮点型<code>&lt;class 'float'&gt;</code></td>
<td style="text-align: center;"><code>3.149, 11.11</code></td>
</tr>
<tr class="odd">
<td style="text-align: center;">bool</td>
<td style="text-align: center;">布尔型<code>&lt;class 'bool'&gt;</code></td>
<td style="text-align: center;"><code>True, False</code></td>
</tr>
</tbody>
</table>


可以通过`type`或者`isinstance`来获取对象的类：

``` python
x = 123
type(x)
>> <class 'int'>
isinstance(x,int)
>> True
```

两者的区别是：`type`不考虑继承关系，而`isinstance`考虑继承关系：

``` python
class A:
    pass
 
class B(A):
    pass
##B(A)表示B继承A

type(A()) == A
>> True
type(B()) == A
>> False
isinstance(B(),A)
>> True
```

对于**浮点数**，有时候我们想要控制其显示的小数点位数，可以使用decimal
包里的 Decimal 对象和 getcontext() 方法：

``` python
import decimal
from decimal import Decimal

a = decimal.getcontext()
a
>> Context(prec=28, rounding=ROUND_HALF_EVEN, Emin=-999999, Emax=999999, capitals=1, clamp=0, flags=[], traps=[InvalidOperation, DivisionByZero, Overflow])
```

其中prec=28表示默认精度是28位：

``` python
1/3
>> 0.3333333333333333
Decimal(1)/Decimal(3)
>> Decimal('0.3333333333333333333333333333')
```

用 getcontext().prec 来调整精度,使其保留小数点后4位：

``` python
decimal.getcontext().prec = 4
Decimal(1)/Decimal(3)
>> Decimal('0.3333')
```

对于**布尔值**,
除了直接将True和False赋值给布尔型变量之外，还可以使用`bool`函数来创建布尔型变量，这个函数的参数可以有两种类型：

-   基本数据类型：整型，浮点型和布尔型
-   容器数据类型：字符串，列表，元组，字典和集合

对于基本数据类型：0(包括整型的0和浮点型的0.0)是False，其他都是True  
对于容器数据类型：空的就是False，非空的就是True

``` python
bool(0)
>> False
bool(0.00)
>> False
bool(1)
>> True
bool("hello")
>> True
bool([])
>> False
bool([1])
>> True
bool({})
>> False
bool({"a":1})
>> True
```

## 位运算

计算机中的数在内存中都是以二进制形式进行存储的，位运算就是直接对数在内存中的二进制位来进行操作，具有较高的效率；二进制有3种表示形式：原码，反码和补码
**计算机内部使用补码来表示**

-   原码：正常的二进制表示(负数有一个符号位)；比如`00 00 00 11`表示的数为3，`10 00 00 11`表示的数为-3，最高位(最左边)为符号位(0表示正，1表示负)

-   反码：正数的反码和原码一样，负数的反码是对应正数原码进行按位取反，比如3的原码和反码是一样的，而-3的反码是`11 11 11 00`

-   补码：正数的补码和原码一样，负数的补码为反码加1，比如-3的补码为`11 11 11 01`

### 按位运算

-   按位非: `~`

`~ num`表示将num的补码进行取反(0变成1，1变成0，包括符号位)

``` python
~ 1 ###1的原码为00 00 00 01;补码和原码一样，所以取反后为11 11 11 10是负数，
##所以转化成十进制为数值位取反加1：0 00 00 01+1=0 00 00 10 为2 再加上符号，因此为-2
>> -2
```

-   按位与操作: `&`

这是一个二元操作符，只有两个对应位都是1时结果才为1

``` python
1 & 1
>> 1
1 & -2
##1的补码为00 00 00 01；-2的原码为10 00 00 10，补码为00 00 00 10按位取反得11 11 11 01再加1为11 11 11 10
##所以 与操作 结果为00 00 00 00 为0
>> 0
```

-   按位或操作：`|`

只要两个对应位中有一个为1结果就为1

``` python
1 | 1
>> 1
1 | -2 ##11 11 11 11转化为10进制：10 00 00 01 = -1
>> -1
```

-   按位异或操作：`^`

两个对应位不同时结果才是1

``` python
1 ^ -2 ##11 11 11 11 结果也是-1
>> -1
```

-   按位左移和右移操作

`num << i`将num得二进制表示(1的位置)向左移动i位，`>>`表示向右移动：

``` python
11 << 3 ##11的补码为00 00 10 11将所有的1向左移3位得到：01 01 10 00为88
>> 88
11 >> 3 ##右移3位得到 00 00 00 01为1
>> 1
```

## 条件语句

-   if语句：

``` python
if expression:
    expr_true_suite
```

只有当expression为真，才执行语句expr\_true\_suite

-   if-else 语句：

``` python
if expression:
    expr_true_suite
else:
    expr_false_suite
```

expression为真执行expr\_true\_suite，否则执行expr\_false\_suite

-   if-elif-else语句：

``` python
if expression1:
    expr1_true_suite
elif expression2:
    expr2_true_suite
    .
    .
elif expressionN:
    exprN_true_suite
else:
    expr_false_suite
```

进行多重判断

``` python
source = 99
if 100 >= source >= 90:
    print('A')
elif 90 > source >= 80:
    print('B')
elif 80 > source >= 60:
    print('C')
elif 60 > source >= 0:
    print('D')
else:
    print('输入错误！')
>> A
```

-   assert 断言关键词

当assert后面的语句为False时，会抛出`AssertionError`异常：

``` python
assert 3 > 6
##AssertionError
```

## 循环语句

-   while 循环

``` python
while expression:
  code
```

当expression为真的时候会一直执行缩进语句中的代码

-   while-else 循环

``` python
while expression:
  code1
else:
  code2
```

expression为真，执行code1，为假则执行code2；需要注意的是如果code1中执行了跳出循环的语句，那么不会执行code2中的代码

``` python
count = 5 

while count > 6:
  print("%d is more than 6" % count)
  count = count - 1
else:
  print("%d is not more than 6" % count)
>> 5 is not more than 6
```

``` python
count = 5 

while count > 4:
  print("%d is more than 4" % count)
  count = count - 1
  break
else:
  print("%d is not more than 4" % count)
>> 5 is more than 4
```

``` python
while count > 4:
  print("%d is more than 4" % count)
  count = count - 1
else:
  print("%d is not more than 4" % count)
>> 4 is not more than 4
```

-   for 循环

for循环是一个通用的序列迭代器，可以遍历任何可迭代对象(str,list,tuple,dict等)

``` python
for var in object:
  code
```

``` python
for i in "abcd":
  print(i,end="\t")
>> a    b   c   d   
```

``` python
dic = {"a":1,"b":2,"c":3}
for i in dic:
  print(i,end=" ")##默认是keys
>> a b c
```

``` python
for i in dic.values():
  print(i,end=" ")
>> 1 2 3
```

-   for-else 循环

``` python
for var in object:
  code1
else:
  code2
```

和while-else循环类似，在for循环执行完后执行else下的语句，如果执行了code1中的跳出循环的语句将不会执行code2

-   range()函数

range函数的用法为`range(start,stop,step=1)`，可以用来生成从start到stop步长为step的数字序列(注意：不包含stop，左闭右开)

``` python
for i in range(1,3):
  print(i,end=" ")
>> 1 2
```

``` python
for i in range(3):##默认为stop
  print(i,end=" ")
>> 0 1 2
```

``` python
for i in range(1,10,2):
  print(i,end=" ")
>> 1 3 5 7 9
```

-   enumerate()函数

enumerate的用法为`enumerate(seq,start=0)`,seq为可迭代对象，返回的也是一个可迭代对象(数据加上索引，索引默认从0开始，可以通过start参数指定)

``` python
seasons = ['Spring', 'Summer', 'Fall', 'Winter']

seasons_enu = enumerate(seasons)
print(seasons_enu,type(seasons_enu))
>> <enumerate object at 0x00000209F6583380> <class 'enumerate'>
print(list(seasons_enu))
>> [(0, 'Spring'), (1, 'Summer'), (2, 'Fall'), (3, 'Winter')]
seasons_enu1 = enumerate(seasons,start=1)
print(list(seasons_enu1))
>> [(1, 'Spring'), (2, 'Summer'), (3, 'Fall'), (4, 'Winter')]
```

``` python
for i in seasons_enu1:
  print(i,end=" ")
```

-   break 和 continue 语句

break是跳出整个循环语句，而continue是跳出本次循环

``` python
counts = 4

while counts < 7:
  print("%d is less than 7" % counts)
  counts = counts +1
  if counts == 5:
    break
  print("i run!")
>> 4 is less than 7
```

``` python
counts = 4

while counts < 7:
  print("%d is less than 7" % counts)
  counts = counts +1
  if counts == 5:
    continue
  print("i run!")
>> 4 is less than 7
>> 5 is less than 7
>> i run!
>> 6 is less than 7
>> i run!
```

-   pass 语句

pass 是空语句，不做任何操作，起到占位作用(暂时不确定在该位置写什么代码)

## 异常处理

### Python 标准异常总结

-   BaseException：所有异常的 **基类**
-   Exception：常规异常的 **基类**
-   StandardError：所有的内建标准异常的基类
-   ArithmeticError：所有数值计算异常的基类
-   FloatingPointError：浮点计算异常
-   <u>OverflowError</u>：数值运算超出最大限制
-   <u>ZeroDivisionError</u>：除数为零
-   <u>AssertionError</u>：断言语句（assert）失败
-   <u>AttributeError</u>：尝试访问未知的对象属性
-   EOFError：没有内建输入，到达EOF标记
-   EnvironmentError：操作系统异常的基类
-   IOError：输入/输出操作失败
-   <u>OSError</u>：操作系统产生的异常（例如打开一个不存在的文件）
-   WindowsError：系统调用失败
-   <u>ImportError</u>：导入模块失败的时候
-   KeyboardInterrupt：用户中断执行
-   LookupError：无效数据查询的基类
-   <u>IndexError</u>：索引超出序列的范围
-   <u>KeyError</u>：字典中查找一个不存在的关键字
-   <u>MemoryError</u>：内存溢出（可通过删除对象释放内存）
-   <u>NameError</u>：尝试访问一个不存在的变量
-   UnboundLocalError：访问未初始化的本地变量
-   ReferenceError：弱引用试图访问已经垃圾回收了的对象
-   RuntimeError：一般的运行时异常
-   NotImplementedError：尚未实现的方法
-   <u>SyntaxError</u>：语法错误导致的异常
-   IndentationError：缩进错误导致的异常
-   TabError：Tab和空格混用
-   SystemError：一般的解释器系统异常
-   <u>TypeError</u>：不同类型间的无效操作
-   <u>ValueError</u>：传入无效的参数
-   UnicodeError：Unicode相关的异常
-   UnicodeDecodeError：Unicode解码时的异常
-   UnicodeEncodeError：Unicode编码错误导致的异常
-   UnicodeTranslateError：Unicode转换错误导致的异常

异常体系内部有层次关系，Python异常体系中的部分关系如下所示：

![](https://img-blog.csdnimg.cn/20200710131404548.png)

------------------------------------------------------------------------

### Python标准警告总结

-   Warning：警告的基类
-   DeprecationWarning：关于被弃用的特征的警告
-   FutureWarning：关于构造将来语义会有改变的警告
-   UserWarning：用户代码生成的警告
-   PendingDeprecationWarning：关于特性将会被废弃的警告
-   RuntimeWarning：可疑的运行时行为(runtime behavior)的警告
-   SyntaxWarning：可疑语法的警告
-   ImportWarning：用于在导入模块过程中触发的警告
-   UnicodeWarning：与Unicode相关的警告
-   BytesWarning：与字节或字节码相关的警告
-   ResourceWarning：与资源使用相关的警告

捕获异常可以使用try-except语句：

-   try-except语句

``` python
try:
  code1
except Exception [as reason]:
  code2
```

首先执行code1，如果没有异常发生就忽略code2；如果code1中出现异常，那么就会将异常的类型(见上面的总结)和Exception进行匹配，如果可以匹配上就执行code2,如果不能匹配，异常将传递到上层的try，但是一直没有匹配就会报错

``` python
try:
    f = open('test.txt')
    print(f.read())
    f.close()
except OSError:
    print('open file error')
>> open file error
```

``` python
try:
   1 + "p"
except OSError:
    print('open file error')
>> Error in py_call_impl(callable, dots$args, dots$keywords): TypeError: unsupported operand type(s) for +: 'int' and 'str'
```

还可以加上as来展示具体的错误信息：

``` python
try:
    f = open('test.txt')
    print(f.read())
    f.close()
except OSError as error:
    print('open file error, reason is :' + str(error))
>> open file error, reason is :[Errno 2] No such file or directory: 'test.txt'
```

try后面可以接上多个except语句，用来处理不同的异常，但是需要注意异常之间的关系(见上面那张图)：

``` python
dict1 = {'a': 1, 'b': 2, 'v': 22}
try:
    x = dict1['y']
except LookupError:
    print('查询错误')
except KeyError:
    print('键错误')
>> 查询错误
```

这里KeyError属于LookupError的子类，LookupError又在前面，所以执行的是LookupError里面的语句，所以在使用多个except语句时，要将最底端的异常放在前面(更加具体)

``` python
dict1 = {'a': 1, 'b': 2, 'v': 22}
try:
    x = dict1['y']
except KeyError:
    print('键错误')
except LookupError:
    print('查询错误')
>> 键错误
```

一个except语句也可以处理多个异常，将需要处理的异常放在元组中：

``` python
try:
    s = 1 + 'p'
    f = open('test.txt')
    print(f.read())
    f.close()
except (OSError, TypeError) as error:
    print('error !, the reason for first error is ：' + str(error))
>> error !, the reason for first error is ：unsupported operand type(s) for +: 'int' and 'str'
```

-   try-except-else 语句

``` python
try:
  code2
except  Exception [as reason]:
  code2
else:
  code3
```

如果没有异常则执行code3

-   try-except-finally 语句

``` python
try:
  code2
except  Exception [as reason]:
  code2
finally:
  code3
```

无论code1中有没有异常，code3中的代码都会被执行(如果code1中有异常并且不能被except捕获，那么会在运行code3之后报错)

``` {python}
def divide(x, y):
    try:
        result = x / y
        print("result is", result)
    except ZeroDivisionError:
        print("division by zero!")
    finally:
        print("executing finally clause")
        
divide(2, 1)
divide(2, 0)
divide("2", "1")
```

