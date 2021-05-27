---
title: 【R语言编程指南】元编程
date: 2021-01-14 17:11:27
tags: 编程
index_img: img/r_inter.jpg
---



函数式编程，基于语言的计算，非标准计算

<!-- more -->

本章主要学习3个内容：

-   函数式编程
    -   在函数内部定义的函数→闭包
    -   与其他函数组合使用的函数→高阶函数
-   基于语言的计算
-   非标准计算

## 函数式编程

### 闭包

闭包(closure)就是在函数内部定义的函数,下面创建一个简单的闭包：

``` r
addn <- function(y){
  function(x){
    x+y
  }
}
```

上面的函数在其内部创建一个函数，所以其返回值是一个闭包；这个addn就相当于一个“函数工厂”，通过提供不同的参数就可以创建不同的函数：

``` r
add1 <- addn(1)
add2 <- addn(2)

add1
>> function(x){
>>     x+y
>>   }
>> <environment: 0x00000209f3243f80>
##当函数不在当前工作环境，输出该函数时会显示其所在的环境

add1(10)
>> [1] 11
add2(10)
>> [1] 12
```

我们可以通过在[R内部机制](http://localhost:4321/p/r-inter/)中提到的ennironment()来查看这两个闭包的封闭环境：

``` r
environment(add1)$y
>> [1] 1
environment(add2)$y
>> [1] 2
```

既然闭包可以用来创建函数，我们就可以将一些大部分情况下只会用到一部分参数的函数包装起来形成一个简版的“专用函数”.
比如画图的时候可能在不同的图之间只需要更改线条的颜色，那么就可以将其他参数包装起来，使代码更简洁：

``` r
color_line <- function(col){
  function(...){
    plot(...,type="l",lty=1,col=col)
  }
}

##生成专用函数
red_line <- color_line("red")
red_line(rnorm(30))
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/ex4-1.png)<!-- -->

``` r
##也可以设置其他参数，因为上面的闭包中使用...来处理其他的参数
red_line(rnorm(30),main="Red line plot")
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/ex4-2.png)<!-- -->

``` r
###与下面的代码相比较，使用闭包创建专用函数可以使代码更简洁
#plot(rnorm(30),type = "l",lty=1,col="red",main = "Red line plot")
```

### 高阶函数

高阶函数指的是：将另外一个函数作为参数的函数
将函数作为参数也就是将现有的函数和一个变量名绑定，那么将一个函数赋给一个变量会影响函数的封闭环境吗(这可能会影响到变量的搜索路径)，答案是不会改变，下面的实验证明了这一点：

``` r
f1 <- function(){
  cat("f1 的执行环境为 ")
  print(environment())
  cat("f1 的封闭环境为 ")
  print(parent.env(environment()))
  cat("f1 的调用环境为 ")
  print(parent.frame())
}

f2 <- function(){
  cat("f2 的执行环境为 ")
  print(environment())
  cat("f2 的封闭环境为 ")
  print(parent.env(environment()))
  cat("f2 的调用环境为 ")
  print(parent.frame())
  p <- f1
  p()
}

f1()
>> f1 的执行环境为 <environment: 0x00000209f26542b0>
>> f1 的封闭环境为 <environment: R_GlobalEnv>
>> f1 的调用环境为 <environment: R_GlobalEnv>
f2()
>> f2 的执行环境为 <environment: 0x00000209f26000d0>
>> f2 的封闭环境为 <environment: R_GlobalEnv>
>> f2 的调用环境为 <environment: R_GlobalEnv>
>> f1 的执行环境为 <environment: 0x00000209f25f5928>
>> f1 的封闭环境为 <environment: R_GlobalEnv>
>> f1 的调用环境为 <environment: 0x00000209f26000d0>
```

可以看到在f2内部将f1赋给p并调用p，并不改变f1的封闭环境(定义的地方)，所以将一个函数赋给一个变量仅仅是给了函数一个“别名”:

``` r
f3 <- function(x,y){
  if(x > y){
    x-y
  }else{
    x+y
  }
}

###给+,-函数起别名
f4 <- function(x,y){
  op <- if(x>y) `-` else `+`
  op(x,y)
}

f3(1,2)
>> [1] 3
f4(1,2)
>> [1] 3
```

既然函数可以作为变量使用，那么函数也可以作为参数来传递：

``` r
add <- function(x,y,z){
  x+y+z
}

product <- function(x,y,z){
  x*y*z
}

###定义高阶函数，以其他函数作为参数
combine <- function(f,x,y,z){
  f(x,y,z)
}
```

这里的参数名f就相当于在高阶函数内部给传入的函数起了个别名，在combine函数内部这个函数就叫f了

``` r
###将add和product传给combine
combine(add,3,4,5)
>> [1] 12
combine(product,3,4,5)
>> [1] 60
```

我们经常使用的apply函数族就是高阶函数，接受其他的函数作为参数：

``` r
sapply(1:3,addn(3))
>> [1] 4 5 6
```

## 基于语言的计算(元编程)

元编程可以允许我们调整语言本身，使特定的语言结构在特定情况下更方便使用(不是很懂)
先来看一个元编程的用处：

``` r
###将iris数据集中数值列的大于80%分位数的数挑出来
iris[iris$Sepal.Length > quantile(iris$Sepal.Length,0.8),
     iris$Sepal.Width > quantile(iris$Sepal.Width,0.8),
     iris$Petal.Length > quantile(iris$Petal.Length,0.8),
     iris$Petal.Width > quantile(iris$Petal.Width,0.8)]
>> Error in `[.data.frame`(iris, iris$Sepal.Length > quantile(iris$Sepal.Length, : 参数没有用(iris$Petal.Width > quantile(iris$Petal.Width, 0.8))

##上面的代码就比较繁琐，需要多次写iris$
##subset函数可以简化上述代码
subset(iris,
       Sepal.Length > quantile(Sepal.Length,0.8) &
       Sepal.Width > quantile(Sepal.Width,0.8) &
       Petal.Length > quantile(Petal.Length,0.8) &
       Petal.Width > quantile(Petal.Width,0.8))
>>     Sepal.Length Sepal.Width Petal.Length Petal.Width   Species
>> 110          7.2         3.6          6.1         2.5 virginica
>> 118          7.7         3.8          6.7         2.2 virginica
>> 132          7.9         3.8          6.4         2.0 virginica


###但是下面的代码就不能运行：
iris[Sepal.Length > quantile(Sepal.Length,0.8) &
     Sepal.Width > quantile(Sepal.Width,0.8) &
     Petal.Length > quantile(Petal.Length,0.8) &
     Petal.Width > quantile(Petal.Width,0.8)]
>> Error in `[.data.frame`(iris, Sepal.Length > quantile(Sepal.Length, 0.8) & : 找不到对象'Sepal.Length'
```

这是因为subset函数使用**元编程调整了其参数的计算环境**(也就是不是去全局环境寻找这些变量，所以不会产生后面那样的报错)，这个过程分为两步：*捕获表达式；调整表达式的计算(修改计算的环境等)*

### 捕获表达式

捕获表达式指的是：将表达式本身存储为变量的形式，防止表达式的直接执行(执行了还修改什么)；使用函数quote()

``` r
call1 <- quote(rnorm(5))
call1
>> rnorm(5)

typeof(call1)
>> [1] "language"
class(call1)
>> [1] "call"

name1 <- quote(rnorm)
name1
>> rnorm

typeof(name1)
>> [1] "symbol"
class(name1)
>> [1] "name"
```

可以看到当我们捕获一个函数调用的时候返回的是一个语言对象(language)/函数调用(call),当捕获一个函数名(变量名)返回的是符号(symbol)/名称(name)

因此这里面我们需要区分的是：

-   变量和符号对象：变量表示的是一个对象的名称，而这个名称本身也是一个对象，这个对象就是符号对象

-   函数和调用对象：函数是可以被调用(计算)的对象，而调用对象是函数调用的语言对象，是不会被计算的

可以将对象转化成列表以便查看其内部结构：

``` r
as.list(call1)
>> [[1]]
>> rnorm
>> 
>> [[2]]
>> [1] 5

typeof(call1[[1]])
>> [1] "symbol"
class(call1[[1]])
>> [1] "name"

typeof(call1[[2]])
>> [1] "double"
class(call1[[2]])
>> [1] "numeric"
```

符号对象和调用对象都是语言对象，可以使用is.symbol/is.name检查对象是否为符号对象，使用is.call()检查是否为调用对象；也可以使用is.language()同时检查：

``` r
is.call(call1)
>> [1] TRUE
is.call(name1)
>> [1] FALSE
is.symbol(name1)
>> [1] TRUE
is.symbol(call1)
>> [1] FALSE

is.language(call1)
>> [1] TRUE
is.language(name1)
>> [1] TRUE
```

捕获已知的表达式可以使用quote()，但是需要捕获用户输入的参数就不行了:

``` r
func1 <- function(x){
  quote(x)
}

func1(rnorm(5))
>> x
```

这个时候可以使用函数substitute,substitute基本用法为：substitute(expr,
env),expr为表达式，env为环境或者列表，默认是当前的执行环境,*将表达式中的变量替换成环境中的值*：

``` r
fun2 <- function(x){
  substitute(x)##默认是当前的执行环境
}

fun2(rnorm(5))
>> rnorm(5)

##substitute 用法
##将x替换成执行环境中x所绑定的值，也就是函数的输入
substitute(x+y,list(x=1))
>> 1 + y

substitute(f(x+f(y)),list(f = quote(sin)))
>> sin(x + sin(y))
```

除了将表达式捕获为语言对象之外，还可以直接创建语言对象:

``` r
##quote捕获
call2 <- quote(rnorm(5,mean = 3))

##call 创建函数调用
call3 <- call("rnorm",5,mean=3)
call3
>> rnorm(5, mean = 3)

##as.call将列表转化成函数调用
call4 <- as.call(list(quote(rnorm),5,mean=3))
call4
>> rnorm(5, mean = 3)

identical(call3,call4)
>> [1] TRUE
identical(call2,call3)
>> [1] TRUE
```

### 修改表达式

当一个表达式被捕获为调用对象后，可以将其当作列表来修改:

``` r
call1
>> rnorm(5)

##可以修改第一个元素来更改要调用的函数
call1[[1]] <- quote(runif())
call1
>> runif()(5)

##也可以添加新的元素来添加参数
call1[[3]] <- "min" 
call1
>> runif()(5, "min")
```

### 执行表达式

捕获表达式后，下一步就是对其求值，可使用eval()函数  

eval的基本用法为：eval(expr,envir,enclos),expr是需要被计算的对象，envir是执行环境，enclos是封闭环境(在执行环境中找不到变量就会到这里找)，enclos如果不指定就取决于envir的类型，如果envir是列表则enclos是当前执行函数的调用环境(parent.frame()),如果envir不是列表则enclos是baseenv()

``` r
call1 <- quote(sin(x))
eval(call1)
>> Error in eval(call1): 找不到对象'x'

x <- 1
eval(call1)
>> [1] 0.841471

call2 <- quote(x^2+y^2)
eval(call2)
>> Error in eval(call2): 找不到对象'y'

eval(call2,list(y=1))
>> [1] 2

rm(x,y)
>> Warning in rm(x, y): 找不到对象'y'

e1 <- new.env()
e1$x <- 1

eval(call2,envir = e1)
>> Error in eval(call2, envir = e1): 找不到对象'y'

###新建一个环境 其父环境是e1
e2 <- new.env(parent = e1)
e2$y <- 2

eval(call2,e2)
>> [1] 5

###也可以指定封闭环境
e3 <- new.env()
e3$y  <- 1
eval(call2,list(x=2),e3)
>> [1] 5
```

所以通过捕获表达式–执行表达式，我们可以**调整表达式的执行环境和封闭环境来定制计算过程**,这就是一开始subset函数的“魔力”所在,我们来看一下subset函数的源码：

``` r
subset.data.frame
>> function (x, subset, select, drop = FALSE, ...) 
>> {
>>     r <- if (missing(subset)) 
>>         rep_len(TRUE, nrow(x))
>>     else {
>>         e <- substitute(subset)
>>         r <- eval(e, x, parent.frame())
>>         if (!is.logical(r)) 
>>             stop("'subset' must be logical")
>>         r & !is.na(r)
>>     }
>>     vars <- if (missing(select)) 
>>         TRUE
>>     else {
>>         nl <- as.list(seq_along(x))
>>         names(nl) <- names(x)
>>         eval(substitute(select), nl, parent.frame())
>>     }
>>     x[r, vars, drop = drop]
>> }
>> <bytecode: 0x00000209ea31b388>
>> <environment: namespace:base>
```

可以看到这里也使用了substitute和eval的组合来选择特定的实例或者变量

## 非标准计算

为了理解非标准计算，我们先来看一个例子，从向量中取子集：
假设现在有一个整数向量，我们想从中提取第3个到倒数第5个元素

``` r
x <- 1:10
x[3:(length(x)-5)]
>> [1] 3 4 5
```

上面的表达式用了两次x，有点繁琐；我们可以使用之前讲过的元编程技术来定义一个函数：

``` r
qs <- function(x,range){
  range <- substitute(range)
  selector <- eval(range,list(.=length(x)))###用x向量的长度来代替点号
  x[selector]
}
```

这个函数可以使用点号来表示向量的长度：

``` r
qs(x,3:(.-5))
>> [1] 3 4 5

###
qs(x,.-1)
>> [1] 9
```

基于qs，下面的函数用于修剪x两端的n个元素，返回去掉前n个和后n个元素的中间部分

``` r
trim_margin <- function(x,n){
  qs(x,(n+1):(.-n-1))
}
```

但是我们调用这个函数的时候，会出现错误：

``` r
trim_margin(x,3)
>> Error in n + 1: 二进列运算符中有非数值参数
```

我们来分析一下为什么会报错：
当调用trim\_margin(x,3)时，会在一个新的执行环境中调用qs(x,(n+1):(.-n-1))，而在qs内部使用eval来执行捕获到的表达式，回忆一下eval的用法,

> eval的基本用法为：eval(expr,envir,enclos),expr是需要被计算的对象，envir是执行环境，enclos是封闭环境(在执行环境中找不到变量就会到这里找)，enclos如果不指定就取决于envir的类型，如果envir是列表则enclos是当前执行函数的调用环境(parent.frame()),如果envir不是列表则enclos是baseenv()

而qs内部的eval提供的就是一个列表并且只有点号，所以当找不到n的时候就会到eval的调用环境(parent.frame())中去找，也就是qs的执行环境，而qs的执行环境当然时没有n的(n在qs的调用环境，也是trim\_margin的执行环境中)，所以会报错

解决这个报错也比较简单，我们只需要指定eval的封闭环境就行

``` r
qs <- function(x,range){
  range <- substitute(range)
  selector <- eval(range,list(.=length(x)),enclos = parent.frame())###用x向量的长度来代替点号
  x[selector]
}
```
