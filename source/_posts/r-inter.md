---
title: 【R语言编程指南】R内部机制
date: 2021-01-04 17:03:30
tags: 编程
index_img: img/r_inter.jpg
categories:
  - R
---



本章主要是学习四个方面：

-   惰性求值
-   复制修改机制
-   词法作用域
-   环境

<!-- more -->

## 惰性求值

惰性求值指的是：在函数调用时，参数的值只在用到的时侯才会被调用/执行

比如下面这个函数：

``` r
test0 <- function(x,y){
  if(x > 0){
    x
  }else{
    y
  }
}

test0(1,stop("Stop nrow"))
>> [1] 1
test0(-1,stop("Stop nrow"))
>> Error in test0(-1, stop("Stop nrow")): Stop nrow
```

可以看到当调用`test0(1,stop("Stop nrow"))`并不会报错，因为这种情况下函数不会运行y(因为不需要y的值)；而在调用`test0(-1,stop("Stop nrow"))`的时候会发生报错是因为输入x是负数，因此会进入`else`运行y，而y的值是表达式`stop("Stop nrow")`所以会报错

在这一节中还有一个巧妙的用法:使用`stop`和`switch`来控制函数的输入:

``` r
check_input <- function(x){
  switch(x,
         y = message("yes"),
         n = message("no"),
         stop("Invalid input")
  )
}

check_input("y")
>> yes
check_input("n")
>> no
check_input("a")
>> Error in check_input("a"): Invalid input
```

## 复制——修改机制

复制修改机制指的是：当有多个变量指向同一个对象，那么修改一个变量(包括值和属性)就会生成该对象的一个副本

我们可以看一个例子：

``` r
x1 <- c(1,2,3)
x2 <- x1

##使用tracemem可以追踪变量的内存地址
tracemem(x1)
>> [1] "<00000209EAA600E0>"
tracemem(x2)
>> [1] "<00000209EAA600E0>"

x1[1] <- 0
>> tracemem[0x00000209eaa600e0 -> 0x00000209ea315e78]: eval eval withVisible withCallingHandlers handle timing_fn evaluate_call <Anonymous> evaluate in_dir block_exec call_block process_group.block process_group withCallingHandlers process_file <Anonymous> render
```

可以看到在赋值操作中变量所指向的内存地址是一样的，但是在改变其中一个变量的值的时候，该变量的内存地址发生了变化，也就是说修改操作会生成一个副本，然后在该副本上进行修改

对于函数的参数也是这样，当我们传一个变量给函数的参数时，就相当于该变量和函数参数所表示的变量都指向我们传入的数据，所以在函数内部进行修改时并不会修改传入的变量，而是将该变量的值复制后再进行修改:

``` r
modify <- function(x){
  x[1] <- 2
  x
}

v1 <- c(1,2,3)
modify(v1)
>> [1] 2 2 3
v1
>> [1] 1 2 3
```

可用下图来说明：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220226110746801.png)



## 词法作用域

本节中有几个知识点：

1.  函数只有在被调用的时候才寻找变量：

``` r
##定义函数
fun1 <- function(x){
  c(a,x,b)
}
#现在并没有定义a和b,但是创建函数时不会报错

fun1(1)
>> Error in fun1(1): 找不到对象'a'
##由于调用函数时找不到相应的变量所以报错

a <- b <- 1
fun1(1)
>> [1] 1 1 1
```

1.  当函数被调用时，先会在函数内部搜索变量，如果在内部找不到相应的变量，就会在自己被定义的地方(所在的域或者环境)而不是被调用的地方搜索相应的变量—-**词法作用域**

``` r
f1 <- function(x){
  x + p 
}

g1 <- function(x){
  p <- 1 
  f1(x)
}

g1(0)
>> Error in f1(x): 找不到对象'p'
##在g1内部调用f1，f1先在其内部找p，找不到；接着f1到其被定义的域/环境中找p，也找不到(因为p是在g1内部被定义的)，所以会报错

p <- 1
g1(0)
>> [1] 1
##在g1内部调用f1，f1先在其内部找p，找不到；接着f1到其被定义的域/环境中找p,这时就可以找到了
```

接下来看一个有意思的例子：

``` r
f1 <- function(x){
  p <- 1
  q <- 2
  cat(sprintf("1. [f1] p: %d, q: %d\n",p,q))
  f2 <- function(x){
    p <- 3
    cat(sprintf("2. [f2] p: %d, q: %d\n",p,q))
    c(x=x, p=p, q=q)
  }
  cat(sprintf("3. [f1] p: %d, q: %d",p,q))
  f2(x)
}

f1(0)
>> 1. [f1] p: 1, q: 2
>> 3. [f1] p: 1, q: 22. [f2] p: 3, q: 2
>> x p q 
>> 0 3 2

##注意运行的顺序和展示的值
```

## 环境的工作方式

环境是一组名称组成的对象，每个名称(变量)都指向一个对象，并且每个环境(除了空环境)都要一个父环境，当我们寻找某个名称的时候会沿着“环境链”进行搜索

我们可以通过`new.env`来创建新环境，通过$和\[\]来在新环境中创建变量(和列表操作类似)

``` r
e1 <- new.env()
e1
>> <environment: 0x00000209ed933aa0>
##环境是用内存地址来表示

##创建变量
e1$x <- 1
e1[["y"]] <- 2
```

但是在访问变量的时候，环境不能像列表那样通过索引来提取元素：

``` r
e1[[1]]
>> Error in e1[[1]]: 取子集环境时的参数不对
e1[1:2]
>> Error in e1[1:2]: 类别为'environment'的对象不可以取子集
```

可以通过使用变量的名称或者专门的函数(exist/get/ls)来访问环境中的变量：

``` r
e1[["x"]]
>> [1] 1

exists("x",e1) ##exist判断某个变量是否在环境中
>> [1] TRUE
get("x",e1)##get从环境中获取相应的对象
>> [1] 1
ls(e1)##ls查看环境中所有变量
>> [1] "x" "y"
```

环境还有两个重要的特征：

-   环境有父环境
-   环境有引用语义

### 链接环境

环境有父环境，当我们寻找一个变量的时候，在当前环境中找不到就会去父环境中寻找
在创建环境时可以指定其父环境：

``` r
e2 <- new.env(parent = e1) ##创建新环境e2，其父环境是e1
e1
>> <environment: 0x00000209ed933aa0>
e2
>> <environment: 0x00000209ece0a5b0>

##可以使用parent.env查看环境的父环境
parent.env(e2)
>> <environment: 0x00000209ed933aa0>
##可以看到和e1的内存地址一样
```

需要注意的是：只有环境访问函数(exists/get)会沿着环境链寻找变量，操作符($/\[\])不会.

``` r
e2$y <- 2
ls(e2)
>> [1] "y"
e2[["y"]]
>> [1] 2
e2$y
>> [1] 2

exists("y",e2)
>> [1] TRUE
exists("x",e2)##x在e2中并没有，但是exists会到e2的父环境e1中找到x
>> [1] TRUE
get("x",e2)
>> [1] 1

e2[["x"]]
>> NULL
e2$x
>> NULL

##我们也可以让这些函数不去在父环境中寻找，加上参数inherits = FALSE
exists("x",e2,inherits = FALSE)
>> [1] FALSE
get("x",e2,inherits = FALSE)
>> Error in get("x", e2, inherits = FALSE): 找不到对象'x'
```

当每次开启一个新的R会话时工作环境都会时R的一个内置环境，即全局环境(R\_GlobalEnv)：

``` r
environment()##使用environment()查看当前工作环境
>> <environment: R_GlobalEnv>

##还可以通过其他方式访问全局环境
globalenv()
>> <environment: R_GlobalEnv>
.GlobalEnv
>> <environment: R_GlobalEnv>
```

那么全局环境的父环境是什么？全局环境的父环境的父环境是什么？最终有没有尽头呢？
我们可以通过下面的函数来探索一下：

``` r
parents <- function(env){
  while(TRUE){
    name <- environmentName(env)
    txt <- if (nzchar(name)){
      name
    }else{
      format(env)
    }
    cat(txt,"\n")
    env <- parent.env(env)
  }
}

parents(globalenv())
>> R_GlobalEnv 
>> package:rtracklayer 
>> package:GenomicRanges 
>> package:GenomeInfoDb 
>> package:IRanges 
>> package:S4Vectors 
>> package:BiocGenerics 
>> package:parallel 
>> package:stats4 
>> package:dplyr 
>> package:rmarkdown 
>> tools:rstudio 
>> package:stats 
>> package:graphics 
>> package:grDevices 
>> package:utils 
>> package:datasets 
>> package:methods 
>> Autoloads 
>> base 
>> R_EmptyEnv
>> Error in parent.env(env): 空环境没有父母环境
```

可以看到这个环境链条从空环境起始经过多个拓展包的环境最后终止于空环境，并且空环境没有父环境，这个结果和search(搜索路径)的结果相似：

``` r
search()
>>  [1] ".GlobalEnv"            "package:rtracklayer"   "package:GenomicRanges" "package:GenomeInfoDb" 
>>  [5] "package:IRanges"       "package:S4Vectors"     "package:BiocGenerics"  "package:parallel"     
>>  [9] "package:stats4"        "package:dplyr"         "package:rmarkdown"     "tools:rstudio"        
>> [13] "package:stats"         "package:graphics"      "package:grDevices"     "package:utils"        
>> [17] "package:datasets"      "package:methods"       "Autoloads"             "package:base"
```

需要注意的是：我们每加载一个包，该包的环境就会加到全局路径的后面，所以如果需要调用两个包中同名函数，会优先选取后加载的包的函数(后加载的包mask了前面包的同名函数)：

``` r
library(dplyr)
search()
>>  [1] ".GlobalEnv"            "package:rtracklayer"   "package:GenomicRanges" "package:GenomeInfoDb" 
>>  [5] "package:IRanges"       "package:S4Vectors"     "package:BiocGenerics"  "package:parallel"     
>>  [9] "package:stats4"        "package:dplyr"         "package:rmarkdown"     "tools:rstudio"        
>> [13] "package:stats"         "package:graphics"      "package:grDevices"     "package:utils"        
>> [17] "package:datasets"      "package:methods"       "Autoloads"             "package:base"

library(data.table)
>> Warning: package 'data.table' was built under R version 3.6.2
search()
>>  [1] ".GlobalEnv"            "package:data.table"    "package:rtracklayer"   "package:GenomicRanges"
>>  [5] "package:GenomeInfoDb"  "package:IRanges"       "package:S4Vectors"     "package:BiocGenerics" 
>>  [9] "package:parallel"      "package:stats4"        "package:dplyr"         "package:rmarkdown"    
>> [13] "tools:rstudio"         "package:stats"         "package:graphics"      "package:grDevices"    
>> [17] "package:utils"         "package:datasets"      "package:methods"       "Autoloads"            
>> [21] "package:base"

##可以看到data.table到前面去了
```

### 引用语义

引用语义指的是：修改环境并不会复制环境的副本(也就是没有复制修改机制)：

``` r
e3 <- e1
e1$x
>> [1] 1
e3$x
>> [1] 1

e3$x <- 2
e1$x
>> [1] 2
##因此e3和e1指向同一个对象，改变e3中的变量值，e1中的变量值也会改变
```

### 与函数相关的环境

有3个与函数及其运行过程相关的环境：

-   执行环境(executing environment):
    每次函数执行时，R都会新建一个环境来管理函数的执行过程，所以函数的参数和在函数内部创建的变量都是执行环境中的变量
-   封闭环境(enclosing
    environment)：定义函数的环境，也是执行环境的父环境，所以在函数执行的时候，没有在执行环境中找到的变量就会到其父环境，也就是封闭环境中寻找(词法作用域)，可以使用environment()来获取函数的封闭环境
-   调用环境(calling environment):
    调用函数的环境，可以使用parent.frame()来获取调用环境
