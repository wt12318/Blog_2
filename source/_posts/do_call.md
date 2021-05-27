---
title: do.call 用法    
date: 2021-01-21 10:00:00    
index_img: img/do_call.png    
---



do.call函数的用法

<!-- more -->

do.call从名称或者函数构建和执行函数调用，并且使用list来存放函数的参数，基本用法为：

> do.call(what, args, quote = FALSE, envir = parent.frame())

-   what 是一个函数或者表示函数名称的字符

-   args 是函数调用的参数，列表形式，列表的names属性就是参数名

-   quote 逻辑值，表示是否对参数进行捕获

-   envir 执行调用的环境

如果quote是FALSE，那么参数会被计算(执行的环境是调用环境，而不是envir指定的环境)；如果quote是TRUE，那么每个参数会被捕获(也就是在调用被构建的时候不计算参数,从而使我们可以通过envir调整计算的环境)

``` r
x1 <- c(1:10,NA)

do.call(sum,list(x1,na.rm=TRUE))
>> [1] 55
```

do.call函数的文档中有一个从哪里寻找对象的例子：

``` r
A <- 2
f <- function(x) print(x^2)
env <- new.env()

assign("A", 10, envir = env)
env$A
>> [1] 10

assign("f", f, envir = env)
env$f
>> function(x) print(x^2)

###改变了全局环境中的f，但是env中的f没有变
f <- function(x) print(x)
f
>> function(x) print(x)
env$f
>> function(x) print(x^2)

f(A)     
>> [1] 2

do.call("f", list(A))##在调用环境(全局环境)中寻找f和A
>> [1] 2

do.call("f", list(A), envir = env)##在env中找到了f，但是由于参数没有被捕获，所以参数在调用环境(全局环境)中计算  
>> [1] 4

do.call(f, list(A), envir = env)##在全局中寻找f，但是由于参数没有被捕获，所以参数在调用环境(全局环境)中计算    
>> [1] 2

do.call("f", list(quote(A)), envir = env)##由于A被捕获所以在env中寻找f和A
>> [1] 100
##为什么和下面的不一样？
do.call("f", list(A),quote = TRUE, envir = env)
>> [1] 4

do.call(f, list(quote(A)), envir = env)#由于A被捕获所以在env中寻找A，但是f并不是语言对象(symbol)，所以在env中找不到   
>> [1] 10

do.call("f", list(as.name("A")), envir = env)##这里面as.name和quote的作用一样,都是获取symbol对象
>> [1] 100
```
