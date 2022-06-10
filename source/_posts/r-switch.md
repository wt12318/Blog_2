---
title: 【R】switch函数用法
date: 2021-01-04 17:14:18
tags: 编程
index_img: img/switch.png
categories:
  - R
---

`switch`函数的基本用法

<!-- more -->

`switch`的基本用法为`switch (expression, list)`    
第一个参数是表达式，第二个参数是列表；基于表达式的值返回列表中相应元素(按照名称)的值  
来看一些具体的例子：  
如果expression的结果是整数，那么会按照位置返回值    
如果expression的结果是字符，那么会按照其后参数构成的列表中元素的名称返回相应的值

``` r
switch(1,x=1,y=2)
## [1] 1

centre <- function(x, type) {
  switch(type,
         mean = mean(x),
         median = median(x),
         trimmed = mean(x, trim = .1))
}

centre(c(1,2,3),"mean")
## [1] 2
```

对于数字的情况，如果输入是负数或者超出范围，不会报错，返回的是NULL(`print`后才可见)

对于字符，如果输入没有匹配的名称也会返回NULL，这个时候可以在list的最后加上没有名称的值捕获其他的输入：

``` r
for(i in c(-1:3, 9))  print(switch(i, 1, 2 , 3, 4))
## NULL
## NULL
## [1] 1
## [1] 2
## [1] 3
## NULL

print(switch("a",c=1,b=2))
## NULL
print(switch("a",c=1,b=2,3))
## [1] 3
```
