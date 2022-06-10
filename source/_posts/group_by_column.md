---
title: 【R】按列分组计算 
date: 2021-01-05 10:00:00    
index_img: img/OIP.jpg
categories:
  - R
---

R按列进行分组计算
<!-- more -->
现在有一个数据框`dt`:

``` r
dt <- data.frame(
  x = c(1,2,3),
  y = c(2,3,4),
  z = c("a","a","b")
)
dt
##   x y z
## 1 1 2 a
## 2 2 3 a
## 3 3 4 b
```

想要依据`z`列分组并计算`x`,`y`列的均值：

``` r
library(dplyr)
dt %>% 
  group_by(z) %>% 
  summarise(mean_x=mean(x),mean_y=mean(y))
## # A tibble: 2 x 3
##   z     mean_x mean_y
##   <fct>  <dbl>  <dbl>
## 1 a        1.5    2.5
## 2 b        3      4
```

但是如果传入的是字符就会出现我们不想要的结果：

``` r
dt %>% 
  group_by("z") %>% 
  summarise(mean_x=mean(x),mean_y=mean(y))
## `summarise()` ungrouping output (override with `.groups` argument)
## # A tibble: 1 x 3
##   `"z"` mean_x mean_y
##   <chr>  <dbl>  <dbl>
## 1 z          2      3
```

这里实际上是创建了一个新的变量`z`并且他的值也是`z`，然后计算了x和y列的均值   
这种情况下可以使用`group_by_at`来选择变量

``` r
dt %>% 
  group_by_at("z") %>% 
  summarise(mean_x=mean(x),mean_y=mean(y))
## `summarise()` ungrouping output (override with `.groups` argument)
## # A tibble: 2 x 3
##   z     mean_x mean_y
##   <fct>  <dbl>  <dbl>
## 1 a        1.5    2.5
## 2 b        3      4
```

在`dplyr`的最新版本中(1.0+)有新的函数`across`也可以做同样的事:

``` r
dt %>% 
  group_by(across("z")) %>% 
  summarise(mean_x=mean(x),mean_y=mean(y))
## `summarise()` ungrouping output (override with `.groups` argument)
## # A tibble: 2 x 3
##   z     mean_x mean_y
##   <fct>  <dbl>  <dbl>
## 1 a        1.5    2.5
## 2 b        3      4
```

另外我们也可以使用`eval`加`parse`来将字符解析为对象：

``` r
dt %>% 
  group_by(eval(parse(text = "z"))) %>% 
  summarise(mean_x=mean(x),mean_y=mean(y))
## `summarise()` ungrouping output (override with `.groups` argument)
## # A tibble: 2 x 3
##   `eval(parse(text = "z"))` mean_x mean_y
##   <fct>                      <dbl>  <dbl>
## 1 a                            1.5    2.5
## 2 b                            3      4
```
