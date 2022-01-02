---
title: dplyr行式操作    
date: 2021-07-10 10:00:00    
index_img: img/dplyr.png
categories:
  - R
---



dplyr 按行操作，主要是 rowwise 函数的用法

<!-- more -->

## 创建

按行操作需要一种特殊类型的分组，每一组中只有一行，通过 `rowwise` 实现：

``` r
df <- tibble(x = 1:2, y = 3:4, z = 5:6)
df %>% rowwise()
>> # A tibble: 2 x 3
>> # Rowwise: 
>>       x     y     z
>>   <int> <int> <int>
>> 1     1     3     5
>> 2     2     4     6
```

像 `group_by` 一样，`rowwise` 并不会进行计算，仅仅会改变其他动词的行为：

``` r
df %>% mutate(m = mean(c(x, y, z)))
>> # A tibble: 2 x 4
>>       x     y     z     m
>>   <int> <int> <int> <dbl>
>> 1     1     3     5   3.5
>> 2     2     4     6   3.5
df %>% rowwise() %>% mutate(m = mean(c(x, y, z)))
>> # A tibble: 2 x 4
>> # Rowwise: 
>>       x     y     z     m
>>   <int> <int> <int> <dbl>
>> 1     1     3     5     3
>> 2     2     4     6     4
```

可以看到不使用 `rowwise` 之前计算的是所有行的均值（6个值的均值，实际上是一个值），而使用 `rowwsise` 之后计算的是每行的均值。

`rowwise` 也可以加上列名作为参数，表示在输出结果中保留这一列：

``` r
df <- tibble(name = c("Mara", "Hadley"), x = 1:2, y = 3:4, z = 5:6)

##不加列名输出的就是最终结果
df %>% 
  rowwise() %>% 
  summarise(m = mean(c(x, y, z)))
>> `summarise()` has ungrouped output. You can override using the `.groups` argument.
>> # A tibble: 2 x 1
>>       m
>>   <dbl>
>> 1     3
>> 2     4

##加上列名最后会保存该列
df %>% 
  rowwise(name) %>% 
  summarise(m = mean(c(x, y, z)))
>> `summarise()` has grouped output by 'name'. You can override using the `.groups` argument.
>> # A tibble: 2 x 2
>> # Groups:   name [2]
>>   name       m
>>   <chr>  <dbl>
>> 1 Mara       3
>> 2 Hadley     4
```

## 按行进行汇总统计

`rowwise` 可以和 `summarise` 一起使用来方便的进行汇总统计：

``` r
##创建数据
df <- tibble(id = 1:6, w = 10:15, x = 20:25, y = 30:35, z = 40:45)
df
>> # A tibble: 6 x 5
>>      id     w     x     y     z
>>   <int> <int> <int> <int> <int>
>> 1     1    10    20    30    40
>> 2     2    11    21    31    41
>> 3     3    12    22    32    42
>> 4     4    13    23    33    43
>> 5     5    14    24    34    44
>> 6     6    15    25    35    45

rf <- df %>% rowwise(id)

##mutate来添加一列汇总
rf %>% mutate(total = sum(c(w, x, y, z)))
>> # A tibble: 6 x 6
>> # Rowwise:  id
>>      id     w     x     y     z total
>>   <int> <int> <int> <int> <int> <int>
>> 1     1    10    20    30    40   100
>> 2     2    11    21    31    41   104
>> 3     3    12    22    32    42   108
>> 4     4    13    23    33    43   112
>> 5     5    14    24    34    44   116
>> 6     6    15    25    35    45   120
##或者summarise只保留汇总
rf %>% summarise(total = sum(c(w, x, y, z)))
>> `summarise()` has grouped output by 'id'. You can override using the `.groups` argument.
>> # A tibble: 6 x 2
>> # Groups:   id [6]
>>      id total
>>   <int> <int>
>> 1     1   100
>> 2     2   104
>> 3     3   108
>> 4     4   112
>> 5     5   116
>> 6     6   120
```

当有很多变量的时候，可以结合 `c_across` 来进行 `tidy selection`：

``` r
rf %>% mutate(total = sum(c_across(w:z)))
>> # A tibble: 6 x 6
>> # Rowwise:  id
>>      id     w     x     y     z total
>>   <int> <int> <int> <int> <int> <int>
>> 1     1    10    20    30    40   100
>> 2     2    11    21    31    41   104
>> 3     3    12    22    32    42   108
>> 4     4    13    23    33    43   112
>> 5     5    14    24    34    44   116
>> 6     6    15    25    35    45   120
rf %>% mutate(total = sum(c_across(where(is.numeric))))
>> # A tibble: 6 x 6
>> # Rowwise:  id
>>      id     w     x     y     z total
>>   <int> <int> <int> <int> <int> <int>
>> 1     1    10    20    30    40   100
>> 2     2    11    21    31    41   104
>> 3     3    12    22    32    42   108
>> 4     4    13    23    33    43   112
>> 5     5    14    24    34    44   116
>> 6     6    15    25    35    45   120
```

在这里可以结合行式操作和列式操作计算每个元素占行总和的比例：

``` r
rf %>% 
  mutate(total = sum(c_across(w:z))) %>% 
  ungroup() %>% 
  mutate(across(w:z, ~ . / total))
>> # A tibble: 6 x 6
>>      id     w     x     y     z total
>>   <int> <dbl> <dbl> <dbl> <dbl> <int>
>> 1     1 0.1   0.2   0.3   0.4     100
>> 2     2 0.106 0.202 0.298 0.394   104
>> 3     3 0.111 0.204 0.296 0.389   108
>> 4     4 0.116 0.205 0.295 0.384   112
>> 5     5 0.121 0.207 0.293 0.379   116
>> 6     6 0.125 0.208 0.292 0.375   120
```

## List-columns

当列中有列表元素时，行式操作就比较方便。现在假设有下列的数据，想要计算一列中每行元素的长度：

``` r
df <- tibble(
  x = list(1, 2:3, 4:6)
)
df
>> # A tibble: 3 x 1
>>   x        
>>   <list>   
>> 1 <dbl [1]>
>> 2 <int [2]>
>> 3 <int [3]>

df %>% mutate(l = length(x))
>> # A tibble: 3 x 2
>>   x             l
>>   <list>    <int>
>> 1 <dbl [1]>     3
>> 2 <int [2]>     3
>> 3 <int [3]>     3
```

可以看到当使用 `length` 时返回的是列的长度，而不是列中每个元素的长度；这个问题可以使用 R 内置的 `lengths` 函数或者直接用循环来解决：

``` r
df %>% mutate(l = lengths(x))
>> # A tibble: 3 x 2
>>   x             l
>>   <list>    <int>
>> 1 <dbl [1]>     1
>> 2 <int [2]>     2
>> 3 <int [3]>     3

df %>% mutate(l = sapply(x, length))
>> # A tibble: 3 x 2
>>   x             l
>>   <list>    <int>
>> 1 <dbl [1]>     1
>> 2 <int [2]>     2
>> 3 <int [3]>     3

##或者使用purrr
df %>% mutate(l = purrr::map_int(x, length))
>> # A tibble: 3 x 2
>>   x             l
>>   <list>    <int>
>> 1 <dbl [1]>     1
>> 2 <int [2]>     2
>> 3 <int [3]>     3
```

另一个方法就是使用 `rowwise` 操作：

``` r
df %>% 
  rowwise() %>% 
  mutate(l = length(x))
>> # A tibble: 3 x 2
>> # Rowwise: 
>>   x             l
>>   <list>    <int>
>> 1 <dbl [1]>     1
>> 2 <int [2]>     2
>> 3 <int [3]>     3
```

那么这里 `mutate` 和 `rowwise` 之后再 `mutate` 有什么区别呢？主要不同就是：`mutate` 对列切片的方式是一个中括号 `[`，而 `rowwise mutate` 则是通过两个中括号 `[[` 来切片，所以当列的元素是列表的时候，我们通过 `[` 只能获取整个列表的内容，而 `[[` 能获取列表的元素：

``` r
df$x
>> [[1]]
>> [1] 1
>> 
>> [[2]]
>> [1] 2 3
>> 
>> [[3]]
>> [1] 4 5 6

# mutate
out1 <- integer(3)
for (i in 1:3) {
  out1[[i]] <- length(df$x[i])
}
out1
>> [1] 1 1 1

# rowwise mutate
out2 <- integer(3)
for (i in 1:3) {
  out2[[i]] <- length(df$x[[i]])
}
out2
>> [1] 1 2 3
```

## 建模

`rowwise` 的按行操作的特征特别适合建模以及存放模型及数据：

``` r
##nest_by 和 group_by 类似，不过在视觉上改变了数据框的结构，返回的是rowwise的数据框
by_cyl <- mtcars %>% nest_by(cyl)
by_cyl
>> # A tibble: 3 x 2
>> # Rowwise:  cyl
>>     cyl                data
>>   <dbl> <list<tibble[,10]>>
>> 1     4           [11 x 10]
>> 2     6            [7 x 10]
>> 3     8           [14 x 10]
```

将数据按行存放之后就可以对每行进行建模及存放数据：

``` r
##建立线性回归模型
mods <- by_cyl %>% mutate(mod = list(lm(mpg ~ wt, data = data)))

##添加预测数据
mods <- mods %>% mutate(pred = list(predict(mod, data)))
mods
>> # A tibble: 3 x 4
>> # Rowwise:  cyl
>>     cyl                data mod    pred      
>>   <dbl> <list<tibble[,10]>> <list> <list>    
>> 1     4           [11 x 10] <lm>   <dbl [11]>
>> 2     6            [7 x 10] <lm>   <dbl [7]> 
>> 3     8           [14 x 10] <lm>   <dbl [14]>
```

这个用 `list` 的原因是：`predict` 返回的是列表，上面说过对于 `rowwise mutate` 使用 `[[` 来切片，因此返回的数据的长度是 11，而 `mutate` 返回的长度必须是 1。

``` r
mods %>% mutate(pred = predict(mod, data))
>> Error: Problem with `mutate()` column `pred`.
>> i `pred = predict(mod, data)`.
>> i `pred` must be size 1, not 11.
>> i Did you mean: `pred = list(predict(mod, data))` ?
>> i The error occurred in row 1.
```

接着可以自由添加一些模型的汇总信息：

``` r
mods %>% summarise(rmse = sqrt(mean((pred - data$mpg) ^ 2)))
>> `summarise()` has grouped output by 'cyl'. You can override using the `.groups` argument.
>> # A tibble: 3 x 2
>> # Groups:   cyl [3]
>>     cyl  rmse
>>   <dbl> <dbl>
>> 1     4 3.01 
>> 2     6 0.985
>> 3     8 1.87

mods %>% summarise(rsq = summary(mod)$r.squared)
>> `summarise()` has grouped output by 'cyl'. You can override using the `.groups` argument.
>> # A tibble: 3 x 2
>> # Groups:   cyl [3]
>>     cyl   rsq
>>   <dbl> <dbl>
>> 1     4 0.509
>> 2     6 0.465
>> 3     8 0.423

mods %>% summarise(broom::glance(mod))
>> `summarise()` has grouped output by 'cyl'. You can override using the `.groups` argument.
>> # A tibble: 3 x 12
>> # Groups:   cyl [3]
>>     cyl r.squared adj.r.squared sigma statistic p.value
>>   <dbl>     <dbl>         <dbl> <dbl>     <dbl>   <dbl>
>> 1     4     0.509         0.454  3.33      9.32  0.0137
>> 2     6     0.465         0.357  1.17      4.34  0.0918
>> 3     8     0.423         0.375  2.02      8.80  0.0118
>> # ... with 6 more variables: df <int>, logLik <dbl>,
>> #   AIC <dbl>, BIC <dbl>, deviance <dbl>,
>> #   df.residual <int>

##获取模型参数
mods %>% summarise(broom::tidy(mod))
>> `summarise()` has grouped output by 'cyl'. You can override using the `.groups` argument.
>> # A tibble: 6 x 6
>> # Groups:   cyl [3]
>>     cyl term       estimate std.error statistic   p.value
>>   <dbl> <chr>         <dbl>     <dbl>     <dbl>     <dbl>
>> 1     4 (Intercep~    39.6      4.35       9.10   7.77e-6
>> 2     4 wt            -5.65     1.85      -3.05   1.37e-2
>> 3     6 (Intercep~    28.4      4.18       6.79   1.05e-3
>> 4     6 wt            -2.78     1.33      -2.08   9.18e-2
>> 5     8 (Intercep~    23.9      3.01       7.94   4.05e-6
>> 6     8 wt            -2.19     0.739     -2.97   1.18e-2
```

## 重复调用函数

和前面的思路一样，如果在各列中存放函数的参数，就可以方便的重复调用函数：

``` r
df <- tribble(
  ~ n, ~ min, ~ max,
    1,     0,     1,
    2,    10,   100,
    3,   100,  1000,
)

#从不同参数的均匀分布中产生随机数
df %>% 
  rowwise() %>% 
  mutate(data = list(runif(n, min, max)))
>> # A tibble: 3 x 4
>> # Rowwise: 
>>       n   min   max data     
>>   <dbl> <dbl> <dbl> <list>   
>> 1     1     0     1 <dbl [1]>
>> 2     2    10   100 <dbl [2]>
>> 3     3   100  1000 <dbl [3]>

##扩充不同的组合
df <- expand.grid(mean = c(-1, 0, 1), sd = c(1, 10, 100))

df %>% 
  rowwise() %>% 
  mutate(data = list(rnorm(10, mean, sd)))
>> # A tibble: 9 x 3
>> # Rowwise: 
>>    mean    sd data      
>>   <dbl> <dbl> <list>    
>> 1    -1     1 <dbl [10]>
>> 2     0     1 <dbl [10]>
>> 3     1     1 <dbl [10]>
>> 4    -1    10 <dbl [10]>
>> 5     0    10 <dbl [10]>
>> 6     1    10 <dbl [10]>
>> 7    -1   100 <dbl [10]>
>> 8     0   100 <dbl [10]>
>> 9     1   100 <dbl [10]>
```

当我们想要变化的不是函数的参数，而是调用不同的函数，可以将 `rowwise` 和 `do.call` 结合：

``` r
df <- tribble(
   ~rng,     ~params,
   "runif",  list(n = 10), 
   "rnorm",  list(n = 20),
   "rpois",  list(n = 10, lambda = 5),
) %>%
  rowwise()

df %>% 
  mutate(data = list(do.call(rng, params)))
>> # A tibble: 3 x 3
>> # Rowwise: 
>>   rng   params           data      
>>   <chr> <list>           <list>    
>> 1 runif <named list [1]> <dbl [10]>
>> 2 rnorm <named list [1]> <dbl [20]>
>> 3 rpois <named list [2]> <int [10]>
```
