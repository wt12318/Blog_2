---
title: dplyr基础
date: 2021-01-20 10:00:00    
index_img: img/dplyr.png
---





dplyr基础知识

<!-- more -->

参考： [dplyrbase](https://github.com/tidyverse/dplyr/blob/master/vignettes/base.Rmd) [twotable](https://github.com/tidyverse/dplyr/blob/master/vignettes/two-table.Rmd)

这篇文章主要比较dplyr函数和base R的区别

## Overview

1.  dplyr动词输入和输出都是数据框，而base R大部分是单独的向量    
2.  dplyr依赖非标准计算，所以不需要$来选择变量(列)
3.  dplyr使用一系列具有单个目的的动词，而在baseR中通常使用\[\]
4.  dplyr的动词通常可以通过管道(%\>%)连在一起，而baseR中常常需要将中间结果保存为变量
5.  所有的dplyr动词都可以处理分组数据并且和处理整个数据框类似，但是在baseR中可能每个组的处理都有着不同的形式

## One table verbs

| dplyr                         | base                                             |
|-------------------------------|--------------------------------------------------|
| `arrange(df, x)`              | `df[order(x), , drop = FALSE]`                   |
| `distinct(df, x)`             | `df[!duplicated(x), , drop = FALSE]`, `unique()` |
| `filter(df, x)`               | `df[which(x), , drop = FALSE]`, `subset()`       |
| `mutate(df, z = x + y)`       | `df$z <- df$x + df$y`, `transform()`             |
| `pull(df, 1)`                 | `df[[1]]`                                        |
| `pull(df, x)`                 | `df$x`                                           |
| `rename(df, y = x)`           | `names(df)[names(df) == "x"] <- "y"`             |
| `relocate(df, y)`             | `df[union("y", names(df))]`                      |
| `select(df, x, y)`            | `df[c("x", "y")]`, `subset()`                    |
| `select(df, starts_with("x")` | `df[grepl(names(df), "^x")]`                     |
| `summarise(df, mean(x))`      | `mean(df$x)`, `tapply()`, `aggregate()`, `by()`  |
| `slice(df, c(1, 2, 5))`       | `df[c(1, 2, 5), , drop = FALSE]`                 |

首先载入示例数据：

``` r
library(dplyr)
>> Warning: package 'dplyr' was built under R version 3.6.3
>> 
>> Attaching package: 'dplyr'
>> The following objects are masked from 'package:stats':
>> 
>>     filter, lag
>> The following objects are masked from 'package:base':
>> 
>>     intersect, setdiff, setequal, union

mtcars <- as_tibble(mtcars)
iris <- as_tibble(iris)
```

### `arrange()` 通过变量来组织行

`dplyr::arrange()`通过一列或多列的值来对数据框的行进行排序：

``` r
mtcars %>% arrange(cyl,disp)
>> # A tibble: 32 x 11
>>      mpg   cyl  disp    hp  drat    wt  qsec    vs    am  gear  carb
>>    <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl>
>>  1  33.9     4  71.1    65  4.22  1.84  19.9     1     1     4     1
>>  2  30.4     4  75.7    52  4.93  1.62  18.5     1     1     4     2
>>  3  32.4     4  78.7    66  4.08  2.2   19.5     1     1     4     1
>>  4  27.3     4  79      66  4.08  1.94  18.9     1     1     4     1
>>  5  30.4     4  95.1   113  3.77  1.51  16.9     1     1     5     2
>>  6  22.8     4 108      93  3.85  2.32  18.6     1     1     4     1
>>  7  21.5     4 120.     97  3.7   2.46  20.0     1     0     3     1
>>  8  26       4 120.     91  4.43  2.14  16.7     0     1     5     2
>>  9  21.4     4 121     109  4.11  2.78  18.6     1     1     4     2
>> 10  22.8     4 141.     95  3.92  3.15  22.9     1     0     4     2
>> # ... with 22 more rows
```

`desc()`辅助函数可以进行降序排序：

``` r
mtcars %>% arrange(desc(cyl),desc(disp))
>> # A tibble: 32 x 11
>>      mpg   cyl  disp    hp  drat    wt  qsec    vs    am  gear  carb
>>    <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl>
>>  1  10.4     8   472   205  2.93  5.25  18.0     0     0     3     4
>>  2  10.4     8   460   215  3     5.42  17.8     0     0     3     4
>>  3  14.7     8   440   230  3.23  5.34  17.4     0     0     3     4
>>  4  19.2     8   400   175  3.08  3.84  17.0     0     0     3     2
>>  5  18.7     8   360   175  3.15  3.44  17.0     0     0     3     2
>>  6  14.3     8   360   245  3.21  3.57  15.8     0     0     3     4
>>  7  15.8     8   351   264  4.22  3.17  14.5     0     1     5     4
>>  8  13.3     8   350   245  3.73  3.84  15.4     0     0     3     4
>>  9  15.5     8   318   150  2.76  3.52  16.9     0     0     3     2
>> 10  15.2     8   304   150  3.15  3.44  17.3     0     0     3     2
>> # ... with 22 more rows
```

在base R中可以使用\[+order函数对行进行排序：

``` r
mtcars[order(mtcars$cyl,mtcars$disp),,drop= FALSE]
>> # A tibble: 32 x 11
>>      mpg   cyl  disp    hp  drat    wt  qsec    vs    am  gear  carb
>>    <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl>
>>  1  33.9     4  71.1    65  4.22  1.84  19.9     1     1     4     1
>>  2  30.4     4  75.7    52  4.93  1.62  18.5     1     1     4     2
>>  3  32.4     4  78.7    66  4.08  2.2   19.5     1     1     4     1
>>  4  27.3     4  79      66  4.08  1.94  18.9     1     1     4     1
>>  5  30.4     4  95.1   113  3.77  1.51  16.9     1     1     5     2
>>  6  22.8     4 108      93  3.85  2.32  18.6     1     1     4     1
>>  7  21.5     4 120.     97  3.7   2.46  20.0     1     0     3     1
>>  8  26       4 120.     91  4.43  2.14  16.7     0     1     5     2
>>  9  21.4     4 121     109  4.11  2.78  18.6     1     1     4     2
>> 10  22.8     4 141.     95  3.92  3.15  22.9     1     0     4     2
>> # ... with 22 more rows
```

记得加上drop=
FALSE，不然如果输入是只有一列的数据框，输出就是一个向量而不是数据框了：

``` r
dt <- data.frame(
  x = c(1,2,3)
)

dt[order(dt$x),]
>> [1] 1 2 3
dt[order(dt$x),,drop=FALSE]
>>   x
>> 1 1
>> 2 2
>> 3 3
```

进行倒序排序，base R有两种选择：

-   对于数值变量可以加上负号-

-   在order函数中指定参数decreasing=TRUE

``` r
mtcars[order(mtcars$cyl, mtcars$disp, decreasing = TRUE), , drop = FALSE]
>> # A tibble: 32 x 11
>>      mpg   cyl  disp    hp  drat    wt  qsec    vs    am  gear  carb
>>    <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl>
>>  1  10.4     8   472   205  2.93  5.25  18.0     0     0     3     4
>>  2  10.4     8   460   215  3     5.42  17.8     0     0     3     4
>>  3  14.7     8   440   230  3.23  5.34  17.4     0     0     3     4
>>  4  19.2     8   400   175  3.08  3.84  17.0     0     0     3     2
>>  5  18.7     8   360   175  3.15  3.44  17.0     0     0     3     2
>>  6  14.3     8   360   245  3.21  3.57  15.8     0     0     3     4
>>  7  15.8     8   351   264  4.22  3.17  14.5     0     1     5     4
>>  8  13.3     8   350   245  3.73  3.84  15.4     0     0     3     4
>>  9  15.5     8   318   150  2.76  3.52  16.9     0     0     3     2
>> 10  15.2     8   304   150  3.15  3.44  17.3     0     0     3     2
>> # ... with 22 more rows

###or
mtcars[order(-mtcars$cyl, -mtcars$disp), , drop = FALSE]
>> # A tibble: 32 x 11
>>      mpg   cyl  disp    hp  drat    wt  qsec    vs    am  gear  carb
>>    <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl>
>>  1  10.4     8   472   205  2.93  5.25  18.0     0     0     3     4
>>  2  10.4     8   460   215  3     5.42  17.8     0     0     3     4
>>  3  14.7     8   440   230  3.23  5.34  17.4     0     0     3     4
>>  4  19.2     8   400   175  3.08  3.84  17.0     0     0     3     2
>>  5  18.7     8   360   175  3.15  3.44  17.0     0     0     3     2
>>  6  14.3     8   360   245  3.21  3.57  15.8     0     0     3     4
>>  7  15.8     8   351   264  4.22  3.17  14.5     0     1     5     4
>>  8  13.3     8   350   245  3.73  3.84  15.4     0     0     3     4
>>  9  15.5     8   318   150  2.76  3.52  16.9     0     0     3     2
>> 10  15.2     8   304   150  3.15  3.44  17.3     0     0     3     2
>> # ... with 22 more rows
```

### `distinct()`:选择唯一的行

`dplyr::distinct()`选择唯一的行:

``` r
df <- tibble(
  x = sample(10, 100, rep = TRUE),
  y = sample(10, 100, rep = TRUE)
)

df %>% distinct(x)
>> # A tibble: 10 x 1
>>        x
>>    <int>
>>  1     1
>>  2    10
>>  3     3
>>  4     6
>>  5     7
>>  6     8
>>  7     9
>>  8     5
>>  9     4
>> 10     2

###使用.keep_all保留其他的列
df %>% distinct(x,.keep_all = TRUE)
>> # A tibble: 10 x 2
>>        x     y
>>    <int> <int>
>>  1     1     6
>>  2    10     8
>>  3     3     3
>>  4     6     5
>>  5     7     9
>>  6     8     5
>>  7     9     3
>>  8     5     7
>>  9     4     4
>> 10     2     3
```

在base R中基于想要选择的列还是全部的数据框也有两种实现方法：

``` r
unique(df["x"])
>> # A tibble: 10 x 1
>>        x
>>    <int>
>>  1     1
>>  2    10
>>  3     3
>>  4     6
>>  5     7
>>  6     8
>>  7     9
>>  8     5
>>  9     4
>> 10     2

df[!duplicated(df$x), , drop = FALSE]
>> # A tibble: 10 x 2
>>        x     y
>>    <int> <int>
>>  1     1     6
>>  2    10     8
>>  3     3     3
>>  4     6     5
>>  5     7     9
>>  6     8     5
>>  7     9     3
>>  8     5     7
>>  9     4     4
>> 10     2     3
```

### `filter()`返回符合条件的行

`dplyr::filter()` 返回表达式是TRUE的行

``` r
starwars %>% filter(species == "Human")
>> # A tibble: 35 x 14
>>    name  height  mass hair_color skin_color eye_color birth_year sex   gender homeworld species films vehicles
>>    <chr>  <int> <dbl> <chr>      <chr>      <chr>          <dbl> <chr> <chr>  <chr>     <chr>   <lis> <list>  
>>  1 Luke~    172    77 blond      fair       blue            19   male  mascu~ Tatooine  Human   <chr~ <chr [2~
>>  2 Dart~    202   136 none       white      yellow          41.9 male  mascu~ Tatooine  Human   <chr~ <chr [0~
>>  3 Leia~    150    49 brown      light      brown           19   fema~ femin~ Alderaan  Human   <chr~ <chr [1~
>>  4 Owen~    178   120 brown, gr~ light      blue            52   male  mascu~ Tatooine  Human   <chr~ <chr [0~
>>  5 Beru~    165    75 brown      light      blue            47   fema~ femin~ Tatooine  Human   <chr~ <chr [0~
>>  6 Bigg~    183    84 black      light      brown           24   male  mascu~ Tatooine  Human   <chr~ <chr [0~
>>  7 Obi-~    182    77 auburn, w~ fair       blue-gray       57   male  mascu~ Stewjon   Human   <chr~ <chr [1~
>>  8 Anak~    188    84 blond      fair       blue            41.9 male  mascu~ Tatooine  Human   <chr~ <chr [2~
>>  9 Wilh~    180    NA auburn, g~ fair       blue            64   male  mascu~ Eriadu    Human   <chr~ <chr [0~
>> 10 Han ~    180    80 brown      fair       brown           29   male  mascu~ Corellia  Human   <chr~ <chr [0~
>> # ... with 25 more rows, and 1 more variable: starships <list>

starwars %>% filter(mass > 1000)
>> # A tibble: 1 x 14
>>   name  height  mass hair_color skin_color eye_color birth_year sex   gender homeworld species films vehicles
>>   <chr>  <int> <dbl> <chr>      <chr>      <chr>          <dbl> <chr> <chr>  <chr>     <chr>   <lis> <list>  
>> 1 Jabb~    175  1358 <NA>       green-tan~ orange           600 herm~ mascu~ Nal Hutta Hutt    <chr~ <chr [0~
>> # ... with 1 more variable: starships <list>

starwars %>% filter(hair_color == "none" & eye_color == "black")
>> # A tibble: 9 x 14
>>   name  height  mass hair_color skin_color eye_color birth_year sex   gender homeworld species films vehicles
>>   <chr>  <int> <dbl> <chr>      <chr>      <chr>          <dbl> <chr> <chr>  <chr>     <chr>   <lis> <list>  
>> 1 Nien~    160    68 none       grey       black             NA male  mascu~ Sullust   Sullus~ <chr~ <chr [0~
>> 2 Gasg~    122    NA none       white, bl~ black             NA male  mascu~ Troiken   Xexto   <chr~ <chr [0~
>> 3 Kit ~    196    87 none       green      black             NA male  mascu~ Glee Ans~ Nautol~ <chr~ <chr [0~
>> 4 Plo ~    188    80 none       orange     black             22 male  mascu~ Dorin     Kel Dor <chr~ <chr [0~
>> 5 Lama~    229    88 none       grey       black             NA male  mascu~ Kamino    Kamino~ <chr~ <chr [0~
>> 6 Taun~    213    NA none       grey       black             NA fema~ femin~ Kamino    Kamino~ <chr~ <chr [0~
>> 7 Shaa~    178    57 none       red, blue~ black             NA fema~ femin~ Shili     Togruta <chr~ <chr [0~
>> 8 Tion~    206    80 none       grey       black             NA male  mascu~ Utapau    Pau'an  <chr~ <chr [0~
>> 9 BB8       NA    NA none       none       black             NA none  mascu~ <NA>      Droid   <chr~ <chr [0~
>> # ... with 1 more variable: starships <list>
```

在baseR中有相似功能的函数是subset

``` r
subset(starwars, species == "Human")
>> # A tibble: 35 x 14
>>    name  height  mass hair_color skin_color eye_color birth_year sex   gender homeworld species films vehicles
>>    <chr>  <int> <dbl> <chr>      <chr>      <chr>          <dbl> <chr> <chr>  <chr>     <chr>   <lis> <list>  
>>  1 Luke~    172    77 blond      fair       blue            19   male  mascu~ Tatooine  Human   <chr~ <chr [2~
>>  2 Dart~    202   136 none       white      yellow          41.9 male  mascu~ Tatooine  Human   <chr~ <chr [0~
>>  3 Leia~    150    49 brown      light      brown           19   fema~ femin~ Alderaan  Human   <chr~ <chr [1~
>>  4 Owen~    178   120 brown, gr~ light      blue            52   male  mascu~ Tatooine  Human   <chr~ <chr [0~
>>  5 Beru~    165    75 brown      light      blue            47   fema~ femin~ Tatooine  Human   <chr~ <chr [0~
>>  6 Bigg~    183    84 black      light      brown           24   male  mascu~ Tatooine  Human   <chr~ <chr [0~
>>  7 Obi-~    182    77 auburn, w~ fair       blue-gray       57   male  mascu~ Stewjon   Human   <chr~ <chr [1~
>>  8 Anak~    188    84 blond      fair       blue            41.9 male  mascu~ Tatooine  Human   <chr~ <chr [2~
>>  9 Wilh~    180    NA auburn, g~ fair       blue            64   male  mascu~ Eriadu    Human   <chr~ <chr [0~
>> 10 Han ~    180    80 brown      fair       brown           29   male  mascu~ Corellia  Human   <chr~ <chr [0~
>> # ... with 25 more rows, and 1 more variable: starships <list>
```

也可以使用\[来选择行：

``` r
starwars[starwars$species == "Human",]
>> # A tibble: 39 x 14
>>    name  height  mass hair_color skin_color eye_color birth_year sex   gender homeworld species films vehicles
>>    <chr>  <int> <dbl> <chr>      <chr>      <chr>          <dbl> <chr> <chr>  <chr>     <chr>   <lis> <list>  
>>  1 Luke~    172    77 blond      fair       blue            19   male  mascu~ Tatooine  Human   <chr~ <chr [2~
>>  2 Dart~    202   136 none       white      yellow          41.9 male  mascu~ Tatooine  Human   <chr~ <chr [0~
>>  3 Leia~    150    49 brown      light      brown           19   fema~ femin~ Alderaan  Human   <chr~ <chr [1~
>>  4 Owen~    178   120 brown, gr~ light      blue            52   male  mascu~ Tatooine  Human   <chr~ <chr [0~
>>  5 Beru~    165    75 brown      light      blue            47   fema~ femin~ Tatooine  Human   <chr~ <chr [0~
>>  6 Bigg~    183    84 black      light      brown           24   male  mascu~ Tatooine  Human   <chr~ <chr [0~
>>  7 Obi-~    182    77 auburn, w~ fair       blue-gray       57   male  mascu~ Stewjon   Human   <chr~ <chr [1~
>>  8 Anak~    188    84 blond      fair       blue            41.9 male  mascu~ Tatooine  Human   <chr~ <chr [2~
>>  9 Wilh~    180    NA auburn, g~ fair       blue            64   male  mascu~ Eriadu    Human   <chr~ <chr [0~
>> 10 Han ~    180    80 brown      fair       brown           29   male  mascu~ Corellia  Human   <chr~ <chr [0~
>> # ... with 29 more rows, and 1 more variable: starships <list>
```

但是这样处理会出现NA的情况，为了避免NA，可以结合使用which：

``` r
starwars[which(starwars$species == "Human"), , drop = FALSE]
>> # A tibble: 35 x 14
>>    name  height  mass hair_color skin_color eye_color birth_year sex   gender homeworld species films vehicles
>>    <chr>  <int> <dbl> <chr>      <chr>      <chr>          <dbl> <chr> <chr>  <chr>     <chr>   <lis> <list>  
>>  1 Luke~    172    77 blond      fair       blue            19   male  mascu~ Tatooine  Human   <chr~ <chr [2~
>>  2 Dart~    202   136 none       white      yellow          41.9 male  mascu~ Tatooine  Human   <chr~ <chr [0~
>>  3 Leia~    150    49 brown      light      brown           19   fema~ femin~ Alderaan  Human   <chr~ <chr [1~
>>  4 Owen~    178   120 brown, gr~ light      blue            52   male  mascu~ Tatooine  Human   <chr~ <chr [0~
>>  5 Beru~    165    75 brown      light      blue            47   fema~ femin~ Tatooine  Human   <chr~ <chr [0~
>>  6 Bigg~    183    84 black      light      brown           24   male  mascu~ Tatooine  Human   <chr~ <chr [0~
>>  7 Obi-~    182    77 auburn, w~ fair       blue-gray       57   male  mascu~ Stewjon   Human   <chr~ <chr [1~
>>  8 Anak~    188    84 blond      fair       blue            41.9 male  mascu~ Tatooine  Human   <chr~ <chr [2~
>>  9 Wilh~    180    NA auburn, g~ fair       blue            64   male  mascu~ Eriadu    Human   <chr~ <chr [0~
>> 10 Han ~    180    80 brown      fair       brown           29   male  mascu~ Corellia  Human   <chr~ <chr [0~
>> # ... with 25 more rows, and 1 more variable: starships <list>
```

### `mutate()`创建或转化变量

`dplyr::mutate`从已存在的变量中创建新的变量

``` r
df %>% mutate(z = x + y, z2 = z ^ 2)
>> # A tibble: 100 x 4
>>        x     y     z    z2
>>    <int> <int> <int> <dbl>
>>  1     1     6     7    49
>>  2    10     8    18   324
>>  3    10     7    17   289
>>  4     3     3     6    36
>>  5     6     5    11   121
>>  6     7     9    16   256
>>  7     8     5    13   169
>>  8     1     9    10   100
>>  9     6     8    14   196
>> 10    10     7    17   289
>> # ... with 90 more rows
```

在base
R里面相似的有transform函数，但是要注意的是transform函数不能使用刚创建的变量，只能使用已有的变量：

``` r
head(transform(df,z=x+y,z2=z^2))
>> Error in eval(substitute(list(...)), `_data`, parent.frame()): 找不到对象'z'

head(transform(df,z=x+y,z2=(x+y)^2))
>>    x y  z  z2
>> 1  1 6  7  49
>> 2 10 8 18 324
>> 3 10 7 17 289
>> 4  3 3  6  36
>> 5  6 5 11 121
>> 6  7 9 16 256
```

也可以使用`$<-`来创建新的变量：

``` r
mtcars$cy12 <- mtcars$cyl * 2
mtcars$cy14 <- mtcars$cy12 *2
```

当应用到分组的数据框上，mutate可以对每个组别计算新的变量：

``` r
gf <- tibble(g = c(1, 1, 2, 2), x = c(0.5, 1.5, 2.5, 3.5))
gf %>% 
  group_by(g) %>% 
  mutate(x_mean = mean(x), x_rank = rank(x))
>> # A tibble: 4 x 4
>> # Groups:   g [2]
>>       g     x x_mean x_rank
>>   <dbl> <dbl>  <dbl>  <dbl>
>> 1     1   0.5      1      1
>> 2     1   1.5      1      2
>> 3     2   2.5      3      1
>> 4     2   3.5      3      2
```

在baseR中可以用使用`ave`函数

``` r
transform(gf, 
  x_mean = ave(x, g, FUN = mean), 
  x_rank = ave(x, g, FUN = rank)
)
>>   g   x x_mean x_rank
>> 1 1 0.5      1      1
>> 2 1 1.5      1      2
>> 3 2 2.5      3      1
>> 4 2 3.5      3      2
```

### `pull()` 抽提变量

`dplyr::pull()`可以通过名称或者位置提取变量：

``` r
mtcars %>% pull(1)
>>  [1] 21.0 21.0 22.8 21.4 18.7 18.1 14.3 24.4 22.8 19.2 17.8 16.4 17.3 15.2 10.4 10.4 14.7 32.4 30.4 33.9 21.5
>> [22] 15.5 15.2 13.3 19.2 27.3 26.0 30.4 15.8 19.7 15.0 21.4

mtcars %>% pull(cyl)
>>  [1] 6 6 4 6 8 6 8 4 4 6 6 8 8 8 8 8 8 4 4 4 4 8 8 8 8 4 4 4 8 6 8 4
```

在base R中相当于\[\[和$:

``` r
mtcars[["cyl"]]
>>  [1] 6 6 4 6 8 6 8 4 4 6 6 8 8 8 8 8 8 4 4 4 4 8 8 8 8 4 4 4 8 6 8 4
mtcars[[1]]
>>  [1] 21.0 21.0 22.8 21.4 18.7 18.1 14.3 24.4 22.8 19.2 17.8 16.4 17.3 15.2 10.4 10.4 14.7 32.4 30.4 33.9 21.5
>> [22] 15.5 15.2 13.3 19.2 27.3 26.0 30.4 15.8 19.7 15.0 21.4

mtcars$cyl
>>  [1] 6 6 4 6 8 6 8 4 4 6 6 8 8 8 8 8 8 4 4 4 4 8 8 8 8 4 4 4 8 6 8 4
```

### `relocate()` 改变列的顺序

`dplyr::relocate()`
可以方便的将列移到新的位置(默认是最前面,下面要讲的`select`只能将列移到最前面):

``` r
# to front
mtcars %>% relocate(gear, carb) 
>> # A tibble: 32 x 13
>>     gear  carb   mpg   cyl  disp    hp  drat    wt  qsec    vs    am  cy12  cy14
>>    <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl>
>>  1     4     4  21       6  160    110  3.9   2.62  16.5     0     1    12    24
>>  2     4     4  21       6  160    110  3.9   2.88  17.0     0     1    12    24
>>  3     4     1  22.8     4  108     93  3.85  2.32  18.6     1     1     8    16
>>  4     3     1  21.4     6  258    110  3.08  3.22  19.4     1     0    12    24
>>  5     3     2  18.7     8  360    175  3.15  3.44  17.0     0     0    16    32
>>  6     3     1  18.1     6  225    105  2.76  3.46  20.2     1     0    12    24
>>  7     3     4  14.3     8  360    245  3.21  3.57  15.8     0     0    16    32
>>  8     4     2  24.4     4  147.    62  3.69  3.19  20       1     0     8    16
>>  9     4     2  22.8     4  141.    95  3.92  3.15  22.9     1     0     8    16
>> 10     4     4  19.2     6  168.   123  3.92  3.44  18.3     1     0    12    24
>> # ... with 22 more rows

# to back
mtcars %>% relocate(mpg, cyl, .after = last_col()) 
>> # A tibble: 32 x 13
>>     disp    hp  drat    wt  qsec    vs    am  gear  carb  cy12  cy14   mpg   cyl
>>    <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl>
>>  1  160    110  3.9   2.62  16.5     0     1     4     4    12    24  21       6
>>  2  160    110  3.9   2.88  17.0     0     1     4     4    12    24  21       6
>>  3  108     93  3.85  2.32  18.6     1     1     4     1     8    16  22.8     4
>>  4  258    110  3.08  3.22  19.4     1     0     3     1    12    24  21.4     6
>>  5  360    175  3.15  3.44  17.0     0     0     3     2    16    32  18.7     8
>>  6  225    105  2.76  3.46  20.2     1     0     3     1    12    24  18.1     6
>>  7  360    245  3.21  3.57  15.8     0     0     3     4    16    32  14.3     8
>>  8  147.    62  3.69  3.19  20       1     0     4     2     8    16  24.4     4
>>  9  141.    95  3.92  3.15  22.9     1     0     4     2     8    16  22.8     4
>> 10  168.   123  3.92  3.44  18.3     1     0     4     4    12    24  19.2     6
>> # ... with 22 more rows

# to after disp
mtcars %>% relocate(mpg, cyl, .after = disp) 
>> # A tibble: 32 x 13
>>     disp   mpg   cyl    hp  drat    wt  qsec    vs    am  gear  carb  cy12  cy14
>>    <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl>
>>  1  160   21       6   110  3.9   2.62  16.5     0     1     4     4    12    24
>>  2  160   21       6   110  3.9   2.88  17.0     0     1     4     4    12    24
>>  3  108   22.8     4    93  3.85  2.32  18.6     1     1     4     1     8    16
>>  4  258   21.4     6   110  3.08  3.22  19.4     1     0     3     1    12    24
>>  5  360   18.7     8   175  3.15  3.44  17.0     0     0     3     2    16    32
>>  6  225   18.1     6   105  2.76  3.46  20.2     1     0     3     1    12    24
>>  7  360   14.3     8   245  3.21  3.57  15.8     0     0     3     4    16    32
>>  8  147.  24.4     4    62  3.69  3.19  20       1     0     4     2     8    16
>>  9  141.  22.8     4    95  3.92  3.15  22.9     1     0     4     2     8    16
>> 10  168.  19.2     6   123  3.92  3.44  18.3     1     0     4     4    12    24
>> # ... with 22 more rows
```

在base R中就有一点复杂：

``` r
##to front
mtcars[union(c("gear", "carb"), names(mtcars))]
>> # A tibble: 32 x 13
>>     gear  carb   mpg   cyl  disp    hp  drat    wt  qsec    vs    am  cy12  cy14
>>    <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl>
>>  1     4     4  21       6  160    110  3.9   2.62  16.5     0     1    12    24
>>  2     4     4  21       6  160    110  3.9   2.88  17.0     0     1    12    24
>>  3     4     1  22.8     4  108     93  3.85  2.32  18.6     1     1     8    16
>>  4     3     1  21.4     6  258    110  3.08  3.22  19.4     1     0    12    24
>>  5     3     2  18.7     8  360    175  3.15  3.44  17.0     0     0    16    32
>>  6     3     1  18.1     6  225    105  2.76  3.46  20.2     1     0    12    24
>>  7     3     4  14.3     8  360    245  3.21  3.57  15.8     0     0    16    32
>>  8     4     2  24.4     4  147.    62  3.69  3.19  20       1     0     8    16
>>  9     4     2  22.8     4  141.    95  3.92  3.15  22.9     1     0     8    16
>> 10     4     4  19.2     6  168.   123  3.92  3.44  18.3     1     0    12    24
>> # ... with 22 more rows

###to back
##先将要移动的列去掉，再重组到后面
to_back <- c("mpg", "cyl")
mtcars[c(setdiff(names(mtcars), to_back), to_back)]
>> # A tibble: 32 x 13
>>     disp    hp  drat    wt  qsec    vs    am  gear  carb  cy12  cy14   mpg   cyl
>>    <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl>
>>  1  160    110  3.9   2.62  16.5     0     1     4     4    12    24  21       6
>>  2  160    110  3.9   2.88  17.0     0     1     4     4    12    24  21       6
>>  3  108     93  3.85  2.32  18.6     1     1     4     1     8    16  22.8     4
>>  4  258    110  3.08  3.22  19.4     1     0     3     1    12    24  21.4     6
>>  5  360    175  3.15  3.44  17.0     0     0     3     2    16    32  18.7     8
>>  6  225    105  2.76  3.46  20.2     1     0     3     1    12    24  18.1     6
>>  7  360    245  3.21  3.57  15.8     0     0     3     4    16    32  14.3     8
>>  8  147.    62  3.69  3.19  20       1     0     4     2     8    16  24.4     4
>>  9  141.    95  3.92  3.15  22.9     1     0     4     2     8    16  22.8     4
>> 10  168.   123  3.92  3.44  18.3     1     0     4     4    12    24  19.2     6
>> # ... with 22 more rows
```

### `rename()` 重命名变量

`dplyr::rename()`可以通过旧的名称或者位置来重命名变量：

``` r
iris %>% rename(sepal_length = Sepal.Length, sepal_width = 2)
>> # A tibble: 150 x 5
>>    sepal_length sepal_width Petal.Length Petal.Width Species
>>           <dbl>       <dbl>        <dbl>       <dbl> <fct>  
>>  1          5.1         3.5          1.4         0.2 setosa 
>>  2          4.9         3            1.4         0.2 setosa 
>>  3          4.7         3.2          1.3         0.2 setosa 
>>  4          4.6         3.1          1.5         0.2 setosa 
>>  5          5           3.6          1.4         0.2 setosa 
>>  6          5.4         3.9          1.7         0.4 setosa 
>>  7          4.6         3.4          1.4         0.3 setosa 
>>  8          5           3.4          1.5         0.2 setosa 
>>  9          4.4         2.9          1.4         0.2 setosa 
>> 10          4.9         3.1          1.5         0.1 setosa 
>> # ... with 140 more rows
```

在base R中根据位置来重命名变量是比较直接的：

``` r
iris2 <- iris
names(iris2)[2] <- "sepal_width"
```

通过旧的变量名来重命名有一点繁琐：

``` r
names(iris2)[names(iris2) == "Sepal.Length"] <- "sepal_length"
```

### `rename_with()`通过函数来重命名变量

`dplyr::rename_with()`通过函数来转化列名：

``` r
iris %>% rename_with(toupper)
>> # A tibble: 150 x 5
>>    SEPAL.LENGTH SEPAL.WIDTH PETAL.LENGTH PETAL.WIDTH SPECIES
>>           <dbl>       <dbl>        <dbl>       <dbl> <fct>  
>>  1          5.1         3.5          1.4         0.2 setosa 
>>  2          4.9         3            1.4         0.2 setosa 
>>  3          4.7         3.2          1.3         0.2 setosa 
>>  4          4.6         3.1          1.5         0.2 setosa 
>>  5          5           3.6          1.4         0.2 setosa 
>>  6          5.4         3.9          1.7         0.4 setosa 
>>  7          4.6         3.4          1.4         0.3 setosa 
>>  8          5           3.4          1.5         0.2 setosa 
>>  9          4.4         2.9          1.4         0.2 setosa 
>> 10          4.9         3.1          1.5         0.1 setosa 
>> # ... with 140 more rows

###也可以选择范围，默认是所有列
rename_with(iris, toupper, starts_with("Petal"))
>> # A tibble: 150 x 5
>>    Sepal.Length Sepal.Width PETAL.LENGTH PETAL.WIDTH Species
>>           <dbl>       <dbl>        <dbl>       <dbl> <fct>  
>>  1          5.1         3.5          1.4         0.2 setosa 
>>  2          4.9         3            1.4         0.2 setosa 
>>  3          4.7         3.2          1.3         0.2 setosa 
>>  4          4.6         3.1          1.5         0.2 setosa 
>>  5          5           3.6          1.4         0.2 setosa 
>>  6          5.4         3.9          1.7         0.4 setosa 
>>  7          4.6         3.4          1.4         0.3 setosa 
>>  8          5           3.4          1.5         0.2 setosa 
>>  9          4.4         2.9          1.4         0.2 setosa 
>> 10          4.9         3.1          1.5         0.1 setosa 
>> # ... with 140 more rows

###也可以自定义函数
rename_with(iris, function(x){
  gsub(".","_",x,fixed = TRUE)
  },starts_with("Petal"))
>> # A tibble: 150 x 5
>>    Sepal.Length Sepal.Width Petal_Length Petal_Width Species
>>           <dbl>       <dbl>        <dbl>       <dbl> <fct>  
>>  1          5.1         3.5          1.4         0.2 setosa 
>>  2          4.9         3            1.4         0.2 setosa 
>>  3          4.7         3.2          1.3         0.2 setosa 
>>  4          4.6         3.1          1.5         0.2 setosa 
>>  5          5           3.6          1.4         0.2 setosa 
>>  6          5.4         3.9          1.7         0.4 setosa 
>>  7          4.6         3.4          1.4         0.3 setosa 
>>  8          5           3.4          1.5         0.2 setosa 
>>  9          4.4         2.9          1.4         0.2 setosa 
>> 10          4.9         3.1          1.5         0.1 setosa 
>> # ... with 140 more rows

###或者公式类型的函数
rename_with(iris, ~ tolower(gsub(".", "_", .x, fixed = TRUE)))
>> # A tibble: 150 x 5
>>    sepal_length sepal_width petal_length petal_width species
>>           <dbl>       <dbl>        <dbl>       <dbl> <fct>  
>>  1          5.1         3.5          1.4         0.2 setosa 
>>  2          4.9         3            1.4         0.2 setosa 
>>  3          4.7         3.2          1.3         0.2 setosa 
>>  4          4.6         3.1          1.5         0.2 setosa 
>>  5          5           3.6          1.4         0.2 setosa 
>>  6          5.4         3.9          1.7         0.4 setosa 
>>  7          4.6         3.4          1.4         0.3 setosa 
>>  8          5           3.4          1.5         0.2 setosa 
>>  9          4.4         2.9          1.4         0.2 setosa 
>> 10          4.9         3.1          1.5         0.1 setosa 
>> # ... with 140 more rows
```

在base R中可以使用`setNames()`来实现：

``` r
setNames(iris, toupper(names(iris)))
>> # A tibble: 150 x 5
>>    SEPAL.LENGTH SEPAL.WIDTH PETAL.LENGTH PETAL.WIDTH SPECIES
>>           <dbl>       <dbl>        <dbl>       <dbl> <fct>  
>>  1          5.1         3.5          1.4         0.2 setosa 
>>  2          4.9         3            1.4         0.2 setosa 
>>  3          4.7         3.2          1.3         0.2 setosa 
>>  4          4.6         3.1          1.5         0.2 setosa 
>>  5          5           3.6          1.4         0.2 setosa 
>>  6          5.4         3.9          1.7         0.4 setosa 
>>  7          4.6         3.4          1.4         0.3 setosa 
>>  8          5           3.4          1.5         0.2 setosa 
>>  9          4.4         2.9          1.4         0.2 setosa 
>> 10          4.9         3.1          1.5         0.1 setosa 
>> # ... with 140 more rows
```

### `select()`通过列名选择变量

`dplyr::select()`根据列名，位置，和列名相关的函数或者其他特征来选择列：

``` r
###位置
iris %>% select(1:3)
>> # A tibble: 150 x 3
>>    Sepal.Length Sepal.Width Petal.Length
>>           <dbl>       <dbl>        <dbl>
>>  1          5.1         3.5          1.4
>>  2          4.9         3            1.4
>>  3          4.7         3.2          1.3
>>  4          4.6         3.1          1.5
>>  5          5           3.6          1.4
>>  6          5.4         3.9          1.7
>>  7          4.6         3.4          1.4
>>  8          5           3.4          1.5
>>  9          4.4         2.9          1.4
>> 10          4.9         3.1          1.5
>> # ... with 140 more rows

##列名
iris %>% select(Species, Sepal.Length)
>> # A tibble: 150 x 2
>>    Species Sepal.Length
>>    <fct>          <dbl>
>>  1 setosa           5.1
>>  2 setosa           4.9
>>  3 setosa           4.7
>>  4 setosa           4.6
>>  5 setosa           5  
>>  6 setosa           5.4
>>  7 setosa           4.6
>>  8 setosa           5  
>>  9 setosa           4.4
>> 10 setosa           4.9
>> # ... with 140 more rows

##函数
iris %>% select(starts_with("Petal"))
>> # A tibble: 150 x 2
>>    Petal.Length Petal.Width
>>           <dbl>       <dbl>
>>  1          1.4         0.2
>>  2          1.4         0.2
>>  3          1.3         0.2
>>  4          1.5         0.2
>>  5          1.4         0.2
>>  6          1.7         0.4
>>  7          1.4         0.3
>>  8          1.5         0.2
>>  9          1.4         0.2
>> 10          1.5         0.1
>> # ... with 140 more rows
iris %>% select(where(is.factor))
>> # A tibble: 150 x 1
>>    Species
>>    <fct>  
>>  1 setosa 
>>  2 setosa 
>>  3 setosa 
>>  4 setosa 
>>  5 setosa 
>>  6 setosa 
>>  7 setosa 
>>  8 setosa 
>>  9 setosa 
>> 10 setosa 
>> # ... with 140 more rows
```

在base R中通过位置选择变量是比较直接的:

``` r
iris[1:3]##单个参数是取列的
>> # A tibble: 150 x 3
>>    Sepal.Length Sepal.Width Petal.Length
>>           <dbl>       <dbl>        <dbl>
>>  1          5.1         3.5          1.4
>>  2          4.9         3            1.4
>>  3          4.7         3.2          1.3
>>  4          4.6         3.1          1.5
>>  5          5           3.6          1.4
>>  6          5.4         3.9          1.7
>>  7          4.6         3.4          1.4
>>  8          5           3.4          1.5
>>  9          4.4         2.9          1.4
>> 10          4.9         3.1          1.5
>> # ... with 140 more rows

iris[1:3, , drop = FALSE]##也可以加多个参数，第二个参数是列
>> # A tibble: 3 x 5
>>   Sepal.Length Sepal.Width Petal.Length Petal.Width Species
>>          <dbl>       <dbl>        <dbl>       <dbl> <fct>  
>> 1          5.1         3.5          1.4         0.2 setosa 
>> 2          4.9         3            1.4         0.2 setosa 
>> 3          4.7         3.2          1.3         0.2 setosa
```

按照名称选择列可以有两种选择：

``` r
###直接按照字符向量选择
iris[c("Species", "Sepal.Length")]
>> # A tibble: 150 x 2
>>    Species Sepal.Length
>>    <fct>          <dbl>
>>  1 setosa           5.1
>>  2 setosa           4.9
>>  3 setosa           4.7
>>  4 setosa           4.6
>>  5 setosa           5  
>>  6 setosa           5.4
>>  7 setosa           4.6
>>  8 setosa           5  
>>  9 setosa           4.4
>> 10 setosa           4.9
>> # ... with 140 more rows

###使用subset函数，subset使用了和dplyr相同的机制(元编程)
subset(iris, select = c(Species, Sepal.Length))
>> # A tibble: 150 x 2
>>    Species Sepal.Length
>>    <fct>          <dbl>
>>  1 setosa           5.1
>>  2 setosa           4.9
>>  3 setosa           4.7
>>  4 setosa           4.6
>>  5 setosa           5  
>>  6 setosa           5.4
>>  7 setosa           4.6
>>  8 setosa           5  
>>  9 setosa           4.4
>> 10 setosa           4.9
>> # ... with 140 more rows
```

通过名称的函数来选择列，可以使用`grep`函数来匹配：

``` r
iris[grep("^Petal", names(iris))]
>> # A tibble: 150 x 2
>>    Petal.Length Petal.Width
>>           <dbl>       <dbl>
>>  1          1.4         0.2
>>  2          1.4         0.2
>>  3          1.3         0.2
>>  4          1.5         0.2
>>  5          1.4         0.2
>>  6          1.7         0.4
>>  7          1.4         0.3
>>  8          1.5         0.2
>>  9          1.4         0.2
>> 10          1.5         0.1
>> # ... with 140 more rows
```

也可以通过Filter函数根据变量的类型来选择列：Filter是高阶函数，接受别的函数作为参数，高阶函数的内容见[review](https://wutaoblog.com.cn/p/meta_r_prom/)

``` r
###
Filter(is.factor,iris)
>> # A tibble: 150 x 1
>>    Species
>>    <fct>  
>>  1 setosa 
>>  2 setosa 
>>  3 setosa 
>>  4 setosa 
>>  5 setosa 
>>  6 setosa 
>>  7 setosa 
>>  8 setosa 
>>  9 setosa 
>> 10 setosa 
>> # ... with 140 more rows
```

### `summarise()`将多个值汇总成单个值

`dplyr::summarise` 计算每个组别的汇总信息：

``` r
mtcars %>% 
  group_by(cyl) %>% 
  summarise(mean = mean(disp), n = n())
>> `summarise()` ungrouping output (override with `.groups` argument)
>> # A tibble: 3 x 3
>>     cyl  mean     n
>>   <dbl> <dbl> <int>
>> 1     4  105.    11
>> 2     6  183.     7
>> 3     8  353.    14
```

在base R里面比较相似的是by函数，但是by函数返回的是list：

``` r
###先来看一下by函数的用法
##by(data, data$byvar, FUN)
##data是数据，data$byvar是分组依据，fun是函数


mtcars_by <- by(mtcars, mtcars$cyl, function(df) {
  with(df, data.frame(cyl = cyl[[1]], mean = mean(disp), n = nrow(df)))
})
mtcars_by
>> mtcars$cyl: 4
>>   cyl     mean  n
>> 1   4 105.1364 11
>> ------------------------------------------------------------------------------------ 
>> mtcars$cyl: 6
>>   cyl     mean n
>> 1   6 183.3143 7
>> ------------------------------------------------------------------------------------ 
>> mtcars$cyl: 8
>>   cyl  mean  n
>> 1   8 353.1 14
```

我们可以使用[do.call函数]()来将这个列表合并成数据框：

``` r
do.call(rbind,mtcars_by)
>>   cyl     mean  n
>> 4   4 105.1364 11
>> 6   6 183.3143  7
>> 8   8 353.1000 14
```

### `slice()` 根据位置选择行

``` r
###n表示行数
slice(mtcars, 25:n())
>> # A tibble: 8 x 13
>>     mpg   cyl  disp    hp  drat    wt  qsec    vs    am  gear  carb  cy12  cy14
>>   <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl>
>> 1  19.2     8 400     175  3.08  3.84  17.0     0     0     3     2    16    32
>> 2  27.3     4  79      66  4.08  1.94  18.9     1     1     4     1     8    16
>> 3  26       4 120.     91  4.43  2.14  16.7     0     1     5     2     8    16
>> 4  30.4     4  95.1   113  3.77  1.51  16.9     1     1     5     2     8    16
>> 5  15.8     8 351     264  4.22  3.17  14.5     0     1     5     4    16    32
>> 6  19.7     6 145     175  3.62  2.77  15.5     0     1     5     6    12    24
>> 7  15       8 301     335  3.54  3.57  14.6     0     1     5     8    16    32
>> 8  21.4     4 121     109  4.11  2.78  18.6     1     1     4     2     8    16
```

在base R中可以直接使用\[来选取：

``` r
mtcars[25:nrow(mtcars), , drop = FALSE]
>> # A tibble: 8 x 13
>>     mpg   cyl  disp    hp  drat    wt  qsec    vs    am  gear  carb  cy12  cy14
>>   <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl> <dbl>
>> 1  19.2     8 400     175  3.08  3.84  17.0     0     0     3     2    16    32
>> 2  27.3     4  79      66  4.08  1.94  18.9     1     1     4     1     8    16
>> 3  26       4 120.     91  4.43  2.14  16.7     0     1     5     2     8    16
>> 4  30.4     4  95.1   113  3.77  1.51  16.9     1     1     5     2     8    16
>> 5  15.8     8 351     264  4.22  3.17  14.5     0     1     5     4    16    32
>> 6  19.7     6 145     175  3.62  2.77  15.5     0     1     5     6    12    24
>> 7  15       8 301     335  3.54  3.57  14.6     0     1     5     8    16    32
>> 8  21.4     4 121     109  4.11  2.78  18.6     1     1     4     2     8    16
```

## Two-table verbs

增加：[two-table](https://github.com/tidyverse/dplyr/blob/master/vignettes/two-table.Rmd)

two-table
verbs指的是合并两个数据框的操作，在dplyr中使用\*\_join操作代替base
R中的各种merge操作：

| dplyr                  | base                                     |
|------------------------|------------------------------------------|
| `inner_join(df1, df2)` | `merge(df1, df2)`                        |
| `left_join(df1, df2)`  | `merge(df1, df2, all.x = TRUE)`          |
| `right_join(df1, df2)` | `merge(df1, df2, all.y = TRUE)`          |
| `full_join(df1, df2)`  | `merge(df1, df2, all = TRUE)`            |
| `semi_join(df1, df2)`  | `df1[df1$x %in% df2$x, , drop = FALSE]`  |
| `anti_join(df1, df2)`  | `df1[!df1$x %in% df2$x, , drop = FALSE]` |

在dplyr中有3类动词可以同时对两个table进行操作：

-   Mutating join 根据匹配的行来添加变量

-   Filtering joins 根据匹配的行来筛选变量

-   Set operations 将数据集的行作为集合的元素来操作

### Mutating joins

Mutating
join可以将两个table的变量结合到一起；比如在nycflights13数据中一个table有航班信息，并且每个航班有相应的航空公司的缩写，另一个table有航空公司的缩写和全称的对应信息，我们可以将这两个table合并：

``` r
library("nycflights13")
>> Warning: package 'nycflights13' was built under R version 3.6.3

flights2 <- flights %>% select(year:day, hour, origin, dest, tailnum, carrier)

flights2 %>% 
  left_join(airlines)
>> Joining, by = "carrier"
>> # A tibble: 336,776 x 9
>>     year month   day  hour origin dest  tailnum carrier name                    
>>    <int> <int> <int> <dbl> <chr>  <chr> <chr>   <chr>   <chr>                   
>>  1  2013     1     1     5 EWR    IAH   N14228  UA      United Air Lines Inc.   
>>  2  2013     1     1     5 LGA    IAH   N24211  UA      United Air Lines Inc.   
>>  3  2013     1     1     5 JFK    MIA   N619AA  AA      American Airlines Inc.  
>>  4  2013     1     1     5 JFK    BQN   N804JB  B6      JetBlue Airways         
>>  5  2013     1     1     6 LGA    ATL   N668DN  DL      Delta Air Lines Inc.    
>>  6  2013     1     1     5 EWR    ORD   N39463  UA      United Air Lines Inc.   
>>  7  2013     1     1     6 EWR    FLL   N516JB  B6      JetBlue Airways         
>>  8  2013     1     1     6 LGA    IAD   N829AS  EV      ExpressJet Airlines Inc.
>>  9  2013     1     1     6 JFK    MCO   N593JB  B6      JetBlue Airways         
>> 10  2013     1     1     6 LGA    ORD   N3ALAA  AA      American Airlines Inc.  
>> # ... with 336,766 more rows
```

#### 控制table如何匹配

每一个Mutating join函数都有一个by参数，控制哪个变量被用来进行匹配

-   `NULL`
    默认值，使用两个table中共有的变量，比如flights和weather两个表的共有列为year,
    month, day, hour 和origin

    ``` r
    flights2 %>% left_join(weather)
    >> Joining, by = c("year", "month", "day", "hour", "origin")
    >> # A tibble: 336,776 x 18
    >>     year month   day  hour origin dest  tailnum carrier  temp  dewp humid wind_dir wind_speed wind_gust precip
    >>    <int> <int> <int> <dbl> <chr>  <chr> <chr>   <chr>   <dbl> <dbl> <dbl>    <dbl>      <dbl>     <dbl>  <dbl>
    >>  1  2013     1     1     5 EWR    IAH   N14228  UA       39.0  28.0  64.4      260       12.7      NA        0
    >>  2  2013     1     1     5 LGA    IAH   N24211  UA       39.9  25.0  54.8      250       15.0      21.9      0
    >>  3  2013     1     1     5 JFK    MIA   N619AA  AA       39.0  27.0  61.6      260       15.0      NA        0
    >>  4  2013     1     1     5 JFK    BQN   N804JB  B6       39.0  27.0  61.6      260       15.0      NA        0
    >>  5  2013     1     1     6 LGA    ATL   N668DN  DL       39.9  25.0  54.8      260       16.1      23.0      0
    >>  6  2013     1     1     5 EWR    ORD   N39463  UA       39.0  28.0  64.4      260       12.7      NA        0
    >>  7  2013     1     1     6 EWR    FLL   N516JB  B6       37.9  28.0  67.2      240       11.5      NA        0
    >>  8  2013     1     1     6 LGA    IAD   N829AS  EV       39.9  25.0  54.8      260       16.1      23.0      0
    >>  9  2013     1     1     6 JFK    MCO   N593JB  B6       37.9  27.0  64.3      260       13.8      NA        0
    >> 10  2013     1     1     6 LGA    ORD   N3ALAA  AA       39.9  25.0  54.8      260       16.1      23.0      0
    >> # ... with 336,766 more rows, and 3 more variables: pressure <dbl>, visib <dbl>, time_hour <dttm>
    ```

-   字符向量，`by="x"` 使用指定的变量进行匹配

    ``` r
    flights2 %>% left_join(planes, by = "tailnum")
    >> # A tibble: 336,776 x 16
    >>    year.x month   day  hour origin dest  tailnum carrier year.y type  manufacturer model engines seats speed
    >>     <int> <int> <int> <dbl> <chr>  <chr> <chr>   <chr>    <int> <chr> <chr>        <chr>   <int> <int> <int>
    >>  1   2013     1     1     5 EWR    IAH   N14228  UA        1999 Fixe~ BOEING       737-~       2   149    NA
    >>  2   2013     1     1     5 LGA    IAH   N24211  UA        1998 Fixe~ BOEING       737-~       2   149    NA
    >>  3   2013     1     1     5 JFK    MIA   N619AA  AA        1990 Fixe~ BOEING       757-~       2   178    NA
    >>  4   2013     1     1     5 JFK    BQN   N804JB  B6        2012 Fixe~ AIRBUS       A320~       2   200    NA
    >>  5   2013     1     1     6 LGA    ATL   N668DN  DL        1991 Fixe~ BOEING       757-~       2   178    NA
    >>  6   2013     1     1     5 EWR    ORD   N39463  UA        2012 Fixe~ BOEING       737-~       2   191    NA
    >>  7   2013     1     1     6 EWR    FLL   N516JB  B6        2000 Fixe~ AIRBUS INDU~ A320~       2   200    NA
    >>  8   2013     1     1     6 LGA    IAD   N829AS  EV        1998 Fixe~ CANADAIR     CL-6~       2    55    NA
    >>  9   2013     1     1     6 JFK    MCO   N593JB  B6        2004 Fixe~ AIRBUS       A320~       2   200    NA
    >> 10   2013     1     1     6 LGA    ORD   N3ALAA  AA          NA <NA>  <NA>         <NA>       NA    NA    NA
    >> # ... with 336,766 more rows, and 1 more variable: engine <chr>

    ##两个table的其他的共有列会加上后缀
    ```

-   具名字符向量，`by=c("a"="b")`
    将一个table中的a变量与另一个table中的b变量进行匹配(输出中保留a)

    ``` r
    flights2 %>% left_join(airports, c("dest" = "faa"))
    >> # A tibble: 336,776 x 15
    >>     year month   day  hour origin dest  tailnum carrier name                lat   lon   alt    tz dst   tzone    
    >>    <int> <int> <int> <dbl> <chr>  <chr> <chr>   <chr>   <chr>             <dbl> <dbl> <dbl> <dbl> <chr> <chr>    
    >>  1  2013     1     1     5 EWR    IAH   N14228  UA      George Bush Inte~  30.0 -95.3    97    -6 A     America/~
    >>  2  2013     1     1     5 LGA    IAH   N24211  UA      George Bush Inte~  30.0 -95.3    97    -6 A     America/~
    >>  3  2013     1     1     5 JFK    MIA   N619AA  AA      Miami Intl         25.8 -80.3     8    -5 A     America/~
    >>  4  2013     1     1     5 JFK    BQN   N804JB  B6      <NA>               NA    NA      NA    NA <NA>  <NA>     
    >>  5  2013     1     1     6 LGA    ATL   N668DN  DL      Hartsfield Jacks~  33.6 -84.4  1026    -5 A     America/~
    >>  6  2013     1     1     5 EWR    ORD   N39463  UA      Chicago Ohare In~  42.0 -87.9   668    -6 A     America/~
    >>  7  2013     1     1     6 EWR    FLL   N516JB  B6      Fort Lauderdale ~  26.1 -80.2     9    -5 A     America/~
    >>  8  2013     1     1     6 LGA    IAD   N829AS  EV      Washington Dulle~  38.9 -77.5   313    -5 A     America/~
    >>  9  2013     1     1     6 JFK    MCO   N593JB  B6      Orlando Intl       28.4 -81.3    96    -5 A     America/~
    >> 10  2013     1     1     6 LGA    ORD   N3ALAA  AA      Chicago Ohare In~  42.0 -87.9   668    -6 A     America/~
    >> # ... with 336,766 more rows
    
    flights2 %>% left_join(airports, c("origin" = "faa"))
    >> # A tibble: 336,776 x 15
    >>     year month   day  hour origin dest  tailnum carrier name             lat   lon   alt    tz dst   tzone       
    >>    <int> <int> <int> <dbl> <chr>  <chr> <chr>   <chr>   <chr>          <dbl> <dbl> <dbl> <dbl> <chr> <chr>       
    >>  1  2013     1     1     5 EWR    IAH   N14228  UA      Newark Libert~  40.7 -74.2    18    -5 A     America/New~
    >>  2  2013     1     1     5 LGA    IAH   N24211  UA      La Guardia      40.8 -73.9    22    -5 A     America/New~
    >>  3  2013     1     1     5 JFK    MIA   N619AA  AA      John F Kenned~  40.6 -73.8    13    -5 A     America/New~
    >>  4  2013     1     1     5 JFK    BQN   N804JB  B6      John F Kenned~  40.6 -73.8    13    -5 A     America/New~
    >>  5  2013     1     1     6 LGA    ATL   N668DN  DL      La Guardia      40.8 -73.9    22    -5 A     America/New~
    >>  6  2013     1     1     5 EWR    ORD   N39463  UA      Newark Libert~  40.7 -74.2    18    -5 A     America/New~
    >>  7  2013     1     1     6 EWR    FLL   N516JB  B6      Newark Libert~  40.7 -74.2    18    -5 A     America/New~
    >>  8  2013     1     1     6 LGA    IAD   N829AS  EV      La Guardia      40.8 -73.9    22    -5 A     America/New~
    >>  9  2013     1     1     6 JFK    MCO   N593JB  B6      John F Kenned~  40.6 -73.8    13    -5 A     America/New~
    >> 10  2013     1     1     6 LGA    ORD   N3ALAA  AA      La Guardia      40.8 -73.9    22    -5 A     America/New~
    >> # ... with 336,766 more rows
    ```

#### 匹配的类型

有4种类型的mutating join，他们的区别在于如何处理找不到匹配的情况

``` r
##示例数据
df1 <- tibble(x = c(1, 2), y = 2:1)
df2 <- tibble(x = c(3, 1), a = 10, b = "a")

df1
>> # A tibble: 2 x 2
>>       x     y
>>   <dbl> <int>
>> 1     1     2
>> 2     2     1

df2
>> # A tibble: 2 x 3
>>       x     a b    
>>   <dbl> <dbl> <chr>
>> 1     3    10 a    
>> 2     1    10 a
```

-   `inner_join(x,y)` 只包含x和y中都有的行

    ``` r
    df1 %>% inner_join(df2)
    >> Joining, by = "x"
    >> # A tibble: 1 x 4
    >>       x     y     a b    
    >>   <dbl> <int> <dbl> <chr>
    >> 1     1     2    10 a
    ```

-   `left_join(x,y)` 包含x的所有行，不管有没有匹配(没有匹配的为NA)

    ``` r
    df1 %>% left_join(df2)
    >> Joining, by = "x"
    >> # A tibble: 2 x 4
    >>       x     y     a b    
    >>   <dbl> <int> <dbl> <chr>
    >> 1     1     2    10 a    
    >> 2     2     1    NA <NA>
    ```

-   `right_join(x,y)` 包含y的所有行(和`left_join(y,x)`
    的差别在于行和列的顺序不一样)

    ``` r
    df1 %>% right_join(df2)
    >> Joining, by = "x"
    >> # A tibble: 2 x 4
    >>       x     y     a b    
    >>   <dbl> <int> <dbl> <chr>
    >> 1     1     2    10 a    
    >> 2     3    NA    10 a
    
    df2 %>% left_join(df1)
    >> Joining, by = "x"
    >> # A tibble: 2 x 4
    >>       x     a b         y
    >>   <dbl> <dbl> <chr> <int>
    >> 1     3    10 a        NA
    >> 2     1    10 a         2
    ```

-   `full_join(x,y)` 包含x和y的所有行

    ``` r
    df1 %>% full_join(df2)
    >> Joining, by = "x"
    >> # A tibble: 3 x 4
    >>       x     y     a b    
    >>   <dbl> <int> <dbl> <chr>
    >> 1     1     2    10 a    
    >> 2     2     1    NA <NA> 
    >> 3     3    NA    10 a
    ```

需要注意的一点是：如果匹配不唯一(比如x里面用来匹配的变量有几行是一样的)，那么进行join时会加上所有可能的组合(笛卡尔积)：

``` r
df1 <- tibble(x = c(1, 1, 2), y = 1:3)
df2 <- tibble(x = c(1, 1, 2), z = c("a", "b", "a"))
df1 %>% left_join(df2)
>> Joining, by = "x"
>> # A tibble: 5 x 3
>>       x     y z    
>>   <dbl> <int> <chr>
>> 1     1     1 a    
>> 2     1     1 b    
>> 3     1     2 a    
>> 4     1     2 b    
>> 5     2     3 a
```

### Filtering joins

filter join影响的是观测不是变量，有两种类型：

-   `semi_join(x,y)` 保留x在y中有匹配的观测

-   `anti_join(x,y)` 丢弃x在y中有匹配的观测
