---
title: dplyr列式操作    
date: 2021-01-23 10:00:00    
index_img: img/across.png
---



dplyr 按列操作，主要是 across 函数的用法

<!-- more -->

我们在数据分析过程中经常要做的一件事就是对数据框的多列进行同样的操作，但是如果采用粘贴复制的方法比较繁琐也容易出错，比如：

``` r
df %>% 
  group_by(g1, g2) %>% 
  summarise(a = mean(a), b = mean(b), c = mean(c), d = mean(d))
```

使用across函数就可以较简洁的重写上面的代码：

``` r
df %>% 
  group_by(g1, g2) %>% 
  summarise(across(a:d, mean))
```

## 基础用法

`across`有两个基本的参数：

-   第一个参数是`.cols` 选择想要操作的列，使用的方法是*tidy selection* (也就是和select一样，可以根据位置/名称/类型来选择)

-   第二个参数是`.fns` 是对每列进行操作的函数，可以是purrr风格的公式(比如\~.x/2，具体见[迭代—purrr](sss))

across最常见的是与summarise一起使用(别的动词也可以)：

``` r
starwars %>% 
  summarise(across(where(is.character), ~ length(unique(.x))))
>> # A tibble: 1 x 8
>>    name hair_color skin_color eye_color   sex gender
>>   <int>      <int>      <int>     <int> <int>  <int>
>> 1    87         13         31        15     5      3
>> # ... with 2 more variables: homeworld <int>,
>> #   species <int>

starwars %>% 
  group_by(species) %>% 
  filter(n() > 1) %>% 
  summarise(across(c(sex, gender, homeworld), ~ length(unique(.x))))
>> # A tibble: 9 x 4
>>   species    sex gender homeworld
>>   <chr>    <int>  <int>     <int>
>> 1 Droid        1      2         3
>> 2 Gungan       1      1         1
>> 3 Human        2      2        16
>> 4 Kaminoan     2      2         1
>> 5 Mirialan     1      1         1
>> 6 Twi'lek      2      2         1
>> 7 Wookiee      1      1         1
>> 8 Zabrak       1      1         2
>> 9 <NA>         1      1         3

starwars %>% 
  group_by(homeworld) %>% 
  filter(n() > 1) %>% 
  summarise(across(where(is.numeric), ~ mean(.x, na.rm = TRUE)))
>> # A tibble: 10 x 4
>>    homeworld height  mass birth_year
>>    <chr>      <dbl> <dbl>      <dbl>
>>  1 Alderaan    176.  64         43  
>>  2 Corellia    175   78.5       25  
>>  3 Coruscant   174.  50         91  
>>  4 Kamino      208.  83.1       31.5
>>  5 Kashyyyk    231  124        200  
>>  6 Mirial      168   53.1       49  
>>  7 Naboo       175.  64.2       55  
>>  8 Ryloth      179   55         48  
>>  9 Tatooine    170.  85.4       54.6
>> 10 <NA>        139.  82        334.
```

需要注意的是：across在进行操作的时候不会选择分组变量：

``` r
df <- data.frame(g = c(1, 1, 2), x = c(-1, 1, 3), y = c(-1, -4, -9))
df %>% 
  group_by(g) %>% 
  summarise(across(where(is.numeric), sum))
>> # A tibble: 2 x 3
>>       g     x     y
>>   <dbl> <dbl> <dbl>
>> 1     1     0    -5
>> 2     2     3    -9
```

### 多个函数

也可以使用多个函数对列进行操作，只需要在第二个参数中提供具名函数的列表：

``` r
min_max <- list(
  min = ~min(.x, na.rm = TRUE), 
  max = ~max(.x, na.rm = TRUE)
)
starwars %>% summarise(across(where(is.numeric), min_max))
>> # A tibble: 1 x 6
>>   height_min height_max mass_min mass_max birth_year_min
>>        <int>      <int>    <dbl>    <dbl>          <dbl>
>> 1         66        264       15     1358              8
>> # ... with 1 more variable: birth_year_max <dbl>

starwars %>% summarise(across(c(height, mass, birth_year), min_max))
>> # A tibble: 1 x 6
>>   height_min height_max mass_min mass_max birth_year_min
>>        <int>      <int>    <dbl>    <dbl>          <dbl>
>> 1         66        264       15     1358              8
>> # ... with 1 more variable: birth_year_max <dbl>
```

我们可以看到默认的输出结果的列名是原来的列名加上函数的名称中间用下划线分割，也可以通过`.name`参数来指定输出的名称(以`glue`包中的格式)：

``` r
starwars %>% summarise(across(where(is.numeric), min_max, .names = "{.fn}.{.col}"))##调换位置，并以点号分割
>> # A tibble: 1 x 6
>>   min.height max.height min.mass max.mass min.birth_year
>>        <int>      <int>    <dbl>    <dbl>          <dbl>
>> 1         66        264       15     1358              8
>> # ... with 1 more variable: max.birth_year <dbl>

starwars %>% summarise(across(c(height, mass, birth_year), min_max, .names = "{.fn}.{.col}"))
>> # A tibble: 1 x 6
>>   min.height max.height min.mass max.mass min.birth_year
>>        <int>      <int>    <dbl>    <dbl>          <dbl>
>> 1         66        264       15     1358              8
>> # ... with 1 more variable: max.birth_year <dbl>
```

如果想要同一个函数操作得到的结果放在一起，我们可以把上面两个函数拆开执行：

``` r
starwars %>% summarise(
  across(c(height, mass, birth_year), ~min(.x, na.rm = TRUE), .names = "min_{.col}"),
  across(c(height, mass, birth_year), ~max(.x, na.rm = TRUE), .names = "max_{.col}")
)
>> # A tibble: 1 x 6
>>   min_height min_mass min_birth_year max_height max_mass
>>        <int>    <dbl>          <dbl>      <int>    <dbl>
>> 1         66       15              8        264     1358
>> # ... with 1 more variable: max_birth_year <dbl>
```

注意：在上面的代码中不能直接使用`where(is.numeric)` 因为第二个across会对新生成的数值变量(“min_height”, “min_mass” and“min_birth_year”)进行操作（执行有顺序）:

``` r
starwars %>% summarise(
  across(where(is.numeric), ~min(.x, na.rm = TRUE), .names = "min_{.col}"),
    across(where(is.numeric), ~max(.x, na.rm = TRUE), .names = "max_{.col}")  
)
>> # A tibble: 1 x 9
>>   min_height min_mass min_birth_year max_height max_mass
>>        <int>    <dbl>          <dbl>      <int>    <dbl>
>> 1         66       15              8        264     1358
>> # ... with 4 more variables: max_birth_year <dbl>,
>> #   max_min_height <int>, max_min_mass <dbl>,
>> #   max_min_birth_year <dbl>
```

可以看到生成了额外的三列：max_min_height \<int\>, max_min_mass \<dbl\>,
max_min_birth_year \<dbl\>

要解决这个问题可以将 `across` 的结果返回为一个 tibble，再输出：

``` r
starwars %>% summarise(
  tibble(
    across(where(is.numeric), ~min(.x, na.rm = TRUE), .names = "min_{.col}"),
    across(where(is.numeric), ~max(.x, na.rm = TRUE), .names = "max_{.col}")  
  )
)
>> # A tibble: 1 x 6
>>   min_height min_mass min_birth_year max_height max_mass
>>        <int>    <dbl>          <dbl>      <int>    <dbl>
>> 1         66       15              8        264     1358
>> # ... with 1 more variable: max_birth_year <dbl>
```

另外，我们也可以使用 `relocate` 函数来调整列的顺序：

``` r
starwars %>% 
  summarise(across(where(is.numeric), min_max, .names = "{.fn}.{.col}")) %>% 
  relocate(starts_with("min"))
>> # A tibble: 1 x 6
>>   min.height min.mass min.birth_year max.height max.mass
>>        <int>    <dbl>          <dbl>      <int>    <dbl>
>> 1         66       15              8        264     1358
>> # ... with 1 more variable: max.birth_year <dbl>
```

## 当前列

通过 `cur_column` 来获取当前列的名称：

``` r
df <- tibble(x = 1:3, y = 3:5, z = 5:7)
mult <- list(x = 1, y = 10, z = 100)

df %>% mutate(across(all_of(names(mult)), ~ .x * mult[[cur_column()]]))
>> # A tibble: 3 x 3
>>       x     y     z
>>   <dbl> <dbl> <dbl>
>> 1     1    30   500
>> 2     2    40   600
>> 3     3    50   700
```

在 [stackoverflow](https://stackoverflow.com/questions/65543579/can-you-use-dplyr-across-to-iterate-across-pairs-of-columns) 看到一个有意思的例子：

``` r
library(glue)
library(stringr)
df <- data.frame("label" = c('a','b','c','d'),
                 "A" = c(4, 3, 8, 9),
                 "B" = c(10, 0, 4, 1),
                 "error_A" = c(0.4, 0.3, 0.2, 0.1),
                 "error_B" = c(0.3, 0, 0.4, 0.1))
##要成对计算
##get 获取对象的值
df %>% 
    mutate(across(c(A, B), ~ 
     ./get(str_c('error_', cur_column() )), .names = 'R_{.col}' ))
>>   label A  B error_A error_B R_A      R_B
>> 1     a 4 10     0.4     0.3  10 33.33333
>> 2     b 3  0     0.3     0.0  10      NaN
>> 3     c 8  4     0.2     0.4  40 10.00000
>> 4     d 9  1     0.1     0.1  90 10.00000
```

## 其他动词

`across` 也可以和 `dplyr` 的其他动词联用：

-   和 `mutate` 一起用：

    ``` r
    ##将数值变量的范围缩放到0-1
    rescale01 <- function(x) {
    rng <- range(x, na.rm = TRUE)
    (x - rng[1]) / (rng[2] - rng[1])
    }
    df <- tibble(x = 1:4, y = rnorm(4))
    df %>% mutate(across(where(is.numeric), rescale01))
    >> # A tibble: 4 x 2
    >>       x     y
    >>   <dbl> <dbl>
    >> 1 0     1    
    >> 2 0.333 0.475
    >> 3 0.667 0    
    >> 4 1     0.145
    ```

-   和 `distinct` 一起用：

    ``` r
    ##看所有含有颜色变量有多少种类
    starwars %>% distinct(across(contains("color")))
    >> # A tibble: 67 x 3
    >>    hair_color    skin_color  eye_color
    >>    <chr>         <chr>       <chr>    
    >>  1 blond         fair        blue     
    >>  2 <NA>          gold        yellow   
    >>  3 <NA>          white, blue red      
    >>  4 none          white       yellow   
    >>  5 brown         light       brown    
    >>  6 brown, grey   light       blue     
    >>  7 brown         light       blue     
    >>  8 <NA>          white, red  red      
    >>  9 black         light       brown    
    >> 10 auburn, white fair        blue-gray
    >> # ... with 57 more rows
    ```

-   和 `count` 一起用：

    ``` r
    starwars %>% count(across(contains("color")), sort = TRUE)
    >> # A tibble: 67 x 4
    >>    hair_color skin_color eye_color     n
    >>    <chr>      <chr>      <chr>     <int>
    >>  1 brown      light      brown         6
    >>  2 brown      fair       blue          4
    >>  3 none       grey       black         4
    >>  4 black      dark       brown         3
    >>  5 blond      fair       blue          3
    >>  6 black      fair       brown         2
    >>  7 black      tan        brown         2
    >>  8 black      yellow     blue          2
    >>  9 brown      fair       brown         2
    >> 10 none       white      yellow        2
    >> # ... with 57 more rows
    ```

-   和 `filter` 一起用：

    ``` r
    ###没有缺失值的变量的所有行
    starwars %>% filter(across(everything(), ~ !is.na(.x)))
    >> # A tibble: 29 x 14
    >>    name      height  mass hair_color skin_color eye_color
    >>    <chr>      <int> <dbl> <chr>      <chr>      <chr>    
    >>  1 Luke Sky~    172    77 blond      fair       blue     
    >>  2 Darth Va~    202   136 none       white      yellow   
    >>  3 Leia Org~    150    49 brown      light      brown    
    >>  4 Owen Lars    178   120 brown, gr~ light      blue     
    >>  5 Beru Whi~    165    75 brown      light      blue     
    >>  6 Biggs Da~    183    84 black      light      brown    
    >>  7 Obi-Wan ~    182    77 auburn, w~ fair       blue-gray
    >>  8 Anakin S~    188    84 blond      fair       blue     
    >>  9 Chewbacca    228   112 brown      unknown    blue     
    >> 10 Han Solo     180    80 brown      fair       brown    
    >> # ... with 19 more rows, and 8 more variables:
    >> #   birth_year <dbl>, sex <chr>, gender <chr>,
    >> #   homeworld <chr>, species <chr>, films <list>,
    >> #   vehicles <list>, starships <list>
    
    ##选择至少有一列不是 NA 的行
    starwars %>% 
    filter(if_any(everything(), ~ !is.na(.x)))
    >> # A tibble: 87 x 14
    >>    name      height  mass hair_color skin_color eye_color
    >>    <chr>      <int> <dbl> <chr>      <chr>      <chr>    
    >>  1 Luke Sky~    172    77 blond      fair       blue     
    >>  2 C-3PO        167    75 <NA>       gold       yellow   
    >>  3 R2-D2         96    32 <NA>       white, bl~ red      
    >>  4 Darth Va~    202   136 none       white      yellow   
    >>  5 Leia Org~    150    49 brown      light      brown    
    >>  6 Owen Lars    178   120 brown, gr~ light      blue     
    >>  7 Beru Whi~    165    75 brown      light      blue     
    >>  8 R5-D4         97    32 <NA>       white, red red      
    >>  9 Biggs Da~    183    84 black      light      brown    
    >> 10 Obi-Wan ~    182    77 auburn, w~ fair       blue-gray
    >> # ... with 77 more rows, and 8 more variables:
    >> #   birth_year <dbl>, sex <chr>, gender <chr>,
    >> #   homeworld <chr>, species <chr>, films <list>,
    >> #   vehicles <list>, starships <list>
    
    ##选择所有的列都不是NA的行
    starwars %>% 
    filter(if_all(everything(), ~ !is.na(.x)))
    >> # A tibble: 29 x 14
    >>    name      height  mass hair_color skin_color eye_color
    >>    <chr>      <int> <dbl> <chr>      <chr>      <chr>    
    >>  1 Luke Sky~    172    77 blond      fair       blue     
    >>  2 Darth Va~    202   136 none       white      yellow   
    >>  3 Leia Org~    150    49 brown      light      brown    
    >>  4 Owen Lars    178   120 brown, gr~ light      blue     
    >>  5 Beru Whi~    165    75 brown      light      blue     
    >>  6 Biggs Da~    183    84 black      light      brown    
    >>  7 Obi-Wan ~    182    77 auburn, w~ fair       blue-gray
    >>  8 Anakin S~    188    84 blond      fair       blue     
    >>  9 Chewbacca    228   112 brown      unknown    blue     
    >> 10 Han Solo     180    80 brown      fair       brown    
    >> # ... with 19 more rows, and 8 more variables:
    >> #   birth_year <dbl>, sex <chr>, gender <chr>,
    >> #   homeworld <chr>, species <chr>, films <list>,
    >> #   vehicles <list>, starships <list>
    ```

