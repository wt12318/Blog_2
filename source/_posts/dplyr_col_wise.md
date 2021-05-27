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

-   第一个参数是`.cols` 选择想要操作的列，使用的方法是*tidy selection*
    (也就是和select一样，可以根据位置/名称/类型来选择)

-   第二个参数是`.fns`
    是对每列进行操作的函数，可以是purrr风格的公式(比如\~.x
    /2，具体见[迭代—purrr](sss))

across最常见的是与summarise一起使用(别的动词也可以)：

``` r
starwars %>% 
  summarise(across(where(is.character), ~ length(unique(.x))))
>> # A tibble: 1 x 8
>>    name hair_color skin_color eye_color   sex gender homeworld species
>>   <int>      <int>      <int>     <int> <int>  <int>     <int>   <int>
>> 1    87         13         31        15     5      3        49      38

starwars %>% 
  group_by(species) %>% 
  filter(n() > 1) %>% 
  summarise(across(c(sex, gender, homeworld), ~ length(unique(.x))))
>> `summarise()` ungrouping output (override with `.groups` argument)
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
>> `summarise()` ungrouping output (override with `.groups` argument)
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
>> `summarise()` ungrouping output (override with `.groups` argument)
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
>>   height_min height_max mass_min mass_max birth_year_min birth_year_max
>>        <int>      <int>    <dbl>    <dbl>          <dbl>          <dbl>
>> 1         66        264       15     1358              8            896

starwars %>% summarise(across(c(height, mass, birth_year), min_max))
>> # A tibble: 1 x 6
>>   height_min height_max mass_min mass_max birth_year_min birth_year_max
>>        <int>      <int>    <dbl>    <dbl>          <dbl>          <dbl>
>> 1         66        264       15     1358              8            896
```

我们可以看到默认的输出结果的列名是原来的列名加上函数的名称中间用下划线分割，也可以通过`.name`参数来指定输出的名称(以`glue`包中的格式)：

``` r
starwars %>% summarise(across(where(is.numeric), min_max, .names = "{.fn}.{.col}"))##调换位置，并以点号分割
>> # A tibble: 1 x 6
>>   min.height max.height min.mass max.mass min.birth_year max.birth_year
>>        <int>      <int>    <dbl>    <dbl>          <dbl>          <dbl>
>> 1         66        264       15     1358              8            896

starwars %>% summarise(across(c(height, mass, birth_year), min_max, .names = "{.fn}.{.col}"))
>> # A tibble: 1 x 6
>>   min.height max.height min.mass max.mass min.birth_year max.birth_year
>>        <int>      <int>    <dbl>    <dbl>          <dbl>          <dbl>
>> 1         66        264       15     1358              8            896
```

如果想要同一个函数操作得到的结果放在一起，我们可以把上面两个函数拆开执行：

``` r
starwars %>% summarise(
  across(c(height, mass, birth_year), ~min(.x, na.rm = TRUE), .names = "min_{.col}"),
  across(c(height, mass, birth_year), ~max(.x, na.rm = TRUE), .names = "max_{.col}")
)
>> # A tibble: 1 x 6
>>   min_height min_mass min_birth_year max_height max_mass max_birth_year
>>        <int>    <dbl>          <dbl>      <int>    <dbl>          <dbl>
>> 1         66       15              8        264     1358            896
```

注意：在上面的代码中不能直接使用`where(is.numeric)`
因为第二个across会对新生成的数值变量(“min_height”, “min_mass” and
“min_birth_year”)进行操作:

``` r
starwars %>% summarise(
  across(where(is.numeric), ~min(.x, na.rm = TRUE), .names = "min_{.col}"),
    across(where(is.numeric), ~max(.x, na.rm = TRUE), .names = "max_{.col}")  
)
>> # A tibble: 1 x 9
>>   min_height min_mass min_birth_year max_height max_mass max_birth_year max_min_height max_min_mass
>>        <int>    <dbl>          <dbl>      <int>    <dbl>          <dbl>          <int>        <dbl>
>> 1         66       15              8        264     1358            896             66           15
>> # ... with 1 more variable: max_min_birth_year <dbl>
```

可以看到生成了额外的三列：max_min_height \<int\>, max_min_mass \<dbl\>,
max_min_birth_year \<dbl\>

另外，我们也可以使用`relocate`函数来调整列的顺序：

``` r
starwars %>% 
  summarise(across(where(is.numeric), min_max, .names = "{.fn}.{.col}")) %>% 
  relocate(starts_with("min"))
>> # A tibble: 1 x 6
>>   min.height min.mass min.birth_year max.height max.mass max.birth_year
>>        <int>    <dbl>          <dbl>      <int>    <dbl>          <dbl>
>> 1         66       15              8        264     1358            896
```
