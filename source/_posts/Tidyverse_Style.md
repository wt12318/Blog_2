---
title: Tidyverse 代码规范
date: 2022-06-28 19:14:18
tags: code
index_img: img/tidyverse-default.png
categories:
  - 编程
---



Tidyverse 代码规范

<!-- more -->

分为两部分：数据分析和包开发

## 数据分析

### 文件

文件名应该是有意义的并且以 `.R` 结尾；避免在文件名中使用特殊字符（比如空格之类的），较好的选择是使用字母，数字，下划线和 `-` :

```r
# Good
fit_models.R
utility_functions.R

# Bad
fit models.R
foo.r
stuff.r
```

当文件是以特定的次序运行的，应该给文件名加上数字前缀：

```r
00_download.R
01_explore.R
...
09_model.R
10_visualize.R
```

注意文件名的大小写，因为不同的操作系统或者版本控制系统可能对大小写识别的行为不一致（比如 Windows 系统对大小写是不敏感的），因此最好文件名全部用小写表示并且两个文件名不应该只有首字母大小写不一样（比如下图windows上不允许，但是linux上就可以）：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220614155241-j87o1qy.png" style="zoom:50%;" />

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220614155319-ywjjarv.png)


使用 `-` 或者 `==` 来将代码分割成易于阅读的不同代码块，每个代码块可以处理一个或几个相关的任务：

```r
# Load data ---------------------------

# Plot data ---------------------------
```

另外最好在脚本开始就载入所需要的包，而不是将 `library` 分散在脚本的不同部分。

### 语法

#### 对象命名

变量和函数名应该只包含小写字母，数字和下划线 `_` ；使用下划线分割文件名中的不同单词（snake case）：

```r
# Good
day_one
day_1

# Bad
DayOne
dayone
```

基础R在一些函数名和类名中使用了点（比如 `contrib.url()` 和 `data.frame`），但是最好为 S3 面向对象系统保留点号的用法（S3中，方法使用 `function.class` 来命名，不同的类有不同的 S3 方法），而不是在函数名或者类名中使用，容易造成混淆，比如 `as.data.frame.data.frame()` 表示 `data.frame` 类的 `as.data.frame` 方法。

通常来讲，变量名一般是名词，而函数名应该是动词：

```r
# Good
day_one

# Bad
first_day_of_the_month
djm1
```

另外，避免使用常用函数和内置变量作为新的函数名或变量名：

```r
# Bad
T <- FALSE
c <- 10
mean <- function(x) sum(x)
```

#### 空格

在逗号后面加上一个空格：

```r
# Good
x[, 1]

# Bad
x[,1]
x[ ,1]
x[ , 1]
```

对于函数调用：不要在括号前面或者后面添加空格：

```r
# Good
mean(x, na.rm = TRUE)

# Bad
mean (x, na.rm = TRUE)
mean( x, na.rm = TRUE )
```

在 `if`，`for` 或者 `while` 和括号之间加上空格（和 `{` 之间也要加上空格）：

```r
# Good
if (debug) {
  show(x)
}

# Bad
if(debug){
  show(x)
}
```

对于函数的参数，在 `()` 后面加上一个空格：

```r
# Good
function(x) {}

# Bad
function (x) {}
function(x){}
```

Embrace `{{}}` 和里面的内容之间需要加上空格，来强调其特殊的行为：

```r
# Good
max_by <- function(data, var, by) {
  data %>%
    group_by({{ by }}) %>%
    summarise(maximum = max({{ var }}, na.rm = TRUE))
}

# Bad
max_by <- function(data, var, by) {
  data %>%
    group_by({{by}}) %>%
    summarise(maximum = max({{var}}, na.rm = TRUE))
}
```

大部分 infix 操作符（`==`，`+`，`-`，`<-` 等）应该在两端加上空格：

```r
# Good
height <- (feet * 12) + inches
mean(x, na.rm = TRUE)

# Bad
height<-feet*12+inches
mean(x, na.rm=TRUE)
```

但是有一些例外，不应该加空格：

* 有着高优先级的操作符：`::`，`:::`，`$`，`@`，`[`，`[[`，`^`，一元的 `-`，一元的 `+`，`:`:

  ```r
  # Good
  sqrt(x^2 + y^2)
  df$z
  x <- 1:10
  
  # Bad
  sqrt(x ^ 2 + y ^ 2)
  df $ z
  x <- 1 : 10
  ```
* 单侧公式，当右侧是单个变量：

  ```r
  # Good
  ~foo
  tribble(
    ~col1, ~col2,
    "a",   "b"
  )
  
  # Bad
  ~ foo
  tribble(
    ~ col1, ~ col2,
    "a", "b"
  )
  ```

  但是注意当右侧是多个变量的复合表达式时还是需要加上空格的：

  ```r
  # Good
  ~ .x + .y
  
  # Bad
  ~.x + .y
  ```
* tidy 计算符 `!!` 和 `!!!` （因为这些操作符和一元 `-` `+` 的优先级相同）：

  ```r
  # Good
  call(!!xyz)
  
  # Bad
  call(!! xyz)
  call( !! xyz)
  call(! !xyz)
  ```
* `?` :

  ```r
  # Good
  package?stats
  ?mean
  
  # Bad
  package ? stats
  ? mean
  ```

有时候为了对其也可以添加一些额外的空格：

```r
# Good
list(
  total = a + b + c,
  mean  = (a + b + c) / n
)

# Also fine
list(
  total = a + b + c,
  mean = (a + b + c) / n
)
```

#### 函数调用

函数参数一般可以分为两大类，一类是用来计算的数据，另一类是控制计算的细节。当调用函数时，通常会省略数据参数的名称，当需要覆盖参数的默认值时要使用全称，不要用部分匹配：

```r
# Good
mean(1:10, na.rm = TRUE)

# Bad
mean(x = 1:10, , FALSE)
mean(, TRUE, x = c(1:10, NA))
```

避免在函数调用中进行赋值：

```r
# Good
x <- complicated_function()
if (nzchar(x) < 1) {
  # do something
}

# Bad
if (nzchar(x <- complicated_function()) < 1) {
  # do something
}
```

除非某些特别的函数需要这样使用（比如 `capture.output`）：

```r
output <- capture.output(x <- f())
```

#### 控制流

大括号定义了 R代码的层级结构，为了使这种层级结构比较易读，需要：

* `{` 应该是一行的最后一个字符，相关的代码（如 `if`，函数声明等）需要和这个开始的大括号在同一行
* 大括号里面的内容应该缩进两个空格
* `}` 应该是每行的第一个字符

```r
# Good
if (y < 0 && debug) {
  message("y is negative")
}

if (y == 0) {
  if (x > 0) {
    log(x)
  } else {
    message("x is negative or zero")
  }
} else {
  y^x
}

test_that("call1 returns an ordered factor", {
  expect_s3_class(call1(x, y), c("factor", "ordered"))
})

tryCatch(
  {
    x <- scan()
    cat("Total: ", sum(x), "\n", sep = "")
  },
  interrupt = function(e) {
    message("Aborted by user")
  }
)

# Bad
if (y < 0 && debug) {
message("Y is negative")
}

if (y == 0)
{
    if (x > 0) {
      log(x)
    } else {
  message("x is negative or zero")
    }
} else { y ^ x }
```

对于 `if` 需要有几点注意的地方：

* 如果使用 `else`，`else` 应该和右大括号 `}` 在同一行上
* 不要在 if 的判断语句中使用 `&` 和 `|` ，因为这些操作符返回的是向量（逐元素比较，较短的会循环成较长的），应该使用 `&&` 和 `||` （从左到右并且只比较第一个元素，惰性）
* 在 `if` 的条件判断中避免隐式的类型转化（比如从数值转化为逻辑值）：

  ```r
  # Good
  if (length(x) > 0) {
    # do something
  }
  
  # Bad
  if (length(x)) {
    # do something
  }
  ```
* 注意 `ifelse(x, a, b)` 并不是 `if (x) a else b` 的替代，`ifelse` 是向量化的操作，也就是如果 `x` 的长度大于1，那么 `a` 和 `b` 会被循环到 `x` 的长度来匹配：

  ```r
  > ifelse(c(T,T,F),1,2)
  [1] 1 1 2
  ```

如果 `if else` 的代码块比较简单，可以写成一行：

```r
if (x > 10) {
  message <- "big"
} else {
  message <- "small"
}
##==>
message <- if (x > 10) "big" else "small"
```

但是如果有影响控制流的函数调用（比如 `return`，`stop`，`continue` ），那么仍然需要大括号：

```r
# Good
if (y < 0) {
  stop("Y is negative")
}

find_abs <- function(x) {
  if (x > 0) {
    return(x)
  }
  x * -1
}

# Bad
if (y < 0) stop("Y is negative")

if (y < 0)
  stop("Y is negative")

find_abs <- function(x) {
  if (x > 0) return(x)
  x * -1
}
```

`switch` 也是一种常用的流程控制方法，需要注意：

* 避免以位置来选择条件，最好是用名称：

  ```r
  # Good 
  switch(x, 
    a = ,
    b = 1, 
    c = 2,
    stop("Unknown `x`", call. = FALSE)
  )
  #bad
  switch(1,"a","b","c")
  ```
* 每个条件应该放在一行
* 在等号后面需要有个空格

  ```r
  # Bad
  switch(x, a = , b = 1, c = 2)
  switch(x, a =, b = 1, c = 2)
  ```
* 对于不符合条件的输入应该提供一个错误信息，像上面那个 `stop`

#### 长的代码行

每行的代码应该限制在 80 个字符，因为这和合适的打印格式相适应；如果一个函数调用太长而不能在一行中放下，应该在单独的行上写函数名称，参数以及最后的 `)`，以方便阅读和修改:

```r
# Good
do_something_very_complicated(
  something = "that",
  requires = many,
  arguments = "some of which may be long"
)

# Bad
do_something_very_complicated("that", requires, many, arguments,
                              "some of which may be long"
                              )
```

需要注意的是对于不写参数名的参数（前面讲过的一些常用的参数，比如 `data`），需要和函数名在一行:

```rib
map(x, f,
  extra_argument_a = 10,
  extra_argument_b = c(1, 43, 390, 210209)
)
```

也可以将一些紧密联系的参数写在一行：

```r
# Good
paste0(
  "Requirement: ", requires, "\n",
  "Result: ", result, "\n"
)

# Bad
paste0(
  "Requirement: ", requires,
  "\n", "Result: ",
  result, "\n")
```

#### 分号

不要将 `;` 放在一行的末尾，也不要用 `;` 来在一行中分割多个命令。

#### 赋值

使用 `<-` 而不是 `=` 来赋值：

```r
# Good
x <- 5

# Bad
x = 5
```

#### 数据

对于文本数据，尽量使用双引号 `"` ，不使用 `'` ；除非文本中已经有双引号了（并且没有单引号）：

```r
# Good
"Text"
'Text with "quotes"'
'<a href="http://style.tidyverse.org">A link</a>'

# Bad
'Text'
'Text with "double" and \'single\' quotes'
```

使用 `TRUE` / `FALSE` 全称，而不是 `T` / `F` 简写。

#### 注释

每行注释应该由单个注释符加上一个空格：`# ` ；在数据分析代码中，使用注释来记录重要的发现和分析决策（如果需要注释来解释代码在干什么，最好考虑重写代码使其更清晰！）。

### 函数

前面在对象命名中提过，函数名最好是用动词：

```r
# Good
add_row()
permute()

# Bad
row_adder()
permutation()
```

对于太长的函数名和参数定义，可以有两种方法：

* 函数缩进，每个参数一行，缩进匹配函数的开始括号 `(`:

  ```r
  long_function_name <- function(a = "a long argument",
                                 b = "another argument",
                                 c = "another long argument") {
    # As usual code is indented by two spaces.
  }
  ```
* 两个空格缩进，每个参数一行并使用首行2空格缩进：

  ```r
  long_function_name <- function(
      a = "a long argument",
      b = "another argument",
      c = "another long argument") {
    # As usual code is indented by two spaces.
  }
  ```

在两种情况中 `)` 和 `{` 都应该和最后一个参数在同一行，更倾向于使用函数缩进形式。

只有在“提前”返回值的时候才需要使用 `return` 语句，否则依赖 R 来自动返回最后的计算结果：

```r
# Good
find_abs <- function(x) {
  if (x > 0) {
    return(x)
  }
  x * -1
}
add_two <- function(x, y) {
  x + y
}

# Bad
add_two <- function(x, y) {
  return(x + y)
}
```

并且 `return` 语句应该单独一行：

```r
# Good
find_abs <- function(x) {
  if (x > 0) {
    return(x)
  }
  x * -1
}

# Bad
find_abs <- function(x) {
  if (x > 0) return(x)
  x * -1
}
```

如果我们的函数调用是为了一些“副作用”（比如打印，画图，或者保存数据），在函数内部没有使用参数值进行计算，这个时候使用 `return` 就会打印参数值（printed if not assigned）：

```r
test <- function(x){
  print(x)
  return(x)
}
test(1)
[1] 1
[1] 1
```

因此需要使用 `invisible` 将这个打印行为隐藏（do not print when they are not assigned）：

```r
test <- function(x){
  print(x)
  invisible(x)
}
test(1)
[1] 1
```

使用注释来解释为什么，而不是是什么和怎么做，每一行注释应该由单个注释符和一个空格 `# ` :

```r
# Good

# Objects like data frames are treated as leaves
x <- map_if(x, is_bare_list, recurse)


# Bad

# Recurse only with bare lists
x <- map_if(x, is_bare_list, recurse)
```

注释应该是以句子的形式呈现，仅在有至少两个句子时才需要使用点号来表示结束：

```r
# Good

# Objects like data frames are treated as leaves
x <- map_if(x, is_bare_list, recurse)

# Do not use `is.list()`. Objects like data frames must be treated
# as leaves.
x <- map_if(x, is_bare_list, recurse)


# Bad

# objects like data frames are treated as leaves
x <- map_if(x, is_bare_list, recurse)

# Objects like data frames are treated as leaves.
x <- map_if(x, is_bare_list, recurse)
```

### Pipe

管道符 `%>%` 前面需要一个空格并且后面的内容是另起一行（缩进两个空格），这样方便添加新的步骤或者重新组织现有的步骤：

```r
# Good
iris %>%
  group_by(Species) %>%
  summarize_if(is.numeric, mean) %>%
  ungroup() %>%
  gather(measure, value, -Species) %>%
  arrange(value)

# Bad
iris %>% group_by(Species) %>% summarize_all(mean) %>%
ungroup %>% gather(measure, value, -Species) %>%
arrange(value)
```

如果在管道中有函数的参数在一行中放不下，则每行放一个参数并且缩进两个空格：

```r
iris %>%
  group_by(Species) %>%
  summarise(
    Sepal.Length = mean(Sepal.Length),
    Sepal.Width = mean(Sepal.Width),
    Species = n_distinct(Species)
  )
```

只有一步的管道可以放在一行中，但是除非后面要进行扩充，这样的情况最好是写成一个常规的函数调用：

```r
# Good
iris %>% arrange(Species)

iris %>% 
  arrange(Species)

arrange(iris, Species)
```

有些时候一个长的管道操作在函数的参数中使用短的管道是有用的，但是要考虑这样是否更易读，将这样短的管道移到外面并赋一个有意义的名称是否更好：

```r
# Good
x %>%
  select(a, b, w) %>%
  left_join(y %>% select(a, b, v), by = c("a", "b"))

# Better
x_join <- x %>% select(a, b, w)
y_join <- y %>% select(a, b, v)
left_join(x_join, y_join, by = c("a", "b"))
```

注意 `magrittr` 包可以允许我们在使用管道时对于那些没有额外参数的函数可以省略 `()`，尽量避免这种用法：

```r
# Good
x %>% 
  unique() %>%
  sort()

# Bad
x %>% 
  unique %>%
  sort
```

对于将管道的结果赋值，有 3 种方法：

* 变量名和赋值在不同的行：

  ```r
  iris_long <-
    iris %>%
    gather(measure, value, -Species) %>%
    arrange(-value)
  ```
* 变量名和赋值在相同的行：

  ```r
  iris_long <- iris %>%
    gather(measure, value, -Species) %>%
    arrange(-value)
  ```
* 在管道的末尾使用 `->` 来赋值：

  ```r
  iris %>%
    gather(measure, value, -Species) %>%
    arrange(-value) ->
    iris_long
  ```

`magrittr` 包也提供了原位操作符 `%<>%` ，避免使用这种方式：

```r
# Good
x <- x %>% 
  abs() %>% 
  sort()
  
# Bad
x %<>%
  abs() %>% 
  sort()
```

### ggplot2

对于 ggplot2 中的 `+` ，格式建议和上面的管道符是差不多的。`+` 前面有个空格，后面是新行，新行应该缩进两空格；如果从 dplyr 管道来创建 ggplot2 图像，那么只应该有一层缩进：

```r
# Good
iris %>%
  filter(Species == "setosa") %>%
  ggplot(aes(x = Sepal.Width, y = Sepal.Length)) +
  geom_point()

# Bad
iris %>%
  filter(Species == "setosa") %>%
  ggplot(aes(x = Sepal.Width, y = Sepal.Length)) +
    geom_point()

# Bad
iris %>%
  filter(Species == "setosa") %>%
  ggplot(aes(x = Sepal.Width, y = Sepal.Length)) + geom_point()
```

如果 ggplot2 层的函数参数不能在一行中放下，将每个参数放在单独的行中并且有缩进：

```r
# Good
ggplot(aes(x = Sepal.Width, y = Sepal.Length, color = Species)) +
  geom_point() +
  labs(
    x = "Sepal width, in cm",
    y = "Sepal length, in cm",
    title = "Sepal length vs. width of irises"
  ) 

# Bad
ggplot(aes(x = Sepal.Width, y = Sepal.Length, color = Species)) +
  geom_point() +
  labs(x = "Sepal width, in cm", y = "Sepal length, in cm", title = "Sepal length vs. width of irises") 
```

虽然可以在 ggplot2 的 `data` 参数内进行数据的操作，但是最好将数据操作的过程在画图之前独立出来：

```r
# Good
iris %>%
  filter(Species == "setosa") %>%
  ggplot(aes(x = Sepal.Width, y = Sepal.Length)) +
  geom_point()

# Bad
ggplot(filter(iris, Species == "setosa"), aes(x = Sepal.Width, y = Sepal.Length)) +
  geom_point()
```

## R 包

### 文件

第一部分中对文件的建议也使用于 R 包中的文件，下面是一些不同的地方。

对于文件的命名：

* 如果一个文件只有一个函数，那么这个文件应该和该函数同名
* 如果一个文件含有多个相关的函数，那么应该起一个简洁并概况的名字
* 弃用的函数应该放在有着 `deprec` 前缀的文件中
* 兼容的函数应该放在有着 `compat` 前缀的文件中

对于一个文件中含有多个函数，应该把公开的函数（export 的函数）和其文档放在前面，私有的函数放在后面，如果多个函数共享同一个文档，这些函数都需要跟在文档的后面：

```r
# Bad
help_compute <- function() {
  # ... Lots of code ...
}

#' My public function
#'
#' This is where the documentation of my function begins.
#' ...
#' @export
do_something_cool <- function() {
  # ... even more code ...
  help_compute()
}

# Good
#' Lots of functions for doing something cool
#'
#' ... Complete documentation ...
#' @name something-cool
NULL

#' @describeIn something-cool Get the mean
#' @export
get_cool_mean <- function(x) {
  # ...
}

#' @describeIn something-cool Get the sum
#' @export
get_cool_sum <- function(x) {
  # ...
}
```

### 文档

包中代码（和数据）的文档化是非常重要的，实验· `roxygen2` 和 `markdown` 使得代码和其文档邻近。

#### 标题和描述

代码文档的第一行作为标题，简要的描述函数，数据或者类；标题应该是一个句子的形式，但是不要以点号结尾：

```r
#' Combine values into a vector or list
#' 
#' This is a generic function which combines its arguments.
#'
```

不需要用 `@title` 或 `@description` 来显式的注明标题和描述，除非描述有多段或者包含复杂的格式，比如列表：

```r
#' Apply a function to each element of a vector
#'
#' @description
#' The map function transform the input, returning a vector the same length
#' as the input. 
#' 
#' * `map()` returns a list or a data frame
#' * `map_lgl()`, `map_int()`, `map_dbl()` and `map_chr()` return 
#'    vectors of the corresponding type (or die trying); 
#' * `map_dfr()` and `map_dfc()` return data frames created by row-binding 
#'    and column-binding respectively. They require dplyr to be installed.
```

#### 缩进和空行

在 `#'` 后缩进一个空格，如果对于一个 tag 的描述有多行，那么新起的行需要额外的两个空格缩进：

```r
#' @param key The bare (unquoted) name of the column whose values will be used 
#'   as column headings. 
```

另外对于跨多行的 tag（像 `@description` ，`@example` 和 `@section`），可以将这种 tag 放到单独的一行上，具体的内容在接下来的行并且不需要缩进：

```r
#' @examples
#' 1 + 1
#' sin(pi)
```

在不同的部分之间可以根据需要添加空行来分割：

```r
#' @section Tidy data:
#' When applied to a data frame, row names are silently dropped. To preserve,
#' convert to an explicit variable with [tibble::rownames_to_column()].
#'
#' @section Scoped filtering:
#' The three [scoped] variants ([filter_all()], [filter_if()] and
#' [filter_at()]) make it easy to apply a filtering condition to a
#' selection of variables.
```

#### 文档化参数

对于大部分的 tag（像 `@param`，`@seealso`，`@return`）描述的文字应该是句子的形式，并且由大写字母开头，以点号结尾：

```r
#' @param key The bare (unquoted) name of the column whose values will be used 
#'   as column headings. 
```

如果一些函数共享参数，可以使用 `@inhertParams` 来避免内容的重复：

```r
#' @inheritParams function_to_inherit_from
```

#### 交叉链接

将关系密切的一些函数放在 `@seealso` 里面，如果只有一个函数，可以写成句子的形式：

```r
#' @seealso [fct_lump()] to automatically convert the rarest (or most common)
#'   levels to "other".
```

如果有多个推荐组织成列表的形式：

```r
#' @seealso
#' * [tibble()] constructs from individual columns.
#' * [enframe()] converts a named vector into a two-column tibble (names and 
#'   values).
#' * [name-repair] documents the details of name repair.
```

如果有一个相关函数构成的家族，可以使用 `@family` tag 来自动将这些函数添加到 `@seealso` 部分，家族名称要复数；比如在 `dplyr` 中，`arrange`, `filter`, `mutate`, `slice`, `summarize` 这些动词构成 `single table verbs` 家族：

```r
#' @family single table verbs
```


当我们问号的时候看到的就是 `seealso`:

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220619210559-ywcyut5.png" style="zoom:50%;" />

#### R 代码

含有 R 代码的文本应该用反引号标记（markdown）包括：

* 函数名，后面应该跟个括号 `()` ，比如 `tibble()`
* 函数参数，`na.rm`
* 值，比如 `TRUE`, `NA`, `NULL`
* R 代码，`mean(x, na.rm=TRUE)`
* 类名

而对于包名则不要使用代码样式标记，即使包名称出现在句子的开始也不需要首字母大写。

#### 内部函数

内部函数和一般的函数一样进行文档化，区别就是需要添加 `@noRd` 标签来阻止生成相应的 `.Rd` 文件：

```r
#' Drop last
#'
#' Drops the last element from a vector.
#'
#' @param x A vector object to be trimmed.
#'
#' @noRd
```

### 错误信息

一个错误信息应该由问题的一般性陈述开头，然后给出一个对错误的简洁描述。

#### 问题陈述

每个句子应该含有一个短语，并且只提到一个变量；避免复杂的句子，因此更倾向于将信息用列表的方式展示：开始是一个包含语境信息的列表，以关于用户输入的错误信息结束（如果支持 UTF-8 的话，这些列表可以是 ℹ 和 ✖ 开头，如果支持颜色也可以用蓝色和红色，或者使用 `*`）。

* 如果造成问题的原因是清晰的，可以使用 `must`:

```r
dplyr::nth(1:10, "x")
#> Error: `n` must be a numeric vector:
#> ✖ You've supplied a character vector.

dplyr::nth(1:10, 1:2)
#> Error: `n` must have length 1
#> ✖ You've supplied a vector of length 2.
```

* 如果无法说明预期的情况，使用 `can't`:

```r
mtcars %>% pull(b)
#> Error: Can't find column `b` in `.data`.

as_vector(environment())
#> Error: Can't coerce `.x` to a vector.

purrr::modify_depth(list(list(x = 1)), 3, ~ . + 1)
#> Error: Can't find specified `.depth` in `.x`.
```

在产生错误信息时使用 `stop(call.=FALSE)` ，`rlang::abort()`，和 `Rf_errorcall(R_NilValue,...)` 来避免产生错误的函数名称混淆了错误信息，这些函数名称的信息可以通过 `trackback()` 来获得；比如下面的函数是 `dplyr` 中的 `nth` 函数，如果不使用 `abort` 就会显示 `trunc` 错误，但是这个错误对我们来说是没有什么用处的：

```r
nth <- function(x, n, order_by = NULL, default = default_missing(x)) {
  if (length(n) != 1 || !is.numeric(n)) {
    message("`n` must be a single integer.")
  }
  n <- trunc(n)
  
  if (n == 0 || n > length(x) || n < -length(x)) {
    return(default)
  }
  
  # Negative values index from RHS
  if (n < 0) {
    n <- length(x) + n + 1
  }
  
  if (is.null(order_by)) {
    x[[n]]
  } else {
    x[[ order(order_by)[[n]] ]]
  }
}
> nth(c(1:10),"a")
`n` must be a single integer.
Error in trunc(n) : non-numeric argument to mathematical function
```

#### 错误位置

尽量展示出现错误的位置，名称和内容，以更好的帮助用户找到并解决错误：

```r
purrr::map_int(1:5, ~ "x")
Error: Can't coerce element 1 from a character to a integer

# Bad
map_int(1:5, ~ "x")
#> Error: Each result must be a single integer
```

如果错误的来源的不清晰，避免我们对错误来源的观点误导了用户：

```r
##good
pull(mtcars, b)
Error:
! object 'b' not found

# Bad: implies one argument at fault
pull(mtcars, b)
#> Error: Column `b` must exist in `.data`
```

```r
##good 展示信息 不带观点
tibble(x = 1:2, y = 1:3, z = 1)
Error:
! Tibble columns must have compatible sizes.
• Size 2: Existing data.
• Size 3: Column `y`.
ℹ Only values of size one are recycled

##bad 可能是 y 的长度不对
tibble(x = 1:2, y = 1:3, z = 1)
#> Error: Column `x` must be length 1 or 3, not 2 
```

如果有多个错误，或者在不同的参数或实例间不统一，更倾向于使用一个列表来展示信息（上面那个就是一个例子）：

```r
# Good
purrr::reduce2(1:4, 1:2, `+`)
#> Error: `.x` and `.y` must have compatible lengths:
#> ✖ `.x` has length 4
#> ✖ `.y` has length 2

# Bad: harder to scan 貌似现在就是这样
purrr::reduce2(1:4, 1:2, `+`)
#> Error: `.x` and `.y` must have compatible lengths: `.x` has length 4 and 
#> `.y` has length 2
```

如果有问题的地方太多，只需要展示前几个元素：

```r
# Good
#> Error: NAs found at 1,000,000 locations: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ...
```

#### 提示

对于清晰常见的一些错误，可以提供解决问题的提示，使用 `ℹ` （UTF8 支持）或者蓝色（颜色支持）：

```r
dplyr::filter(iris, Species = "setosa")
Error in `dplyr::filter()`:
! We detected a named input.
ℹ This usually means that you've used `=` instead of `==`.
ℹ Did you mean `Species == "setosa"`?

ggplot2::ggplot(ggplot2::aes())
Error in `fortify()`:
! `data` must be a data frame, or other object coercible by `fortify()`, not an S3 object with class uneval.
Did you accidentally pass `aes()` to the `data` argument?
```

提示需要以问号结尾。

#### 标点

* 错误信息应该以句子的形式展现，并且以点号结尾；项目列表的格式应该一致；首字母大写（除非是参数或者列名）
* 在问题陈述中倾向于使用单数：

  ```r
  # Good
  map_int(1:2, ~ "a")
  #> Error: Each result must be coercible to a single integer:
  #> ✖ Result 1 is a character vector.
  
  # Bad
  map_int(1:2, ~ "a")
  #> Error: Results must be coercible to single integers: 
  #> ✖ Result 1 is a character vector
  ```
* 如果有多个问题，最多展示五个：

  ```r
  # BETTER
  map_int(1:10, ~ "a")
  #> Error: Each result must be coercible to a single integer:
  #> ✖ Result 1 is a character vector
  #> ✖ Result 2 is a character vector
  #> ✖ Result 3 is a character vector
  #> ✖ Result 4 is a character vector
  #> ✖ Result 5 is a character vector
  #> ... and 5 more problems
  
  `\`x\``
  ```
* 在问题陈述和展示问题位置之间选择合适的连接符，可以是 `,` 或者 `:`
* 对参数名使用反引号，可以加上 `cloumn` 来避免列名和参数名歧义，比如某个参数表示为 \`x\`，另一个同名的列名用 Column \`x\` 表示
* 每个错误信息的组成应该不超过 80 个字符，对于长的信息不要使用换行符（因为如果控制台比较窄或者比较宽的时候就会显示不正常），尽量使用项目符号将长的信息拆分成几个短的信息

### News

包中任何面向用户的改变都应该在 `NEWS.md` 中有一个对应的列表说明。（文档的微小改变不需要这样做）

#### 列表

项目符号列表的作用是简要的描述改变以让用户了解到包的变化，内容上和 commit 信息类似，但是要以用户的角度去写（而不是开发者）；新的项目符号列表应该添加到文件的开始。

##### 总体风格

将函数的名字尽量放到函数的开始，因为一致的位置便于阅读，并且方便在新的版本释放之前组织列表：

```r
# Good
* `ggsave()` now uses full argument names to avoid partial match warning (#2355).

# Bad
* Fixed partial argument matches in `ggsave()` (#2355).
```

每行限制在 80 个字符，并且以点号结尾；描述现在发生了什么，而不是过去发生了什么（在现在改过来了），使用一般现在时态：

```r
# Good
* `ggsave()` now uses full argument names to avoid partial match warnings (#2355).

# Bad
* `ggsave()` no longer partially matches argument names (#2355).
```

对于错误修复或者微小的改进使用单行的句子就足够描述了，但是如果有一些新的复杂的特征加入，则需要更详细的描述，有些时候还要包含一些代码的例子（使用\```）

##### 致谢

如果一个列表和 Github 的 issue 相关，需要包含该 issue 的编码；如果该贡献是非包作者提出的 PR，需要将用户名加上：

```r
# Good
* `ggsave()` now uses full argument names to avoid partial match warnings 
  (@wch, #2355).

# Bad
* `ggsave()` now uses full argument names to avoid partial match warnings.

* `ggsave()` now uses full argument names to avoid partial match warnings.
  (@wch, #2355)
```

##### 代码风格

函数，参数，文件名应该使用反引号；函数名要包括后面的括号；省略 `the argument` 和 `the function` 指代：

```r
# Good
* In `stat_bin()`, `binwidth` now also takes functions.

# Bad
* In the stat_bin function, "binwidth" now also takes functions.
```

##### 通用模版

下面是 `tidyverse` 包中的 `NEWs` 可以作为模板：

* 新的函数家族：

  ```r
  * Support for ordered factors is improved. Ordered factors throw a warning 
    when mapped to shape (unordered factors do not), and do not throw warnings 
    when mapped to size or alpha (unordered factors do). Viridis is used as 
    default colour and fill scale for ordered factors (@karawoo, #1526).
  
  * `possibly()`, `safely()` and friends no longer capture interrupts: this
    means that you can now terminate a mapper using one of these with
    Escape or Ctrl + C (#314).
  ```
* 新的函数：

  ```r
  * New `position_dodge2()` provides enhanced dogding for boxplots...
  
  * New `stat_qq_line()` makes it easy to add a simple line to a Q-Q plot. 
    This line makes it easier to judge the fit of the theoretical distribution 
    (@nicksolomon).
  ```
* 已有函数的新参数：

  ```r
  * `geom_segment()` gains a `linejoin` parameter.
  ```
* 函数参数行为变更：

  ```r
  * In `separate()`, `col = -1` now refers to the far right position. 
    Previously, and incorrectly, `col = -2` referred to the far-right 
    position.
  ```
* 函数行为变更：

  ```r
  * `map()` and `modify()` now work with calls and pairlists (#412).
  
  * `flatten_dfr()` and `flatten_dfc()` now aborts with informative 
     message if dplyr is not installed (#454).
  
  * `reduce()` now throws an error if `.x` is empty and `.init` is not
    supplied.
  ```

#### 内容组织

##### 开发

在包的开发阶段，新的列表应该添加到文件的最前面：

```r
# haven (development version)

* Second update.

* First update.
```

##### 版本释放

在释放新版本前，`NEWS` 文件应该要仔细校对和修改；每次版本释放应该有一个含有包名和版本号的一级标题：

```r
# modelr 0.1.2

* `data_grid()` no longer fails with modern tidyr (#58).

* New `mape()` and `rsae()` model quality statistics (@paulponcet, #33).

* `rsquare()` use more robust calculation 1 - SS_res / SS_tot rather 
  than SS_reg / SS_tot (#37).

* `typical()` gains `ordered` and `integer` methods (@jrnold, #44), 
  and `...` argument (@jrnold, #42).
```

如果有很多列表，需要考虑以二级标题来组织，一些常用的二级标题：

```r
# package 1.1.0

## Breaking changes

## New features

## Minor improvements and fixes
```

##### 破坏性改动

如果有 API 的破坏性改动（breaking change），这些改动应该以一个单独的部分组织起来，该部分的每个列表应该描述这种改动带来的变化以及如何修复：

```r
## Breaking changes

* `separate()` now correctly uses -1 to refer to the far right position, 
  instead of -2. If you depended on this behaviour, you'll need to condition
  on `packageVersion("tidyr") > "0.7.2"`.
```



参考：

- [Welcome | The tidyverse style guide](https://style.tidyverse.org/)
