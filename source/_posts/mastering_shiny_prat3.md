---
title: Shiny基础-3
date: 2022-03-15 19:14:18
tags: 编程
index_img: img/shiny.jpg
categories:
  - R

---

R Shiny 基础知识 参考：[Mastering Shiny](https://mastering-shiny.org/index.html)

<!-- more -->

这一部分主要是构建 shiny app 的 "best practices" ，涉及以 R 包，Shiny 模块的形式来组织代码，从而方便地构建较大型的 app，自动化测试以及 app 性能的分析与提升。

## 函数

- 在 UI 中，可能会有一些组件重复出现在多个地方，但是改动非常小；我们可以将这种代码组织到函数中以减少重复，还可以使我们在一个地方控制多个组件的行为
- 在 server 中尽量把单个响应抽离出来作为一个一个的函数，更容易进行 debug

另外将代码组织成函数可以将代码分散到多个文件中，而不是创建一个巨大的 `app.R` 文件，这样就更易于管理。

### 文件组织

基于函数的大小，可以有两个地方存放这些函数文件：

- 一些大的函数（以及这些大函数需要的辅助函数）可以放在 `R/{function name}.R` 文件中
- 一些小的，简单的函数可以放在 `R/utils.R` 中，如果直接被 UI 使用也可以放到 `R/ui.R` 中

### UI 函数

举个例子：如果需要创建多个 `sliderInput` 范围都是 0-1，从 0.5 开始，步长为 0.1，最直接的方法就是复制多个有着不同 ID 的 `sliderInput` :

```R
ui <- fluidRow(
  sliderInput("alpha", "alpha", min = 0, max = 1, value = 0.5, step = 0.1),
  sliderInput("beta",  "beta",  min = 0, max = 1, value = 0.5, step = 0.1),
  sliderInput("gamma", "gamma", min = 0, max = 1, value = 0.5, step = 0.1),
  sliderInput("delta", "delta", min = 0, max = 1, value = 0.5, step = 0.1)
)
```

这种情况下我们可以将其组织成一个函数，只有 ID 是需要指定的参数：

```R
sliderInput01 <- function(id) {
  sliderInput(id, label = id, min = 0, max = 1, value = 0.5, step = 0.1)
}

ui <- fluidRow(
  sliderInput01("alpha"),
  sliderInput01("beta"),
  sliderInput01("gamma"),
  sliderInput01("delta")
)
```

如果使用函数式编程则能够更进一步精简代码：

```R
library(purrr)

vars <- c("alpha", "beta", "gamma", "delta")
sliders <- map(vars, sliderInput01)
ui <- fluidRow(sliders)
```

`map` 对每一个 var 中的值都会调用 `sliderInput01` ，返回一个列表；当将列表传递给 `fluidRow` 时会自动将列表拆解，因此每个列表中的元素会成为 `fluidRow` 容器的一个子元素。更进一步，函数式编程可以使我们用数据框来存储函数变量：

```R
vars <- tibble::tribble(
  ~ id,   ~ min, ~ max,
  "alpha",     0,     1,
  "beta",      0,    10,
  "gamma",    -1,     1,
  "delta",     0,     1,
)
```

使用 `tribble` 创建一个和参数对应的 tibble，然后使用 `pmap` 每次对数据框的一行调用函数：

```R
mySliderInput <- function(id, label = id, min = 0, max = 1) {
  sliderInput(id, label, min = min, max = max, value = 0.5, step = 0.1)
}
sliders <- pmap(vars, mySliderInput)
```

### Server 函数

相比较 UI 中函数的主要作用是减少代码的重复，Server 中函数的作用就是隔离和测试（debug），一般来说当一个响应表达式较长时（可能大于 10 行）就需要将其独立为一个不使用响应的函数，下面以一个数据上传的简单 app 为例：

```R
server <- function(input, output, session) {
  data <- reactive({
    req(input$file)
    
    ext <- tools::file_ext(input$file$name)
    switch(ext,
      csv = vroom::vroom(input$file$datapath, delim = ","),
      tsv = vroom::vroom(input$file$datapath, delim = "\t"),
      validate("Invalid file; Please upload a .csv or .tsv file")
    )
  })
  
  output$head <- renderTable({
    head(data(),n=5)
  })
}
```

上面的 app 根据用户上传的文件后缀选择不同的读取方式并展示文件的前5行，我们可以将这个响应给隔离出来作为一个函数，参数是文件名和路径：

```R
load_file <- function(name, path) {
  ext <- tools::file_ext(name)
  switch(ext,
    csv = vroom::vroom(path, delim = ","),
    tsv = vroom::vroom(path, delim = "\t"),
    validate("Invalid file; Please upload a .csv or .tsv file")##validate 和 stop 差不多
  )
}
```

这样就可以把这个函数放到 `R/load_file.R` 文件中，然后在响应表达式中直接调用这个函数：

```R
server <- function(input, output, session) {
  data <- reactive({
    req(input$file)
    load_file(input$file$name, input$file$datapath)
  })
  
  output$head <- renderTable({
    head(data(),n=5)
  })
}
```

通过隔离出没有响应的函数，我们可以直接在 console 中测试或者 debug 这个函数，另外也可以较明确的知道这个函数的输入和输出分别是什么。

## Shiny 模块

















