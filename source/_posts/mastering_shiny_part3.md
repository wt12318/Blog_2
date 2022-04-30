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

shiny 模块和一般的 shiny 区别在于每一个模块都会构建一个命名空间（namespace），而不像普通的 app 那样是共享的（对于每个控件的 ID ，所有的 server 函数都可以通过该 ID 获取输入的内容）；只有具有相同命名空间的函数才可以共享这些变量，那么这些函数就构成了一个模块。因此模块就像一个一个隔离的黑盒子，在模块外面只能获取到模块的输入，下面是一个 app 的例子：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220425162058363.png" style="zoom:50%;" />

### 模块基础

我们以一个非常简单的画直方图的 app 为例：

```R
ui <- fluidPage(
  selectInput("var", "Variable", names(mtcars)),
  numericInput("bins", "bins", 10, min = 1),
  plotOutput("hist")
)
server <- function(input, output, session) {
  data <- reactive(mtcars[[input$var]])
  output$hist <- renderPlot({
    hist(data(), breaks = input$bins, main = input$var)
  }, res = 96)
}
```

一个模块和一个 app 是类似的，由两部分构成：

- 模块 UI 函数：产生 UI
- 模块 server 函数：运行 server 函数内的代码

这两个函数都是以 `id` 作为参数，并将其作为该模块的命名空间。

#### 模块 UI

构建模块 UI 函数需要两步：将 UI 代码放到一个接受 id 为参数的函数中；将之前的 ID 放到 `NS` (NameSpace) 的函数调用中，例如原来 ID 是 `var` ，现在就变成 `NS(id,"var")` ，将上面 app 的 UI 部分构建成模块 UI 为：

```R
histogramUI <- function(id) {
  tagList(
    selectInput(NS(id, "var"), "Variable", choices = names(mtcars)),
    numericInput(NS(id, "bins"), "bins", value = 10, min = 1),
    plotOutput(NS(id, "hist"))
  )
}
```

注意这里的 `tagList` 将多个 UI 控件放在一起，并没有指定布局是什么，我们可以在调用这个模块 UI 时选择合适的布局函数（比如 `fluidRow` , `fluidPage` 等）

#### 模块 server

模块的 server 函数是一个两层的函数，第一层和 UI 类似，以 `id` 作为输入；第二层是一个 `moduleServer` 函数，这个函数和 server 类似，但是需要有个 `id`：

```R
histogramServer <- function(id) {
  moduleServer(id, function(input, output, session) {
    data <- reactive(mtcars[[input$var]])
    output$hist <- renderPlot({
      hist(data(), breaks = input$bins, main = input$var)
    }, res = 96)
  })
}
```

`moduleServer` 函数可以自动附加命名空间，比如在这个函数里面 `input$bins` 就会自动找命名空间为 `id` 的 `input$bins` 变量，不需要像 UI 里面一样加上 `NS`。

现在可以把原来的 app 改写成模块的形式：

```R
histogramApp <- function() {
  ui <- fluidPage(
    histogramUI("hist1")
  )
  server <- function(input, output, session) {
    histogramServer("hist1")
  }
  shinyApp(ui, server)  
}
```

{% note warning %}
注意和 shiny 控件一样，一个模块的 UI 和 server 的 `id` 要一样，不然变量无法获取
{% endnote %}

前面讲过模块就像一个黑盒子，从外面 “看不到” 里面的东西，比如下面这个 app：

 ```R
 ui <- fluidPage(
   histogramUI("hist1"),
   textOutput("out")
 )
 server <- function(input, output, session) {
   histogramServer("hist1")
   output$out <- renderText(paste0("Bins: ", input$bins))
 }
 ```

`output$out` 不会依据 `input$bins` 的值进行更新，因为没有 `input$bins`，只有 `hist1` 模块可以看到见 `input$bins` 这个变量（相当于局部变量）。像函数一样，为了精简 `app.R` 文件，我们可以将这些模块函数放到一个单独的文件中，比如 `R/histogram.R`。

------

再举个例子，构建四个一样的控件，但是命名空间不一样，也就是四个模块：

```R
# module UI
randomUI <- function(id) {
  fluidRow(
    column(width = 1,
           textOutput(NS(id, "val"))),
    column(width = 11,
           actionButton(NS(id, "go"), "Go!"))
  )
}

# module server
randomServer <- function(id) {
  moduleServer(id, function(input, output, session) {
    rand <- eventReactive(input$go, sample(100, 1))
    output$val <- renderText(rand())
  })
}

```

```R
library(shiny)

ui <- fluidPage(
  randomUI("rand1"),
  randomUI("rand2"),
  randomUI("rand3"),
  randomUI("rand4")
)
server <- function(input, output, session) {
  randomServer("rand1")
  randomServer("rand2")
  randomServer("rand3")
  randomServer("rand4")
}
shinyApp(ui, server)  
```

或者：

```R
library(shiny)

# generate app
randomApp <- function() {
  ui <- fluidPage(
    randomUI("rand1"),
    randomUI("rand2"),
    randomUI("rand3"),
    randomUI("rand4")
  )
  server <- function(input, output, session) {
    randomServer("rand1")
    randomServer("rand2")
    randomServer("rand3")
    randomServer("rand4")
  }
  shinyApp(ui, server)  
}

# run app
randomApp()
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/%E5%8A%A8%E7%94%BB249.gif)

------

### 输入和输出

对于模块的 UI 和 server 除了 id 参数外还可以输入其他参数，给模块 UI 其他的参数可以让我们使用同一个模块函数在不同的地方表现多样的 UI 外观，但是模型 UI 函数和普通的函数是一样，因此加上其他的参数并没有什么特殊的地方，而对于模块 server 函数来说更需要注意输入和输出是什么。

#### UI 输入 + server 输出

以一个选择数据集的模块为例：

对模块 UI ，我们添加了一个 `filter` 参数（`is.data.frame` 或者 `is.matrix` 来筛选供选择的数据集）：

```R
datasetInput <- function(id, filter = NULL) {
  names <- ls("package:datasets")
  if (!is.null(filter)) {
    data <- lapply(names, get, "package:datasets")
    names <- names[vapply(data, filter, logical(1))]
  }
  
  selectInput(NS(id, "dataset"), "Pick a dataset", choices = names)
}
```

`vapply` 可以类型判断，返回的值必须和 `FUN.VALUE` (这里就是 logical(1)，也就是逻辑值) 的长度和数据类型一致，因此返回的是一个布尔型标量，依据这个值来选择数据集的名称。

对于模块 server，我们只需要使用 `get` 来依据用户提供的名称提取数据就行了，但是要注意的是模块 server 函数更像一般的函数而不是 server 函数，也就是模块 server 需要返回一个**响应**值（函数最后一个代码的值会被自动返回，因此不需要 return）：

```R
datasetServer <- function(id) {
  moduleServer(id, function(input, output, session) {
    reactive(get(input$dataset, "package:datasets"))
  })
}
```

在 app 中使用模块 server 返回的值直接用赋值语句（`<-`）就可以了：

```R
datasetApp <- function(filter = NULL) {
  ui <- fluidPage(
    datasetInput("dataset", filter = filter),
    tableOutput("data")
  )
  server <- function(input, output, session) {
    data <- datasetServer("dataset")
    output$data <- renderTable(head(data()))
  }
  shinyApp(ui, server)
}
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/%E5%8A%A8%E7%94%BB250.gif)

我们再进一步：让用户基于已选择的数据集来选择列展示；这个 UI 很简单，就是一个 `selectInput` ，但是要注意由于刚开始用户没有选择数据集，因此启动 app 时需要将 `choices` 设定为 `NULL`:

```R
selectVarInput <- function(id) {
  selectInput(NS(id, "var"), "Variable", choices = NULL) 
}
```

模块 server 函数需要另外两个参数：`data` 用来选择变量的数据，这个需要是响应的，可以使用上面那个 `dataset` 模块的输出作为这个参数；`filter` ，用来控制什么样的变量可供用户选择，因此这个参数不需要响应（由开发者定义的），先将选择变量名的函数独立出来：

```R
find_vars <- function(data, filter) {
  names(data)[vapply(data, filter, logical(1))]
}
```

模块 server 中应该使用 `observeEvent` 来响应 `data` 的变化，从而更新 `inputSelect` ，并且返回的仍然是一个可响应的值：

```R
selectVarServer <- function(id, data, filter = is.numeric) {
  moduleServer(id, function(input, output, session) {
    observeEvent(data(), {
      updateSelectInput(session, "var", choices = find_vars(data(), filter))
    })
    
    reactive(data()[[input$var]])
  })
}
```

对于整个 app 来说，此时就有两个模块了，`dataset` 和 `var` 模块，并且 **`var` 模块接受 `dataset` 模块的输出作为输入**：

```R
selectVarApp <- function(filter = is.numeric) {
  ui <- fluidPage(
    datasetInput("data", is.data.frame),
    selectVarInput("var"),
    verbatimTextOutput("out")
  )
  server <- function(input, output, session) {
    data <- datasetServer("data")
    var <- selectVarServer("var", data, filter = filter)
    output$out <- renderPrint(var())
  }
  
  shinyApp(ui, server)
}
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/%E5%8A%A8%E7%94%BB251.gif)

#### Server 输入

从上面的例子也可以看出：我们设计一个模块 server 时需要考虑函数的每个参数是由谁提供的，如果是由其他开发者或者调用这个模块的人提供，那么这种参数就是在 app 的生命周期中不变的，也就是不需要响应（如上面的 `filter` 参数）；如果参数是由用户提供的，那么就需要是响应的（如上面的 `data` 参数）。除了在设计 server 时考虑好，还可以在模块中检查某个值是不是响应的：

```R
selectVarServer <- function(id, data, filter = is.numeric) {
  stopifnot(is.reactive(data))
  stopifnot(!is.reactive(filter))
  
  moduleServer(id, function(input, output, session) {
    observeEvent(data(), {
      updateSelectInput(session, "var", choices = find_vars(data(), filter))
    })
    
    reactive(data()[[input$var]])
  })
}
```

#### 在模块内的模块

模块是可以组合的，比如我们将上面两个模块 (`dataset` 和 `var` ) 组合成一个模块 `datavar` 来进行选择数据和变量：

```R
selectDataVarUI <- function(id) {
  tagList(
    datasetInput(NS(id, "data"), filter = is.data.frame),
    selectVarInput(NS(id, "var"))
  )
}
selectDataVarServer <- function(id, filter = is.numeric) {
  moduleServer(id, function(input, output, session) {
    data <- datasetServer("data")
    var <- selectVarServer("var", data, filter = filter)
    var
  })
}

selectDataVarApp <- function(filter = is.numeric) {
  ui <- fluidPage(
    sidebarLayout(
      sidebarPanel(selectDataVarUI("datavar")),
      mainPanel(verbatimTextOutput("out"))
    )
  )
  server <- function(input, output, session) {
    var <- selectDataVarServer("datavar", filter)
    output$out <- renderPrint(var(), width = 40)
  }
  shinyApp(ui, server)
}
```

------

再进一步，依据我们选择的变量可以绘制上面那个直方图，这里对于模块 server 提供两个额外的参数：`x` 需要绘制的变量和 `title` 直方图的标题，这两个参数都要是可响应的，因此对于标题来说开始需要用 `reavtive` 返回一个常数：

```R
histogramOutputBins <- function(id) {
  numericInput(NS(id, "bins"), "bins", 10, min = 1, step = 1)
}
histogramOutputPlot <- function(id) {
  plotOutput(NS(id, "hist"))
}

histogramServer <- function(id, x, title = reactive("Histogram")) {
  stopifnot(is.reactive(x))
  stopifnot(is.reactive(title))
  
  moduleServer(id, function(input, output, session) {
    output$hist <- renderPlot({
      req(is.numeric(x()))
      main <- paste0(title(), " [", input$bins, "]")
      hist(x(), breaks = input$bins, main = main)
    }, res = 96)
  })
}

histogramApp <- function() {
  ui <- fluidPage(
  sidebarLayout(
    sidebarPanel(
      datasetInput("data", is.data.frame),
      selectVarInput("var"),
      histogramOutputBins("hist")
    ),
    mainPanel(
      histogramOutputPlot("hist")
    )
  )
)
  
  server <- function(input, output, session) {
    data <- datasetServer("data")
    x <- selectVarServer("var", data)
    histogramServer("hist", x)
  }
  shinyApp(ui, server)
} 
histogramApp()
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/%E5%8A%A8%E7%94%BB252.gif)

------

#### 多输出



