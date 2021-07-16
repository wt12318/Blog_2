---
title: Shiny基础-2
date: 2021-07-15 19:14:18
tags: 编程
index_img: img/shiny.jpg

---

Shiny app 基础知识 参考：[Mastering Shiny](https://mastering-shiny.org/index.html)

<!-- more -->

## 单页面布局

布局函数提供了一个 app的可视化结构，布局由函数调用的层次结构创建，其中 R 中的层次结构与生成的 HTML 中的层次结构相匹配。

### 页面函数

最常用的页面函数是 `fluidPage()`，页面函数会设置好 Shiny 运行需要的 HTML, CSS, 和JavaScript，另外还有其他的一些页面函数，比如 `fixedPage()` 固定了页面的最大宽度，`fillPage()` 填充浏览器的全部高度。

### 有sidebar的页面

为了创建更复杂的布局，我们需要在页面函数内部调用布局函数；比如想要创建一个两列布局的界面（左边是输入，右边是输出），就需要使用 `sidebarLayout()` 函数，基本的框架为：

```R
fluidPage(
  titlePanel(
    # app title/description
  ),
  sidebarLayout(
    sidebarPanel(
      # inputs
    ),
    mainPanel(
      # outputs
    )
  )
)
```

会产生如下的布局形式：

<img src="https://d33wubrfki0l68.cloudfront.net/37aa2b1c61a6141cc95188bffd0cfc782fdb27d5/b6aa6/diagrams/action-layout/sidebar.png" alt="Structure of a basic app with sidebar" style="zoom:33%;" />

下面的代码创建一个展示中心极限定理的简单 app 的例子：

```R
ui <- fluidPage(
  titlePanel("Central limit theorem"),
  sidebarLayout(
    sidebarPanel(
      numericInput("m", "Number of samples:", 2, min = 1, max = 100)
    ),
    mainPanel(
      plotOutput("hist")
    )
  )
)
server <- function(input, output, session) {
  output$hist <- renderPlot({
    means <- replicate(1e4, mean(runif(input$m)))
    hist(means, breaks = 20)
  }, res = 96)
}
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210715164555291.png)

`sidebarLayout` 默认 `sidebarPanel` 和 `mainPanel` 是一比二的比例，可以通过 `width` 参数进行更改。

### 多行布局

`sidebarLayout` 是基于更灵活的多行布局来构造的，我们也可以直接使用 shiny 中的多行布局的函数：`fluidRow`；`fluidRow` 创建行，一行的宽度为12，可以在内部使用 `column` 函数来指定每个部件的宽度（因此指定的宽度值必须小于等于12），如果想要创建一个和上面类型的界面可以使用下面的代码：

```R
library(shiny)

## 创建两行，第一行是标题，第二行又分成两列，比例是1：2
ui <- fluidPage(
  fluidRow(
    column(8,
          h1("Central limit theorem")
    )
  ),
  fluidRow(
    column(4,
           numericInput("m", "Number of samples:", 2, min = 1, max = 100)
      ),
    column(8,
           plotOutput("hist")
    )
  )
)
server <- function(input, output, session) {
  output$hist <- renderPlot({
    means <- replicate(1e4, mean(runif(input$m)))
    hist(means, breaks = 20)
  }, res = 96)
}

shinyApp(ui, server)
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210715165706121.png)

这两个界面有一点区别：在 `sidebarPanel` 是有灰色的背景，而使用 `fluidRow` 创建的没有

## 多页面布局

对于一个复杂的 app，将所有的内容放到一个界面上是不合理的，因此需要多页面的布局来组织内容，在 `shiny` 中是通过 `tabPanel` 来实现。

### Tabsets

`tabPanel` 经常是和 `tabsetPanel` 一起使用的，`tabsetPanel` 创建一个多页面的容器，然后可以在里面放置不同的 `tabPanel` 页面：

```R
ui <- fluidPage(
  tabsetPanel(
    tabPanel("Import data", 
      fileInput("file", "Data", buttonLabel = "Upload..."),
      textInput("delim", "Delimiter (leave blank to guess)", ""),
      numericInput("skip", "Rows to skip", 0, min = 0),
      numericInput("rows", "Rows to preview", 10, min = 1)
    ),
    tabPanel("Set parameters"),
    tabPanel("Visualise results")
  )
)
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210715170813633.png)

每个 `tabPanel` 都有一个名称，如果想要知道用户选择了哪个页面，可以在为 `tabsetPanel` 加上标签选项，然后在 `server` 函数中通过标签获取 `tabPanel` 的名称：

```R
ui <- fluidPage(
  sidebarLayout(
    sidebarPanel(
      textOutput("panel")
    ),
    mainPanel(
      tabsetPanel(
        id = "tabset",
        tabPanel("panel 1", "one"),
        tabPanel("panel 2", "two"),
        tabPanel("panel 3", "three")
      )
    )
  )
)
server <- function(input, output, session) {
  output$panel <- renderText({
    paste("Current panel: ", input$tabset)
  })
}
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210715171313171.png)

`tabsetPanel` 能用在任何地方（上面在 `mainPanel` 内部）。

### 导航栏

上面讲到的都是利用 tab 进行多页面的水平布局，但有的时候 tab 过多或者 tab 的标题过长就不适合水平放置，这个时候就可以利用导航栏的形式进行竖直排布。

和 `tabsetPanel` 类似的是 `navlistPanel` ，不同的是 `navlistPanel` 是将 tab 垂直放置：

```R
ui <- fluidPage(
  navlistPanel(
    id = "tabset",
    "Heading 1",
    tabPanel("panel 1", "Panel one contents"),
    "Heading 2",
    tabPanel("panel 2", "Panel two contents"),
    tabPanel("panel 3", "Panel three contents")
  )
)
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210715182754399.png)

另一种方法是使用 `navbarPage` ，这个函数和 `tabsetPanel` 一样，也是将 tab 水平排布，但是可以通过 `narbarMenu` 来添加下拉列表，相当于结合了水平排布和竖直排布：

```R
ui <- navbarPage(
  "Page title",   
  tabPanel("panel 1", "one"),
  tabPanel("panel 2", "two"),
  tabPanel("panel 3", "three"),
  navbarMenu("subpanels", 
    tabPanel("panel 4a", "four-a"),
    tabPanel("panel 4b", "four-b"),
    tabPanel("panel 4c", "four-c")
  )
)
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210715183133798.png)

