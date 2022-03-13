---
title: Shiny基础-2
date: 2021-07-15 19:14:18
tags: 编程
index_img: img/shiny.jpg
categories:
  - R
---

R Shiny 基础知识 参考：[Mastering Shiny](https://mastering-shiny.org/index.html)

<!-- more -->

# 布局

## 单页面布局

布局函数提供了一个 app的可视化结构，布局由函数调用的层次结构创建，其中 R 中的层次结构与生成的 HTML 中的层次结构相匹配，比如：

```R
fluidPage(
  titlePanel("Hello Shiny!"),
  sidebarLayout(
    sidebarPanel(
      sliderInput("obs", "Observations:", min = 0, max = 1000, value = 500)
    ),
    mainPanel(
      plotOutput("distPlot")
    )
  )
)
```

如果只看这些函数的话，有着如下的层级结构：

```R
fluidPage(
  titlePanel(),
  sidebarLayout(
    sidebarPanel(),
    mainPanel()
  )
)
```

我们可以从这个代码就可以想象这个页面的构成：最上面是一个标题栏，然后是一个侧边栏，含有一个滑动条和一个用来展示图像的主页面，所以为了保持代码的可读性，使用这种一致的层级结构是必要的。

### 页面函数

最常用的页面函数是 `fluidPage()`，页面函数会设置好 Shiny 运行需要的 HTML, CSS, 和 JavaScript，另外还有其他的一些页面函数，比如 `fixedPage()` 固定了页面的最大宽度，防止 app 在更大的屏幕上显示不正常；`fillPage()` 填充浏览器的全部高度，在想要绘制占据整个屏幕的图的时候有用。

### 有侧边栏（ sidebar） 的页面

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

这两个界面有一点区别：在 `sidebarPanel` 是有灰色的背景，而使用 `fluidRow` 创建的没有。

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

`tabsetPanel` 能用在任何地方！（如上面在 `mainPanel` 内部）。

### 导航栏

上面讲到的都是利用 tab 进行多页面的水平布局，但有的时候 tab 过多或者 tab 的标题过长就不适合水平放置，这个时候就可以利用导航栏的形式进行竖直排布。和 `tabsetPanel` 类似的是 `navlistPanel` ，不同的是 `navlistPanel` 是将 tab 垂直放置：

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

另一种方法是使用 `navbarPage` ，这个函数和 `tabsetPanel` 一样，也是将 tab 水平排布，但是可以通过  `narbarMenu` 来添加下拉列表，相当于结合了水平排布和竖直排布：

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

## Boostrap

Bootstrap 是一个前端组件库，对于 shiny 来说并不需要过多地关注 Bootstrap， 因为 shiny 函数会自动生成 Bootstrap 兼容的 HTML，但是我们也可以自己进行定制：

- 使用 `bslib::bs_theme()` 定制代码的外观
- 使用 `class` 参数定制一些布局，输入和输出（class 为 bootstrap 的类名）
- 也可以自己写函数产生一些 shiny 未提供的 bootstrap 组件

除了 bootstrap 外，一些 R 包也提供了其他不同的 CSS 框架，比如：

- `shiny.semantic` 基于 Fomantic UI 组件库构建
- `shinyMobile` 基于 framework 7 构建，适用于移动设备
- `shinymaterial` 基于谷歌的 Material design 框架
- `shinydashboard ` 可以使用 shiny 来创建仪表盘 app

## 主题

可以使用 `bslib` 包去修改很多的 Bootstrap  默认选项来创建更加独特的 app 外观。使用 `bslib::bs_theme()` 函数来创建主题，通过布局函数的 `theme` 参数来应用这个主题：

```R
fluidPage(
  theme = bslib::bs_theme(...)
)
```

改变 app 整体外观的最简单的方式就是使用预制的 `bootwatch` [主题]([Bootswatch: Free themes for Bootstrap](https://bootswatch.com/)), 在 `bslib::bs_theme()` 的 `bootwatch` 参数中设置想要的主题名称（` bslib::bootswatch_themes()` 获取可选主题）:

```
#可选的主题
> bslib::bootswatch_themes()
 [1] "cerulean"  "cosmo"     "cyborg"    "darkly"    "flatly"    "journal"   "litera"   
 [8] "lumen"     "lux"       "materia"   "minty"     "pulse"     "sandstone" "simplex"  
[15] "sketchy"   "slate"     "solar"     "spacelab"  "superhero" "united"    "yeti"  
```

```R
ui <- fluidPage(
  theme = bslib::bs_theme(bootswatch = "spacelab"),
  sidebarLayout(
    sidebarPanel(
      textInput("txt", "Text input:", "text here"),
      sliderInput("slider", "Slider input:", 1, 100, 30)
    ),
    mainPanel(
      h1(paste0("Theme: spacelab")),
      h2("Header 2"),
      p("Some text")
    )
  )
)
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220308220045131.png)

也可以使用 `bs_theme` 的其他参数来构建新的主题，比如 `bg` 参数表示背景颜色，`fg` 参数表示前景色，`base_font` 表示字体等：

```R
theme <- bslib::bs_theme(
  bg = "#0b3d91", 
  fg = "white", 
  base_font = "Source Sans Pro"
)
```

为了方便设计主题，该包提供了一个 `bslib::bs_theme_preview(theme)` 函数，可以预览主题的外观：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/%E5%8A%A8%E7%94%BB111.gif)

在定制了 app 的外观后，app 中的绘图风格最好也要一致，`thematic ` 包提供了自动匹配 ggplot2，lattice，和 base R 绘图的风格，只需在 server 函数中调用 `thematic_shiny()` 即可：

```R
library(ggplot2)

ui <- fluidPage(
  theme = bslib::bs_theme(bootswatch = "darkly"),
  titlePanel("A themed plot"),
  plotOutput("plot"),
)

server <- function(input, output, session) {
  thematic::thematic_shiny()
  
  output$plot <- renderPlot({
    ggplot(mtcars, aes(wt, mpg)) +
      geom_point() +
      geom_smooth()
  }, res = 96)
}
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/%E5%8A%A8%E7%94%BB112.gif)

# 图

## 交互性

`plotOutput` 除了展示图形输出外还可以作为鼠标相应事件的输入，因此可以允许用户和图形进行交互。

一个图可以响应四种不同的鼠标事件：单击（click），双击（dblclick），悬浮（hover，鼠标在某处停留一段时间）和笔刷（brush，矩形选择工具）；可以使用 `plotOutput` 中的 `click` 参数来将这些事件转化为 shiny 的输入，比如：`plotOutput("plot", click = "plot_click")` 就会生成一个 `input$plot_click` 的输入标签，可以在 server 中使用，下面是一个例子，当用户点击图上某个点的时候输出该点的坐标：

```R
ui <- fluidPage(
  plotOutput("plot", click = "plot_click"),
  verbatimTextOutput("info")
)

server <- function(input, output) {
  output$plot <- renderPlot({
    plot(mtcars$wt, mtcars$mpg)
  }, res = 96)

  output$info <- renderPrint({
    req(input$plot_click)##保证只在用户产生动作时才会运行下面的代码
    x <- round(input$plot_click$x, 2)
    y <- round(input$plot_click$y, 2)
    cat("[", x, ", ", y, "]", sep = "")
  })
}
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/动画17.gif)

### 点击

点击事件（`input$plot_click`）返回一个内容非常丰富的列表，在上面的例子中用到了 `x` 和 `y` 的坐标，但是这种数据结构在实际中很少用到，常常用的是 `nearPoints()` 函数，返回一个数据框，行是源数据中离我们点击的位置比较近的数据点（默认是 5 个像素以内的所有点）：

```R
ui <- fluidPage(
  plotOutput("plot", click = "plot_click"),
  tableOutput("data")
)
server <- function(input, output, session) {
  output$plot <- renderPlot({
    ggplot(mtcars, aes(wt, mpg)) + geom_point()
  }, res = 96)
  
  output$data <- renderTable({
    req(input$plot_click)
    nearPoints(mtcars, input$plot_click)
  })
}
```

这里给 `nearPoints` 提供了两个参数，用于画图的数据以及点击事件（如果不是用 ggplot2 画的图，还需要提供 `xvar` 表示 x 轴是哪个变量，以及 `yvar` 表示 y 轴的变量），还可以设置 `threshold` 和 `maxpoints` 参数，分别表示离点击点的像素距离以及最多返回多少个点：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/动画18.gif)

另外在 `nearPoints` 中还可以使用参数 `allRows = TRUE` 以及 `addDist = TRUE` 来返回多了两列的原始数据框，多了的两列分别为：

- `dist_` ：表示该行的样本点与所选择的点的距离
- `selected_` ：表示改行的样本点是否被选中（也就是上面例子中展示的行）

关于双击和悬浮，由于用的不多，这里就没有具体讲解，以后用到的时候再学习。

### 笔刷

另一个在图上选择点的方法是使用笔刷（brush），和 `click` 类似，响应笔刷事件的函数为 `brushedPoints()`，下面是一个例子：

```R
ui <- fluidPage(
  plotOutput("plot", brush = "plot_brush"),
  tableOutput("data")
)
server <- function(input, output, session) {
  output$plot <- renderPlot({
    ggplot(mtcars, aes(wt, mpg)) + geom_point()
  }, res = 96)
  
  output$data <- renderTable({
    brushedPoints(mtcars, input$plot_brush)
  })
}
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/动画19.gif)

另外对于 plotOutput 的 `click` 或 `brush` 等参数可以传入 `brushOpts` 而不是字符串，从而可以控制事件的行为（比如笔刷中的填充颜色，限制只能在 x 或 y 方向上进行选择等）：

```R
ui <- fluidPage(
  plotOutput("plot", brush = brushOpts(
    "plot_brush",
    fill = "red",##填充红色
    stroke = "green",##边缘绿色
    direction = c("x")##只能在x方向上选择
  )),
  tableOutput("data")
)
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/动画20.gif)

### 修饰图

到目前为止，只是在另一个输出中展示交互的结果（如在另一个表中输出选择的样本点），但是真正的交互应该是在交互的图上直接展示交互后的结果。这就需要一个技术：`reactiveVal`（这个函数在第 3 部分还会讲），`reactiveval` 和 `reactive()` 类似，通过这个函数创建的响应变量可以当作函数来使用，但是不同之处在于`reactiveval`可以用来更新值:

```R
val <- reactiveVal(10)
val()
#> [1] 10
val(20)
val()
#> [1] 20
val(val() + 1)
val()
#> [1] 21
##实际上直接在终端运行这些命令会报错，因为响应变量只有在响应式环境，比如observeEvent中才可以被执行
```

下面的例子展示了当点击一个点后会显示图上其他点离该点的距离（以点的大小表示）：

```R
set.seed(1014)
df <- data.frame(x = rnorm(100), y = rnorm(100))

ui <- fluidPage(
  plotOutput("plot", click = "plot_click", )
)
server <- function(input, output, session) {
  dist <- reactiveVal(rep(1, nrow(df)))##初始化，全为1
  observeEvent(input$plot_click,
    dist(nearPoints(df, input$plot_click, allRows = TRUE, addDist = TRUE)$dist_)##用nearPoints中的距离值更新 dist  
  )
  
  output$plot <- renderPlot({
    df$dist <- dist()
    ggplot(df, aes(x, y, size = dist)) + 
      geom_point() + 
      scale_size_area(limits = c(0, 1000), max_size = 10, guide = NULL)##标准化点的大小
  }, res = 96)
}
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/动画21.gif)

下面是一个复杂一点的例子，使用笔刷逐步增加所选择的点，和上面一样，先初始化 `reactiveVal` 中的值为 FALSE，然后设置 `brushedPoints()` 的参数 `allRows = TRUE` ，得到 `selected_` 列，最后使用 `selected_` 来更新 `reactiveVal` 的值：

```R
ui <- fluidPage(
  plotOutput("plot", brush = "plot_brush", dblclick = "plot_reset")
)
server <- function(input, output, session) {
  selected <- reactiveVal(rep(FALSE, nrow(mtcars)))##初始化

  observeEvent(input$plot_brush, {
    brushed <- brushedPoints(mtcars, input$plot_brush, allRows = TRUE)$selected_
    selected(brushed | selected())##更新
  })
  observeEvent(input$plot_reset, {
    selected(rep(FALSE, nrow(mtcars)))
  })

  output$plot <- renderPlot({
    mtcars$sel <- selected()
    ggplot(mtcars, aes(wt, mpg)) + 
      geom_point(aes(colour = sel)) +
      scale_colour_discrete(limits = c("TRUE", "FALSE"))##使得在没有选的时候，legend也有两个
  }, res = 96)
}
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/动画22.gif)

## 动态高宽

我们也可以让用户来决定图的高宽，需要分别向 `renderPlot` 的 `height` 和 `weight` 参数提供一个没有参数的函数，返回是高和宽的值（像素），下面是一个简单的例子：

```R
ui <- fluidPage(
  sliderInput("height", "height", min = 100, max = 500, value = 250),
  sliderInput("width", "width", min = 100, max = 500, value = 250),
  plotOutput("plot", width = 250, height = 250)
)
server <- function(input, output, session) {
  output$plot <- renderPlot(
    width = function() input$width,
    height = function() input$height,
    res = 96,
    {
      plot(rnorm(20), rnorm(20))
    }
  )
}
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/%E5%8A%A8%E7%94%BB113.gif)

需要注意的是，在改变图的高宽的时候并没有改变数据，也就是说 `  plot(rnorm(20), rnorm(20))` 并没有重新运行。

## 图片

可以使用 `renderImage()` 来展示已有的图片（非代码绘出的图）：

```R
ui <- fluidPage(
  imageOutput("photo")
)
server <- function(input, output, session) {
  output$photo <- renderImage({
    list(
      src = file.path("../test.webp"),
      contentType = "image/webp",
      width = 500,
      height = 650
    )
  }, deleteFile = FALSE)
}
shinyApp(ui, server)
```

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220309103028686.png" alt="" style="zoom:50%;" />

`renderImage` 需要输入的是一个列表，必须参数是 `src` 表示图片的路径，其他可选参数有：

- `contentType` ：图片的 MIME 格式（[Multipurpose Internet Mail Extensions]([What Is MIME Type? (metadata2go.com)](https://www.metadata2go.com/file-info/mime-type#:~:text=MIME Types are structured in a certain way%2C,(%2F) is used to separate type from subtype.)），如果有后缀就无需提供
- `width` 和 `height` ：图片的宽高
- 其他的 HTML `<img>` 标签参数，如 `class` , `alt` 等

注意 shiny 1.5 之前的版本在渲染完之后会删除图片，因此需要加上 `deleteFile = FALSE` 参数。

# 用户反馈

本章主要介绍对用户的输入进行反馈以及输出程序运行中信息的技术，包括：

- 确认（validation），当输入不正确时提醒用户；
- 信息（notification），输出程序运行的信息；
- 进度条（process bar），展示由多个小步骤构成的耗时操作的细节；
- 对于某些危险的操作给予确定或撤销选项。

## 确认（Validation）

### 确认输入

`shinyFeedback` 包可用来向用户提供额外的反馈，首先需要在 `ui` 中添加 `useShinyFeedback()` 函数，该函数设置了显示错误信息所需要的 HTML 和 JavaScript：

```R
ui <- fluidPage(
  shinyFeedback::useShinyFeedback(),
  numericInput("n", "n", value = 10),
  textOutput("half")
)
```

然后在 `server` 函数中可以使用下面四个函数（输出不同的信息）：``feedback()`, `feedbackWarning()`, `feedbackDanger()`, 和 `feedbackSuccess()` ，这些函数都有 3 个关键的参数：

- `inputId`：和 UI 中的输入 id 相匹配，说明反馈信息应该放在哪里
- `show`：是否显示反馈信息
- `text`：要显示的内容

```R
server <- function(input, output, session) {
  half <- reactive({
    even <- input$n %% 2 == 0 ##是不是偶数
    shinyFeedback::feedbackWarning("n", !even, "Please select an even number")##不是偶数输出信息
    input$n / 2    
  })
  
  output$half <- renderText(half())
}
```

![image-20210719081106770](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210719081106770.png)

可以看到当输入 11 时会出现提示信息，但是这个时候也会输出结果；为了避免这种情况发生，我们需要一个新的工具：`req`函数（required）：

```R
server <- function(input, output, session) {
  half <- reactive({
    even <- input$n %% 2 == 0
    shinyFeedback::feedbackWarning("n", !even, "Please select an even number")
    req(even)
    input$n / 2    
  })
  
  output$half <- renderText(half())
}
```

当 `req` 的输入是 FALSE 时，会认为响应表达式所需要的输入没有全部被满足，从而会告诉 shiny 暂停对输入的响应：

![image-20210719081536513](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210719081536513.png)

下面进一步看一下这个函数：

### 使用 `req` 取消执行

在启动 APP 之后会基于提供的默认值构建一个完整的响应图，但有的时候采取默认值会出现问题，比如下面的三种情况：

- 在 `textInput` 中，使用默认值 `value = ""`，在用户输入之前不会做任何的响应
- 在 `selectInput` 中，使用空的选项作为默认值
- 在 `fileInput` 中，在用户上传文件之前是没有任何结果的

```R
ui <- fluidPage(
  selectInput("language", "Language", choices = c("", "English", "Maori")),
  textInput("name", "Name"),
  textOutput("greeting")
)

server <- function(input, output, session) {
  greetings <- c(
    English = "Hello", 
    Maori = "Kia ora"
  )
  output$greeting <- renderText({
    paste0(greetings[[input$language]], " ", input$name, "!")
  })
}
```

![image-20210719083343793](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210719083343793.png)

上面的 app 中由于我们提供的选择默认值为空，而 greetings 向量中没有这个元素，所以会报错，因此我们想要的是在用户没有输入操作之前是不会运行 app 的，这里就需要 `req` 函数：

```R
server <- function(input, output, session) {
  greetings <- c(
    English = "Hello", 
    Maori = "Kia ora"
  )
  output$greeting <- renderText({
    req(input$language, input$name)
    paste0(greetings[[input$language]], " ", input$name, "!")
  })
}
```

![image-20210719083558854](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210719083558854.png)

![image-20210719083740800](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210719083740800.png)

可以看到在没有输入前是没有任何输出的；`req` 被调用后会发出特殊的信号，造成下游的响应和输出停止执行；`req` 可以以两种方式工作：

- 提供输入 `req(input$x)` ：在 x 被提供的情况下才会继续执行下游的命令
- 提供条件表达式：`req(input$a > 0)` ：在输入的值大于 0 的时候才会继续执行下游的命令

（**可以理解为参数为 TRUE 的时候才会放行，数据存在也是一种 TRUE**）

下面来创建一个相对复杂的 app（和第一部分中的展示 R 内部数据一样），需要用户输入数据集的名称，然后展示该数据集；这个 app 的关键在于需要判断用户输入的数据集的名称是否在内置的数据集中，如果不在，需要打印反馈信息：

```R
ui <- fluidPage(
  shinyFeedback::useShinyFeedback(),
  textInput("dataset", "Dataset name"), 
  tableOutput("data")
)

server <- function(input, output, session) {
  data <- reactive({
    req(input$dataset)
    
    exists <- exists(input$dataset, "package:datasets")
    shinyFeedback::feedbackDanger("dataset", !exists, "Unknown dataset")
    req(exists, cancelOutput = TRUE)

    get(input$dataset, "package:datasets")
  })
  
  output$data <- renderTable({
    head(data())
  })
}
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/动画1.gif)

注意 `req` 的参数 `cancelOutput = TRUE`，这个选项会保留最后一个正确的输入得到的输出，如果设为 FALSE 会清除最后正确的结果：

 ![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/%E5%8A%A8%E7%94%BB221.gif)

### 在输出中确认

上面讲到的都是输出和单个输入相关联，可以在 UI 中放置 `useShinyFeedback`，对应在 `server` 中放置 `feedback**` 函数；但是有时**输出是和多个输入相关联**的，这个时候就不适合在 UI 中放置 `useShinyFeedback`，因为不知道要对哪个输入做出反馈确认，所以解决方法为在输出中使用`validate` 来做出反馈；当 `validate` 被调用时，会停止执行剩下的代码并输出指定的信息：

```R
##当输入为负数并且计算类型是开方或log时，打印信息
ui <- fluidPage(
  numericInput("x", "x", value = 0),
  selectInput("trans", "transformation", 
    choices = c("square", "log", "square-root")
  ),
  textOutput("out")
)

server <- function(input, output, session) {
  output$out <- renderText({
    if (input$x < 0 && input$trans %in% c("log", "square-root")) {
      validate("x can not be negative for this transformation")
    }
    
    switch(input$trans,
      square = input$x ^ 2,
      "square-root" = sqrt(input$x),
      log = log(input$x)
    )
  })
}
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/动画2.gif)

## 通知（Notifications）

当程序运行过程中没有产生问题，但是想要用户知道发生了什么，可以使用 notification。在 shiny 中 `showNotification` 函数可以用来创建 notification，有 3 种方式：

- 短时间的信息，在一定时间后会自动消失
- 在任务开始时展示信息，在结束时移除
- 随着任务的进行逐步更新信息

### 短时 Notification

最简单的方法就是给 `showNotification` 提供一个参数，表示要展示的信息：

```R
ui <- fluidPage(
  actionButton("goodnight", "Good night")
)
server <- function(input, output, session) {
  observeEvent(input$goodnight, {
    showNotification("So long")
    Sys.sleep(1)
    showNotification("Farewell")
    Sys.sleep(1)
    showNotification("Auf Wiedersehen")
    Sys.sleep(1)
    showNotification("Adieu")
  })
}
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/动画3.gif)

默认信息在 5 秒后会自动消失，也可以人为设定（通过 duration 参数设定或者直接点击叉号）；另外设置 `type` 参数可以改变输出信息的文本框的背景色（不同类型的信息）：

```R
server <- function(input, output, session) {
  observeEvent(input$goodnight, {
    showNotification("So long",duration = 1)
    Sys.sleep(1)
    showNotification("Farewell", type = "message")
    Sys.sleep(1)
    showNotification("Auf Wiedersehen", type = "warning")
    Sys.sleep(1)
    showNotification("Adieu", type = "error")
  })
}
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/动画5.gif)

### 在任务完成后移除

要在完成任务后删除信息，需要设置：

- `duration = NULL` 和 `closeButton = FALSE` 使 notification 在任务完成前可见
- 将 `showNotification` 的返回存到一个变量中，然后将其作为 `removeNotification` 的参数，并且最好使用 `on.exit` 函数，用来保证不管是任务成功运行或者报错时 notification 都会被移除

```R
library(shiny)

write.csv(cars,file = "cars.csv")
write.csv(iris,file = "iris.csv")

ui <- fluidPage(
  selectInput("file","Files",choices = c("","cars","iris")),
  tableOutput("data")
)
server <- function(input, output, session) {
  
  dt <- c("cars.csv","iris.csv")
  names(dt) <- c("cars","iris")
  
  data <- reactive({
    id <- showNotification("Reading data...", duration = NULL, closeButton = FALSE)
    on.exit(removeNotification(id), add = TRUE)
    read.csv(dt[input$file][[1]])
  })
  output$data <- renderTable({
    req(input$file)
    head(data())
  })
}
shinyApp(ui, server)
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/动画6.gif)

### 随进度更新

在前面的短时确认中，我们调用了多次 `showNotification` ，产生了多个信息，当一个任务运行时间比较长，并且有多个子任务时，更好的方法是：只显示一个信息，但是随着任务的进度更新这个信息，主要区别就是**将上一个信息的调用作为下一个信息的 ID**：

```R
ui <- fluidPage(
  tableOutput("data")
)

server <- function(input, output, session) {
  notify <- function(msg, id = NULL) {
    showNotification(msg, id = id, duration = NULL, closeButton = FALSE)
  }

  data <- reactive({ 
    id <- notify("Reading data...")
    on.exit(removeNotification(id), add = TRUE)
    Sys.sleep(1)
      
    notify("Reticulating splines...", id = id)
    Sys.sleep(1)
    
    notify("Herding llamas...", id = id)
    Sys.sleep(1)

    notify("Orthogonalizing matrices...", id = id)
    Sys.sleep(1)
        
    mtcars
  })
  
  output$data <- renderTable(head(data()))
}
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/动画7.gif)

## 进度条

对于需要长时间运行的 app 来说，进度条是最直观的方式来展现任务运行的进度，这部分主要使用两个技术来展示进度条：`Shiny` 的自带方法以及 `waiter` 包中的方法。

### Shiny

shiny 中的 `withProcess` 和 `incProgress` 搭配可以创建进度条，主要步骤为：将我们需要进度条显示的任务（一般是循环）用 `withProcess` 包起来，在每一个子任务完成的时候使用 `incProcess` 增加进度条中的进度，类似于：

```R
withProgress({
  for (i in seq_len(step)) {
    x <- function_that_takes_a_long_time(x)
    incProgress(1 / length(step))##incProcess 的第一个参数是进度条增加的数量
  }
})
```

下面是一个例子（使用 `Sys.sleep(0.5)` 模拟需要长时间运行的函数）：

```R
library(shiny)

ui <- fluidPage(
  numericInput("steps", "How many steps?", 10),
  actionButton("go", "go"),
  textOutput("result")
)

server <- function(input, output, session) {
  data <- eventReactive(input$go, {
    withProgress(message = "Computing random number", {
      for (i in seq_len(input$steps)) {
        Sys.sleep(0.5)
        incProgress(1 / input$steps)
      }
      runif(1)
    })
  })
  
  output$result <- renderText(round(data(), 2))
}
shinyApp(ui, server)
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/动画8.gif)

这里的 `eventReactive` 在第一部分讲过，第一个参数是依赖，只有依赖发生改变时第二个参数中的代码才会运行（也就是只有在用户点击率 go 按钮之后才会显示进度条）。

### Waiter

`waiter` 包提供了进度条的更多选项，使用 `waiter` 需要首先在 UI 中添加 `use_waitress()`：

```R
ui <- fluidPage(
  waiter::use_waitress(),
  numericInput("steps", "How many steps?", 10),
  actionButton("go", "go"),
  textOutput("result")
)
```

在 server 中使用 `waiter::Waitress$new` 创建新的进度条，使用 `on.exit(waitress$close())` 使得进度完成时进度条可以被移除，用 `waitress$inc` 代替上面的 `incProgress` 来增加进度：

```R
server <- function(input, output, session) {
  data <- eventReactive(input$go, {
    waitress <- waiter::Waitress$new(max = input$steps)
    on.exit(waitress$close())
    
    for (i in seq_len(input$steps)) {
      Sys.sleep(0.5)
      waitress$inc(1)
    }
    runif(1)
  })
  
  output$result <- renderText(round(data(), 2))
}
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/动画9.gif)

可以看到进度条默认是在页面的顶部出现并且覆盖整个页面，除了这个默认的之外，我们可以选择不同的进度条样式（使用 `theme` 参数，可以是 `overlay`,`overlay-opacity` 或者 `overlay-percent`）;另外也可以选择不覆盖整个页面，而是在某个输入或输出组件中展示（使用 `selector ` 参数），下面是在一个输入框中展示进度条的例子：

```R
library(shiny)

ui <- fluidPage(
  waiter::use_waitress(),
  numericInput("steps", "How many steps?", 10),
  numericInput("shows","process",NULL),
  actionButton("go", "go"),
  textOutput("result")
)

server <- function(input, output, session) {
  data <- eventReactive(input$go, {
    waitress <- waiter::Waitress$new(max = input$steps)
    waitress <- waiter::Waitress$new(max = input$steps,selector = "#shows", theme = "overlay")
    on.exit(waitress$close())
    
    for (i in seq_len(input$steps)) {
      Sys.sleep(0.5)
      waitress$inc(1)
    }
    
    runif(1)
  })
  
  output$result <- renderText(round(data(), 2))
}
shinyApp(ui, server)
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/动画10.gif)

有些时候显示进度条不是一件容易的事（特别是代码中没有循环），因此可以使用动态加载的页面来表示正在运行，需要将上面的 `use_waitress()` 改成 `use_waiter()`:

```R
ui <- fluidPage(
  waiter::use_waiter(),
  actionButton("go", "go"),
  textOutput("result")
)

server <- function(input, output, session) {
  data <- eventReactive(input$go, {
    waiter <- waiter::Waiter$new()
    waiter$show()
    on.exit(waiter$hide())
    
    Sys.sleep(sample(5, 1))
    runif(1)
  })
  output$result <- renderText(round(data(), 2))
}
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/动画11.gif)

## 确认和撤销

防止用户意外进行危险操作的最简单的方法就是给出一个明确的警告，并且需要用户确定，在 shiny 中可以使用 `modalDialog` 来实现，这个函数会创建一个对话框来提示用户。下面是一个对删除文件进行确认的对话框：

```R
modal_confirm <- modalDialog(
  "Are you sure you want to continue?",
  title = "Deleting files",
  footer = tagList(
    actionButton("cancel", "Cancel"),
    actionButton("ok", "Delete", class = "btn btn-danger")
  )
)

ui <- fluidPage(
  actionButton("delete", "Delete all files?")
)

server <- function(input, output, session) {
  observeEvent(input$delete, {
    showModal(modal_confirm)
  })
  
  observeEvent(input$ok, {
    showNotification("Files deleted")
    removeModal()
  })
  observeEvent(input$cancel, {
    removeModal()
  })
}
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/动画12.gif)

`showModal` 和 `removeModal` 可以展示和移除对话框。这里就是当用户点击了按钮就展示对话框，接下来如果选择了 Delete 也就是标签为 ok 的按钮，那么就会展示通知并且移除对话框，如果点击了 Cancel 就会直接移除对话框。

>  什么时候用 observeEvent ，什么时候用 eventReactive？
>
>  当想要对一个事件做出一个行为的时候用 observeEvent ；当想对一个事件的更新算出一个值的时候就用 eventReactive，通常 eventReactive 要赋给一个变量作为响应表达式吗，随后再调用，而 observeEvent 则是直接反应

# 上传和下载

## 上传

### UI

在第一部分已经遇到过上传文件的 UI，也就是 `fileInput()`:

```R
ui <- fluidPage(
  fileInput("upload", "Upload a file")##还可以有一些其他的参数
)
```

### Server

从 UI 的 `fileInput` 中传入 `server` 的是有四列的数据框：

- `name` ：在用户电脑上的原始文件名
- `size` ：文件大小，单位是字节，默认只能输入最大 5 MB 的文件，可以通过设置环境变量 `shiny.maxRequestSize` 来控制，比如：`options(shiny.maxRequestSize = 10 * 1024^2)` 表示最大上传文件大小设置为 10 M
- `type` ：MIME 文件类型，通常是从文件拓展名推测的
- `datapath`：上传后，文件所在的临时位置（和临时文件名）

下面是一个具体的展示：

```R
library(shiny)

ui <- fluidPage(
  fileInput("upload", NULL, buttonLabel = "Upload...", multiple = TRUE),##multiple表示可以多选
  tableOutput("files")
)
server <- function(input, output, session) {
  output$files <- renderTable(input$upload)
}
shinyApp(ui, server)
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/动画13.gif)

### 上传数据

上面只是演示了一下数据读进来的格式，当我们需要获取用户上传的文件中的数据时，有两个细节需要注意：

- `input$upload` 在页面载入时是初始化为 `NULL` 的（`upload` 是 `fileInput` 的标签 ID），因此我们需要使用 `req(input$upload)` 来判断用户是否上传了文件
- `fileInput` 还有一个 `accept` 的参数，这个参数可以用来限制用户上传文件的类型，但是要注意：这种方法只能是对用户的“建议”，用户还可以自己更改（具体见下面的动图）；因此需要我们再次手动确认用户输入的文件格式（可以使用 `tools::file_ext()` 函数来获取文件的拓展名）

下面是一个例子，接受用户上传的 `csv` 或 `tsv` 文件，并读取用户定义的前 `n` 行：

```R
ui <- fluidPage(
  fileInput("upload", NULL, accept = c(".csv", ".tsv")),
  numericInput("n", "Rows", value = 5, min = 1, step = 1),
  tableOutput("head")
)

server <- function(input, output, session) {
  data <- reactive({
    req(input$upload)
    
    ext <- tools::file_ext(input$upload$name)
    switch(ext,
      csv = vroom::vroom(input$upload$datapath, delim = ","),
      tsv = vroom::vroom(input$upload$datapath, delim = "\t"),
      validate("Invalid file; Please upload a .csv or .tsv file")
    )
  })
  
  output$head <- renderTable({
    head(data(), input$n)
  })
}
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/动画14.gif)

## 下载

下载需要在 UI 里面加上 `downloadButton(id)` 或者 `downloadlink(id)`，这两者的页面如下：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/download.png)

在 server 中与之匹配的是 `downloadHandler` :

```R
output$download <- downloadHandler(
  filename = function() {
    paste0(input$dataset, ".csv")
  },
  content = function(file) {
    write.csv(data(), file)
  }
)
```

该函数有两个参数，并且两个参数都是函数：

- `filename` ：一个没有参数的函数，返回的是文件名（字符串），这个函数的任务是生成在下载对话框中展示给用户的文件名
- `content` ：有一个参数的函数，参数是存储文件的路径，这个参数是 shiny 自己创建的临时文件，**不需要我们指定**

下面是一个例子，在内置数据集中选择一个数据集供用户下载：

```R
ui <- fluidPage(
  selectInput("dataset", "Pick a dataset", ls("package:datasets")),
  tableOutput("preview"),
  downloadButton("download", "Download .tsv")
)

server <- function(input, output, session) {
  data <- reactive({
    out <- get(input$dataset, "package:datasets")##获取数据
    if (!is.data.frame(out)) {##判断是不是数据框
      validate(paste0("'", input$dataset, "' is not a data frame"))
    }
    out
  })
  
  output$preview <- renderTable({
    head(data())
  })
    
  output$download <- downloadHandler(
    filename = function() {
      paste0(input$dataset, ".tsv")
    },
    content = function(file) {
      vroom::vroom_write(data(), file)
    }
  )
}
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/动画15.gif)

## 案例

最后是一个小的案例，对用户上传的数据（可以指定分隔符）进行预览，进行一些可选的清理和转化（使用 `janitor` 包），最后可以让用户下载清理后的数据。这里我们可以将整个页面分成三部分，第一部分是上传数据以及对原始数据的预览，第二部分是处理数据以及对处理后的数据的预览，第三部分是下载数据的按钮，因此使用行式布局 `sidebarLayout()`和 `fluidRow()`:

```R
##第一部分
ui_upload <- sidebarLayout(
  sidebarPanel(
    fileInput("file", "Data", buttonLabel = "Upload..."),
    textInput("delim", "Delimiter (leave blank to guess)", ""),
    numericInput("skip", "Rows to skip", 0, min = 0),
    numericInput("rows", "Rows to preview", 10, min = 1)
  ),
  mainPanel(
    h3("Raw data"),
    tableOutput("preview1")
  )
)

##第二部分
ui_clean <- sidebarLayout(
  sidebarPanel(
    checkboxInput("snake", "Rename columns to snake case?"),##蛇形命名法，用下划线将单词连起来
    checkboxInput("constant", "Remove constant columns?"),
    checkboxInput("empty", "Remove empty cols?")
  ),
  mainPanel(
    h3("Cleaner data"),
    tableOutput("preview2")
  )
)

##第三部分
ui_download <- fluidRow(
  column(width = 12, downloadButton("download", class = "btn-block"))
)

##合并三个部分
ui <- fluidPage(
  ui_upload,
  ui_clean,
  ui_download
)

###server
server <- function(input, output, session) {
  # Upload ---------------------------------------------------------
  raw <- reactive({
    req(input$file)
    delim <- if (input$delim == "") NULL else input$delim
    vroom::vroom(input$file$datapath, delim = delim, skip = input$skip)
  })
  output$preview1 <- renderTable(head(raw(), input$rows))
  
  # Clean ----------------------------------------------------------
  tidied <- reactive({
    out <- raw()
    if (input$snake) {
      names(out) <- janitor::make_clean_names(names(out))
    }
    if (input$empty) {
      out <- janitor::remove_empty(out, "cols")
    }
    if (input$constant) {
      out <- janitor::remove_constant(out)
    }
    
    out
  })
  output$preview2 <- renderTable(head(tidied(), input$rows))
  
  # Download -------------------------------------------------------
  output$download <- downloadHandler(
    filename = function() {
      paste0(tools::file_path_sans_ext(input$file$name), ".tsv")##获取不带拓展名的文件名
    },
    content = function(file) {
      vroom::vroom_write(tidied(), file)
    }
  )
}
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/动画16.gif)

# 动态 UI

前面讲的都是**静态UI** ，也就是在 APP 启动后，UI 的界面不会发生改变，这一部分将介绍动态 UI，可以通过在 server 函数中的代码运行来改变 UI。创建动态 UI 有 3 个关键的技术：

- 使用 `update` 系列函数来修改输入控制的参数
- 使用 `tabsetPanel` 来有条件的展示或隐藏 UI 的一部分
- 使用 `uiOutput` 和 `renderUI` 来生成 UI

## 更新输入

`update` 家族的函数可以在输入被创建后更改输入的 UI；每一个输入控制函数，比如 `textInput`， 都和一个 `update` 函数相匹配，如 `updateTextInput`。下面是一个例子来展示在 server 中接受 `numericInput` 的输入来改变 `sliderInput` 展示的最小最大值：

```R
ui <- fluidPage(
  numericInput("min", "Minimum", 0),
  numericInput("max", "Maximum", 3),
  sliderInput("n", "n", min = 0, max = 3, value = 1)
)
server <- function(input, output, session) {
  observeEvent(input$min, {
    updateSliderInput(inputId = "n", min = input$min)
  })  
  observeEvent(input$max, {
    updateSliderInput(inputId = "n", max = input$max)
  })
}
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/%E5%8A%A8%E7%94%BB_0904.gif)

`update` 系列函数的特征是：`inputId` 参数是相应输入的名称，剩下的参数是我们想要改变 UI 的输入组件的参数（如这里的 min 和 max）。

 接下来再看两个例子：

1. 点击 `Reset` 按钮后 Slider 的值归 0：

   ```R
   ui <- fluidPage(
     sliderInput("x1", "x1", 0, min = -10, max = 10),
     sliderInput("x2", "x2", 0, min = -10, max = 10),
     sliderInput("x3", "x3", 0, min = -10, max = 10),
     actionButton("reset", "Reset")
   )
   
   server <- function(input, output, session) {
     observeEvent(input$reset, {##actionButton的状态发生改变就会触发运行
       updateSliderInput(inputId = "x1", value = 0)
       updateSliderInput(inputId = "x2", value = 0)
       updateSliderInput(inputId = "x3", value = 0)
     })
   }
   ```

   ![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/动画_090601.gif)

2. 依据我们输入的值改变按钮的显示：

   ```R
   ui <- fluidPage(
     numericInput("n", "Simulations", 10),
     actionButton("simulate", "Simulate")
   )
   
   server <- function(input, output, session) {
     observeEvent(input$n, {
       label <- paste0("Simulate ", input$n, " times")
       updateActionButton(inputId = "simulate", label = label)
     })
   }
   ```

   ![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/动画23_99.gif)

需要注意，有些时候输出依赖于多个输入，并且这多个输入间也是相互依赖的（根据一个输入来改变另一个输入），此时可能会出现一些我们不想要的中间状态，下面以一个例子说明，这个例子在一个选择框中可以选择数据集，在另一个选择框中根据我们选择的数据来展示可以选择的列，最后输出该列的汇总统计：

```R
ui <- fluidPage(
  selectInput("dataset", "Choose a dataset", c("pressure", "cars")),
  selectInput("column", "Choose column", character(0)),
  verbatimTextOutput("summary")
)

server <- function(input, output, session) {
  dataset <- reactive(get(input$dataset, "package:datasets"))
  
  observeEvent(input$dataset, {
    updateSelectInput(inputId = "column", choices = names(dataset()))
  })
  
  output$summary <- renderPrint({
    summary(dataset()[[input$column]])
  })
}
```



![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/动画24_99.gif)

可以看到会出现一个中间的暂时状态，数据框和列名不匹配，因此出现的结果是`summary(NULL)`：

```R
> summary(NULL)
Length  Class   Mode 
     0   NULL   NULL 
```

> 但是在我自己的电脑上不会出现这个问题，可能是云服务的延迟问题？

可以通过 `freezeReactiveValue()` 来 “冻住” 输入解决这个问题，具体来说就是当 shiny 读取 “冻住” 的输入时会触发运行 `req(False)` （前面讲过 `req(False)` 会停止响应过程），并且不需要我们手动 “解冻” 已经 “冻住” 的值，因为当所有的值改变后，shiny 会自动更新（不是很理解）。在实践中，如果要动态地改变输入值，最好都要使用这个技术：

```R
server <- function(input, output, session) {
  dataset <- reactive(get(input$dataset, "package:datasets"))
  
  observeEvent(input$dataset, {
    freezeReactiveValue(input, "column")
    updateSelectInput(inputId = "column", choices = names(dataset()))
  })
  
  output$summary <- renderPrint({
    summary(dataset()[[input$column]])
  })
}
```

需要注意的是由于 update* 函数会自动更新输入，但是 server 中的行为又会依赖于输入，因此要避免无限循环的出现，比如下面这个 app，每次 `updateNumericInput` 运行时都会改变 `input$n` 的值，造成 `updateNumericInput` 再次运行，陷入无限循环：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/%E5%8A%A8%E7%94%BB223.gif)

## 动态可见性

比较复杂一点的是选择性的部分展示和隐藏 UI ，在 shiny 中可以利用一个小技巧来解决这种问题：用 tabset 来隐藏可选的 UI，比如：

```R
ui <- fluidPage(
  sidebarLayout(
    sidebarPanel(
      selectInput("controller", "Show", choices = paste0("panel", 1:3))
    ),
    mainPanel(
      tabsetPanel(
        id = "switcher",
        type = "hidden",
        tabPanelBody("panel1", "Panel 1 content"),
        tabPanelBody("panel2", "Panel 2 content"),
        tabPanelBody("panel3", "Panel 3 content")
      )
    )
  )
)

server <- function(input, output, session) {
  observeEvent(input$controller, {
    updateTabsetPanel(inputId = "switcher", selected = input$controller)
  })
}
```



![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/%E5%8A%A8%E7%94%BB224.gif)

这里使用了隐藏标签的 tabset （`type = "hidden"`）,并且用 `updateTablesetPanel` 来更新选择的标签达到隐藏标签 UI 的目的。

另外一个需要隐藏 UI 的场景是根据用户的输入展示不同的参数页面（之前的参数页面被隐藏），比如我们想要根据用户选择的分布类型（正态分布，均匀分布，指数分布）来展示可以选择的参数，再根据用户选择的参数来绘制相应的分布图。首先创建一个 `tabsetPanel` ，这个 tabset 有 3 个标签对应 3 个不同的分布参数，但是我们把标签给隐藏起来，然后再把这个 tabset UI 和 `selectInput` ，`numericInput` 合并起来，与绘图区域用侧边栏布局整合，根据用户选择的参数展示相应的 panel 和绘图：

```R
parameter_tabs <- tabsetPanel(
  id = "params",
  type = "hidden",
  tabPanel("normal",
           numericInput("mean", "mean", value = 1),
           numericInput("sd", "standard deviation", min = 0, value = 1)
  ),
  tabPanel("uniform", 
           numericInput("min", "min", value = 0),
           numericInput("max", "max", value = 1)
  ),
  tabPanel("exponential",
           numericInput("rate", "rate", value = 1, min = 0),
  )
)
ui <- fluidPage(
  sidebarLayout(
    sidebarPanel(
      selectInput("dist", "Distribution", 
                  choices = c("normal", "uniform", "exponential")
      ),
      numericInput("n", "Number of samples", value = 100),
      parameter_tabs,
    ),
    mainPanel(
      plotOutput("hist")
    )
  )
)
server <- function(input, output, session) {
  observeEvent(input$dist, {
    updateTabsetPanel(inputId = "params", selected = input$dist)
  }) 
  
  sample <- reactive({
    switch(input$dist,
           normal = rnorm(input$n, input$mean, input$sd),
           uniform = runif(input$n, input$min, input$max),
           exponential = rexp(input$n, input$rate)
    )
  })
  output$hist <- renderPlot(hist(sample()), res = 96)
}
shinyApp(ui, server)
```

下面分别是隐藏了 tabset 的标签和没有隐藏的效果：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220309145928520.png)

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220309150005649.png)

因此这种方法在改变参数时底层的 HTML 并没有消失，只是我们看不到而已。

## 利用代码创建 UI

上面讲到的那些方法只能允许我们去改变已经存在的输入，tabset 只能在已知可能的标签时也可以发挥作用，但是有些时候我们想要根据用户的输入来创建不同类型或者不同数量的输入或者输出 UI，就可以使用这种技术来在 app 运行时创建或修改 UI 界面（之前都是在运行前已经确定了 UI）。这种行为主要通过 `uiOutput` 和 `renderUI` 来实现：

- `uiOutput` 在 UI 中放置一个占位符，等待之后的 server 代码（创建的UI）插入
- `renderUI` 则放在 server 中，利用代码生成相应的 UI 去填充 `uiOutput` 的占位符

下面可以看一个例子：依据用户选择的输入类型（数值或者滑动条）和标签来创建相应的 UI：

```R
ui <- fluidPage(
  textInput("label", "label"),
  selectInput("type", "type", c("slider", "numeric")),
  uiOutput("numeric")
)
server <- function(input, output, session) {
  output$numeric <- renderUI({
    if (input$type == "slider") {
      sliderInput("dynamic", input$label, value = 0, min = 0, max = 10)
    } else {
      numericInput("dynamic", input$label, value = 0, min = 0, max = 10) 
    }
  })
}
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/%E5%8A%A8%E7%94%BB227.gif)

但是如果在 app 中过分依赖这种行为还导致 app 响应速度变慢（因为 app 需要先载入，然后触发一个调用 server 函数的响应事件，接着生成 HTML，将其插入相应的位置），因此如果更关注性能，还是尽量使用固定的 UI。这种方法还有一个其他的问题，从上面的图可以看出，当我们改变输入 UI 的类型后，之前输入的值就会消失，变成默认值 0 了，我们可以通过将新输入的值设置为现有控件的当前值来解决问题：

```R
server <- function(input, output, session) {
  output$numeric <- renderUI({
    value <- isolate(input$dynamic)
    if (input$type == "slider") {
      sliderInput("dynamic", input$label, value = value, min = 0, max = 10)
    } else {
      numericInput("dynamic", input$label, value = value, min = 0, max = 10)
    }
  })
}
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/%E5%8A%A8%E7%94%BB228.gif)

**注意**两点：

- 这里使用了 `isolate`，其实不使用 `isolate` 在行为上看起来是一样的（直接 `value <-input$dynamic`），但是 `isolate` 使得响应图独立出来，也就是当 `input$dynamic` 改变时，`value <-input$dynamic` 并没有重新运行，而是当 `input$type` 或者 `input$label` 改变时才会运行这行代码

- 在 `selectInput` 中需要将 `slider` 和 `numeric` 交换位置，不然就会产生如下报错，原因在于刚启动 shiny 时 `input$dynamic` 为 NULL，而 滑动窗的初始值不能为 NULL：

  ![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/%E5%8A%A8%E7%94%BB229.gif)

### 多个控件

当需要生成不确定数量或者类型（所谓的不确定是对开发者而言不知道用户需要的输入）的控件时使用动态 UI 是比较有用的。对于这种任务使用函数式编程可以可以使得代码更加清晰（比如 purrr 包中的 map 和 reduce 系列函数）。举个例子：我们想要根据用户的输入产生特定的调色板，首先需要使用 `numericInput` 得到用户需要的颜色数目，`uiOutput` 生成对应数量的输入框，使用户可以输入颜色，然后 `plotOutput` 生成需要的调色板。

```R
ui <- fluidPage(
  sidebarLayout(
    sidebarPanel(
      numericInput("n", "Number of colours", value = 5, min = 1),
      uiOutput("col"),
    ),
    mainPanel(
      plotOutput("plot")  
    )
  )
)

server <- function(input, output, session) {
  col_names <- reactive(paste0("col", seq_len(input$n)))
  
  output$col <- renderUI({
    map(col_names(), ~ textInput(.x, NULL, value = isolate(input[[.x]])))
  })
  
  output$plot <- renderPlot({
    cols <- map_chr(col_names(), ~ input[[.x]] %||% "")
    # convert empty inputs to transparent
    cols[cols == ""] <- NA
    
    barplot(
      rep(1, length(cols)), 
      col = cols,
      space = 0, 
      axes = FALSE
    )
  }, res = 96)
}
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/%E5%8A%A8%E7%94%BB230.gif)

- `reactive` 表达式可以用来存储值，这里用 `col_names()` 来存储需要的输入控件的 ID

- 使用 `map` 函数来产生一系列的 `textInput` 控件，这里 value 中使用了 `isolate` 来保留每次的输入，使得在改变控件数量时已输入的内容不会消失：

  ![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/%E5%8A%A8%E7%94%BB247.gif)

- 注意，这里面使用了一种新方法来获取 input 中的元素，之前一直使用的是 `$` ，但是现在元素的名称是一个字符向量，所以需要用 `[[`  来获取元素

- `map_chr` 返回的是字符向量：

  ```R
  x <- list(a="x",b="y")
  map_chr(c("a","b"),~x[[.x]])
  ##[1] "x" "y"
  ```

  但是如果选出的元素是 NULL ，`map_chr` 就会报错：

  ```R
  x <- list(a="x",b="y",c=NULL)
  map_chr(c("a","b","c"),~x[[.x]])
  ##Error: Result 3 must be a single string, not NULL of length 0
  ##Run `rlang::last_error()` to see where the error occurred.
  ```

  而在浏览器渲染成功之前会有一小段的瞬间，此时值为 NULL，因此会出现报错（见下图，可以看到一闪而过的红色报错），所以这里使用了 `%||%` 函数，该函数当左边是 NULL 时会返回右边的值：

  ```R
  > "a" %||% "b"
  [1] "a"
  > NULL %||% "b"
  [1] "b"
  ```

  ![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/%E5%8A%A8%E7%94%BB248.gif)

### 动态筛选

这一部分是一个例子，创建一个 app 可以动态筛选任何数据框：每一个数值变量都有一个相应的滑动条来筛选变量范围，每一个因子变量都有一个多选择框来筛选因子水平；首先需要两个函数，一个根据变量类型来创建 UI，另一个函数接受变量和输入控件返回的值，决定要包括那些观测值（返回和变量长度一样的逻辑向量）：

```R
make_ui <- function(x, var) {
  if (is.numeric(x)) {
    rng <- range(x, na.rm = TRUE)
    sliderInput(var, var, min = rng[1], max = rng[2], value = rng)
  } else if (is.factor(x)) {
    levs <- levels(x)
    selectInput(var, var, choices = levs, selected = levs, multiple = TRUE)
  } else {
    # Not supported
    NULL
  }
}

filter_var <- function(x, val) {
  if (is.numeric(x)) {
    !is.na(x) & x >= val[1] & x <= val[2]
  } else if (is.factor(x)) {
    x %in% val
  } else {
    # No control, so don't filter
    TRUE
  }
}
```

接着结合上面讲到的技术创建 app：

```R
dfs <- keep(ls("package:datasets"), ~ is.data.frame(get(.x, "package:datasets")))

ui <- fluidPage(
  sidebarLayout(
    sidebarPanel(
      selectInput("dataset", label = "Dataset", choices = dfs),
      uiOutput("filter")
    ),
    mainPanel(
      tableOutput("data")
    )
  )
)
server <- function(input, output, session) {
  data <- reactive({
    get(input$dataset, "package:datasets")
  })
  vars <- reactive(names(data()))
  
  output$filter <- renderUI(
    map(vars(), ~ make_ui(data()[[.x]], .x))
  )
  
  selected <- reactive({
    each_var <- map(vars(), ~ filter_var(data()[[.x]], input[[.x]]))
    reduce(each_var, `&`)
  })
  
  output$data <- renderTable(head(data()[selected(), ], 12))
}
```

- `keep` 是 `purrr` 包中的一个函数，第一个参数是列表或者向量，第二个参数是函数，只有第二个参数返回是 TRUE 的列表或者向量元素才会被保留：

  ```R
  > keep(c(1,2,3),~.x>2)
  [1] 3
  ```

- 这里使用了两个 `map` ，第一个 `map` 生成 UI 的列表，第二个 `map` 生成变量的 T/F 列表，然后使用 `reduce` 对列表中的元素向量相应的位置做 `&` 运算，也就是保留全为 T 的观测：

  ```R
  > a <- list(x=c(T,T,F,T),y=c(T,F,F,T))
  > a
  $x
  [1]  TRUE  TRUE FALSE  TRUE
  
  $y
  [1]  TRUE FALSE FALSE  TRUE
  
  > reduce(a,`&`)
  [1]  TRUE FALSE FALSE  TRUE
  ```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/%E5%8A%A8%E7%94%BB250.gif)



# 书签

