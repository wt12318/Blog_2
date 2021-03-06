---
title: R 启动设置
date: 2022-03-06 19:14:18
tags: 编程
index_img: img/r_start_up.png
categories:
  - R
---

R 及 Rstudio 启动时的环境配置，主要包括 `.Rprofile`, `.Renviron`, `Rprofile.site`, `Renviron.site`, `rsession.conf`, `repos.conf` 等文件

<!-- more -->

最近升级了系统了的 `gcc` 版本，但是在 R 中编译包时还是提示找不到动态库，搜寻了一番才发现原来 Rstudio 中的环境变量不是从 `.bashrc` 中继承的，[r - RStudio shows a different $PATH variable - Stack Overflow](https://stackoverflow.com/questions/31121645/rstudio-shows-a-different-path-variable)：

> When you start R from the command line and then run `system(echo $PATH)`, you are inheriting the Bash environment from your command line session. When you launch RStudio from, say, the Dock or Finder on a Mac or as a system application in Ubuntu, and not from the command line, RStudio does not gets its environment from your `/.bashrc`. Instead it will get the environment variables from system-wide settings. How it finds those system settings will depend on the operating system.

因此学习一下 R 中环境变量的设置是有必要的。

R 启动时有 3 个步骤：启动 R ，设置环境变量，载入 profile 脚本；在后两个步骤中 R 会寻找服务器层面或者用户/项目层面的文件，这 3 个步骤及相关的文件可用下图来表示：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/R_STARTUP.jpeg)

R 启动时的选项和环境变量一般由下表列出的文件控制：

| File            | Who Controls  | Level           | Limitations                     |
| --------------- | ------------- | --------------- | ------------------------------- |
| `.Rprofile`     | User or Admin | User or Project | None, sourced as R code.        |
| `.Renviron`     | User or Admin | User or Project | Set environment variables only. |
| `Rprofile.site` | Admin         | Version of R    | None, sourced as R code.        |
| `Renviron.site` | Admin         | Version of R    | Set environment variables only. |

### .Rprofile

`.Rprofile ` 文件是用户可控制的文件，用来设置选项（一些默认值，如通过 options() 函数设置的全局选项）；`.Rprofile ` 文件可以是用户或者项目级别的，用户级别的则放到用户的 `home` 路径下，而项目级别的则放到特定项目的路径下。虽然有两种 `.Rprofile ` 文件，但是 R 只会载入一个，当存在项目级别的 `.Rprofile `  时会优先载入项目级别，而不是用户级别，因此如果想要载入两个，则需要在项目级别的 `.Rprofile ` 开始就加上 `source("~/.Rprofile")`。从这里我们也可以看出，`.Rprofile` 文件中必须是 R 代码的形式，因此如果想要在这个文件中设置环境变量，必须要使用 `Sys.setenv(key="value")` 命令。

可以方便地使用 `usethis::edit_r_profile()` 函数来编辑 `.Rprofile` 文件，这个函数有一个 `scope` 参数，可以指定修改的是用户（user）还是项目（project）级别的 `.Rprofile` 文件（注意修改后要重启）。

主要注意的是一般不要将影响代码复现性或者可移植性的选项放在 `.Rprofile` 文件中，比如 `options(stringsAsFactors = FALSE)` , 还有读数据，载入包，定义函数等一般也不放在 `profile` 文件中。

### .Renviron

`.Renviron ` 也是一个用户可控制的文件，用来设置环境变量，这个文件不像 `.Rprofile` 是代码形式的，而是键值形式的（key=value），和 `.bashrc` 差不多；在 R 会话里面可以使用 `Sys.getenv("key")` 来获取该文件中对应的环境变量值。和  `.Rprofile`  一样，这个文件也可以是用户级别或者项目级别的，同样地，只会有一个能被导入，如果同时存在用户和项目级别的 `.Renviron ` 文件，优先导入项目级别的文件。

可以使用 `usethis::edit_r_environ()` 函数来编辑 `.Renviron` 文件。

### Rprofile.site 和 Renviron.site

 `.Rprofile` 和  `.Renviron` 都是用户层面或者项目层面的配置文件，相对应的就是`Rprofile.site` 和`Renviron.site `，为系统层面的，是管理员设定的，对于用户来说可以使用  `.Rprofile` 文件中的设置覆盖这些文件中的设置；这些文件可以被放置在 `R_HOME` 的 `etc` 目录下。

------

可以在项目的 `.Rprofile` 文件中加入以下代码，使得每次启动 session 时都会自动生成 `sessionInfoLog` 文件：

```R
.Last <- function(){
  if (interactive()) {
    
    ## check to see if we're in an RStudio project (requires the rstudioapi package)
    if (!requireNamespace("rstudioapi"))
      return(NULL)
    pth <- rstudioapi::getActiveProject()
    if (is.null(pth))
      return(NULL)
    
    ## append date + sessionInfo to a file called sessionInfoLog
    cat("Recording session info into the project's sesionInfoLog file...")
    info <-  capture.output(sessionInfo())
    info <- paste("\n----------------------------------------------",
                  paste0('Session Info for ', Sys.time()),
                  paste(info, collapse = "\n"),
                  sep  = "\n")
    f <- file.path(pth, "sessionInfoLog")
    cat(info, file = f, append = TRUE)
  }
}
```

------

参考：

1. [Managing R with .Rprofile, .Renviron, Rprofile.site, Renviron.site, rsession.conf, and repos.conf – RStudio Support](https://support.rstudio.com/hc/en-us/articles/360047157094-Managing-R-with-Rprofile-Renviron-Rprofile-site-Renviron-site-rsession-conf-and-repos-conf)

2. [R for Enterprise: Understanding R’s Startup · R Views (rstudio.com)](https://rviews.rstudio.com/2017/04/19/r-for-enterprise-understanding-r-s-startup/)

