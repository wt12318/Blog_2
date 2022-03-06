---
title: R 启动设置
date: 2021-08-05 19:14:18
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

[Managing R with .Rprofile, .Renviron, Rprofile.site, Renviron.site, rsession.conf, and repos.conf – RStudio Support](https://support.rstudio.com/hc/en-us/articles/360047157094-Managing-R-with-Rprofile-Renviron-Rprofile-site-Renviron-site-rsession-conf-and-repos-conf)

[R for Enterprise: Understanding R’s Startup · R Views (rstudio.com)](https://rviews.rstudio.com/2017/04/19/r-for-enterprise-understanding-r-s-startup/)

