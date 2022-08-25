---
title: proj4安装报错解决
date: 2022-07-11 19:14:18
tags: issue_fix
index_img: img/error_message.jpg
categories:
  - tips
---

proj4 包安装

<!-- more -->

正常按照 `install.package` 会报错：

```r
install.packages("proj4")

configure: error: Cannot find working proj.h headers and library.
*** You may need to install libproj-dev or similar! ***

ERROR: configuration failed for package ‘proj4’
* removing ‘/home/data/public/R/library/proj4’
```

提示找不到 `proj.h`，因此通过 `conda` 来安装一下 :

```r
conda install -c conda-forge proj4
```

在 `miniconda` 下的 `include` 中就会出现这个文件，添加到环境变量中：

```r
Sys.setenv("PROJ_LIB"="/root/miniconda3/include/")
install.packages("proj4")
```

但是此时安装还是会报另一种错：

```r
checking whether we are cross compiling... configure: error: in `/tmp/RtmpQwqrNq/R.INSTALL18c82825aafb/proj4':
configure: error: cannot run C++ compiled programs.
If you meant to cross compile, use `--host'.
See `config.log' for more details
ERROR: configuration failed for package ‘proj4’
* removing ‘/home/data/public/R/library/proj4’
```

提示无法运行 C++ 的编译程序，上网搜索发现有一种解决方法 [(20条消息) configure: error: cannot run C compiled programs 解决办法_weixin_34178244的博客-CSDN博客](https://blog.csdn.net/weixin_34178244/article/details/93059566)：

> checking whether the C compiler works... configure: error: in `/home/programming/bootloader/blob-xscale/blob-xscale': configure: error: cannot run C compiled programs. If you meant to cross compile, use `--host'.  
> See `config.log' for more details.
>
> 在源码安装文件的过程中发现这个报错，GCC也安装好了，就是不通过，后查资料，在配置过程中加上一个参数即可解决这个问题。
>
> 参数如何：
>
> ./configure --prefix=/usr/local/XXX ...... --host=arm

通过阅读 `install.package` 文档发现有个 `configure.args` 参数可以控制源码包的编译安装，上网搜索这个参数的用法 [install.packages - pass configure arguments to install packages in R - Stack Overflow](https://stackoverflow.com/questions/37287226/pass-configure-arguments-to-install-packages-in-r)：

> I just stumbled upon this problem myself, trying to install [udunits2](https://cran.r-project.org/web/packages/udunits2/index.html) as a dependency of [ggforce](https://cran.r-project.org/web/packages/ggforce/index.html). [This answer](http://r.789695.n4.nabble.com/install-packages-and-configure-args-tp917554p917555.html) on the R devel mailing list worked in my case: I needed to pass a named character vector to `configure.args` keyed by the  *package name* . This should would work for your case, then:
>
> ```r
> install.packages("Rmpfr",
> configure.args = c(Rmpfr = "--with-mpfr-include=/path/to/mpfr/include"))
> ```

因此尝试：

```r
install.packages("proj4",configure.args = c(proj4 = "--host=arm"))
```

还是有问题：

```r
Error: package or namespace load failed for ‘proj4’ in dyn.load(file, DLLpath = DLLpath, ...):
 unable to load shared object '/home/data/public/R/library/00LOCK-proj4/00new/proj4/libs/proj4.so':
  libproj.so.15: cannot open shared object file: No such file or directory
Error: loading failed
Execution halted
```

提示动态库没有找到，但是由于刚才是通过 `conda` 安装的，所以这个动态库应该是在 `miniconda` 下的 `lib` 里面：

```r
(base) [wt@localhost ~]$ ls miniconda3/lib | grep pro
libproj.a
libproj.so
libproj.so.15
libproj.so.15.1.1
libreproc.so
libreproc++.so
libreproc.so.14
libreproc++.so.14
libreproc.so.14.2.1
libreproc++.so.14.2.1
libtbbmalloc_proxy.so.2
libtbbmalloc_proxy.so.2.3
```

把这个路劲加到 `LD_LIBRARY_PATH` 就行了：

```r
usethis::edit_r_environ()
##在LD_LIBRARY_PATH 末尾添加 /home/wt/miniconda3/lib
## restart Rstudio
install.packages("proj4",configure.args = c(proj4 = "--host=arm"))
```

