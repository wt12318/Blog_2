---
title: 利用blogdown创建博客
author: wutao
date: 2021-01-06 10:00:00
categories:
  - R
index_img : /img/hugoCover.jpg
---

使用blogdown来创建静态网页

<!-- more -->

### 创建静态网页

首先需要安装blogdown包

```R
install.packages('blogdown')
```

由于blogdown是基于静态网页生成器Hugo的，所以我们也需要安装Hugo，可以通过blogdown的函数`install_hugo()`来安装：

```R
blogdown::install_hugo()
```

然后在Rstudio中创建一个新的项目，然后使用函数`new_site()`就可以快速的创建一个静态网页(注意，如果不是新创建的项目则需要在空文件夹下运行该命令)，该命令会在当前目录下创建多个文件

![image-20201229200413627](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20201229200413627.png) \  
当我们直接运行`new_site()`时是使用默认的模板("yihui/hugo-lithium"),如果想要使用其他的模板，可以在这个网页中https://themes.gohugo.io/选择想要的模板：

![image-20210106233046381](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210106233046381.png) 

比如这里选择的是Minimal,可以直接点击Minimal就会出来详细的介绍界面：

![image-20210107000102918](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210107000102918.png)

接着点击Download就会跳转到Github界面，这时我们需要记下这个仓库的名称，这里是calintat/minimal，然后就可以回到Rstudio，运行：

```R
new_site(theme = "calintat/minimal")
```

在新版的Rstudio中，在创建项目的时候可以直接选择创建**website using blogdown**   

![image-20210106232404163](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210106232404163.png)

随后在设置的时候可以在Hugo theme选项里面填上calintat/minimal

![image-20210107000355358](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210107000355358.png)

点击create project 我们就可以创建一个静态网页了

在写博客之前，为了随时可以预览所写的内容，我们需要运行`blogdown::serve_site()`函数，该函数使用LiveReload技术，可以实时预览内容(注意只需要在启动Rstudio或者restart session的时候才需要运行这个函数)，另外也可以使用Rstudio的Addin(Serve Site)来运行这个函数：

![image-20210107000650949](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210107000650949.png)

接下来我们就可以开始写博客了，使用`blogdown::new_post()`函数可以创建新的博客，但是需要指定作者，日期;这个时候可以使用Rstudio的New Post(见上图)快捷直观的创建博客：

![image-20210107001022040](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210107001022040.png)

填写相应的信息，按照想要的类型(markdown,Rmarkdown)写作

### 部署静态网页
当我们在本地写好博客之后，有很多方案可以将本地的静态网页部署到web上，这里选用的是Github+Netlify+阿里云(域名)，其他的方案可以查看https://bookdown.org/yihui/blogdown/deployment.html  

我们首先需要在Github上新建一个仓库(这里创建的仓库名为blog_test)用来存放博客，然后将本地的文件和仓库连接并将内容推送到仓库中：
```shell
git init
git remote add origin git@github.com:wt12318/blog_test.git
git add .
git commit -m "init"
git push origin master
```
接着我们需要到Netlify官网：https://www.netlify.com/，没有账号的需要注册一个账号，依次点击New site from Git,选择Github，选择之前创建的仓库，点击Deploy site
![image-20210107002631823](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210107002631823.png)
![image-20210107002724523](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210107002724523.png)
![image-20210107002834588](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210107002834588.png)

等待一会后就部署成功了，netlify会生成一个随机的域名(比如https://pensive-neumann-5a4669.netlify.app),我们可以通过Domain Setting---Edit site name 来修改：

![image-20210107195330204](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210107195330204.png)
![image-20210107195352396](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210107195352396.png)
![image-20210107195413647](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210107195413647.png)
![image-20210107195518614](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210107195518614.png)

总结一下，主要有以下几个步骤：

- 在https://themes.gohugo.io/选择想要的模板
- 在Rstudio中创建新项目(website using blogdown)
- 运行Rstudio的Addin(Serve Site)
- 运行Rstudio的Addin(New Post)新建博客，开始写作
- 部署博客：
  - 创建Github仓库，并将本地内容与仓库连接
  - 将Github的内容部署到Netlify上
- 每次写完博客只需要将内容push到Github上，Netlify就会自动更新


### 绑定域名
通过上面的步骤，Netlify提供的域名都是有后缀的(.netlify.app)，我们也可以绑定自己的域名，具体可以参考这篇文章[Netlify搭建个人博客设置域名](https://blog.csdn.net/mqdxiaoxiao/article/details/96365253)





