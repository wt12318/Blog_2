---
title: 下载EGA数据
author: wutao
date: 2021-09-08 10:00:00 
categories:
  - bioinformatics
index_img: img/logo.png
---



通过 pyEGA3 下载 EGA 数据

<!-- more -->

目前比较简单的方法是利用[EGA download client---pyEGA3](https://github.com/EGA-archive/ega-download-client)来下载

### 下载和安装

这个工具是python版本的，在下载的时候需要端口8443和8052开启，可以通过下面的命令来检查：

```{bash}
openssl s_client -connect ega.ebi.ac.uk:8443
openssl s_client -connect ega.ebi.ac.uk:8052
```

如果返回CONNECTED就表示端口可以连接： ![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210125182954515.png)

接下来就需要安装pyEGA3，可以通过pip，conda或者克隆github仓库来安装，这里选择使用mamba来安装(加速版的conda)：

```{bash}
mamba install pyega3
```

要下载controlled的数据，需要先配置一下EGA的用户名和密码：新建一个**credential_file.json**文件(在pyEGA3运行的文件夹下)，在里面填入用户名和密码(需要在EGA上申请)：

```{bash}
{
    "username": "ega-test-data@ebi.ac.uk",
    "password": "egarocks"
}
```

### 使用

查看当前账号下有权限的数据集，会列出可以下载的数据集(如果想要下载的数据集不在里面，需要在EGA上申请)：

```{bash}
pyega3 -cf ./credential_file.json datasets
```

这个工具命令运行后的输出会输出到`pyega3_output.log`中，方便我们查看

也可以查看某个数据集下有哪些文件可以下载(一般是EGAF\*\*\*\*)：

```{bash}
pyega3 -cf credential_file.json files EGAD*****
```

接下来就可以下载某个样本的测序数据了：

```{bash}
nohup pyega3 -c 30 -cf credential_file.json fetch EGAF***** --saveto <要保存的文件名> > nohup.log 2>&1 &
```

`-c`表示要使用的连接数(**connections**)
