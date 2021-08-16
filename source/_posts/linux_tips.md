---
title: Linux & R Tips
date: 2021-08-05 19:14:18
tags: 编程
index_img: img/linux.jpg
---



常用的 Linux Shell ，R 命令/脚本和报错解决方法

<!-- more -->

###  添加用户，改密码，改权限

```shell
#添加用户组dev
groupadd dev
#添加libb用户，并归属于dev组
useradd -g dev libb
#给libb用户改密码
passwd libb
##更改文件权限
chown -R user:group filename
chown -R wt wt/ ##改变文件主
chmod -R 741 wt/ ##3个数字分别表示User、Group、及Other的权限。 r=4，w=2，x=1
```

```shell
##查看用户
cat /etc/passwd|grep -v nologin|grep -v halt|grep -v shutdown|awk -F":" '{ print $1"|"$3"|"$4 }'|more
```

### 动态库无法载入

有时候在 R 里面加载包会出现下面的报错：

> Error: package or namespace load failed for ‘NeoEnrichment’ in dyn.load(file, DLLpath = DLLpath, ...): unable to load shared object '/home/public/R/library/Rsamtools/libs/Rsamtools.so': /lib64/libstdc++.so.6: version `GLIBCXX_3.4.20' not found (required by /home/public/R/library/Rsamtools/libs/Rsamtools.so)

```shell
##错误提示在/lib64/libstdc++.so.6中无法找到GLIBCXX_3.4.20，所以先检查一下/lib64/libstdc++.so.6 链接的库
strings /usr/lib64/libstdc++.so.6 | grep GLIBCXX
GLIBCXX_3.4
GLIBCXX_3.4.1
GLIBCXX_3.4.2
GLIBCXX_3.4.3
GLIBCXX_3.4.4
GLIBCXX_3.4.5
GLIBCXX_3.4.6
GLIBCXX_3.4.7
GLIBCXX_3.4.8
GLIBCXX_3.4.9
GLIBCXX_3.4.10
GLIBCXX_3.4.11
GLIBCXX_3.4.12
GLIBCXX_3.4.13
GLIBCXX_3.4.14
GLIBCXX_3.4.15
GLIBCXX_3.4.16
GLIBCXX_3.4.17
GLIBCXX_3.4.18
GLIBCXX_3.4.19
GLIBCXX_DEBUG_MESSAGE_LENGTH

##然后再在别的地方找找有么有
sudo find / -name "libstdc++.so.6*"

##检查一下
ll 有需要版本的GLIBCXX的库
##将有需要版本的GLIBCXX的库替换掉/lib64/libstdc++.so.6就行了
```

参考：[搞定 libstdc++.so.6: version `GLIBCXX_3.4.21' not found_phd的专栏-CSDN博客](https://blog.csdn.net/phdsky/article/details/84104769)

### 安装 miniconda 和 mamba

```shell
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh
bash Miniconda3-py39_4.9.2-Linux-x86_64.sh
##重启
conda install mamba -n base -c conda-forge
mamba install -c conda-forge htop
conda create -n your_env_name python=x.x
```

### 设置 history 文件显示时间

更改前：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210816153240709.png)

更改：

```shell
##显示时间
HISTTIMEFORMAT="%d/%m/%y %T "
source ~/.bash_profile
##命令数量
HISTFILESIZE=100000
# append to the history file, don't overwrite it
shopt -s histappend
```

更改后：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20210816153333074.png)

### 不改变时间戳移动文件

不改变时间戳（创建，更改文件的时间）在服务器间移动文件：

```shell
scp -p 
```

### 安装 HDF5Array 编译错误

报错： `cannot find -lcrypto`:

```shell
/usr/bin/ld: cannot find -lcrypto
collect2: error: ld returned 1 exit status
make: *** [HDF5Array.so] Error 1
ERROR: compilation failed for package ‘HDF5Array’


[root@localhost ~]# ll /usr/lib64/libssl*
-rwxr-xr-x 1 root root 387696 Apr 28 09:40 /usr/lib64/libssl3.so
lrwxrwxrwx 1 root root     16 Jun  2 01:56 /usr/lib64/libssl.so.10 -> libssl.so.1.0.2k
-rwxr-xr-x 1 root root 470376 Dec 16  2020 /usr/lib64/libssl.so.1.0.2k
[root@localhost ~]# ln -s /usr/lib64/libssl.so.1.0.2k /usr/lib64/libssl.so
[root@localhost ~]# ll /usr/lib64/libssl*
-rwxr-xr-x 1 root root 387696 Apr 28 09:40 /usr/lib64/libssl3.so
lrwxrwxrwx 1 root root     27 Jul 20 01:42 /usr/lib64/libssl.so -> /usr/lib64/libssl.so.1.0.2k
lrwxrwxrwx 1 root root     16 Jun  2 01:56 /usr/lib64/libssl.so.10 -> libssl.so.1.0.2k
-rwxr-xr-x 1 root root 470376 Dec 16  2020 /usr/lib64/libssl.so.1.0.2k
[root@localhost ~]# ln -s /usr/lib64/libcrypto.so.1.0.2k /usr/lib64/libcrypto.so
```

### 释放 swap 内存：

```shell
sync
swapoff -a
swapon -a
free -m
```

### Win10 右键添加新建 markdown 文件

新建一个 txt 文件，填入以下内容:

```txt
Windows Registry Editor Version 5.00

[HKEY_CLASSES_ROOT\.md]
@="markdown"

[HKEY_CLASSES_ROOT\.md\ShellNew]
"NullFile"=""

[HKEY_CLASSES_ROOT\markdown]
@="Blank Markdown file"
```

然后把后缀改成 `.reg` 双击 一路点是 就可以了

### `xargs`命令

使用管道可以方便地将一个命令的输出作为下一个命令的输入，但是有些命令并不支持管道输入，如`ls`,`mkdir`等，这时就可以用`xargs`命令

`xargs`可以接受文件或者命令的输出，将其传递给其他命令，默认是`echo`

`xargs`可以将多行的输出转化成一行，这就是其可以将输出传递给其他命令的基础（当然也可以用不同的参数得到不同格式的输出）

看几个例子：

#### `xargs`和`wc`

```shell
ls *.gtf | xargs wc -l
 42252 1-6.gtf
   42252 1.gtf
       8 test.gtf
   84512 total
   
##ls 将后缀为gtf的文件名给xargs ，xargs将这些文件名传给wc作为其参数
```

#### 使用`-p`选项，在执行的时候会询问

```shell
 echo 'one two three' | xargs -p touch
 # touch one two three ?...y
```

#### 使用`-I`选项，执行多个命令

`-I`后面跟一个字符，在后面的命令中出现相同的字符会被替换掉：

```shell
 cat directories.txt | xargs -I [  echo [ ## [也可以是其他字符
```

如果我们要同时执行多个命令，可以使用`sh -c`：

```shell
 cat directories.txt | xargs -I [ sh -c 'echo [;mkdir ['
```

用这个我们可以改变`xargs`传递的参数的位置（默认是最后一个参数）：

```shell
ls *.jpg | xargs -n1 -I {} cp {} /data/images ##将当前目录下的所有jpg文件复制到/data/images
```

#### 使用`-n`选项，复制文件到多个位置

`-n`后面接数字，表示一次传递的参数的数目

```shell
 echo ./dir_one/ ./dir_two/ | xargs -n 1 cp -v *.gtf ##-v 报告复制过程
 '1-6.gtf' -> './dir_one/1-6.gtf'
'1.gtf' -> './dir_one/1.gtf'
'test.gtf' -> './dir_one/test.gtf'
'1-6.gtf' -> './dir_two/1-6.gtf'
'1.gtf' -> './dir_two/1.gtf'
'test.gtf' -> './dir_two/test.gtf'
```

#### 结合`find`使用

用 `rm` 删除太多的文件时候，可能得到一个错误信息：`/bin/rm Argument list too long`,用 `xargs `去避免这个问题:

```shell
find . -type f -name "*.log" -print0 | xargs -0 rm -f
```

### 如何将和文件夹同名但是有后缀的文件一次全部移动到相应的文件夹

![](https://raw.githubusercontent.com/wt12318/picgo/master/img/20200403174510.png)

如何将`.pbs`文件一次全部移动到相应的文件夹，`cancer`文件中是这些`pbs`的名字,是下面的命令产生的：

```shell
for i in ./*.pbs ;do basename $i;done > cancer
```

由于`xargs`默认将结果作为下个命令的最后的参数，所以：

```shell
cat cancer |while read i;do basename $i .pbs | xargs -n 1 mv $i;done
```

### 一次得到所有癌症类型的样本名

我的所有癌症的`maf`文件都存在一个相应的文件夹中，怎么将这些文件夹中的样本名提取出来，就是去掉后缀名，对一个癌症可以这样做：

```shell
for i in ./luad/*;do basename $i .maf;done > luadsample.txt
```

批量做：

```shell
#!/bin/bash

for i in /public/home/wangshx/wt/tcga_pancancer_dcc_mafs_082115/mafs/maf/*
do 
   for j in $i/*
   do 
     basename $j .maf >> $(echo $i.txt)
   done
done

```

### 遍历文件每一行进行操作

```shell
cat success_submitted_list.txt |while read i ;do mv $i ./02 ;done
```

### 提取文件名并去掉后缀

```shell
for i in ./luad/*;do basename $i _out.txt;done > sample
```

### `grep`匹配

`grep`的基本模式是：`grep 'pattern' file_to_search` 并且会打印出匹配上`pattern`的行

`gerp`常用的参数有：

- `grep -w` 匹配整个字符：

  ```shell
  echo 'my dog is brown' > sample.txt
  
  grep -w 'dog' sample.txt ##my dog is brown
  grep -w 'do' sample.txt ##not output
  ```

- `grep -i` 匹配时大小写不敏感

- `grep -v`返回不匹配的行

- `grep -o` 只返回匹配的字符，而不是一整行

- `grep -c` 对匹配的行进行计数，相当于：`grep 'pattern' file | wc -l`

- 一些参数返回匹配行的前/后行：

  - `grep -A [n] ` 返回匹配的行和之后的n行
  - `grep -B [n]` 返回匹配的行和之前的n行
  - `grep -C [n]` 返回匹配的行和之前的n行，之后的n行

- `grep -f ` 后面可以接一个文件，里面有多个要匹配的模式，一个模式一行，也就是：`grep -f patterns.txt file.txt`

```shell
grep -v "#" 1.gtf | grep -v "^$" | wc -l ##除去注释行和空行之后的行数，因为`$`是行尾的标记
```

```shell
grep -v '^#' 1.gtf | awk '{print $3, $5-$4+1}' | head ##计算每种特征的长度
gene 1152
transcript 1152
exon 1152
CDS 1149
start_codon 3
stop_codon 3
gene 75
transcript 75
exon 75
CDS 72
```

```shell
grep -v '^#' 1.gtf | awk '{print $3}' | sort | uniq -c ##这个就是看gtf文件中的feature类型的数量
   7050 CDS
   7553 exon
   7126 gene
   6700 start_codon
   6692 stop_codon
   7126 transcript
```

找出文件2中有而文件1中没有的行：

```shell
grep -vf file1 file2
```

### 文件上下翻转和左右翻转

- ```tac``` 文件上下翻转，也就是第一行变为最后一行，第二行 变成倒数第二行

  ```shell
  ##首先创建一个文本
  touch taccommand
  vi taccommand
  #first
  #second
  #third
  cat taccommand | tac
  #third
  #second
  #first
  
  ##我们也可以直接输入文本，使用<<命令
  cat <<END | tac##<<后接的就是终止符,这种方式并不创建一个文本
  
  ```

- ```rev``` 将每列反转，也就是将第一个字符变成最后一个

  ```shell
  cat <<a | rev
  #> abcde 
  #> xyz 
  #> a
  #edcba 
  #zyx
  ```

### 使用```rename```批量重命名

```shell
touch test1.fa test2.fa 
rename 'test' 'new' test*.fa

##批量提取一定长度的文件名，不改变后缀，重命名
for i in *.pbs ; do rename `basename $i .pbs` `echo ${i:0:8}` $i ;done 

#也可以使用
rename 's/names/replace/' file
```

### 一些有用的快捷操作

- ! 加之前输入过的命令的前几个字母，快速获取前面的命令
- 回到命令的行首可以用``ctrl a``或者``home``键，回到行尾可以用``ctrl e``或者``end``键；比如我写好一个命令但是却不想运行了可以回到行首加上#再回车
- ```!!```表示上一条命令
- 替换上一个命令中的字符，再运行一遍命令，用于需要对多个文件执行同样的命令，又不想写循环的情况，使用```:gs(global substitute)``` 如：```!!:gs/test1/test2``` 将上一条命令中的test1替换成test2再运行一次

### 标准输入，标准输出与标准错误

主要是3个符号：``<`` ``>`` ``-``

标准输出与标准错误都是输出到屏幕上的

可以用```>``` 将标准输出重定向到某个文件

```2 >&1``` 将标准错误重定向到标准输出，如果已经将标准输出定向到某个文件，那么这个命令可以将标准错误与标准输出定向到一个文件（**Linux 终端用 2 表示标准错误，1 表示标准输出**）

``<``表示标准输入 后面接可以产生输出的命令 可以将输出转为输入；也可以用```-```表示标准输入

下面看一个例子

```shell
##我们先写一个脚本
#!/bin/bash
echo "i am std output"
unexisted_command

bash stdout_error.sh
#i am std output,这个就是标准输出
#stdout_error.sh: line 4: unexisted_command: command not found，这个就是标准错误

bash stdout_error.sh > stdout_error.stdout##将标准输出重定向到文件，但是标准错误还会在屏幕上显示
bash stdout_error.sh > stdout_error.stdout 2> stdout_error.stderr#将输出和错误都重定向到文件，注意2和>之间没有空格
bash stdout_error.sh > stdout_error.stdout 2>&1 ##将输出和错误重定向到一个文件

cat <<END | diff - <(echo "1 2 3" | tr ' ' '\n') ##diff有两个输入，一个是上一个命令的输出，一个是<后的输出,tr是用来替换字符的
```

### 解压

#### zip

解压用`unzip` ,压缩用`zip`

#### gz

解压用`gunzip`，压缩用`gzip`

#### tar.gz

一些参数：

- `-c`压缩
- `-x`解压
- `-t`查看压缩包的文件名
- `-z`指定`.gz`
- `-f`指定压缩和解压缩的文件名

```shell
tar -zcv -f cp_folder.tar.gz cp_folder       #打包压缩文件夹（gzip格式）

tar -ztv -f cp_folder.tar.gz                 #查看压缩文件夹中的文件名（gzip格式）

tar -zxv -f cp_folder.tar.gz                 #打开包并解压缩（gzip格式）
```

#### `bz2`

压缩：`bzip2`;解压：`bunzip2`

也可以用`tar` 使用`-j`参数调用bzip，如：

```shell
tar -xjf a.tar.bz2
```

### 文件操作

#### 文件生成

`seq()`生成序列

```shell
seq 1 10 ##默认分割符为换行符\n
1
2
3
4
5
6
7
8
9
10
seq -s " " 1 2 10 #也可以指定分隔符和步长
1 3 5 7 9
cat <(seq 0 3 17) <(seq 3 6 18) > text##用标准输入和标准输出
0
3
6
9
12
15
3
9
15
```

#### 文件排序

使用`sort`进行排序，常用参数有：

- `-n` 数值排序
- `-r`逆序
- `-c` 检测文件是否已经排序
- `-k`指定使用哪几列排序
- `-m`合并排序好的文件
- `-u`删除重复行

```shell
sort test ##直接排序
1
12
13
14
14
15
16
2
24
3
35
4

 sort -n test ## 按照数值排序
1
2
3
4
12
13
14
14
15
16
24
35

sort test > on_test
sort -n test > n_test

sort -cn on_test ##注意用cn就表示检查是否按照数值排序的
#sort: on_test:9: disorder: 2
sort -cn n_test ##没有输出，表示排序了

sort -nu test
sort -n test | uniq##两种相同，都是将重复元素去掉
1
2
3
4
12
13
14
15
16
24
35

sort -n test | uniq -d##获取重复的行
14

sort -n test | uniq -c ##获取重复的行重复的次数
      1
      1 1
      1 2
      1 3
      1 4
      1 12
      1 13
      2 14
      1 15
      1 16
      1 24
      1 35
```

#### 一个例子，`fasta`序列提取

```shell
cat test.fasta
>SOX2
ACGAGGGACGCATCGGACGACTGCAGGACTGTC
>POU5F1
ACGAGGGACGCATCGGACGACTGCAGGACTGTC
>NANOG
CGGAAGGTAGTCGTCAGTGCAGCGAGTCCGT

##提取含有SOX2的序列信息
grep -A 1 "SOX2" test.fasta
>SOX2
ACGAGGGACGCATCGGACGACTGCAGGACTGTC
##-A 1表示输出匹配的行和其后一行after
```

也可以用`awk`来操作:

```shell
awk 'BEGIN{OFS=FS="\t"}{if($0 ~ />/) {name=$0;sub(">","",name);}\
else seq[name]=$0;}END{print ">SOX2"; print seq["SOX2"]}' test.fasta

##$0 ~ />/ ~表示模式匹配的运算符，看这一行里面有没有>
##sub 是进行替换的：sub(要替换的部分，要替换成的部分，要替换的字符串
##seq[name] 相当于一个字典，name就是key
```

### 判断某个文件夹下是否有某个文件名(在一个文件中)，如果有将其移动到另一个文件夹

首先要对含有文件名的文件进行遍历：

```shell
cat all.txt |
while read file_name;
do
...
done
```

然后判断是否有某个文件

```shell
#!/bin/bash
cat SRR_Acc_List.txt |
while read file_name
do
	if [ -e ./fastq_gz_2/${file_name}_1.fastq.gz ]
	then
		mv ./fastq_gz_2/${file_name}_1.fastq.gz ./liu_2019
	else
		echo "not exist: $file_name"
	fi
done
```

判断文件夹下是否有某个后缀的文件：

```shell
for i in ./SRR*/;do file=$(ls $i/*.tmp 2>/dev/null | wc -l);if [ $file != 0 ];then echo $i;fi;done ##判断是否有后缀为tmp的文件
```

### 删除空文件夹

```shell
#!/bin/bash

for i in ./SRR*
do 
a=$(ls $i | wc -l)
if [ $a -eq 0 ]
then 
    rm -rf $i
fi
done
```

### 几个取整函数：```ceiling``` ; ```floor``` ; ```trunc``` ; ```round``` ; ```signif```

```ceiling```返回不小于该数字的最小整数，即向上取整

```floor``` 返回不大于该数字的最大整数，即向下取整

```trunc``` 返回距离0最近的整数，即向零取整

![](https://raw.githubusercontent.com/wt12318/picgo/master/img/20200128111606.png)

```round``` 采取**四舍六入五成双**的原则：

- `<=`4舍去
- `>=`6进一位
- 五成双：
  - 当5后面有数字的时候，舍5进一位
  - 当5后面没有数字的时候：
    - 当5前一位为奇数时，舍5进一位
    - 当5前一位为偶数时，舍5不进位

``round``也可以指定保留小数点的位数：```round(x,digits=0)```

```signif``` 保留有效数字，```signif(x,digits=6)``` 默认保留6位

### 安装`github`上的包

首先使用`devtools`来安装：

```R
devtools::install_github("gadenbuie/regexplain")
```

结果报错，应该是网不行：

```R
Error: Failed to install 'unknown package' from GitHub:
  Timeout was reached: [api.github.com] Resolving timed out after 10000 milliseconds
```

上github将包的zip下载下来，用devtools的本地安装：

```R
devtools::install_local("~/packages/regexplain-master.zip")
```

还是不行：

```R
Error: Failed to install 'regexplain-master.zip' from local:
  Timeout was reached: [api.github.com] Resolving timed out after 10000 milliseconds
```

只能用一般的本地安装包的方法：先解压

```R
install.packages("~/packages/regexplain-master/",repos=NULL,type="source")
```

### 翻转字符串

参考：https://www.r-bloggers.com/four-ways-to-reverse-a-string-in-r/

翻转字符串是一项常见的操作，比如我们想要找与一条DNA序列的互补序列就需要用到翻转

先生成一个短序列：

```R
set.seed(1)
dna <- paste(sample(c("A", "T", "C", "G"), 10, 
                    replace = T), collapse = "")
```

第一种方法：先用`strsplit()`将序列拆分成单个字符向量，再用`rev()`翻转向量，最后用`paste()`连起来：

```R
start <- proc.time()
splits <- strsplit(dna, "")[[1]]
reversed <- rev(splits)
final_result <- paste(reversed, collapse = "")
end <- proc.time()

print(end - start)
#  用户   系统   流逝 
#12.931  0.860 13.862 
```

第二种方法：先用R 内置的`utf8ToInt()`将字符转化成整数向量，再用`rev()`进行翻转,最后再用`intToUtf8`转化成字符

```R
utf8ToInt(c("abc"))
#[1] 97 98 99

start <- proc.time()
final_result <- intToUtf8(rev(utf8ToInt(dna)))
end <- proc.time()

print(end - start)
#用户  系统  流逝 
#1.569 0.538 2.136 
```

第三种方法就是用`stringi`包的`stri_reverse`函数进行翻转：

```R
start <- proc.time()
final_result <- stringi::stri_reverse(dna)
end <- proc.time()

print(end - start)
# 用户  系统  流逝 
#0.295 0.028 0.336 
```

第四种方法就是使用`Biostrings`包里面的`str_rev()`函数：

```R
if(require(Biostrings)==F){
  BiocManager::install("Biostrings")
  library(Biostrings)
}else{library(Biostrings)}
start <- proc.time()
final_result <- str_rev(dna)
end <- proc.time()

print(end - start)
# 用户  系统  流逝 
#0.031 0.001 0.032 
```

### 设置了 SSH key 但是 Github push 仍然需要密码

设置了 SSH key 但是 Github push 仍然需要密码，而 github 在 8 月 13 后只支持已 token 的形式 push：

```shell
git push origin main
Username for 'https://github.com': wt12318
Password for 'https://wt12318@github.com':
remote: Support for password authentication was removed on August 13, 2021. Please use a personal access token instead.
remote: Please see https://github.blog/2020-12-15-token-authentication-requirements-for-git-operations/ for more information.
fatal: unable to access 'https://github.com/wt12318/Blog_2.git/': The requested URL returned error: 403
```

需要输入密码是由于使用了 `https` 的 url，参考[git - SSH Key - Still asking for password and passphrase - Stack Overflow](https://stackoverflow.com/questions/21095054/ssh-key-still-asking-for-password-and-passphrase)，所以需要从 `https` 转到 `ssh`:

```shell
git remote -v
origin  https://github.com/wt12318/Blog_2.git (fetch)
origin  https://github.com/wt12318/Blog_2.git (push)

git remote set-url origin git@github.com:wt12318/Blog_2.git
git remote -v
origin  git@github.com:wt12318/Blog_2.git (fetch)
origin  git@github.com:wt12318/Blog_2.git (push)

git push origin main
```

