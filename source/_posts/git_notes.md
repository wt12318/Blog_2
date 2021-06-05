---
title: Git简介
author: wutao
date: 2021-01-17 10:00:00
slug: git-notes
categories:
  - Git
tags:
  - notes
index_img: img/git.jpg
---

主要参考：[廖雪峰的Git教程](https://www.liaoxuefeng.com/wiki/896043488029600)

## 安装创建Git

Ubuntu 可以直接通过`apt`安装：

```shell
sudo apt-get install git
```

Windows可以上Git 官网下载相应的安装包安装

安装完成后需要设置用户名和邮件：

```shell
git config --global user.name "Your Name"
git config --global user.email "email@example.com"
```

接下来就可以创建一个版本库了，版本库(repository)就相当于一个目录，Git可以跟踪该目录中所有文件的修改，删除：

```shell
mkdir learninggit
```

通过`git init`来初始化目录为可以被管理的仓库

然后就可以将文件添加到版本库中：

```shell
##在该文件夹下新建一个文件
touch readme.txt
vi readme.txt
## Git is a version control system.
## Git is free software.

##将文件添加到仓库
git add readme.txt

##将文件提交到仓库
git commit -m 'write a readme file'
```

`-m`后面加的字符串是提交的文件的描叙，`git commit`命令是提交整个文件夹，也就是他可以检查整个文件的改动，所以在前面可以`add`多个文件，最后一次`commit`：

```shell
git add file1.txt
git add file2.txt file3.txt
git commit -m "add 3 files."
```

![](https://raw.githubusercontent.com/wt12318/picgo/master/img/20200411224439.png)

上面的图简单的说明了`add`和`commit`的作用，在文件夹下所有可见的文件都是工作区，而不可见的`.git`文件夹就是版本库，版本库中有一个暂存区(stage)，当`add`文件后，文件就被储存到这个空间里，`add`结束后，`commit`会一次性将暂存区的文件提交到master的分支上,也就是不`add`是不会被提交的

## 修改，管理，版本控制

### 版本控制

前面讲过Git可以监控文件夹中的文件的修改，现在来修改`readme.txt`看看会发生什么

```shell
vi readme.txt
# Git is a distributed version control system.
# Git is free software.
```

可以用`git status`查看仓库的状态：

```shell
git status
On branch master
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git checkout -- <file>..." to discard changes in working directory)

	modified:   readme.txt

no changes added to commit (use "git add" and/or "git commit -a")
```

上面结果说明`readme.txt`已经被修改，但是还没有准备提交

可以用`git diff`来查看修改的内容：

```shell
git diff readme.txt 
#diff --git a/readme.txt b/readme.txt
#index 46d49bf..9247db6 100644
#--- a/readme.txt
#+++ b/readme.txt
#@@ -1,2 +1,2 @@
#-Git is a version control system.
#+Git is a distributed version control system.
#Git is free software.
```

要将修改后的文件提交到仓库，就和之前一样，先`add`再`commit`：

```shell
git add readme.txt

git status
On branch master
Changes to be committed:
  (use "git reset HEAD <file>..." to unstage)

	modified:   readme.txt
	
git commit -m "add distributed"
[master e475afc] add distributed
 1 file changed, 1 insertion(+), 1 deletion(-)
 
 git status
On branch master
nothing to commit, working tree clean
```

可以看到在`add`之后，文件的状态变成待提交(to be committed)，提交之后变成clean了

每一次`commit`文件就相当于对文件进行快照，可以从这些快照中来恢复之前的文件版本

可以使用`git log`命令来查看这些`commit`历史：

```shell
git log
commit 3288eec6192ff2d876c9d0140ce2ddfdba30f5bc (HEAD -> master)
Author: wt12318 <1650464505@qq.com>
Date:   Sat Apr 11 21:22:36 2020 +0800

    add distributed

commit a524903c1f9612bc4f3b85f29936565818c13e2c
Author: wt12318 <1650464505@qq.com>
Date:   Sat Apr 11 20:34:55 2020 +0800

    wrote a readme file
```

可以看到Git将当前版本用HEAD 标记，在Git中上一个版本就是`HEAD^`,每往前一个版本就在后面加上`^`，当数量较多的时候可以用`HEAD~number`代替

比如，现在要将`readme.txt`恢复到前一个版本：

```shell
git reset --hard HEAD^
# HEAD is now at a524903 wrote a readme file

cat readme.txt
Git is a version control system.
Git is free software.

git log
commit a524903c1f9612bc4f3b85f29936565818c13e2c (HEAD -> master)
Author: wt12318 <1650464505@qq.com>
Date:   Sat Apr 11 20:34:55 2020 +0800

    wrote a readme file
```

现在add distributed的那个版本已经不见了，如果想要恢复该版本（回到未来）可以用该版本的`commit id`来进行(可以不写全）：

```shell
git reset --hard 3288eec
HEAD is now at 3288eec add distributed

cat readme.txt
Git is a distributed version control system.
Git is free software.
```

但是如果关掉了terminal，又想要恢复“未来”的文件该怎么办？可以用`git reflog`，记录了每一次的命令：

```shell
git reflog
3288eec (HEAD -> master) HEAD@{0}: reset: moving to 3288eec
a524903 HEAD@{1}: reset: moving to HEAD^
3288eec (HEAD -> master) HEAD@{2}: commit: add distributed
a524903 HEAD@{3}: commit (initial): wrote a readme file
```

### 撤销修改

`git checkout --file`可以撤销工作区的修改，现在在`readme.txt`中添加一行：

```shell
cat readme.txt
Git is a distributed version control system.
Git is free software.
Git tracks changes
Git check out
```

```shell
git checkout -- readme.txt
cat readme.txt
Git is a distributed version control system.
Git is free software.
Git tracks changes
```

这样就复原来最近的一次`add`或者`commit`的版本了

但是如果现在已经将文件`add`到暂存区了，可以用`git reset HEAD file`将文件返回到工作区：

```shell
cat readme.txt
Git is a distributed version control system.
Git is free software.
Git tracks changes
Git check out

git add readme.txt
git status
On branch master
Changes to be committed:
  (use "git reset HEAD <file>..." to unstage)

        modified:   readme.txt
        
git reset HEAD readme.txt
Unstaged changes after reset:
M       readme.txt
git status
On branch master
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git checkout -- <file>..." to discard changes in working directory)

        modified:   readme.txt

no changes added to commit (use "git add" and/or "git commit -a")

```

可以看到在`reset`之后，暂存区没有文件待提交，现在可以用`checkout`来删除修改了

```shell
git checkout -- readme.txt
git status
On branch master
nothing to commit, working tree clean

cat readme.txt
Git is a distributed version control system.
Git is free software.
Git tracks changes
```

已经将文件提交了，可以用之前讲过的`git reset --hard commit_id/HEAD^^...`来恢复

### 删除文件

先创建一个新文件：`touch test.txt`,再像之前一样提交到版本库：

```shell
git add test.txt
git commit -m 'add test.txt'
[master f1d6024] add test.txt
 1 file changed, 0 insertions(+), 0 deletions(-)
 create mode 100644 test.txt
```

现在删除该文件：

```shell
rm test.txt
git status
On branch master
Changes not staged for commit:
  (use "git add/rm <file>..." to update what will be committed)
  (use "git checkout -- <file>..." to discard changes in working directory)

        deleted:    test.txt

no changes added to commit (use "git add" and/or "git commit -a")
```

Git可以发现删除了一个文件，如果是删错了，可以从版本库中恢复，还是用之前那个`checkout`命令：

```shell
git checkout -- test.txt
ls
readme.txt  test.txt
```

其实`checkout`就是用版本库的内容覆盖工作区的内容，所以可以恢复修改和删除

如果确实想删除该文件，可以用`git rm`然后再提交：

```shell
git rm test.txt
# rm 'test.txt'

git commit -m 'remove test.txt'
[master a6b89a8] remove test.txt
 1 file changed, 0 insertions(+), 0 deletions(-)
 delete mode 100644 test.txt
```

## 远程仓库

创建SSH key：`ssh-keygen -t rsa -C "youremail@example.com` 一路回车，完成后就会在主目录的`.ssh`目录下看到`id_rsa  id_rsa.pub`两个文件

登陆Github，在`setting`里面找到`SSH and GPG keys` 添加`New SSH key`,将`id_rsa.pub`中的内容复制进去

### 添加远程库

在Github上添加一个Repository 名字为`learngit`,然后将本地的库与远程的库关联：

```shell
git remote add origin git@github.com:wt12318/learngit.git ## 或者https://github.com/wt12318/learngit.git
git push -u origin master
```

以后在本地提交了之后就可以通过`git push origin master`将修改推送到远程库了

### 从远程库克隆

现在在Github上新建一个远程库，名为`gitskills` 并初始化`README` ,然后就可以在本地克隆该仓库：

```shell
git clone git@github.com:wt12318/gitskills
```

## 分支管理

在**团队协作**中可以建一个自己的分支，在里面修改，提交别人是看不到的，最后开发完成后再与原来的分支合并

### 创建与合并分支

Git的创建和合并分支就是一个改变指针的动作，上面的那些提交都是在master分支上进行的，master指向最新的提交，HEAD指向master，当我们新建一个分支的时候，在以后操作过程中就是由该新建的分支指向提交，再由HEAD指向这个分支，最后合并的时候就是把指针还给master，HEAD重新指向master

现在新建一个分支`dev`，并切换到该分支：

```shell
git checkout -b dev 
##或者 git switch -c dev
```

`git checkout -b `(`git switch -c dev`)相当于创建并切换，等于下面两个命令：

```shell
git branch dev
git checkout dev## git switch dev
```

使用`git branch`可以查看当前的分支：

```shell
git branch
* dev
  master
```

当前分支前面会有一个星号

现在试试在dev上修改，提交：

```shell
## 加一行:Creating a new branch is quick
cat readme.txt
Git is a distributed version control system.
Git is free software.
Git tracks changes
Creating a new branch is quick

git add readme.txt
git commit -m 'branch test'
```

然后切换到master分支上：

```shell
git checkout master ## git switch master
cat readme.txt
Git is a distributed version control system.
Git is free software.
Git tracks changes
```

就会发现之前在dev上修改的内容不见了，原因就是master和dev分支的提交点不一样：

![](https://raw.githubusercontent.com/wt12318/picgo/master/img/20200412194020.png)

所以要想将master移动到当前的修改，就需要将dev分支合并到master上，抗可以用`git merge`命令：

```shell
git merge dev
cat readme.txt
Git is a distributed version control system.
Git is free software.
Git tracks changes
Creating a new branch is quick
```

现在就可以删除分支dev了：

```shell
git branch -d dev
git branch
* master
```

### 解决冲突

合并冲突可能会出现冲突，需要手动编辑冲突的文件

### 分支管理策略

一般情况下，Git会使用`fast forward`方式进行合并分支，这种方法就是直接将master的指针移动到分支的最新提交上，当分支删除后就不能看成分支信息了，可以用`--no-ff`选项禁用`fast forward`模式：

```shell
git switch -c dev
##修改readme
cat readme.txt
Git is a distributed version control system.
Git is free software.
Git tracks changes
Creating a new branch is quick
no fast forward

git add readme.txt
git commit -m "add merge"
git switch master

git merge --no-ff -m "merge with no-ff" dev ##这种合并会创建一个新的commit，所以要加-m
```

可以用`git log --graph`查看合并图：

```shell
git log --graph --pretty=oneline --abbrev-commit
*   7e4a555 (HEAD -> master) merge with no-ff
|\
| * 0c38472 (dev) add merge
|/
* 58a2bf7 branch test
* a6b89a8 (origin/master) remove test.txt
* f1d6024 add test.txt
* c500238 git tracks changes
* 3288eec add distributed
* a524903 wrote a readme file
```

另外如果想要删除一个没有被合并的分支，可以通过`git branch -D name`来删除

### 多人协作

远程仓库的默认名是`orign`

查看远程库的信息：

```shell
git remote
git remote -v ##详细信息
origin  https://github.com/limbo1996/neoantigens_depletion.git (fetch)
origin  https://github.com/limbo1996/neoantigens_depletion.git (push)
```

推送不同的分支：

```shell
git push origin master
git push origin dev
```

多人协作的时候，首先需要将远程仓库克隆到本地：

```shell
git clone git@github.com:wt12318/learngit.git
```

要在已经创建的分支dev上工作，需要创建远程仓库中的dev分支对应的本地分支：

```shell
git checkout -b dev origin/dev
git branch --set-upstream-to=origin/dev dev#指定本地dev分支与远程origin/dev分支的链接
```

拉取，推送分支：

```shell
git push origin dev
git pull
```

#### `github`

首先要fork别人的仓库，然后clone：

```shell
git clone git@github.com:***

##添加远程仓库
git remote add upstream https://github.com/***

##获取最新源码
git pull upstream branch

##将最新代码提交到自己的分支中
git push origin branch
```

最后提交后，再去作者的`github`里pull request

如何与原仓库同步？

```shell
git remote add upstream https://github.com/origin_rep
##如果报错fatal: remote upstream already exists.
##先git remote rm upstream
git fetch upstream
```

然后合并远程分支：

```shell
git checkout master
git merge upstream/master
```

此时已经将本地库与原仓库合并，要更新自己github上的仓库还需要`git push`：

```shell
git push --set-upstream origin master
```



## 总结

### 创建：

克隆已有的库：

```shell
git clone ssh://user@domain.com/repo.git
```

创建本地新库：

```shell
git init
```

### 本地改变

查看改变的文件：

```shell
git status
```

查看文件修改的内容：

```shell
git diff
```

将改变添加到暂存区：

```shell
git add 
```

将改变提交到分支：

```shell
git commit
```

### 查看提交历史

显示所有的提交（从最新的开始）：

```shell
git log
```

显示对于某个文件的改变：

```shell
git log -p <file>
```

显示谁在什么时候改变了什么：

```shell
git blame <file>
```

### 分支

显示所有的分支：

```shell
git branch -v
```

切换分支：

```shell
git checkout <branch> ##or git switch <branch>
```

创建新分支：

```shell
git branch <new-branch>
```

删除分支

```shell
git branch -d <branch>
```

### 远程协作

查看远程库的信息：

```shell
git remote -v
```

推送分支：

```shell
git push origin <branch>
```

拉取分支：

```shell
git pull
```

### 撤销修改

将文件返回工作区：

```shell
git reset HEAD file
```

撤销工作区的删改：

```shell
git checkout --file
```

### 回退

```shell
git reset --hard <commit>
```

### 一些问题
git出错：“Please make sure you have the correct access rights and the repository exists.

参考：https://cloud.tencent.com/developer/article/1572090

ssh 需要重置

1、重置用户名和邮箱
```shell
git config --global [user.name](http://user.name/) “yourname” git config --global user.email“your@email.com"  
```
**注：yourname是你要设置的名字，your@email是你要设置的邮箱。**

2、删除.ssh文件夹下的known_hosts 
3、git输入命令
```
$ ssh-keygen -t rsa -C your@email.com
```
一路yes和回车 然后系统会自动在.ssh文件夹下生成两个文件，id_rsa和id_rsa.pub，用记事本打开id_rsa.pub将全部的内容复制   
4、打开https://github.com/，登陆你的账户，进入设置--进入ssh设置 在key中将刚刚复制的粘贴进去   
5、在git中输入命令
```
ssh -T git@github.com 
```
然后会跳出一堆话 输入命令：回车