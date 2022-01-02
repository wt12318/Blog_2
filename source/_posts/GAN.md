---
title: 分类模型评估指标
author: wutao
date: 2021-12-15 10:00:00
categories:
  - 深度学习
index_img: img/gans.jpg
---



- 为什么需要 GAN
- 什么是 GAN
- 如何实现 GAN

一般的判别问题都是给定一个输出，通过一定的模型来预测其属于哪个类（分类），或者对应的值应该是多少（回归），也就是这些模型评判的是要预测的样本与训练样本的相似性。但是如果我们想要做一些更具有“创造力”的工作，比如让模型去绘画或者创造一种对话机器人，这时单纯的判别模型就不能很好的发挥作用，而生成模型此时就可以派上用场。

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20211210085819179.png)

上图展示的是生成模型的工作流程：生成模型是一种概率模型，输入的是真实数据以及随机噪声，期望通过学习得到真实数据的**分布**，然后在这个分布中采样得到最终的输出；比如上面的生成模型生成一幅马的图像，但是这个图像并不属于训练数据（分布抽样，不是样本抽样）。

生成模型有很多，比如混合高斯模型，隐马尔可夫模型，这里主要讲生成对抗网络（Generative Adversarial Networks，GAN）

例子：

- [apchenstu/sofgan: [TOG 2021\] SofGAN: A Portrait Image Generator with Dynamic Styling (github.com)](https://github.com/apchenstu/sofgan)

- [hindupuravinash/the-gan-zoo: A list of all named GANs! (github.com)](https://github.com/hindupuravinash/the-gan-zoo)



对抗生成网络的思想和进化生物学上的共进化（Coevolution）类似：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20211210092455743.png)

下面举一个更具体的例子来说明 GAN 的思想：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20211210092925900.png)

我们可以比较容易的训练一个分类器来区分真猫咪和假猫咪的图像，但是我们现在用一个生成器来自动地生成假猫咪的图像，关心的问题是如何使这个生成器能生成以假乱真的猫咪图像，也就是生成器的输出可以骗过分类器，与此同时分类器也必须持续的进步来分辨出真猫咪和逐渐进步的生成器生成的假猫咪：

![image-20211210093714155](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20211210093714155.png)

这里的分类器（一般叫做鉴别器，discriminator）和生成器（generator）之间是竞争对手（adversary）的关系，双方都想要超越对方，在这个过程中逐渐提高鉴别和生成的能力，因此这种模型架构叫做对抗生成网络。

下面来看一下如何训练 GAN：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20211210094015805.png)

第一步固定生成器 G，向判别器输入真实数据，并且这些真实数据的标签为 1，来训练判别器；

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20211210094027666.png)

第二步还是固定生成器 G，这次向判别器输入生成器生成的假数据，这些假数据的标签为0，来训练鉴别器（上面两步其实就是一步）；

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20211210094040997.png)

第三步是固定判别器，向判别器输入生成器生成的假数据，但是要注意这时的数据的标签为 1（因为我们想要生成器 ”骗过“ 鉴别器，通过更新生成器让鉴别器认为生成器生成的假数据是真实数据），然后根据损失来训练更新生成器，**不更新判别器**。

使用 `pytorch` 来实现 GAN 生成手写数字图像：

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20211210122801632.png)

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import pandas
import matplotlib.pyplot as plt
import numpy 
import random

# dataset class

class MnistDataset(Dataset):
  
  def __init__(self, csv_file):
    self.data_df = pandas.read_csv(csv_file, header=None)
    
  def __len__(self):
    return len(self.data_df)
  
  def __getitem__(self, index):
    # image target (label)
    label = self.data_df.iloc[index,0]
    target = torch.zeros((10))
    target[label] = 1.0
    
    # image data, normalised from 0-255 to 0-1
    image_values = torch.FloatTensor(self.data_df.iloc[index,1:].values) / 255.0
    
    # return label, image data tensor and target tensor
    return label, image_values, target
  
  def plot_image(self, index):
    img = self.data_df.iloc[index,1:].values.reshape(28,28)
    plt.title("label = " + str(self.data_df.iloc[index,0]))
    plt.imshow(img, interpolation='none', cmap='Blues')

mnist_dataset = MnistDataset('/home/wt/useful_data/mnist_train.csv')
mnist_dataset.plot_image(9)
plt.show()

mnist_dataset[100]

# functions to generate random data
def generate_random_seed(size):
  random_data = torch.randn(size)
  return random_data
  

# discriminator class

class Discriminator(nn.Module):
  
  def __init__(self):
    # initialise parent pytorch class
    super().__init__()
    
    # define neural network layers
    self.model = nn.Sequential(
        nn.Linear(784, 200),
        nn.LeakyReLU(0.02),

        nn.LayerNorm(200),

        nn.Linear(200, 1),
        nn.Sigmoid()
    )
    
    # create loss function
    self.loss_function = nn.BCELoss()

    # create optimiser, simple stochastic gradient descent
    self.optimiser = torch.optim.Adam(self.parameters(), lr=0.0001)

    # counter and accumulator for progress
    self.counter = 0;
    self.progress = []

  def forward(self, inputs):
    # simply run model
    return self.model(inputs)
  
  
  def train(self, inputs, targets):
    # calculate the output of the network
    outputs = self.forward(inputs)
    
    # calculate loss
    loss = self.loss_function(outputs, targets)

    # increase counter and accumulate error every 10
    self.counter += 1;
    if (self.counter % 10 == 0):
        self.progress.append(loss.item())
    if (self.counter % 10000 == 0):
        print("counter = ", self.counter)

    # zero gradients, perform a backward pass, update weights
    self.optimiser.zero_grad()
    loss.backward()
    self.optimiser.step()

  def plot_progress(self):
    df = pandas.DataFrame(self.progress, columns=['loss'])
    df.plot(ylim=(0), figsize=(16,8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5, 1.0, 5.0))
    
# generator class

class Generator(nn.Module):
    
  def __init__(self):
    # initialise parent pytorch class
    super().__init__()
    
    # define neural network layers
    self.model = nn.Sequential(
        nn.Linear(100, 200),
        nn.LeakyReLU(0.02),

        nn.LayerNorm(200),

        nn.Linear(200, 784),
        nn.Sigmoid()
    )
    
    # create optimiser, simple stochastic gradient descent
    self.optimiser = torch.optim.Adam(self.parameters(), lr=0.0001)

    # counter and accumulator for progress
    self.counter = 0;
    self.progress = []
    
  def forward(self, inputs):        
    # simply run model
    return self.model(inputs)

  
  def train(self, D, inputs, targets):
    # calculate the output of the network
    g_output = self.forward(inputs)
    
    # pass onto Discriminator
    d_output = D.forward(g_output)
    
    # calculate error
    loss = D.loss_function(d_output, targets)

    # increase counter and accumulate error every 10
    self.counter += 1;
    if (self.counter % 10 == 0):
        self.progress.append(loss.item())

    # zero gradients, perform a backward pass, update weights
    self.optimiser.zero_grad()
    loss.backward()
    self.optimiser.step()

  def plot_progress(self):
    df = pandas.DataFrame(self.progress, columns=['loss'])
    df.plot(ylim=(0), figsize=(16,8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5, 1.0, 5.0))

G = Generator()
output = G.forward(generate_random_seed(100))
img = output.detach().numpy().reshape(28,28)
plt.imshow(img, interpolation='none', cmap='Blues')
plt.show()
##训练
# create Discriminator and Generator

D = Discriminator()
G = Generator()

epochs = 4

for epoch in range(epochs):
  print ("epoch = ", epoch + 1)

  # train Discriminator and Generator

  for label, image_data_tensor, target_tensor in mnist_dataset:
    # train discriminator on true
    D.train(image_data_tensor, torch.FloatTensor([1.0]))
    # train discriminator on false
    # use detach() so gradients in G are not calculated
    D.train(G.forward(generate_random_seed(100)).detach(), torch.FloatTensor([0.0]))
    # train generator
    G.train(D, generate_random_seed(100), torch.FloatTensor([1.0]))

# plot several outputs from the trained generator

# plot a 3 column, 2 row array of generated images
f, axarr = plt.subplots(2,3, figsize=(16,8))
for i in range(2):
  for j in range(3):
    output = G.forward(generate_random_seed(100))
    img = output.detach().numpy().reshape(28,28)
    axarr[i,j].imshow(img, interpolation='none', cmap='Blues')
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20211210122128264.png)

当我们训练出一个比较好的生成器时，loss 应该是多少？

对于 MSE Loss：一个好的生成器，鉴别器的输出值应该是 0.5，因为其无法分辨是真实的还是假的，所以给了0.5 的预测值，这个时候标签要么是 0 要么是 1，算出来的 LOSS 都是 0.25。

对于交叉熵损失 (BCEloss)：
$$
Loss = \sum-y*ln(x)=-(1.0)*ln(0.5)-(0)*ln(1-0.5)=0.693
$$

我们可以看一下训练的 Loss：

```python
D.plot_progress()
plt.show()
G.plot_progress()
plt.show()
```

