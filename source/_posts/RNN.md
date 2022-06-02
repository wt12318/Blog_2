---
title: 循环神经网络
date: 2022-06-01 19:14:18
tags: 深度学习
index_img: img/rnn.svg
categories:
  - 深度学习
---

循环神经网络及 Pytorch 实践

<!-- more -->

## 循环神经网络 RNN

对于表格数据和图像来说，我们都是假设数据是独立同分布的，当实际上数据并不都是如此的，比如文本中的单词，视频的帧，对话的声音信息，这些数据都是有序列特征的，也就是数据之间并不是独立的，因此我们需要一种特殊的模型去描述这类数据。

### 序列模型

例子：股票的预测，根据之前时间的股票价格来预测目前的股票价格：

$$
x_t  \sim P(x_t|x_{t-1},..., x_1)
$$

对类似上面的问题使用回归模型的难点在于：变量数量的变化，随着时间的推移我们需要纳入模型的变量数量会逐渐增多；解决这个问题有两个策略：

* 自回归模型
* 马尔可夫模型

#### 自回归模型

自回归模型指的是因变量和自变量的数据是一样的（从总体上来说）；自回归模型有两种：

1. 不考虑整个序列，而是一个固定大小 window 的序列，这样变量的数量就可以固定下来

2. 将过去的观测整合成一个变量 $h_t$ 这样的模型也叫做隐自回归模型，因为这里的 ht 是一个隐变量

   
   
   <img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220411211540-nljbhay.png" style="zoom:67%;" />



隐变量实际是存在的，观测不到，潜变量可以是不存在的，人为设定的，比如聚类的类信息

因此整个序列出现的概率可以计算：

$$
P(x_1,..., x_t)=\prod_{t=1}^TP(x_t|x_{t-1},...x_1)
$$

#### 马尔可夫模型

上面使用固定大小的 window 就可以说这个序列满足马尔可夫条件，如果这个 window 为 1， 那么就可以得到一阶马尔可夫模型:

$$
P(x_1,..., x_t)=\prod_{t=1}^TP(x_t|x_{t-1})
$$

#### 代码

首先产生一些随机的数据，使用正弦函数加上一些噪音：

```python
%matplotlib inline
import torch
from torch import nn
from d2l import torch as d2l

T = 1000  # Generate a total of 1000 points
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
d2l.plot(time, [x], 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
```

接下来就要生成训练数据：过去的窗口内的值作为 x ，当前的值作为 y；这里就会出现一个问题，开始的窗口长度的数值就没有足够的 x 输入，一般可以将这些数值扔掉或者 padding 为 0（这里直接丢弃）：

```python
tau = 4
features = torch.zeros((T - tau, tau))##前4个不要
for i in range(tau):
    features[:, i] = x[i: T - tau + i] ##
labels = x[tau:].reshape((-1, 1))

batch_size, n_train = 16, 600
# Only the first `n_train` examples are used for training
train_iter = d2l.load_array((features[:n_train], labels[:n_train]),
                            batch_size, is_train=True)
```

训练使用两层的 MLP 加上 ReLU 激活函数，loss 使用均方误差（MSEloss）（自回归）：

```python
# Function for initializing the weights of the network
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

# A simple MLP
def get_net():
    net = nn.Sequential(nn.Linear(4, 10),
                        nn.ReLU(),
                        nn.Linear(10, 1))
    net.apply(init_weights)
    return net

# Note: `MSELoss` computes squared error without the 1/2 factor
loss = nn.MSELoss(reduction='none')

##训练
def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.sum().backward()
            trainer.step()
        print(f'epoch {epoch + 1}, '
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')

net = get_net()
train(net, train_iter, loss, 5, 0.01)
```

进行预测：

```python
onestep_preds = net(features)
d2l.plot([time, time[tau:]], [x.detach().numpy(), onestep_preds.detach().numpy()], 'time',
         'x', legend=['data', '1-step preds'], xlim=[1, 1000], figsize=(6, 3))
```

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220411223707-iy3hgos.png" style="zoom:67%;" />



这个只是前进一步的预测，如果我们要预测 604 个之后的就只能根据我们的预测来预测（因为上面训练的数据也就是观测的数据只到600），现在来看这些预测怎么样：

```python
multistep_preds = torch.zeros(T)
multistep_preds[: n_train + tau] = x[: n_train + tau] ##只保留实际的604个值后面都是预测的
for i in range(n_train + tau, T):
    multistep_preds[i] = net(
        multistep_preds[i - tau:i].reshape((1, -1)))

d2l.plot([time, time[tau:], time[n_train + tau:]],
         [x.detach().numpy(), onestep_preds.detach().numpy(),
          multistep_preds[n_train + tau:].detach().numpy()], 'time',
         'x', legend=['data', '1-step preds', 'multistep preds'],
         xlim=[1, 1000], figsize=(6, 3))
```

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220411224245-dlctp7g.png" style="zoom:67%;" />



可以看到在离 604 不久后预测就飞了，原因是误差的不断累积，比较不同窗口的区别：

```python
max_steps = 64

features = torch.zeros((T - tau - max_steps + 1, tau + max_steps))
# Column `i` (`i` < `tau`) are observations from `x` for time steps from
# `i + 1` to `i + T - tau - max_steps + 1`
for i in range(tau):
    features[:, i] = x[i: i + T - tau - max_steps + 1]

# Column `i` (`i` >= `tau`) are the (`i - tau + 1`)-step-ahead predictions for
# time steps from `i + 1` to `i + T - tau - max_steps + 1`
for i in range(tau, tau + max_steps):
    features[:, i] = net(features[:, i - tau:i]).reshape(-1)

steps = (1, 4, 16, 64)
d2l.plot([time[tau + i - 1: T - max_steps + i] for i in steps],
         [features[:, (tau + i - 1)].detach().numpy() for i in steps], 'time', 'x',
         legend=[f'{i}-step preds' for i in steps], xlim=[5, 1000],
         figsize=(6, 3))
```

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220411224545-130vwnu.png" style="zoom:67%;" />

### 文本预处理

文本是最流行的序列数据的例子，文本预处理分为以下步骤：

* 将文本以字符串读入内存
* 将字符串拆分成 token，可以是单个的词或者字符
* 构建一个词汇表，将 token 映射到数字索引
* 将文本转化成数字索引的序列

```python
import collections
import re
from d2l import torch as d2l
```

#### 读入数据

将文本读入成文本行构成的列表，每一行是一个字符串：

```python
#@save
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():  #@save
    """Load the time machine dataset into a list of text lines."""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines] ##非字母的字符去掉，并全部转化成小写

lines = read_time_machine()
print(f'# text lines: {len(lines)}')
print(lines[0])
print(lines[10])

Downloading ../data/timemachine.txt from http://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt...
# text lines: 3221
the time machine by h g wells
twinkled and his usually pale face was flushed and animated the
```

将字符串拆分成词或者字符，这个生成的 token 是一个 list of list，其中每个列表是构成一行的词：

```python
def tokenize(lines, token='word'):  #@save
    """Split text lines into word or character tokens."""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('ERROR: unknown token type: ' + token)

tokens = tokenize(lines)
for i in range(11):
    print(tokens[i])

['the', 'time', 'machine', 'by', 'h', 'g', 'wells']
[]
[]
[]
[]
['i']
[]
[]
['the', 'time', 'traveller', 'for', 'so', 'it', 'will', 'be', 'convenient', 'to', 'speak', 'of', 'him']
['was', 'expounding', 'a', 'recondite', 'matter', 'to', 'us', 'his', 'grey', 'eyes', 'shone', 'and']
['twinkled', 'and', 'his', 'usually', 'pale', 'face', 'was', 'flushed', 'and', 'animated', 'the']
```

接下来就要创建一个词汇表将 token 映射到数字索引：先计算文本中唯一 token 的频率表，叫做语料（corpus），然后根据其频率赋予索引（从大到小，从0开始），有些出现较少的 token 可以去掉以减少复杂性，另外还可以添加一些特殊的token，比如在语料中不存在或者被移除的 token 可以用` <unk>` 来表示，开始的token 用 `<bos>`，结束的 token 用 `<eos>` 表示，padding 可以使用 `<pad>` 表示等：

```python
def count_corpus(tokens):  #@save
    """Count token frequencies."""
    # Here `tokens` is a 1D list or 2D list
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # Flatten a list of token lists into a list of tokens
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)
```

注意这里的一个 flatten 二维列表的技巧 `[token for line in tokens for token in line]`，先运行 `line in tokens` 每次拿出一个列表，然后运行 `token in line` 每次拿出该列表中的一个词作为最开始的 `token`：

```python
a = [[1,2,3],[4,5,6]]
[i for j in a for i in j ]

[1, 2, 3, 4, 5, 6]
```

创建词汇表：

```python
class Vocab:  #@save
    """Vocabulary for text."""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # Sort according to frequencies
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)##按照 key进行排序，key选择的是字典中的值，也就是频率
        # The index for the unknown token is 0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}##enumerate返回的一个元素是从零开始的索引
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1##逐渐增加index

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)##dict.get(key,value)当key 不存在时返回 value，这里就是对 unkown 的 token 返回 0
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # Index for the unknown token
        return 0

    @property
    def token_freqs(self):  # Index for the unknown token
        return self._token_freqs
```

接下来我们就可以用上面的类及函数将time machine 这个文本转化为数字：

```python
def load_corpus_time_machine(max_tokens=-1):  #@save
    """Return token indices and the vocabulary of the time machine dataset."""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')##以字符而不是词
    vocab = Vocab(tokens)
    # Since each text line in the time machine dataset is not necessarily a
    # sentence or a paragraph, flatten all the text lines into a single list
    corpus = [vocab[token] for line in tokens for token in line]##返回每个字的index
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

corpus, vocab = load_corpus_time_machine()
len(corpus), len(vocab)

(170580, 28)
```

### 语言模型和数据集

上面我们将文本序列转化成 token，因此一个长度为 T 的文本序列可以表示成一个 token 的序列：$x_1,x_2,...,x_T$ 语言模型的目的就是估计联合概率：

$$
P(x_1,x_2,...,x_T)
$$

由基本的条件概率我们可以得到：

$$
P(x_1, x_2, \ldots, x_T) = \prod_{t=1}^T P(x_t  \mid  x_1, \ldots, x_{t-1}).
$$

因此要计算这个语言模型，我们需要计算词的概率和给定前面的词的条件概率，对于一些大型的文本可以使用词的频率来估计这种概率，但是这有一个问题：对于一些词的组合，可能出现的次数比较少（越比如对于固定的有3个词的词组，可能就不会出现几次），对于这个问题通常的策略是 *Laplace smoothing*，也就是加上一个小的常数：

$$
\hat{P}(x)  = \frac{n(x) + \epsilon_1/m}{n + \epsilon_1}
$$

$$
\hat{P}(x' \mid x) = \frac{n(x, x') + \epsilon_2 \hat{P}(x')}{n(x) + \epsilon_2}
$$

$$
\hat{P}(x’’ \mid x,x’)  = \frac{n(x, x’,x’’) + \epsilon_3 \hat{P}(x’’)}{n(x, x’) + \epsilon_3}
$$



但是这样近似还是会存在一些问题：

* 需要存储所有的词，词组的 counts
* 这个方法忽略了词的意思
* 对于一些长的词序列，在整个文本中可能一次都没有，那么这种方法也是不行的

我们还可以将前面讲过的马尔可夫假设引进语言模型，可以得到不同 gram 的模型估计（对应着一阶，二阶，三阶马尔可夫假设）：

$$
P(x_1, x_2, x_3, x_4) =  P(x_1) P(x_2) P(x_3) P(x_4)
$$

$$
P(x_1, x_2, x_3, x_4) =  P(x_1) P(x_2  \mid  x_1) P(x_3  \mid  x_2) P(x_4  \mid  x_3)
$$

$$
P(x_1, x_2, x_3, x_4) =  P(x_1) P(x_2  \mid  x_1) P(x_3  \mid  x_1, x_2) P(x_4  \mid  x_2, x_3)
$$

#### 读取长序列数据

对于一些很长的序列，模型不能一次性处理，我们需要将这些长序列分割成短序列；首先假设使用神经网络来训练语言模型，神经网络处理的是小批量的序列输入，这些序列有着预定的长度，接下来的问题就是**如何从长的序列中随机地读取小批量的特征和标签**。

如果短序列长度是5，那么可以有如下的选择：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220414105111-jube13u.png" style="zoom:67%;" />



对于开始的 offset 我们一般是随机的选择 offset 的大小，从而使得所有可能的子序列的覆盖度比较大，并且增加随机性；选择子序列的方法有两种：随机抽样和顺序分割，随机抽样中两个相邻的 minibatch 在原始序列中不一定相邻，而顺序分割则是相邻的。

##### 随机抽样

在序列模型中，目标是基于我们看过的 token 预测下一个token，因此标签为原始的序列向后移动一个 token（比如说上面第一个子序列为 the t 其标签为 he ti）:

```python
def seq_data_iter_random(corpus, batch_size, num_steps):  #@save
    """Generate a minibatch of subsequences using random sampling."""
    # Start with a random offset (inclusive of `num_steps - 1`) to partition a
    # sequence
    corpus = corpus[random.randint(0, num_steps - 1):]
    # Subtract 1 since we need to account for labels
    num_subseqs = (len(corpus) - 1) // num_steps
    # The starting indices for subsequences of length `num_steps`
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # In random sampling, the subsequences from two adjacent random
    # minibatches during iteration are not necessarily adjacent on the
    # original sequence
    random.shuffle(initial_indices)

    def data(pos):
        # Return a sequence of length `num_steps` starting from `pos`
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # Here, `initial_indices` contains randomized starting indices for
        # subsequences
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)
```

```python
my_seq = list(range(35))
for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)

X:  tensor([[10, 11, 12, 13, 14],
        [15, 16, 17, 18, 19]]) 
Y: tensor([[11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20]])
X:  tensor([[20, 21, 22, 23, 24],
        [ 5,  6,  7,  8,  9]]) 
Y: tensor([[21, 22, 23, 24, 25],
        [ 6,  7,  8,  9, 10]])
X:  tensor([[25, 26, 27, 28, 29],
        [ 0,  1,  2,  3,  4]]) 
Y: tensor([[26, 27, 28, 29, 30],
        [ 1,  2,  3,  4,  5]])
```

##### 顺序分割

```python
def seq_data_iter_sequential(corpus, batch_size, num_steps):  #@save
    """Generate a minibatch of subsequences using sequential partitioning."""
    # Start with a random offset to partition a sequence
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y
```

```python
for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)

X:  tensor([[ 4,  5,  6,  7,  8],
        [19, 20, 21, 22, 23]]) 
Y: tensor([[ 5,  6,  7,  8,  9],
        [20, 21, 22, 23, 24]])
X:  tensor([[ 9, 10, 11, 12, 13],
        [24, 25, 26, 27, 28]]) 
Y: tensor([[10, 11, 12, 13, 14],
        [25, 26, 27, 28, 29]])
X:  tensor([[14, 15, 16, 17, 18],
        [29, 30, 31, 32, 33]]) 
Y: tensor([[15, 16, 17, 18, 19],
        [30, 31, 32, 33, 34]])
```

将上面的函数包装：

```python
class SeqDataLoader:  #@save
    """An iterator to load sequence data."""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = d2l.seq_data_iter_random
        else:
            self.data_iter_fn = d2l.seq_data_iter_sequential
        self.corpus, self.vocab = d2l.load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)

def load_data_time_machine(batch_size, num_steps,  #@save
                           use_random_iter=False, max_tokens=10000):
    """Return the iterator and the vocabulary of the time machine dataset."""
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab
```

### 循环神经网络

之前讲过为了避免随着预测步长的增加，模型的变量越来越多，我们可以使用一个隐变量模型：

$$
P(x_t \mid x_{t-1}, \ldots, x_1) \approx P(x_t \mid h_{t-1}),
$$

$h_{t-1}$ 是隐状态或者叫隐变量，保留了在 t-1 步及之前的信息，在t步的隐变量由t步的输入和前一步的隐变量计算得到：

$$
h_t = f(x_{t}, h_{t-1}).
$$

这个 f 可以由神经网络来估计：

$$
{H}_t = \phi({X}_t {W}_{xh} + {H}_{t-1} {W}_{hh}  + {b}_h).
$$

在每个时间步，可以根据 Ht 得到该时间步的输出：

$$
{O}_t = {H}_t {W}_{hq} +{b}_q.
$$

因此一个 RNN 的结构为：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/Whiteboard-20220417113658-2bzy5b5.png" style="zoom:50%;" />



下面以一个单词分割成字符预测为例：



<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220417114034-k39nglx.png" style="zoom:50%;" />



在预测中使用类似交叉熵误差的loss，叫做 *perplexity* （困惑度）：
$$
\exp\left(-\frac{1}{n} \sum_{t=1}^n \log P(x_t \mid x_{t-1}, \ldots, x_1)\right).
$$

在每一个时间步的输出就是每个 token 的概率分布

### 循环神经网络的实现

首先读入数据：

```python
%matplotlib inline
import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

由于每个token 是一个数值的index，我们需要将其转换为 one-hot 编码，ont-hot 向量的长度为所有token 的数量，某个token 的 one-hot向量就是在相应的token index数值位置是1 其余都是0，如：

```python
F.one_hot(torch.tensor([0, 2]), len(vocab))

tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0]])
```

还需要注意的一点是我们每次抽样得到的mini-batch 的大小为 （批量大小*时间步），比如下面数据的批量大小是2，时间步是5：

```python
X = torch.arange(10).reshape((2, 5))
X

tensor([[0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9]])
```

在进行one-hot转换后我们需要将其转换为 （时间步*批量大小\*token数量），这样方便进行时间步的取样：

```python
X.T

tensor([[0, 5],
        [1, 6],
        [2, 7],
        [3, 8],
        [4, 9]])

F.one_hot(X.T,len(vocab))
tensor([[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0]],

        [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0]],

        [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0]],

        [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0]],

        [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0]]])

F.one_hot(X.T, 28).shape
torch.Size([5, 2, 28])
```

接下来我们需要初始化模型的参数（也就是上面讲过的 3 个 W 和两个 b）：

```python
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01##为什么0.01

    # Hidden layer parameters
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    # Output layer parameters
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # Attach gradients
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params
```

在 RNN 模型中第一个时间步是没有上一个隐状态传过来的，因此需要初始化一个状态，这里使用全0来初始化：

```python
def init_rnn_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )
```

下面的 rnn 函数定义了在每个时间步如何计算隐状态和输出，注意前面我们将批量的输入进行了转置，使得最外面的维度表示时间步，因此下面是对每个时间步的批量进行迭代运算：

```python
def rnn(inputs, state, params):
    # Here `inputs` shape: (`num_steps`, `batch_size`, `vocab_size`)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # Shape of `X`: (`batch_size`, `vocab_size`)
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)##基于上一个 H 和这一步的 X 来更新这一步的 H
        Y = torch.mm(H, W_hq) + b_q ##基于这一步的 H来预测下一步的Y
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)
```

注意最后一句输出将 output 在第 0 个维度上拼起来了：

```python
torch.cat(outputs, dim=0).shape
#torch.Size([10, 28])
```

其实 RNN 的每个批量的输出和多分类问题是一样的，也就是说虽然批量是 2，但是有 5 个时间步，因此也就相当于有 10 个 “样本”，每个样本的输出都是词汇表长度的向量（28），也就是进行28 类的预测。

现在所有的函数都有了，我们可以定义一个 RNN 模型将这些函数包装到一起：

```python
class RNNModelScratch: #@save
    """A RNN Model implemented from scratch."""
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)
```

测试一下：

```python
num_hiddens = 512
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)
state = net.begin_state(X.shape[0], d2l.try_gpu())
Y, new_state = net(X.to(d2l.try_gpu()), state)
Y.shape, len(new_state), new_state[0].shape
#(torch.Size([10, 28]), 1, torch.Size([2, 512]))
```

#### 预测

在训练模型之前先来看看这么用这个 RNN 模型进行预测，预测分为两步：

* warm-up：根据用户提供的起始字符（**prefix**）来计算这些字符的隐状态，但是不需要输出（因为已经提供输出了）
* 预测：基于上一步计算的隐状态继续后面的隐状态和输出的生成

```python
def predict_ch8(prefix, num_preds, net, vocab, device):  #@save
    """Generate new characters following the `prefix`."""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]] ##第一个output就是提供的字符的第一个
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))##output 的最后一个作为 input
    for y in prefix[1:]:  # Warm-up period
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # Predict `num_preds` steps
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```

试一下：

```python
predict_ch8('time traveller ', 10, net, vocab, d2l.try_gpu())
'time traveller knr<unk>knr<unk>kn'
```

#### 梯度裁剪

当我们计算的时间步比较长的时候，由于多个矩阵相乘可能会导致梯度爆炸或梯度消失，因此在更新梯度时，如果梯度过小或过大则采取梯度裁剪的方法限制其大小：

$$
\mathbf{g} \leftarrow \min\left(1, \frac{\theta}{\|\mathbf{g}\|}\right) \mathbf{g}.
$$

$||g||$ 表示梯度的 L2 范数：

```python
def grad_clipping(net, theta):  #@save
    """Clip the gradient."""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
```

#### 训练

RNN 的训练有几点不同：

* 序列数据不同的采样方法会导致隐状态的初始化不同

  * 对于顺序采样，每个批量的时间步是相邻的，因此上一个批量的最终隐状态可以传递到下一个批量的初始隐状态，但是需要将梯度移除（`detach_`）
  * 对于随机采样，每个批量不一定相邻，因此每次都需要初始化最初的隐状态
* 在更新模型参数之前将梯度进行裁剪
* 用困惑度（perplexity）来衡量模型

```python
#@save
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """Train a net within one epoch (defined in Chapter 8)."""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # Sum of training loss, no. of tokens
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # Initialize `state` when either it is the first iteration or
            # using random sampling
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # `state` is a tensor for `nn.GRU`
                state.detach_()
            else:
                # `state` is a tuple of tensors for `nn.LSTM` and
                # for our custom scratch implementation
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            # Since the `mean` function has been invoked
            updater(batch_size=1)
        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()
```

接下来就可以进行训练了：

```python
#@save
def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """Train a model (defined in Chapter 8)."""
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # Initialize
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # Train and predict
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    print(f'perplexity {ppl:.1f}, {speed:.1f} tokens/sec on {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))

num_epochs, lr = 500, 1
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())
```

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220428231806-47oxqky.png" style="zoom:67%;" />

#### Pytorch API 实现 RNN

导入数据：

```python
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

使用 pytorch 的高阶 API 来构建循环神经网络需要使用 `nn.RNN` 类，实例化这个类需要提供词汇表和隐藏层的大小：

```python
num_hiddens = 256
rnn_layer = nn.RNN(len(vocab), num_hiddens)
```

RNN 层的输入是序列 X 和初始的隐状态，因此我们需要初始化隐状态，性状是（隐藏层的数量，批量大小，隐藏层的大小）：

```python
state = torch.zeros((1, batch_size, num_hiddens))##一个隐藏层
state.shape

torch.Size([1, 32, 256])
```

RNN 层的输出有两个：所有中间的隐藏状态（也就是 Y）以及最后一个隐藏状态（H），要注意这里的 Y 和上面算的 Y 不一样，这里仅仅是隐状态，没有通过输出层转化，所以 Y 的最后一个元素就是输出的 状态：

```python
X = torch.rand(size=(num_steps, batch_size, len(vocab)))
Y, state_new = rnn_layer(X, state)
Y.shape, state_new.shape

(torch.Size([35, 32, 256]), torch.Size([1, 32, 256]))

torch.all(Y[-1] == state_new)##判断最后一个元素是否等于输出的 state
tensor(True)
```

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220528120925-bqsgwri.png" style="zoom:67%;" />



定义整个的 RNN 类：

```python
#@save
class RNNModel(nn.Module):
    """The RNN model."""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # If the RNN is bidirectional (to be introduced later),
        # `num_directions` should be 2, else it should be 1.
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        # The fully connected layer will first change the shape of `Y` to
        # (`num_steps` * `batch_size`, `num_hiddens`). Its output shape is
        # (`num_steps` * `batch_size`, `vocab_size`).
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # `nn.GRU` takes a tensor as hidden state
            return  torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens),
                                device=device)
        else:
            # `nn.LSTM` takes a tuple of hidden states
            return (torch.zeros((
                self.num_directions * self.rnn.num_layers,
                batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device))
```

训练和预测：

```python
device = d2l.try_gpu()
net = RNNModel(rnn_layer, vocab_size=len(vocab))
net = net.to(device)
d2l.predict_ch8('time traveller', 10, net, vocab, device)

'time travellerlysysssyss'
##训练
num_epochs, lr = 500, 1
d2l.train_ch8(net, train_iter, vocab, lr, num_epochs, device)
```

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220528185452-pk22g58.png" style="zoom:67%;" />

### 经典循环神经网络

#### GRU

前面的 RNN 在计算每个输入时都会考虑包含之前所有输入的隐状态，但是对于一个序列而已不是每一个部分都是同等重要的，有些时候需要更关注某些观测值，有时则需要跳过某些观测值。GRU 通过在 RNN 的基础上引入两个门控单元：重置门（reset gate ，**R**）和更新门（update gate，**Z**）来决定**当前的隐状态的更新是否和当前的输入相关**：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220528212128-ujzx5w5.png" style="zoom:67%;" />



这两个门其实就是有着 sigmoid 激活函数的全连接层，接着通过重置门我们可以得到候选隐状态：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220528212419-og1vwiw.png" style="zoom:67%;" />



这个 $\odot$ 表示按元素相乘，因为这两个门的结果都是经过  sigmoid 函数的，也就是在 0 到 1 之间，如果 R 为 0，那么这个候选隐状态就没有考虑之前的隐状态，相当于将当前输入输进一个 MLP 得到的结果；如果 R 为 1，那么这个候选隐状态就和之前 RNN 得到的结果是一样的了。

接着基于候选隐状态和更新门得到最终的隐状态：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220528212906-0l3b8h1.png" style="zoom:67%;" />



因此这个 Z 决定了如何去更新当前的隐状态：如果 Z 为 0，则候选隐状态为当前的隐状态，如果 Z 为 1，则完全不考虑当前的输入。R 是对之前信息的遗忘程度，Z 是对当前信息的关注程度。

##### GRU 实现

读入数据：

```python
import torch
from torch import nn
from d2l import torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

高斯分布初始化参数：

```python
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device)*0.01

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    W_xz, W_hz, b_z = three()  # Update gate parameters
    W_xr, W_hr, b_r = three()  # Reset gate parameters
    W_xh, W_hh, b_h = three()  # Candidate hidden state parameters
    # Output layer parameters
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # Attach gradients
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

##初始化隐状态
def init_gru_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device), )
```

定义模型：

```python
def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)
```

训练，预测：

```python
vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
model = d2l.RNNModelScratch(len(vocab), num_hiddens, device, get_params,
                            init_gru_state, gru)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

也可以使用 pytorch 的 API 来实现：

```python
num_inputs = vocab_size
gru_layer = nn.GRU(num_inputs, num_hiddens)
model = d2l.RNNModel(gru_layer, len(vocab))
model = model.to(device)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220528214618-m3fy1kp.png" style="zoom:67%;" />



#### LSTM

LSTM 和 GRU 的很多设计类似，但是比 GRU 早了 20 年，LSTM 引入了记忆单元（memory cell），和隐状态的大小一样，也可以看作是另一种隐状态。LSTM 使用了 3 个门控单元：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220528223937-9r308oa.png" style="zoom:50%;" />



这些门和 GRU 里面的一样，都是由 sigmoid 激活函数的全连接网络；LSTM 还有一个候选记忆单元：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220528224149-bz5bc3g.png" style="zoom:50%;" />



这个所谓的候选记忆单元和之前的 RNN 里面的隐状态的计算方式是一样的，没有用到门控，因此这个候选记忆单元相当于储存了当前的记忆。除了隐状态之外，LSTM 还有记忆单元 C：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220528224353-sf5ih2j.png" style="zoom:50%;" />



当前的记忆单元的计算涉及到上一个记忆单元，候选记忆单元以及当前的两个门控单元，注意这里的两个门控单元是独立的，而不像前面 GRU 中一个是 Z 另一个就是 1-Z，也就是说可以同时用到前一个记忆单元和当前的记忆单元，也可以都不用（相当于重置了记忆），接着就是隐状态的更新：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220528224849-i85438t.png" style="zoom:50%;" />



tanh 的目的是对 C 进行缩放，因为从上面计算记忆单元的式子来看，得到的结果不一定处于 -1~1 之间。如果这个输出门为 1，那么隐状态就包含了前一个记忆，前一个隐状态，以及当前的输入，如果为 0，则重置隐状态。总结一下：遗忘门控制着对之前记忆的保留程度，输入门控制着当前记忆的保留程度，输出门则控制着对前两个门的计算结果的输出

##### LSTM 的实现

导入数据：

```python
import torch
from torch import nn
from d2l import torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

初始化参数：

```python
def get_lstm_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device)*0.01

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    W_xi, W_hi, b_i = three()  # Input gate parameters
    W_xf, W_hf, b_f = three()  # Forget gate parameters
    W_xo, W_ho, b_o = three()  # Output gate parameters
    W_xc, W_hc, b_c = three()  # Candidate memory cell parameters
    # Output layer parameters
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # Attach gradients
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
              b_c, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

##初始化状态，这里有状态和记忆单元
def init_lstm_state(batch_size, num_hiddens, device):
    return (torch.zeros((batch_size, num_hiddens), device=device),
            torch.zeros((batch_size, num_hiddens), device=device))
```

定义模型：

```python
def lstm(inputs, state, params):
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c,
     W_hq, b_q] = params
    (H, C) = state
    outputs = []
    for X in inputs:
        I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
        F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
        O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
        C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * torch.tanh(C)
        Y = (H @ W_hq) + b_q##O不参与输出计算
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H, C)
```

训练和预测：

```python
vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
model = d2l.RNNModelScratch(len(vocab), num_hiddens, device, get_lstm_params,
                            init_lstm_state, lstm)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

也可以直接使用 Pytorch 的 API：

```python
num_inputs = vocab_size
lstm_layer = nn.LSTM(num_inputs, num_hiddens)
model = d2l.RNNModel(lstm_layer, len(vocab))
model = model.to(device)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220528230544-s3plklg.png" style="zoom:67%;" />

#### 深度循环神经网络

深度循环神经网络就是使用更多的隐藏层，每个隐藏层接受上一个隐藏层的输入，输出是新的隐状态，这个新的隐状态一方面向下一步传递，另一方面向该步的下一个隐藏层传递：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220529083050-v2y45ji.png" style="zoom:50%;" />



代码：

```python
##数据
import torch
from torch import nn
from d2l import torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

##参数和模型
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size
device = d2l.try_gpu()
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers)
model = d2l.RNNModel(lstm_layer, len(vocab))
model = model.to(device)

##预测和训练
num_epochs, lr = 500, 2
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

#### 双向循环神经网络

双向循环神经网络通过加入隐状态的反向传递，从而利用 "未来" 的信息（因此不适宜做推理任务，因为在推理预测任务中模型看不到未来的观测）：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220529120831-5709iik.png" style="zoom:67%;" />



代码：

```python
import torch
from torch import nn
from d2l import torch as d2l

# Load data
batch_size, num_steps, device = 32, 35, d2l.try_gpu()
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
# Define the bidirectional LSTM model by setting `bidirectional=True`
vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers, bidirectional=True)
model = d2l.RNNModel(lstm_layer, len(vocab))
model = model.to(device)
# Train the model
num_epochs, lr = 50, 1
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
```

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220529120858-y4lcxvg.png" style="zoom:67%;" />

### 机器翻译和数据集

机器翻译也就是从一个序列转化成另一个序列，属于一种序列转化模型（sequence transduction），输入和输出都是长度可变的序列，因此和之前讲的语言模型的数据预处理过程有不同的地方。

#### 数据下载和预处理

这里使用的数据是来自 Tatoeba 项目的双语句子对（英语对法语），每一行是一句英语和对应的法语翻译，因此在这个数据集中英语是源语言（source），法语是目标语言（target）：

```python
import os
import torch
from d2l import torch as d2l

##下载数据
#@save
d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip',
                           '94646ad1522d915e7b0f9296181140edcf86a4f5')

##下载数据http://d2l-data.s3-accelerate.amazonaws.com/fra-eng.zip
##解压
!unzip fra-eng.zip
#@save
def read_data_nmt(data_dir):
    """Load the English-French dataset."""
    #data_dir = d2l.download_extract('fra-eng')##已经下载了
    with open(os.path.join(data_dir, 'fra.txt'), 'r') as f:
        return f.read()

raw_text = read_data_nmt("./fra-eng/")
print(raw_text[:75])

Go.	Va !
Hi.	Salut !
Run!	Cours !
Run!	Courez !
Who?	Qui ?
Wow!	Ça alors !
```

接下来需要进行一些预处理的操作，比如将不换行空格转化成空格，将大写转化成小写，在次和标点之间插入空格：

> 编辑器一般会把自动换行放在空格字符处。但是，有些文本内容在排版时不适合被放在连续的一行行尾与下一行行首。例如：“100 km”，就不应该在其中间的那个空格处换行。所以编辑器应该在"100"与"km"之间放置一个“不换行空格”

```python
#@save
def preprocess_nmt(text):
    """Preprocess the English-French dataset."""
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # Replace non-breaking space with space, and convert uppercase letters to
    # lowercase ones
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # Insert space between words and punctuation marks
    ##不是第一个字符，该字符为标点且前一个字符不是空格，那么就在该字符前面插入一个空格
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)

text = preprocess_nmt(raw_text)
print(text[:80])

go .	va !
hi .	salut !
run !	cours !
run !	courez !
who ?	qui ?
wow !	ça alors !
```

和之前一样，这里也需要进行 Tokenization，只不过之前是以字符进行 Tokenization，在机器翻译中可以用词来 Tokenization：

```python
#@save
def tokenize_nmt(text, num_examples=None):
    """Tokenize the English-French dataset."""
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
     return source, target

source, target = tokenize_nmt(text)
source[:6], target[:6]

go .	va !
hi .	salut !
run !	cours !
run !	courez !
who ?	qui ?
wow !	ça alors !
```

由于机器翻译数据集由一对序列构成，因此我们需要对这一对语言构建两个词汇表，对以词为基础的Tokenization来说，词汇表的大小要比字符的Tokenization大得多（因为字符只有26个），所以将出现频次小于 2 的词弄成 `<unk>` 的 token，另外还要加上一些特殊的 token ，如 `padding` (使得小批量中的序列长度一致)，`<bos>` ，`<eos>` ：

```python
src_vocab = d2l.Vocab(source, min_freq=2,
                      reserved_tokens=['<pad>', '<bos>', '<eos>'])
len(src_vocab)

10012
```

前面在语言模型中，一个小批量中的序列长度都是 `num_steps` ，在机器翻译中，每个实例都是一对句子，并且句子的长度还可能不一样，为了计算效率，仍然需要小批量的训练模型，因此在机器翻译中还是以 `num_steps` 来定义小批量中序列的长度，如果一个文本序列比 `num_steps` 短，那么就在末尾添加 `<pad>` 来使得长度等于 `num_steps`，如果比 `num_steps` 长，那么就直接截断：

```python
#@save
def truncate_pad(line, num_steps, padding_token):
    """Truncate or pad sequences."""
    if len(line) > num_steps:
        return line[:num_steps]  # Truncate
    return line + [padding_token] * (num_steps - len(line))  # Pad

truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>'])

[47, 4, 1, 1, 1, 1, 1, 1, 1, 1]
```

 还需要在每个句子的结尾用 `<eos>` 表示句子的结束，当模型预测出一个 `<eos>` 时表示这个句子该结束了；另外还返回了每个句子的实际长度（除去 `<pad>` token），这个信息在后面的模型中会用到：

```python
#@save
def build_array_nmt(lines, vocab, num_steps):
    """Transform text sequences of machine translation into minibatches."""
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = torch.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len
```

最后将上面这些函数放到一起：

```python
#@save
def load_data_nmt(batch_size, num_steps, num_examples=600):
    """Return the iterator and the vocabularies of the translation dataset."""
    text = preprocess_nmt(read_data_nmt("fra-eng/"))
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = d2l.Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = d2l.Vocab(target, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab

##测试一下
train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)
for X, X_valid_len, Y, Y_valid_len in train_iter:
    print('X:', X.type(torch.int32))
    print('valid lengths for X:', X_valid_len)
    print('Y:', Y.type(torch.int32))
    print('valid lengths for Y:', Y_valid_len)
    break

X: tensor([[ 12, 131, 132,   4,   3,   1,   1,   1],
        [ 68,  60,   4,   3,   1,   1,   1,   1]], dtype=torch.int32)
valid lengths for X: tensor([5, 4])
Y: tensor([[44,  0,  4,  3,  1,  1,  1,  1],
        [64, 53,  4,  3,  1,  1,  1,  1]], dtype=torch.int32)
valid lengths for Y: tensor([4, 4])
```

### 编码-解码架构

对于输入和输出都是可变长度的序列数据，我们可以设计一个有两个元素的模型架构来处理这种类型的数据：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220601160137-mx947ou.png" style="zoom:67%;" />



第一个组件是编码器（encoder）：输入是可变长度的序列，输出是固定性状的中间状态；第二个组件是解码器（edcoder）：输入是 encoder 生成的状态，输出是可变长度的序列，这个架构就是编码-解码架构。

代码（架构，不涉及具体实现）：

```python
from torch import nn

##编码器
#@save
class Encoder(nn.Module):
    """The base encoder interface for the encoder-decoder architecture."""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError
```

在 decoder 中需要一个额外的函数（`init_state`）来将 encoder 的输出转化成状态，这一步可能需要一些其他的输入（比如前面提到的序列除去 padding 的有效长度）：

```python
#@save
class Decoder(nn.Module):
    """The base decoder interface for the encoder-decoder architecture."""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError
```

结合两者：

```python
#@save
class EncoderDecoder(nn.Module):
    """The base class for the encoder-decoder architecture."""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)
```

### Seq2seq 模型

Seq2seq 模型就是上面说的编码-解码架构的一个实例：encoder 和 decoder 都是 RNN。

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220601183741-iewypl2.png" style="zoom: 50%;" />



Encoder 是一个常规的 RNN，将隐状态输入进 Decoder 中（如果是多层 RNN 则将最后一层的 RNN 最后一个时刻的隐状态的输出作为 Decoder 的输入），也就是将输入的序列信息编码进这个隐状态中；Decoder 的设计可以有几种选择，比如上面图所示的，第一个时刻接受 encoder 的隐状态和 `<bos>` token，然后后面和一般的 RNN 差不多，另外一种就是在每个时刻都将 Decoder 输出的隐状态和每个时刻的序列同时作为输入，如下图：

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/image-20220601185525-2kz5uix.png" style="zoom:67%;" />



Decoder 预测就是将上一个时刻的输出作为下一个时刻的输入来生成序列。注意，由于 Encoder 可以看到整个序列，所以也可以使用之前讲过的双向 RNN 作为 Encoder（Encoder 起到一个特征提取的作用）。



参考资料：

- 动手学深度学习
- [(强推)李宏毅2021/2022春机器学习课程_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV1Wv411h7kN?p=40)

