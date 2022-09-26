---
title: 数据分析 Tips
date: 2022-09-20 09:14:18
tags: data analysis
index_img: img/data_analysis.png
categories:
  - data analysis
---

数据分析 Tips

<!-- more -->

### 获取列表元素的 index

如果知道元素，可以使用 `.index` 方法就行：

```python
a = ["a","b","c"]
a.index("a")
```

如果是一个逻辑值的列表，判断 TRUE 的位置：

```python
>>> t = [False, False, False, False, True, True, False, True, False, False, False, False, False, False, False, False]
>>> [i for i, x in enumerate(t) if x]
[4, 5, 7]
```

对于大的列表，可以使用 `itertools` 中的 `compress` 方法：

```python
>>> from itertools import compress
>>> list(compress(xrange(len(t)), t))
[4, 5, 7]
>>> t = t*1000
>>> %timeit [i for i, x in enumerate(t) if x]
100 loops, best of 3: 2.55 ms per loop
>>> %timeit list(compress(xrange(len(t)), t))
1000 loops, best of 3: 696 µs per loop
```

### 如何理解 scipy 中的 csr_matrix

csr_matrix 是稀疏矩阵的存储形式，由3个列表构成：`indptr` 表示前 i - 1 行有几个非零元素, `indices` 表示非零元素的列, `data` 是非零元素:

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/image-20220926145322281.png" alt="image-20220926145322281" style="zoom: 50%;" />

因此矩阵中第i行非零元素的列号为 `indices[indptr[i]:indptr[i+1]]`，相应的值为 `data[indptr[i]:indptr[i+1]]`

### 将 jupyter notebook 转化成 PDF

```shell
jupyter nbconvert --to webpdf --allow-chromium-download your-notebook-file.ipynb
```

### 如何获取模型的中间层的结果

例如一个 Autoencoder 模型如下：

```python
AE(
  (encoder): Sequential(
    (0): Linear(in_features=6343, out_features=2000, bias=True)
    (1): ReLU()
    (2): Linear(in_features=2000, out_features=50, bias=True)
    (3): ReLU()
    (4): Linear(in_features=50, out_features=50, bias=True)
    (5): ReLU()
  )
  (decoder): Sequential(
    (0): Linear(in_features=50, out_features=50, bias=True)
    (1): ReLU()
    (2): Linear(in_features=50, out_features=2000, bias=True)
    (3): ReLU()
    (4): Linear(in_features=2000, out_features=6343, bias=True)
  )
)
```

获取 encoder 的结果：

```python
###获取 encoder 的结果
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output
    return hook
ae_model.eval()
ae_model.encoder.register_forward_hook(get_activation('encoder'))
output = ae_model(cell_net[0][0].ae_gene)
decoder_out = activation['encoder'].cpu().detach()

decoder_out.shape
torch.Size([50])
```

### 如何只使用模型的一部分权重

对于预训练模型，有时我们只需要用到部分层的权重，例如自编码器，如果用作特征提取，我们可能只需要前面的 encoder 部分（参考 [How to load part of pre trained model? - PyTorch Forums](https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/3)）：

```python
encoder_model
AE_encoder(
  (encoder): Sequential(
    (0): Linear(in_features=6343, out_features=2000, bias=True)
    (1): ReLU()
    (2): Linear(in_features=2000, out_features=50, bias=True)
    (3): ReLU()
    (4): Linear(in_features=50, out_features=50, bias=True)
    (5): ReLU()
  )
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pretrained_dict = torch.load("modle_best_para.pt",map_location=device)
encoder_model_dict = encoder_model.state_dict()
# 1. filter out unnecessary keys
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in encoder_model_dict}
# 2. overwrite entries in the existing state dict
encoder_model_dict.update(pretrained_dict) 
# 3. load the new state dict
encoder_model.load_state_dict(encoder_model_dict)
```

### 对数据框进行多线程运算

使用 `pqdm`:

```python
from pqdm.processes import pqdm
pqdm(all_dt.iterrows(),some_function,n_jobs=5)
```

