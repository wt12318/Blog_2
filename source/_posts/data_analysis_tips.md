---
title: 数据分析 Tips
date: 2022-09-20 09:14:18
tags: data analysis
index_img: img/data_analysis.png
sticky: 100
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

### 解析命令行参数

使用 argparse 包解析参数：

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-a", "--aaindex", help="The aaindex file path")
parser.add_argument("-m", "--model", help="The model .pth file path")
parser.add_argument("-i", "--input", help="The input file")
parser.add_argument("-o", "--outdir", help="The dir of output file")

args = parser.parse_args()
input_file = os.path.abspath(os.path.expanduser(args.input))
```

### Pytorch 模型载入已有的权重

[python - PyTorch: passing numpy array for weight initialization - Stack Overflow](https://stackoverflow.com/questions/51628607/pytorch-passing-numpy-array-for-weight-initialization)

[Initialize nn.Linear with specific weights - PyTorch Forums](https://discuss.pytorch.org/t/initialize-nn-linear-with-specific-weights/29005/6)

[deep learning - PyTorch - unexpected shape of model parameters weights - Stack Overflow](https://stackoverflow.com/questions/61532695/pytorch-unexpected-shape-of-model-parameters-weights) pytorch 中的权重形状是反的

```python
class mut(torch.nn.Module):
    def __init__(self,weights):
        super().__init__()
  
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(4539, 1000),
            torch.nn.ReLU(),
            torch.nn.Linear(1000, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 50),
            torch.nn.ReLU()
        )
    
        with torch.no_grad():
            self.encoder[0].weight.copy_(torch.tensor(weights[0][0]).t())
            self.encoder[0].bias.copy_(torch.tensor(weights[0][1]))
        
            self.encoder[2].weight.copy_(torch.tensor(weights[1][0]).t())
            self.encoder[2].bias.copy_(torch.tensor(weights[1][1]))
        
            self.encoder[4].weight.copy_(torch.tensor(weights[2][0]).t())
            self.encoder[4].bias.copy_(torch.tensor(weights[2][1]))
    
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(50, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 1000),
            torch.nn.ReLU(),
            torch.nn.Linear(1000, 4539)
        )
    
        with torch.no_grad():
            self.decoder[0].weight.copy_(torch.tensor(weights[3][0]).t())
            self.decoder[0].bias.copy_(torch.tensor(weights[3][1]))
        
            self.decoder[2].weight.copy_(torch.tensor(weights[4][0]).t())
            self.decoder[2].bias.copy_(torch.tensor(weights[4][1]))
        
            self.decoder[4].weight.copy_(torch.tensor(weights[5][0]).t())
            self.decoder[4].bias.copy_(torch.tensor(weights[5][1]))
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.float()

mut_weight = pickle.load(open('/root/autodl-tmp/premodel_tcga_mut_1000_100_50.pickle', 'rb'))
model = mut(mut_weight) 
```

### 安装 jupyterhub

[jupyterhub-the-hard-way/installation-guide-hard.md at 35ddfb49ad81771551c6549696ccec960564d5e4 · jupyterhub/jupyterhub-the-hard-way (github.com)](https://github.com/jupyterhub/jupyterhub-the-hard-way/blob/HEAD/docs/installation-guide-hard.md)

```shell
apt install python3.8-venv
sudo /opt/jupyterhub/bin/python3 -m pip install wheel -i https://pypi.tuna.tsinghua.edu.cn/simple
sudo /opt/jupyterhub/bin/python3 -m pip install jupyterhub jupyterlab -i https://pypi.tuna.tsinghua.edu.cn/simple
sudo /opt/jupyterhub/bin/python3 -m pip install ipywidgets -i https://pypi.tuna.tsinghua.edu.cn/simple
sudo apt install nodejs npm
sudo npm install -g configurable-http-proxy

sudo mkdir -p /opt/jupyterhub/etc/jupyterhub/
cd /opt/jupyterhub/etc/jupyterhub/
sudo /opt/jupyterhub/bin/jupyterhub --generate-config
###c.Spawner.default_url = '/lab'
sudo mkdir -p /opt/jupyterhub/etc/systemd

vi /opt/jupyterhub/etc/systemd/jupyterhub.service
[Unit]
Description=JupyterHub
After=syslog.target network.target

[Service]
User=root
Environment="PATH=/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/opt/jupyterhub/bin"
ExecStart=/opt/jupyterhub/bin/jupyterhub -f /opt/jupyterhub/etc/jupyterhub/jupyterhub_config.py

[Install]
WantedBy=multi-user.target

sudo ln -s /opt/jupyterhub/etc/systemd/jupyterhub.service /etc/systemd/system/jupyterhub.service
sudo systemctl daemon-reload
sudo systemctl enable jupyterhub.service
sudo systemctl start jupyterhub.service ##或者 reboot
sudo systemctl status jupyterhub.service

###添加自己的环境
/path/to/env/python -m ipykernel install --user --name 'env name' --display-name "env name"
```

### 如何保存 R studio Viewer 上的图片

使用 webshot2:

```shell
##需要安装
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo apt install ./google-chrome-stable_current_amd64.deb
sudo apt install --assume-yes chromium-browser
library(htmlwidgets)
library(webshot2)

p <- get_plot("data/citup/pt1_frq.csv","data/citup/pt1_nodes.csv")
saveWidget(p, "temp.html")
webshot("temp.html", "temp.png")
```

### 词云图

从 Pubmed 上下载摘要绘制词云图，参考 [Pubmedwordcloud](http://felixfan.github.io/PubMedWordcloud/)

```r
install.packages("PubMedWordcloud")
library(PubMedWordcloud)
pmids <- unique(neodb_all$PMID) %>% gsub("_IEDB","",.) %>% unique() %>% na.omit()
abstracts <- getAbstracts(pmids,s = 50)
cleanAbs=cleanAbstracts(abstracts)
colors=colSets(type="Paired")
plotWordCloud(cleanAbs,min.freq = 2, scale = c(2, 0.3),colors=colors)
```

### 绘制序列比对的图

先用 Biostrings 的 pairwiseAlignment 得到比对的结果，然后使用 ggmsa 绘图：

```r
library(ggmsa)
library(Biostrings)
seq1 <- "TRAATGRMV"
seq2 <- "GEKFRASSLKFPGL"
data(BLOSUM50)
globalAligns <- pairwiseAlignment(seq1, seq2, substitutionMatrix = "BLOSUM50", gapOpening = -2,gapExtension = -8, scoreOnly = FALSE)
aa <- list(seq1=globalAligns@pattern,seq2=globalAligns@subject)
write.fasta(aa,names = names(aa),file.out = "data/test_msa.fasta")
readAAStringSet("data/test_msa.fasta") -> test_msa
ggmsa(test_msa, char_width = 0.5, seq_name = T)+
  labs(title=paste0("Score = ",globalAligns@score))##加上序列比对的得分
```

![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/undefinedimage.png)

### 打印特殊字符

[r - How do I deal with special characters like ^$.?*|+()[{ in my regex? - Stack Overflow](https://stackoverflow.com/questions/27721008/how-do-i-deal-with-special-characters-like-in-my-regex)

```python
library(rebus)
literal(".")
```

### 将 R 画的图添加到多页 PDF

```R
pdf(file = "fig/kras_hras_or.pdf", onefile=TRUE, width = 8,height = 6)
for(i in 1:2){
  grid.newpage()
  print(eval(parse(text = paste0("p",i)))) ###这个图是用 grid 绘图系统画的
}
dev.off()
```

