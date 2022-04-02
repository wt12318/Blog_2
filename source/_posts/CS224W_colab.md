---
title: 图机器学习实践
date: 2022-04-02 19:14:18
tags: 深度学习
index_img: img/GNN.png
categories:
  - python
---



图机器学习 CS224W Colab 代码

<!-- more -->

## Colab 1

这个 colab 分为 3 个部分：

- 载入网络科学中经典的图：Karate Club Network 并探索这个图的一些图相关统计量；“空手道俱乐部网络” 有两个俱乐部（由之前的一个俱乐部分裂来的，分裂后的俱乐部各有一个领导），边表示成员之间的社会联系，有 34 个节点
- 将图结构转化为 PyTorch 的 Tensor
- 建立 node embedding 模型

###  Graph Basics

To start, we will load a classic graph in network science, the [Karate Club Network](https://en.wikipedia.org/wiki/Zachary%27s_karate_club). We will explore multiple graph statistics for that graph.

首先导入 NetworkX 包用于网络的操作：


```python
import networkx as nx
```

载入 Karate Club Network：


```python
G = nx.karate_club_graph()
# G is an undirected graph
type(G)
```


    networkx.classes.graph.Graph


```python
# Visualize the graph
nx.draw(G, with_labels = True)
```


![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/colab1_7_0.png)
​    

#### Question 1: karate club network 的平均自由度是多少？


```python
def average_degree(num_edges, num_nodes):
  # TODO: Implement this function that takes number of edges
  # and number of nodes, and returns the average node degree of 
  # the graph. Round the result to nearest integer (for example 
  # 3.3 will be rounded to 3 and 3.7 will be rounded to 4)

  avg_degree = 0

  ############# Your code here ############
  avg_degree = round((2 * num_edges) / num_nodes)
  #########################################

  return avg_degree

num_edges = G.number_of_edges()
num_nodes = G.number_of_nodes()
avg_degree = average_degree(num_edges, num_nodes)
print("Average degree of karate club network is {}".format(avg_degree))
```

    Average degree of karate club network is 5

#### Question 2: karate club network 的平均聚类系数是多少？


```python
def average_clustering_coefficient(G):
  # TODO: Implement this function that takes a nx.Graph
  # and returns the average clustering coefficient. Round 
  # the result to 2 decimal places (for example 3.333 will
  # be rounded to 3.33 and 3.7571 will be rounded to 3.76)

  avg_cluster_coef = 0

  ############# Your code here ############
  ## Note: 
  ## 1: Please use the appropriate NetworkX clustering function
  a = nx.clustering(G)
  avg_cluster_coef = sum(a.values())/len(a)
  avg_cluster_coef = round(avg_cluster_coef,2)

  #########################################

  return avg_cluster_coef

avg_cluster_coef = average_clustering_coefficient(G)
print("Average clustering coefficient of karate club network is {}".format(avg_cluster_coef))
```

    Average clustering coefficient of karate club network is 0.57

#### Question 3: 在一个 PageRank 迭代后，节点 0 的 PageRank 值是多少

PageRank equation: $r_j = \sum_{i \rightarrow j} \beta \frac{r_i}{d_i} + (1 - \beta) \frac{1}{N}$


```python
def one_iter_pagerank(G, beta, r0, node_id):
  # TODO: Implement this function that takes a nx.Graph, beta, r0 and node id.
  # The return value r1 is one interation PageRank value for the input node.
  # Please round r1 to 2 decimal places.

  r1 = 0

  ############# Your code here ############
  ## Note: 
  ## 1: You should not use nx.pagerank
  sum = 0
  adj_nodes = list(G[node_id])
  for i in adj_nodes:
    a = beta * (r0 / dict(G.degree([i]))[i])
    sum = sum + a
  r1 = sum + (1 - beta) * (1 / G.number_of_nodes()) 
  r1 = round(r1, 2)
  #########################################

  return r1

beta = 0.8
r0 = 1 / G.number_of_nodes()
node = 0
r1 = one_iter_pagerank(G, beta, r0, node)
print("The PageRank value for node 0 after one iteration is {}".format(r1))
```

    The PageRank value for node 0 after one iteration is 0.13

#### Question 4: 网络的邻近中心性（closeness centrality）是多少？

closeness centrality ： $c(v) = \frac{1}{\sum_{u \neq v}\text{shortest path length between } u \text{ and } v}$


```python
def closeness_centrality(G, node=5):
  # TODO: Implement the function that calculates closeness centrality 
  # for a node in karate club network. G is the input karate club 
  # network and node is the node id in the graph. Please round the 
  # closeness centrality result to 2 decimal places.

  closeness = 0

  ## Note:
  ## 1: You can use networkx closeness centrality function.
  ## 2: Notice that networkx closeness centrality returns the normalized 
  ## closeness directly, which is different from the raw (unnormalized) 
  ## one that we learned in the lecture.
  normalized = nx.closeness_centrality(G,node)
  closeness = normalized / (G.number_of_nodes() - 1)
  closeness = round(closeness,2)
  #########################################

  return closeness

node = 5
closeness = closeness_centrality(G, node=node)
print("The node 5 has closeness centrality {}".format(closeness))
```

    The node 5 has closeness centrality 0.01

###  Graph to Tensor

将图结构转化为 PyTorch 的 Tensor，便于进行后续的机器学习操作。

检测 pytorch 是否按照以及版本：


```python
import torch
print(torch.__version__)
```

    1.10.0+cu111

#### PyTorch tensor basics


```python
# Generate 3 x 4 tensor with all ones
ones = torch.ones(3, 4)
print(ones)

# Generate 3 x 4 tensor with all zeros
zeros = torch.zeros(3, 4)
print(zeros)

# Generate 3 x 4 tensor with random values on the interval [0, 1)
random_tensor = torch.rand(3, 4)
print(random_tensor)

# Get the shape of the tensor
print(ones.shape)
```

    tensor([[1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.]])
    tensor([[0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.]])
    tensor([[0.2417, 0.9127, 0.7875, 0.6463],
            [0.1192, 0.1317, 0.9079, 0.4481],
            [0.0022, 0.4382, 0.3800, 0.5075]])
    torch.Size([3, 4])


PyTorch tensor 有数据类型，用 `dtype` 表示：


```python
# Create a 3 x 4 tensor with all 32-bit floating point zeros
zeros = torch.zeros(3, 4, dtype=torch.float32)
print(zeros.dtype)

# Change the tensor dtype to 64-bit integer
zeros = zeros.type(torch.long)
print(zeros.dtype)
```

    torch.float32
    torch.int64

#### Question 5: 得到 karate club network 的边列表并转化为  `torch.LongTensor`.


```python
def graph_to_edge_list(G):
  # TODO: Implement the function that returns the edge list of
  # an nx.Graph. The returned edge_list should be a list of tuples
  # where each tuple is a tuple representing an edge connected 
  # by two nodes.
  ## 无向图 (0,1) 和 (1,0) 应该视为一条边

  edge_list = []

  ############# Your code here ############
  nodes = list(G.nodes())
  for i in nodes:
    for j in list(G[i]):
      if (i,j)[::-1] in edge_list:
        continue 
      edge_list.append((i,j))
  #########################################

  return edge_list

def edge_list_to_tensor(edge_list):
  # TODO: Implement the function that transforms the edge_list to
  # tensor. The input edge_list is a list of tuples and the resulting
  # tensor should have the shape [2 x len(edge_list)].##为什么要变成 2 * len(edge_list)

  edge_index = torch.tensor([])

  ############# Your code here ############
  edge_index = torch.tensor(edge_list).T
  #########################################

  return edge_index

pos_edge_list = graph_to_edge_list(G)
pos_edge_index = edge_list_to_tensor(pos_edge_list)
print("The pos_edge_index tensor has shape {}".format(pos_edge_index.shape))
print("The pos_edge_index tensor has sum value {}".format(torch.sum(pos_edge_index)))
```

    The pos_edge_index tensor has shape torch.Size([2, 78])
    The pos_edge_index tensor has sum value 2535

#### Question 6: 实现抽样负例边的函数并回答哪些边（edge_1 to edge_5）是负例边

负例边就是在图中并不存在的边，其标签也就是0。


```python
import random
from itertools import compress

def sample_negative_edges(G, num_neg_samples):
  # TODO: Implement the function that returns a list of negative edges.
  # The number of sampled negative edges is num_neg_samples. You do not
  # need to consider the corner case when the number of possible negative edges
  # is less than num_neg_samples. It should be ok as long as your implementation 
  # works on the karate club network. In this implementation, self loops should 
  # not be considered as either a positive or negative edge. Also, notice that 
  # the karate club network is an undirected graph, if (0, 1) is a positive 
  # edge, do you think (1, 0) can be a negative one?

  neg_edge_list = []

  ############# Your code here ############
  ##先得到所有的neg edges
  ###对于每个节点找出图中与其不相连的其他所有节点，构成 neg edge
  nodes = list(G.nodes()) 
  for i in nodes:
    node_tmp = nodes.copy()
    node_tmp.remove(i)
    neg_nodes = list(compress(node_tmp, [j not in list(G[i]) for j in node_tmp]))
    for k in neg_nodes:
      if (i,k)[::-1] in neg_edge_list:
        continue 
      neg_edge_list.append((i,k))
  ##对所有的neg edge list 进行抽样
  neg_edge_list = random.sample(neg_edge_list,num_neg_samples)
  #########################################

  return neg_edge_list

# Sample 78 negative edges
neg_edge_list = sample_negative_edges(G, len(pos_edge_list))

# Transform the negative edge list to tensor
neg_edge_index = edge_list_to_tensor(neg_edge_list)
print("The neg_edge_index tensor has shape {}".format(neg_edge_index.shape))

# Which of following edges can be negative ones?
edge_1 = (7, 1)
edge_2 = (1, 33)
edge_3 = (33, 22)
edge_4 = (0, 4)
edge_5 = (4, 2)

############# Your code here ############
## Note:
## 1: For each of the 5 edges, print whether it can be negative edge
def is_neg(G,edge):
  if (edge[1] in list(G[edge[0]])):
    print(f'{edge} is not negative edge.')
  else:
    print(f'{edge} is negative edge.')

is_neg(G,edge_1)
is_neg(G,edge_2)
is_neg(G,edge_3)
is_neg(G,edge_4)
is_neg(G,edge_5)
#########################################
```

    The neg_edge_index tensor has shape torch.Size([2, 78])
    (7, 1) is not negative edge.
    (1, 33) is negative edge.
    (33, 22) is not negative edge.
    (0, 4) is not negative edge.
    (4, 2) is negative edge.

### Node Emebedding Learning


```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

print(torch.__version__)
```

    1.10.0+cu111


为了建立 node embedding 模型，我们需要使用到 Pytorch 中的`nn.Embedding`  模块：


```python
# Initialize an embedding layer
# Suppose we want to have embedding for 4 items (e.g., nodes)
# Each item is represented with 8 dimensional vector

emb_sample = nn.Embedding(num_embeddings=4, embedding_dim=8)
print('Sample embedding layer: {}'.format(emb_sample))
```

    Sample embedding layer: Embedding(4, 8)


We can select items from the embedding matrix, by using Tensor indices


```python
# Select an embedding in emb_sample
id = torch.LongTensor([1])
print(emb_sample(id))

# Select multiple embeddings
ids = torch.LongTensor([1, 3])
print(emb_sample(ids))

# Get the shape of the embedding weight matrix
shape = emb_sample.weight.data.shape
print(shape)

# Overwrite the weight to tensor with all ones
emb_sample.weight.data = torch.ones(shape)

# Let's check if the emb is indeed initilized
ids = torch.LongTensor([0, 3])
print(emb_sample(ids))
```

    tensor([[-4.4258e-01, -1.5349e+00,  1.1118e-03, -8.3201e-01,  6.3567e-01,
             -7.7746e-01, -5.5710e-02, -4.4338e-02]], grad_fn=<EmbeddingBackward0>)
    tensor([[-4.4258e-01, -1.5349e+00,  1.1118e-03, -8.3201e-01,  6.3567e-01,
             -7.7746e-01, -5.5710e-02, -4.4338e-02],
            [ 4.5921e-01,  8.7926e-02,  9.4428e-01, -7.5985e-01,  1.6396e+00,
             -1.9154e+00, -1.8657e+00, -6.7076e-01]], grad_fn=<EmbeddingBackward0>)
    torch.Size([4, 8])
    tensor([[1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1.]], grad_fn=<EmbeddingBackward0>)

```python
emb_sample.weight.data##weight.data就是embedding matrix
```


    tensor([[1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1., 1., 1., 1.]])

现在我们可以来创建节点的 embedding 矩阵并初始化：

-  对于 karate club network 中的每个节点，进行 16 维向量的 embedding
- 使用 `[0,1]` 的均匀分布来初始化 embedding 矩阵


```python
# Please do not change / reset the random seed
torch.manual_seed(1)

def create_node_emb(num_node=34, embedding_dim=16):
  # TODO: Implement this function that will create the node embedding matrix.
  # A torch.nn.Embedding layer will be returned. You do not need to change 
  # the values of num_node and embedding_dim. The weight matrix of returned 
  # layer should be initialized under uniform distribution. 

  emb = None

  ############# Your code here ############
  emb = nn.Embedding(num_embeddings=num_node, embedding_dim=embedding_dim)
  emb.weight.data = torch.rand(emb.weight.data.shape)
  #########################################

  return emb

emb = create_node_emb()
ids = torch.LongTensor([0, 3])

# Print the embedding layer
print("Embedding: {}".format(emb))

# An example that gets the embeddings for node 0 and 3
print(emb(ids))
```

    Embedding: Embedding(34, 16)
    tensor([[0.2114, 0.7335, 0.1433, 0.9647, 0.2933, 0.7951, 0.5170, 0.2801, 0.8339,
             0.1185, 0.2355, 0.5599, 0.8966, 0.2858, 0.1955, 0.1808],
            [0.7486, 0.6546, 0.3843, 0.9820, 0.6012, 0.3710, 0.4929, 0.9915, 0.8358,
             0.4629, 0.9902, 0.7196, 0.2338, 0.0450, 0.7906, 0.9689]],
           grad_fn=<EmbeddingBackward0>)

#### Visualize the initial node embeddings

先来可视化未经训练的 embedding，以便于和之后的进行比较，这里使用 PCA 将 embedding 空间映射到二维空间来可视化：


```python
def visualize_emb(emb):
  X = emb.weight.data.numpy()##转化成 numpy
  pca = PCA(n_components=2)
  components = pca.fit_transform(X)
  plt.figure(figsize=(6, 6))
  club1_x = []
  club1_y = []
  club2_x = []
  club2_y = []
  for node in G.nodes(data=True):
    if node[1]['club'] == 'Mr. Hi':
      club1_x.append(components[node[0]][0])
      club1_y.append(components[node[0]][1])
    else:
      club2_x.append(components[node[0]][0])
      club2_y.append(components[node[0]][1])
  plt.scatter(club1_x, club1_y, color="red", label="Mr. Hi")
  plt.scatter(club2_x, club2_y, color="blue", label="Officer")
  plt.legend()
  plt.show()

# Visualize the initial random embeddding
visualize_emb(emb)
```



![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/colab1_38_0.png)
    

#### Question 7: Training the embedding

We want to optimize our embeddings for the task of classifying edges as positive or negative. Given an edge and the embeddings for each node, the dot product of the embeddings, followed by a sigmoid, should give us the likelihood of that edge being either positive (output of sigmoid > 0.5) or negative (output of sigmoid < 0.5).


```python
from torch.optim import SGD
import torch.nn as nn

def accuracy(pred, label):
  # TODO: Implement the accuracy function. This function takes the 
  # pred tensor (the resulting tensor after sigmoid) and the label 
  # tensor (torch.LongTensor). Predicted value greater than 0.5 will 
  # be classified as label 1. Else it will be classified as label 0.
  # The returned accuracy should be rounded to 4 decimal places. 
  # For example, accuracy 0.82956 will be rounded to 0.8296.

  accu = 0.0

  ############# Your code here ############
  a = pred > 0.5
  a = a.type(torch.LongTensor)
  accu = sum(a == label)/len(a)
  accu = round(accu.item(),4)
  #########################################

  return accu

def train(emb, loss_fn, sigmoid, train_label, train_edge):
  # TODO: Train the embedding layer here. You can also change epochs and 
  # learning rate. In general, you need to implement: 
  # (1) Get the embeddings of the nodes in train_edge
  # (2) Dot product the embeddings between each node pair
  # (3) Feed the dot product result into sigmoid
  # (4) Feed the sigmoid output into the loss_fn
  # (5) Print both loss and accuracy of each epoch 
  # (6) Update the embeddings using the loss and optimizer 
  # (as a sanity check, the loss should decrease during training)

  epochs = 500
  learning_rate = 0.1

  optimizer = SGD(emb.parameters(), lr=learning_rate, momentum=0.9)

  for i in range(epochs):

    ############# Your code here ############
    pred = torch.empty(train_label.shape)
    for i in range(train_edge.shape[1]):
      ##两个节点的embedding
      em1 = emb(torch.tensor(train_edge[0,i]))
      em2 = emb(torch.tensor(train_edge[1,i]))
      pred_t = sigmoid(torch.dot(em1,em2))
      pred[i] = pred_t
    loss = loss_fn(pred,train_label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epochs % 50 == 0:
      loss = loss.item()
      acc = accuracy(pred,train_label)
      print(f"loss: {loss:>4f}, acc: {acc:>4f}")
    #########################################

loss_fn = nn.BCELoss()
sigmoid = nn.Sigmoid()

print(pos_edge_index.shape)

# Generate the positive and negative labels
pos_label = torch.ones(pos_edge_index.shape[1], )
neg_label = torch.zeros(neg_edge_index.shape[1], )

# Concat positive and negative labels into one tensor
train_label = torch.cat([pos_label, neg_label], dim=0)

# Concat positive and negative edges into one tensor
# Since the network is very small, we do not split the edges into val/test sets
train_edge = torch.cat([pos_edge_index, neg_edge_index], dim=1)
print(train_edge.shape)

train(emb, loss_fn, sigmoid, train_label, train_edge)
```

    torch.Size([2, 78])
    torch.Size([2, 156])
    loss: 2.059661, acc: 0.500000
    loss: 2.046740, acc: 0.500000
    loss: 2.022378, acc: 0.500000
    loss: 1.988064, acc: 0.500000
    loss: 1.945258, acc: 0.500000
    loss: 1.895376, acc: 0.500000
    loss: 1.839767, acc: 0.500000
    loss: 1.779702, acc: 0.500000
    loss: 1.716365, acc: 0.500000
    loss: 1.650848, acc: 0.500000
    loss: 1.584147, acc: 0.500000
    loss: 1.517160, acc: 0.500000


    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:45: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:46: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).


    loss: 1.450682, acc: 0.500000
    loss: 1.385406, acc: 0.500000
    loss: 1.321924, acc: 0.500000
    loss: 1.260727, acc: 0.500000
    loss: 1.202207, acc: 0.500000
    loss: 1.146660, acc: 0.500000
    loss: 1.094293, acc: 0.500000
    loss: 1.045231, acc: 0.500000
    loss: 0.999524, acc: 0.500000
    loss: 0.957157, acc: 0.500000
    loss: 0.918061, acc: 0.500000
    loss: 0.882122, acc: 0.506400
    loss: 0.849194, acc: 0.506400
    loss: 0.819103, acc: 0.519200
    loss: 0.791663, acc: 0.519200
    loss: 0.766678, acc: 0.532100
    loss: 0.743947, acc: 0.538500
    loss: 0.723275, acc: 0.544900
    loss: 0.704472, acc: 0.544900
    loss: 0.687356, acc: 0.544900
    loss: 0.671758, acc: 0.564100
    loss: 0.657519, acc: 0.583300
    loss: 0.644493, acc: 0.596200
    loss: 0.632548, acc: 0.609000
    loss: 0.621563, acc: 0.615400
    loss: 0.611427, acc: 0.628200
    loss: 0.602045, acc: 0.647400
    loss: 0.593329, acc: 0.653800
    loss: 0.585200, acc: 0.666700
    loss: 0.577589, acc: 0.660300
    loss: 0.570436, acc: 0.679500
    loss: 0.563685, acc: 0.692300
    loss: 0.557289, acc: 0.717900
    loss: 0.551207, acc: 0.724400
    loss: 0.545400, acc: 0.730800
    loss: 0.539836, acc: 0.737200
    loss: 0.534487, acc: 0.750000
    loss: 0.529327, acc: 0.762800
    loss: 0.524334, acc: 0.775600
    loss: 0.519488, acc: 0.814100
    loss: 0.514774, acc: 0.833300
    loss: 0.510174, acc: 0.839700
    loss: 0.505678, acc: 0.846200
    loss: 0.501273, acc: 0.846200
    loss: 0.496948, acc: 0.852600
    loss: 0.492696, acc: 0.852600
    loss: 0.488508, acc: 0.859000
    loss: 0.484377, acc: 0.859000
    loss: 0.480299, acc: 0.859000
    loss: 0.476266, acc: 0.865400
    loss: 0.472276, acc: 0.865400
    loss: 0.468323, acc: 0.865400
    loss: 0.464405, acc: 0.865400
    loss: 0.460518, acc: 0.871800
    loss: 0.456659, acc: 0.871800
    loss: 0.452827, acc: 0.871800
    loss: 0.449019, acc: 0.871800
    loss: 0.445234, acc: 0.878200
    loss: 0.441470, acc: 0.878200
    loss: 0.437725, acc: 0.878200
    loss: 0.433999, acc: 0.878200
    loss: 0.430290, acc: 0.878200
    loss: 0.426598, acc: 0.884600
    loss: 0.422922, acc: 0.878200
    loss: 0.419262, acc: 0.878200
    loss: 0.415616, acc: 0.884600
    loss: 0.411986, acc: 0.884600
    loss: 0.408369, acc: 0.891000
    loss: 0.404766, acc: 0.891000
    loss: 0.401178, acc: 0.897400
    loss: 0.397603, acc: 0.897400
    loss: 0.394041, acc: 0.897400
    loss: 0.390494, acc: 0.897400
    loss: 0.386959, acc: 0.903800
    loss: 0.383439, acc: 0.910300
    loss: 0.379932, acc: 0.910300
    loss: 0.376439, acc: 0.910300
    loss: 0.372960, acc: 0.910300
    loss: 0.369495, acc: 0.910300
    loss: 0.366045, acc: 0.910300
    loss: 0.362609, acc: 0.910300
    loss: 0.359187, acc: 0.916700
    loss: 0.355781, acc: 0.923100
    loss: 0.352389, acc: 0.923100
    loss: 0.349013, acc: 0.923100
    loss: 0.345652, acc: 0.923100
    loss: 0.342307, acc: 0.923100
    loss: 0.338979, acc: 0.923100
    loss: 0.335666, acc: 0.923100
    loss: 0.332370, acc: 0.929500
    loss: 0.329091, acc: 0.935900
    loss: 0.325829, acc: 0.942300
    loss: 0.322585, acc: 0.942300
    loss: 0.319358, acc: 0.948700
    loss: 0.316149, acc: 0.948700
    loss: 0.312959, acc: 0.948700
    loss: 0.309787, acc: 0.955100
    loss: 0.306634, acc: 0.955100
    loss: 0.303499, acc: 0.961500
    loss: 0.300385, acc: 0.961500
    loss: 0.297289, acc: 0.961500
    loss: 0.294214, acc: 0.961500
    loss: 0.291159, acc: 0.961500
    loss: 0.288124, acc: 0.961500
    loss: 0.285110, acc: 0.961500
    loss: 0.282117, acc: 0.961500
    loss: 0.279145, acc: 0.961500
    loss: 0.276195, acc: 0.961500
    loss: 0.273266, acc: 0.961500
    loss: 0.270360, acc: 0.961500
    loss: 0.267475, acc: 0.961500
    loss: 0.264613, acc: 0.961500
    loss: 0.261773, acc: 0.961500
    loss: 0.258956, acc: 0.961500
    loss: 0.256162, acc: 0.961500
    loss: 0.253392, acc: 0.961500
    loss: 0.250645, acc: 0.961500
    loss: 0.247921, acc: 0.961500
    loss: 0.245221, acc: 0.961500
    loss: 0.242546, acc: 0.974400
    loss: 0.239894, acc: 0.974400
    loss: 0.237266, acc: 0.974400
    loss: 0.234663, acc: 0.974400
    loss: 0.232085, acc: 0.974400
    loss: 0.229531, acc: 0.974400
    loss: 0.227001, acc: 0.980800
    loss: 0.224497, acc: 0.980800
    loss: 0.222017, acc: 0.987200
    loss: 0.219563, acc: 0.993600
    loss: 0.217133, acc: 0.993600
    loss: 0.214729, acc: 0.993600
    loss: 0.212350, acc: 0.993600
    loss: 0.209996, acc: 0.993600
    loss: 0.207667, acc: 0.993600
    loss: 0.205363, acc: 0.993600
    loss: 0.203085, acc: 0.993600
    loss: 0.200832, acc: 0.993600
    loss: 0.198603, acc: 0.993600
    loss: 0.196401, acc: 0.993600
    loss: 0.194223, acc: 0.993600
    loss: 0.192070, acc: 0.993600
    loss: 0.189942, acc: 0.993600
    loss: 0.187839, acc: 0.993600
    loss: 0.185761, acc: 1.000000
    loss: 0.183708, acc: 1.000000
    loss: 0.181680, acc: 1.000000
    loss: 0.179675, acc: 1.000000
    loss: 0.177696, acc: 1.000000
    loss: 0.175740, acc: 1.000000
    loss: 0.173809, acc: 1.000000
    loss: 0.171902, acc: 1.000000
    loss: 0.170018, acc: 1.000000
    loss: 0.168158, acc: 1.000000
    loss: 0.166322, acc: 1.000000
    loss: 0.164509, acc: 1.000000
    loss: 0.162719, acc: 1.000000
    loss: 0.160952, acc: 1.000000
    loss: 0.159208, acc: 1.000000
    loss: 0.157487, acc: 1.000000
    loss: 0.155788, acc: 1.000000
    loss: 0.154111, acc: 1.000000
    loss: 0.152456, acc: 1.000000
    loss: 0.150822, acc: 1.000000
    loss: 0.149211, acc: 1.000000
    loss: 0.147620, acc: 1.000000
    loss: 0.146051, acc: 1.000000
    loss: 0.144503, acc: 1.000000
    loss: 0.142975, acc: 1.000000
    loss: 0.141467, acc: 1.000000
    loss: 0.139980, acc: 1.000000
    loss: 0.138513, acc: 1.000000
    loss: 0.137065, acc: 1.000000
    loss: 0.135637, acc: 1.000000
    loss: 0.134228, acc: 1.000000
    loss: 0.132838, acc: 1.000000
    loss: 0.131467, acc: 1.000000
    loss: 0.130114, acc: 1.000000
    loss: 0.128780, acc: 1.000000
    loss: 0.127464, acc: 1.000000
    loss: 0.126165, acc: 1.000000
    loss: 0.124884, acc: 1.000000
    loss: 0.123620, acc: 1.000000
    loss: 0.122374, acc: 1.000000
    loss: 0.121144, acc: 1.000000
    loss: 0.119931, acc: 1.000000
    loss: 0.118734, acc: 1.000000
    loss: 0.117554, acc: 1.000000
    loss: 0.116389, acc: 1.000000
    loss: 0.115240, acc: 1.000000
    loss: 0.114107, acc: 1.000000
    loss: 0.112989, acc: 1.000000
    loss: 0.111886, acc: 1.000000
    loss: 0.110797, acc: 1.000000
    loss: 0.109724, acc: 1.000000
    loss: 0.108665, acc: 1.000000
    loss: 0.107620, acc: 1.000000
    loss: 0.106589, acc: 1.000000
    loss: 0.105571, acc: 1.000000
    loss: 0.104568, acc: 1.000000
    loss: 0.103577, acc: 1.000000
    loss: 0.102600, acc: 1.000000
    loss: 0.101636, acc: 1.000000
    loss: 0.100685, acc: 1.000000
    loss: 0.099746, acc: 1.000000
    loss: 0.098819, acc: 1.000000
    loss: 0.097905, acc: 1.000000
    loss: 0.097003, acc: 1.000000
    loss: 0.096113, acc: 1.000000
    loss: 0.095234, acc: 1.000000
    loss: 0.094367, acc: 1.000000
    loss: 0.093511, acc: 1.000000
    loss: 0.092666, acc: 1.000000
    loss: 0.091833, acc: 1.000000
    loss: 0.091010, acc: 1.000000
    loss: 0.090198, acc: 1.000000
    loss: 0.089396, acc: 1.000000
    loss: 0.088604, acc: 1.000000
    loss: 0.087823, acc: 1.000000
    loss: 0.087052, acc: 1.000000
    loss: 0.086290, acc: 1.000000
    loss: 0.085539, acc: 1.000000
    loss: 0.084796, acc: 1.000000
    loss: 0.084064, acc: 1.000000
    loss: 0.083340, acc: 1.000000
    loss: 0.082626, acc: 1.000000
    loss: 0.081921, acc: 1.000000
    loss: 0.081224, acc: 1.000000
    loss: 0.080537, acc: 1.000000
    loss: 0.079857, acc: 1.000000
    loss: 0.079187, acc: 1.000000
    loss: 0.078525, acc: 1.000000
    loss: 0.077870, acc: 1.000000
    loss: 0.077224, acc: 1.000000
    loss: 0.076586, acc: 1.000000
    loss: 0.075956, acc: 1.000000
    loss: 0.075334, acc: 1.000000
    loss: 0.074719, acc: 1.000000
    loss: 0.074112, acc: 1.000000
    loss: 0.073512, acc: 1.000000
    loss: 0.072919, acc: 1.000000
    loss: 0.072333, acc: 1.000000
    loss: 0.071755, acc: 1.000000
    loss: 0.071184, acc: 1.000000
    loss: 0.070619, acc: 1.000000
    loss: 0.070061, acc: 1.000000
    loss: 0.069510, acc: 1.000000
    loss: 0.068965, acc: 1.000000
    loss: 0.068427, acc: 1.000000
    loss: 0.067895, acc: 1.000000
    loss: 0.067370, acc: 1.000000
    loss: 0.066850, acc: 1.000000
    loss: 0.066337, acc: 1.000000
    loss: 0.065830, acc: 1.000000
    loss: 0.065329, acc: 1.000000
    loss: 0.064833, acc: 1.000000
    loss: 0.064343, acc: 1.000000
    loss: 0.063859, acc: 1.000000
    loss: 0.063381, acc: 1.000000
    loss: 0.062908, acc: 1.000000
    loss: 0.062440, acc: 1.000000
    loss: 0.061978, acc: 1.000000
    loss: 0.061521, acc: 1.000000
    loss: 0.061069, acc: 1.000000
    loss: 0.060622, acc: 1.000000
    loss: 0.060181, acc: 1.000000
    loss: 0.059744, acc: 1.000000
    loss: 0.059312, acc: 1.000000
    loss: 0.058885, acc: 1.000000
    loss: 0.058463, acc: 1.000000
    loss: 0.058046, acc: 1.000000
    loss: 0.057633, acc: 1.000000
    loss: 0.057225, acc: 1.000000
    loss: 0.056821, acc: 1.000000
    loss: 0.056422, acc: 1.000000
    loss: 0.056027, acc: 1.000000
    loss: 0.055636, acc: 1.000000
    loss: 0.055250, acc: 1.000000
    loss: 0.054868, acc: 1.000000
    loss: 0.054490, acc: 1.000000
    loss: 0.054116, acc: 1.000000
    loss: 0.053746, acc: 1.000000
    loss: 0.053380, acc: 1.000000
    loss: 0.053018, acc: 1.000000
    loss: 0.052660, acc: 1.000000
    loss: 0.052305, acc: 1.000000
    loss: 0.051955, acc: 1.000000
    loss: 0.051608, acc: 1.000000
    loss: 0.051265, acc: 1.000000
    loss: 0.050925, acc: 1.000000
    loss: 0.050589, acc: 1.000000
    loss: 0.050257, acc: 1.000000
    loss: 0.049928, acc: 1.000000
    loss: 0.049602, acc: 1.000000
    loss: 0.049280, acc: 1.000000
    loss: 0.048961, acc: 1.000000
    loss: 0.048645, acc: 1.000000
    loss: 0.048333, acc: 1.000000
    loss: 0.048023, acc: 1.000000
    loss: 0.047717, acc: 1.000000
    loss: 0.047414, acc: 1.000000
    loss: 0.047114, acc: 1.000000
    loss: 0.046818, acc: 1.000000
    loss: 0.046524, acc: 1.000000
    loss: 0.046233, acc: 1.000000
    loss: 0.045945, acc: 1.000000
    loss: 0.045660, acc: 1.000000
    loss: 0.045377, acc: 1.000000
    loss: 0.045098, acc: 1.000000
    loss: 0.044821, acc: 1.000000
    loss: 0.044547, acc: 1.000000
    loss: 0.044276, acc: 1.000000
    loss: 0.044007, acc: 1.000000
    loss: 0.043741, acc: 1.000000
    loss: 0.043478, acc: 1.000000
    loss: 0.043217, acc: 1.000000
    loss: 0.042959, acc: 1.000000
    loss: 0.042703, acc: 1.000000
    loss: 0.042450, acc: 1.000000
    loss: 0.042199, acc: 1.000000
    loss: 0.041950, acc: 1.000000
    loss: 0.041704, acc: 1.000000
    loss: 0.041460, acc: 1.000000
    loss: 0.041219, acc: 1.000000
    loss: 0.040980, acc: 1.000000
    loss: 0.040743, acc: 1.000000
    loss: 0.040508, acc: 1.000000
    loss: 0.040275, acc: 1.000000
    loss: 0.040045, acc: 1.000000
    loss: 0.039817, acc: 1.000000
    loss: 0.039591, acc: 1.000000
    loss: 0.039367, acc: 1.000000
    loss: 0.039145, acc: 1.000000
    loss: 0.038925, acc: 1.000000
    loss: 0.038707, acc: 1.000000
    loss: 0.038491, acc: 1.000000
    loss: 0.038277, acc: 1.000000
    loss: 0.038066, acc: 1.000000
    loss: 0.037855, acc: 1.000000
    loss: 0.037647, acc: 1.000000
    loss: 0.037441, acc: 1.000000
    loss: 0.037237, acc: 1.000000
    loss: 0.037034, acc: 1.000000
    loss: 0.036833, acc: 1.000000
    loss: 0.036634, acc: 1.000000
    loss: 0.036437, acc: 1.000000
    loss: 0.036242, acc: 1.000000
    loss: 0.036048, acc: 1.000000
    loss: 0.035856, acc: 1.000000
    loss: 0.035666, acc: 1.000000
    loss: 0.035477, acc: 1.000000
    loss: 0.035290, acc: 1.000000
    loss: 0.035105, acc: 1.000000
    loss: 0.034921, acc: 1.000000
    loss: 0.034739, acc: 1.000000
    loss: 0.034558, acc: 1.000000
    loss: 0.034379, acc: 1.000000
    loss: 0.034202, acc: 1.000000
    loss: 0.034026, acc: 1.000000
    loss: 0.033851, acc: 1.000000
    loss: 0.033678, acc: 1.000000
    loss: 0.033507, acc: 1.000000
    loss: 0.033336, acc: 1.000000
    loss: 0.033168, acc: 1.000000
    loss: 0.033001, acc: 1.000000
    loss: 0.032835, acc: 1.000000
    loss: 0.032670, acc: 1.000000
    loss: 0.032507, acc: 1.000000
    loss: 0.032345, acc: 1.000000
    loss: 0.032185, acc: 1.000000
    loss: 0.032026, acc: 1.000000
    loss: 0.031868, acc: 1.000000
    loss: 0.031712, acc: 1.000000
    loss: 0.031556, acc: 1.000000
    loss: 0.031403, acc: 1.000000
    loss: 0.031250, acc: 1.000000
    loss: 0.031098, acc: 1.000000
    loss: 0.030948, acc: 1.000000
    loss: 0.030799, acc: 1.000000
    loss: 0.030651, acc: 1.000000
    loss: 0.030505, acc: 1.000000
    loss: 0.030359, acc: 1.000000
    loss: 0.030215, acc: 1.000000
    loss: 0.030072, acc: 1.000000
    loss: 0.029930, acc: 1.000000
    loss: 0.029789, acc: 1.000000
    loss: 0.029649, acc: 1.000000
    loss: 0.029511, acc: 1.000000
    loss: 0.029373, acc: 1.000000
    loss: 0.029237, acc: 1.000000
    loss: 0.029101, acc: 1.000000
    loss: 0.028967, acc: 1.000000
    loss: 0.028834, acc: 1.000000
    loss: 0.028701, acc: 1.000000
    loss: 0.028570, acc: 1.000000
    loss: 0.028440, acc: 1.000000
    loss: 0.028310, acc: 1.000000
    loss: 0.028182, acc: 1.000000
    loss: 0.028055, acc: 1.000000
    loss: 0.027929, acc: 1.000000
    loss: 0.027803, acc: 1.000000
    loss: 0.027679, acc: 1.000000
    loss: 0.027555, acc: 1.000000
    loss: 0.027433, acc: 1.000000
    loss: 0.027311, acc: 1.000000
    loss: 0.027190, acc: 1.000000
    loss: 0.027071, acc: 1.000000
    loss: 0.026952, acc: 1.000000
    loss: 0.026834, acc: 1.000000
    loss: 0.026716, acc: 1.000000
    loss: 0.026600, acc: 1.000000
    loss: 0.026485, acc: 1.000000
    loss: 0.026370, acc: 1.000000
    loss: 0.026256, acc: 1.000000
    loss: 0.026143, acc: 1.000000
    loss: 0.026031, acc: 1.000000
    loss: 0.025920, acc: 1.000000
    loss: 0.025809, acc: 1.000000
    loss: 0.025700, acc: 1.000000
    loss: 0.025591, acc: 1.000000
    loss: 0.025483, acc: 1.000000
    loss: 0.025375, acc: 1.000000
    loss: 0.025269, acc: 1.000000
    loss: 0.025163, acc: 1.000000
    loss: 0.025058, acc: 1.000000
    loss: 0.024954, acc: 1.000000
    loss: 0.024850, acc: 1.000000
    loss: 0.024747, acc: 1.000000
    loss: 0.024645, acc: 1.000000
    loss: 0.024544, acc: 1.000000
    loss: 0.024443, acc: 1.000000
    loss: 0.024343, acc: 1.000000
    loss: 0.024244, acc: 1.000000
    loss: 0.024145, acc: 1.000000
    loss: 0.024047, acc: 1.000000
    loss: 0.023950, acc: 1.000000
    loss: 0.023853, acc: 1.000000
    loss: 0.023757, acc: 1.000000
    loss: 0.023662, acc: 1.000000
    loss: 0.023567, acc: 1.000000
    loss: 0.023474, acc: 1.000000
    loss: 0.023380, acc: 1.000000
    loss: 0.023288, acc: 1.000000
    loss: 0.023195, acc: 1.000000
    loss: 0.023104, acc: 1.000000
    loss: 0.023013, acc: 1.000000
    loss: 0.022923, acc: 1.000000
    loss: 0.022833, acc: 1.000000
    loss: 0.022744, acc: 1.000000
    loss: 0.022656, acc: 1.000000
    loss: 0.022568, acc: 1.000000
    loss: 0.022481, acc: 1.000000
    loss: 0.022394, acc: 1.000000
    loss: 0.022308, acc: 1.000000
    loss: 0.022223, acc: 1.000000
    loss: 0.022138, acc: 1.000000
    loss: 0.022053, acc: 1.000000
    loss: 0.021969, acc: 1.000000
    loss: 0.021886, acc: 1.000000
    loss: 0.021803, acc: 1.000000
    loss: 0.021721, acc: 1.000000
    loss: 0.021639, acc: 1.000000
    loss: 0.021558, acc: 1.000000
    loss: 0.021477, acc: 1.000000
    loss: 0.021397, acc: 1.000000
    loss: 0.021318, acc: 1.000000
    loss: 0.021238, acc: 1.000000
    loss: 0.021160, acc: 1.000000
    loss: 0.021082, acc: 1.000000
    loss: 0.021004, acc: 1.000000
    loss: 0.020927, acc: 1.000000
    loss: 0.020850, acc: 1.000000
    loss: 0.020774, acc: 1.000000
    loss: 0.020699, acc: 1.000000
    loss: 0.020623, acc: 1.000000
    loss: 0.020549, acc: 1.000000
    loss: 0.020474, acc: 1.000000
    loss: 0.020400, acc: 1.000000
    loss: 0.020327, acc: 1.000000
    loss: 0.020254, acc: 1.000000
    loss: 0.020182, acc: 1.000000
    loss: 0.020110, acc: 1.000000
    loss: 0.020038, acc: 1.000000
    loss: 0.019967, acc: 1.000000
    loss: 0.019896, acc: 1.000000
    loss: 0.019826, acc: 1.000000
    loss: 0.019756, acc: 1.000000
    loss: 0.019687, acc: 1.000000
    loss: 0.019618, acc: 1.000000

### Visualize the final node embeddings

可视化训练后的 embedding，可以看到两类节点还是可以分开的：


```python
# Visualize the final learned embedding
visualize_emb(emb)
```



![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/colab1_42_0.png)
    

