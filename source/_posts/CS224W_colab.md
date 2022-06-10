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
    

## Colab 2

在这个 colab 中，将会使用 PyTorch Geometric 构建简单的 GNN 进行两类问题的预测 : 1) 节点类别 2) 图类别，使用的数据集是 OGB 包中的两个数据集。

查看 Pytorch 版本并安装相应的包：


```python
import torch
import os
print("PyTorch has version {}".format(torch.__version__))
```

    PyTorch has version 1.10.0+cu111


```python
!pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu111.html
!pip install torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cu111.html
!pip install torch-geometric
```

```python
!pip install ogb
```

### 1) PyTorch Geometric (Datasets and Data)


PyTorch Geometric 有两个类用来存储和转化图数据。一个是 `torch_geometric.datasets`, 含有一些常用的数据集，另一个是 `torch_geometric.data`, 提供了以 tensor 操作图数据的方法。

#### PyG Datasets

以 `ENZYMES` 数据集为例：


```python
from torch_geometric.datasets import TUDataset

if 'IS_GRADESCOPE_ENV' not in os.environ:
  root = './enzymes'
  name = 'ENZYMES'

  # The ENZYMES dataset
  pyg_dataset= TUDataset(root, name)

  # You will find that there are 600 graphs in this dataset
  print(pyg_dataset)
```

    Downloading https://www.chrsmrrs.com/graphkerneldatasets/ENZYMES.zip
    Extracting enzymes/ENZYMES/ENZYMES.zip
    Processing...


    ENZYMES(600)


    Done!

##### Question 1: 在 ENZYMES 数据集中有多少类别和特征？


```python
def get_num_classes(pyg_dataset):
  # TODO: Implement a function that takes a PyG dataset object
  # and returns the number of classes for that dataset.

  num_classes = 0

  ############# Your code here ############
  ## (~1 line of code)
  ## Note
  ## 1. Colab autocomplete functionality might be useful.
  num_classes = pyg_dataset.num_classes
  #########################################

  return num_classes

def get_num_features(pyg_dataset):
  # TODO: Implement a function that takes a PyG dataset object
  # and returns the number of features for that dataset.

  num_features = 0

  ############# Your code here ############
  ## (~1 line of code)
  ## Note
  ## 1. Colab autocomplete functionality might be useful.
  num_features = pyg_dataset.num_node_features
  #########################################

  return num_features

if 'IS_GRADESCOPE_ENV' not in os.environ:
  num_classes = get_num_classes(pyg_dataset)
  num_features = get_num_features(pyg_dataset)
  print("{} dataset has {} classes".format(name, num_classes))
  print("{} dataset has {} features".format(name, num_features))
```

    ENZYMES dataset has 6 classes
    ENZYMES dataset has 3 features

#### PyG Data

##### Question 2: ENZYMES 中索引为 100 的图的标签是什么？


```python
def get_graph_class(pyg_dataset, idx):
  # TODO: Implement a function that takes a PyG dataset object,
  # an index of a graph within the dataset, and returns the class/label 
  # of the graph (as an integer).

  label = -1

  ############# Your code here ############
  ## (~1 line of code)
  ##data.y å­˜å‚¨ lable
  label = pyg_dataset[idx]["y"]
  #########################################
  return label

# Here pyg_dataset is a dataset for graph classification
if 'IS_GRADESCOPE_ENV' not in os.environ:
  graph_0 = pyg_dataset[0]
  print(graph_0)
  idx = 100
  label = get_graph_class(pyg_dataset, idx)
  print('Graph with index {} has label {}'.format(idx, label))
```

    Data(edge_index=[2, 168], x=[37, 3], y=[1])
    Graph with index 100 has label tensor([4])

##### Question 3: 索引为 200 的图有多少个边？ 


```python
def get_graph_num_edges(pyg_dataset, idx):
  # TODO: Implement a function that takes a PyG dataset object,
  # the index of a graph in the dataset, and returns the number of 
  # edges in the graph (as an integer). You should not count an edge 
  # twice if the graph is undirected. For example, in an undirected 
  # graph G, if two nodes v and u are connected by an edge, this edge
  # should only be counted once.

  num_edges = 0

  ############# Your code here ############
  ## Note:
  ## 1. You can't return the data.num_edges directly
  ## 2. We assume the graph is undirected
  ## 3. Look at the PyG dataset built in functions
  ## (~4 lines of code)
  num_edges = pyg_dataset[200]["edge_index"].shape[1]/2
  num_edges = int(num_edges)
  #########################################

  return num_edges

if 'IS_GRADESCOPE_ENV' not in os.environ:
  idx = 200
  num_edges = get_graph_num_edges(pyg_dataset, idx)
  print('Graph with index {} has {} edges'.format(idx, num_edges))
```

    Graph with index 200 has 53 edges

#### OGB 数据

以 OGB 中的 ogbn-arxiv 数据为例：


```python
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset

if 'IS_GRADESCOPE_ENV' not in os.environ:
  dataset_name = 'ogbn-arxiv'
  # Load the dataset and transform it to sparse tensor
  dataset = PygNodePropPredDataset(name=dataset_name,transform=T.ToSparseTensor())
  print('The {} dataset has {} graph'.format(dataset_name, len(dataset)))

  # Extract the graph
  data = dataset[0]
  print(data)
```

    Downloading http://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip


    Downloaded 0.08 GB: 100%|â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆ| 81/81 [00:07<00:00, 10.16it/s]


    Extracting dataset/arxiv.zip


    Processing...


    Loading necessary files...
    This might take a while.
    Processing graphs...


    100%|â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆ| 1/1 [00:00<00:00, 8112.77it/s]


    Converting graphs into PyG objects...


    100%|â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆ| 1/1 [00:00<00:00, 5729.92it/s]
    
    Saving...

    Done!


    The ogbn-arxiv dataset has 1 graph
    Data(num_nodes=169343, x=[169343, 128], node_year=[169343, 1], y=[169343, 1], adj_t=[169343, 169343, nnz=1166243])

##### Question 4: ogbn-arxiv 图有多少特征？


```python
def graph_num_features(data):
  # TODO: Implement a function that takes a PyG data object,
  # and returns the number of features in the graph (as an integer).

  num_features = 0

  ############# Your code here ############
  ## (~1 line of code)
  num_features = data.num_node_features
  #########################################

  return num_features

if 'IS_GRADESCOPE_ENV' not in os.environ:
  num_features = graph_num_features(data)
  print('The graph has {} features'.format(num_features))
```

    The graph has 128 features

### GNN: 节点分类预测

使用 PyG 的 `GCNConv` 层：


```python
import torch
import pandas as pd
import torch.nn.functional as F
print(torch.__version__)

# The PyG built-in GCNConv
from torch_geometric.nn import GCNConv

import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
```

    1.10.0+cu111

```python
data.adj_t
```


    SparseTensor(row=tensor([     0,      0,      0,  ..., 169341, 169341, 169341]),
                 col=tensor([   411,    640,   1162,  ...,  30351,  35711, 103121]),
                 size=(169343, 169343), nnz=1166243, density=0.00%)


```python
data.num_nodes
```


    169343


```python
data.adj_t.to_symmetric()
```


    SparseTensor(row=tensor([     0,      0,      0,  ..., 169341, 169342, 169342]),
                 col=tensor([   411,    640,   1162,  ..., 163274,  27824, 158981]),
                 size=(169343, 169343), nnz=2315598, density=0.01%)

#### Load and Preprocess the Dataset


```python
if 'IS_GRADESCOPE_ENV' not in os.environ:
  dataset_name = 'ogbn-arxiv'
  dataset = PygNodePropPredDataset(name=dataset_name,transform=T.ToSparseTensor())
  data = dataset[0]

  # Make the adjacency matrix to symmetric
  data.adj_t = data.adj_t.to_symmetric()

  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  # If you use GPU, the device should be cuda
  print('Device: {}'.format(device))

  data = data.to(device)
  split_idx = dataset.get_idx_split()
  train_idx = split_idx['train'].to(device)
```

    Device: cuda

```python
split_idx
```


    {'test': tensor([   346,    398,    451,  ..., 169340, 169341, 169342]),
     'train': tensor([     0,      1,      2,  ..., 169145, 169148, 169251]),
     'valid': tensor([   349,    357,    366,  ..., 169185, 169261, 169296])}

#### GCN Model

GNN 的架构如下：


![](https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/aaa.png)


```python
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 dropout, return_embeds=False):
        # TODO: Implement a function that initializes self.convs, 
        # self.bns, and self.softmax.

        super(GCN, self).__init__()

        # A list of GCNConv layers
        self.convs = None

        # A list of 1D batch normalization layers
        self.bns = None

        # The log softmax layer
        self.softmax = None

        ############# Your code here ############
        ## Note:
        ## 1. You should use torch.nn.ModuleList for self.convs and self.bns
        ## 2. self.convs has num_layers GCNConv layers
        ## 3. self.bns has num_layers - 1 BatchNorm1d layers
        ## 4. You should use torch.nn.LogSoftmax for self.softmax
        ## 5. The parameters you can set for GCNConv include 'in_channels' and 
        ## 'out_channels'. For more information please refer to the documentation:
        ## https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv
        ## 6. The only parameter you need to set for BatchNorm1d is 'num_features'
        ## For more information please refer to the documentation: 
        ## https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
        ## (~10 lines of code)
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList(
            [GCNConv(input_dim,hidden_dim)] + 
            [GCNConv(hidden_dim,hidden_dim) for i in range(self.num_layers-2)] +
            [GCNConv(hidden_dim,output_dim)])

        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(hidden_dim) for i in range(self.num_layers - 1)])
        self.softmax = torch.nn.LogSoftmax()
        #########################################

        # Probability of an element getting zeroed
        self.dropout = dropout

        # Skip classification layer and return node embeddings
        self.return_embeds = return_embeds

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        # TODO: Implement a function that takes the feature tensor x and
        # edge_index tensor adj_t and returns the output tensor as
        # shown in the figure.

        out = None

        ############# Your code here ############
        ## Note:
        ## 1. Construct the network as shown in the figure
        ## 2. torch.nn.functional.relu and torch.nn.functional.dropout are useful
        ## For more information please refer to the documentation:
        ## https://pytorch.org/docs/stable/nn.functional.html
        ## 3. Don't forget to set F.dropout training to self.training
        ## 4. If return_embeds is True, then skip the last softmax layer
        ## (~7 lines of code)
        for i in range(self.num_layers - 1):
          x = self.convs[i](x, adj_t)
          x = self.bns[i](x)
          x = F.relu(x)
          x = F.dropout(x,p=self.dropout,training=self.training)
        x = self.convs[self.num_layers-1](x, adj_t)
        if self.return_embeds :
          out = x
        else:
          out = self.softmax(x)
        #########################################

        return out
```


```python
def train(model, data, train_idx, optimizer, loss_fn):
    # TODO: Implement a function that trains the model by 
    # using the given optimizer and loss_fn.
    model.train()
    loss = 0

    ############# Your code here ############
    ## Note:
    ## 1. Zero grad the optimizer
    ## 2. Feed the data into the model
    ## 3. Slice the model output and label by train_idx
    ## 4. Feed the sliced output and label to loss_fn
    ## (~4 lines of code)
    optimizer.zero_grad()
    out = model(data.x,data.adj_t)
    loss = loss_fn(out[train_idx], data.y[train_idx].reshape(-1))
    #########################################

    loss.backward()
    optimizer.step()

    return loss.item()
```


```python
# Test function here
@torch.no_grad()
def test(model, data, split_idx, evaluator, save_model_results=False):
    # TODO: Implement a function that tests the model by 
    # using the given split_idx and evaluator.
    model.eval()

    # The output of model on all data
    out = None

    ############# Your code here ############
    ## (~1 line of code)
    ## Note:
    ## 1. No index slicing here
    out = model(data.x,data.adj_t)
    #########################################

    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    if save_model_results:
      print ("Saving Model Predictions")

      data = {}
      data['y_pred'] = y_pred.view(-1).cpu().detach().numpy()

      df = pd.DataFrame(data=data)
      # Save locally as csv
      df.to_csv('ogbn-arxiv_node.csv', sep=',', index=False)


    return train_acc, valid_acc, test_acc
```


```python
# Please do not change the args
if 'IS_GRADESCOPE_ENV' not in os.environ:
  args = {
      'device': device,
      'num_layers': 3,
      'hidden_dim': 256,
      'dropout': 0.5,
      'lr': 0.01,
      'epochs': 100,
  }
  args
```


```python
if 'IS_GRADESCOPE_ENV' not in os.environ:
  model = GCN(data.num_features, args['hidden_dim'],
              dataset.num_classes, args['num_layers'],
              args['dropout']).to(device)
  evaluator = Evaluator(name='ogbn-arxiv')
```


```python
model.convs
```


    ModuleList(
      (0): GCNConv(128, 256)
      (1): GCNConv(256, 256)
      (2): GCNConv(256, 40)
    )


```python
import copy
if 'IS_GRADESCOPE_ENV' not in os.environ:
  # reset the parameters to initial random value
  model.reset_parameters()

  optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
  loss_fn = F.nll_loss

  best_model = None
  best_valid_acc = 0

  for epoch in range(1, 1 + args["epochs"]):
    loss = train(model, data, train_idx, optimizer, loss_fn)
    result = test(model, data, split_idx, evaluator)
    train_acc, valid_acc, test_acc = result
    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        best_model = copy.deepcopy(model)
    print(f'Epoch: {epoch:02d}, '
          f'Loss: {loss:.4f}, '
          f'Train: {100 * train_acc:.2f}%, '
          f'Valid: {100 * valid_acc:.2f}% '
          f'Test: {100 * test_acc:.2f}%')
```

    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:78: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.


    Epoch: 01, Loss: 4.1033, Train: 27.09%, Valid: 29.73% Test: 26.61%
    Epoch: 02, Loss: 2.4399, Train: 26.64%, Valid: 24.23% Test: 28.44%
    Epoch: 03, Loss: 1.9616, Train: 26.83%, Valid: 16.91% Test: 14.72%
    Epoch: 04, Loss: 1.8343, Train: 28.58%, Valid: 15.74% Test: 12.46%
    Epoch: 05, Loss: 1.6758, Train: 30.91%, Valid: 20.28% Test: 17.73%
    Epoch: 06, Loss: 1.6027, Train: 33.16%, Valid: 27.19% Test: 25.46%
    Epoch: 07, Loss: 1.5436, Train: 34.38%, Valid: 30.09% Test: 29.57%
    Epoch: 08, Loss: 1.4816, Train: 36.40%, Valid: 33.96% Test: 35.30%
    Epoch: 09, Loss: 1.4277, Train: 39.82%, Valid: 39.25% Test: 43.29%
    Epoch: 10, Loss: 1.4086, Train: 43.05%, Valid: 42.01% Test: 46.15%
    Epoch: 11, Loss: 1.3740, Train: 46.01%, Valid: 44.16% Test: 47.89%
    Epoch: 12, Loss: 1.3430, Train: 49.18%, Valid: 47.97% Test: 51.24%
    Epoch: 13, Loss: 1.3192, Train: 52.03%, Valid: 51.83% Test: 54.47%
    Epoch: 14, Loss: 1.2995, Train: 54.05%, Valid: 54.74% Test: 57.26%
    Epoch: 15, Loss: 1.2774, Train: 55.88%, Valid: 57.32% Test: 59.08%
    Epoch: 16, Loss: 1.2582, Train: 57.55%, Valid: 59.13% Test: 60.06%
    Epoch: 17, Loss: 1.2439, Train: 58.61%, Valid: 60.20% Test: 61.11%
    Epoch: 18, Loss: 1.2282, Train: 59.16%, Valid: 60.52% Test: 61.89%
    Epoch: 19, Loss: 1.2170, Train: 59.45%, Valid: 60.19% Test: 62.18%
    Epoch: 20, Loss: 1.1986, Train: 60.01%, Valid: 60.50% Test: 62.57%
    Epoch: 21, Loss: 1.1874, Train: 60.63%, Valid: 61.09% Test: 62.82%
    Epoch: 22, Loss: 1.1724, Train: 61.08%, Valid: 61.48% Test: 63.07%
    Epoch: 23, Loss: 1.1602, Train: 61.42%, Valid: 61.66% Test: 63.16%
    Epoch: 24, Loss: 1.1519, Train: 62.03%, Valid: 62.16% Test: 63.46%
    Epoch: 25, Loss: 1.1451, Train: 62.99%, Valid: 63.15% Test: 64.23%
    Epoch: 26, Loss: 1.1398, Train: 64.04%, Valid: 64.03% Test: 64.89%
    Epoch: 27, Loss: 1.1307, Train: 65.06%, Valid: 65.05% Test: 65.56%
    Epoch: 28, Loss: 1.1206, Train: 65.98%, Valid: 65.64% Test: 65.73%
    Epoch: 29, Loss: 1.1140, Train: 66.77%, Valid: 66.28% Test: 65.64%
    Epoch: 30, Loss: 1.1101, Train: 67.40%, Valid: 66.95% Test: 66.20%
    Epoch: 31, Loss: 1.1046, Train: 67.77%, Valid: 67.42% Test: 67.15%
    Epoch: 32, Loss: 1.0961, Train: 67.67%, Valid: 67.28% Test: 67.73%
    Epoch: 33, Loss: 1.0916, Train: 67.43%, Valid: 67.16% Test: 67.91%
    Epoch: 34, Loss: 1.0865, Train: 67.36%, Valid: 67.18% Test: 67.91%
    Epoch: 35, Loss: 1.0757, Train: 67.53%, Valid: 67.34% Test: 68.08%
    Epoch: 36, Loss: 1.0746, Train: 68.01%, Valid: 67.87% Test: 68.27%
    Epoch: 37, Loss: 1.0709, Train: 68.60%, Valid: 68.52% Test: 68.41%
    Epoch: 38, Loss: 1.0660, Train: 69.08%, Valid: 68.87% Test: 68.58%
    Epoch: 39, Loss: 1.0616, Train: 69.52%, Valid: 69.17% Test: 68.88%
    Epoch: 40, Loss: 1.0581, Train: 69.80%, Valid: 69.42% Test: 69.12%
    Epoch: 41, Loss: 1.0546, Train: 69.92%, Valid: 69.43% Test: 69.35%
    Epoch: 42, Loss: 1.0500, Train: 69.99%, Valid: 69.57% Test: 69.40%
    Epoch: 43, Loss: 1.0427, Train: 70.07%, Valid: 69.57% Test: 69.41%
    Epoch: 44, Loss: 1.0425, Train: 70.27%, Valid: 69.71% Test: 69.46%
    Epoch: 45, Loss: 1.0397, Train: 70.39%, Valid: 69.82% Test: 69.61%
    Epoch: 46, Loss: 1.0344, Train: 70.50%, Valid: 69.98% Test: 69.68%
    Epoch: 47, Loss: 1.0285, Train: 70.66%, Valid: 70.16% Test: 69.82%
    Epoch: 48, Loss: 1.0244, Train: 70.80%, Valid: 70.35% Test: 70.00%
    Epoch: 49, Loss: 1.0254, Train: 71.00%, Valid: 70.48% Test: 70.02%
    Epoch: 50, Loss: 1.0170, Train: 71.11%, Valid: 70.49% Test: 69.76%
    Epoch: 51, Loss: 1.0184, Train: 71.14%, Valid: 70.37% Test: 69.58%
    Epoch: 52, Loss: 1.0163, Train: 71.09%, Valid: 70.43% Test: 69.59%
    Epoch: 53, Loss: 1.0116, Train: 71.03%, Valid: 70.41% Test: 69.82%
    Epoch: 54, Loss: 1.0086, Train: 71.12%, Valid: 70.52% Test: 69.89%
    Epoch: 55, Loss: 1.0072, Train: 71.22%, Valid: 70.71% Test: 70.05%
    Epoch: 56, Loss: 1.0051, Train: 71.36%, Valid: 70.88% Test: 69.98%
    Epoch: 57, Loss: 1.0010, Train: 71.49%, Valid: 70.94% Test: 69.96%
    Epoch: 58, Loss: 0.9991, Train: 71.66%, Valid: 70.75% Test: 69.87%
    Epoch: 59, Loss: 0.9962, Train: 71.69%, Valid: 70.70% Test: 69.88%
    Epoch: 60, Loss: 0.9930, Train: 71.71%, Valid: 70.77% Test: 70.09%
    Epoch: 61, Loss: 0.9925, Train: 71.71%, Valid: 70.82% Test: 70.22%
    Epoch: 62, Loss: 0.9894, Train: 71.81%, Valid: 70.88% Test: 70.22%
    Epoch: 63, Loss: 0.9862, Train: 71.96%, Valid: 71.00% Test: 70.29%
    Epoch: 64, Loss: 0.9880, Train: 72.08%, Valid: 71.11% Test: 70.29%
    Epoch: 65, Loss: 0.9831, Train: 72.18%, Valid: 71.01% Test: 70.15%
    Epoch: 66, Loss: 0.9821, Train: 72.24%, Valid: 71.01% Test: 70.09%
    Epoch: 67, Loss: 0.9791, Train: 72.29%, Valid: 70.99% Test: 70.02%
    Epoch: 68, Loss: 0.9775, Train: 72.35%, Valid: 71.07% Test: 70.08%
    Epoch: 69, Loss: 0.9716, Train: 72.26%, Valid: 71.14% Test: 70.15%
    Epoch: 70, Loss: 0.9716, Train: 72.30%, Valid: 71.01% Test: 70.05%
    Epoch: 71, Loss: 0.9709, Train: 72.38%, Valid: 71.00% Test: 69.94%
    Epoch: 72, Loss: 0.9719, Train: 72.46%, Valid: 70.86% Test: 69.75%
    Epoch: 73, Loss: 0.9659, Train: 72.53%, Valid: 71.04% Test: 69.91%
    Epoch: 74, Loss: 0.9669, Train: 72.58%, Valid: 71.24% Test: 70.23%
    Epoch: 75, Loss: 0.9637, Train: 72.66%, Valid: 71.39% Test: 70.35%
    Epoch: 76, Loss: 0.9607, Train: 72.71%, Valid: 71.13% Test: 69.67%
    Epoch: 77, Loss: 0.9588, Train: 72.72%, Valid: 70.88% Test: 69.13%
    Epoch: 78, Loss: 0.9592, Train: 72.85%, Valid: 71.09% Test: 69.73%
    Epoch: 79, Loss: 0.9569, Train: 72.88%, Valid: 71.56% Test: 70.79%
    Epoch: 80, Loss: 0.9550, Train: 72.81%, Valid: 71.50% Test: 70.85%
    Epoch: 81, Loss: 0.9504, Train: 72.75%, Valid: 71.25% Test: 70.50%
    Epoch: 82, Loss: 0.9495, Train: 72.78%, Valid: 71.20% Test: 70.13%
    Epoch: 83, Loss: 0.9503, Train: 72.85%, Valid: 71.17% Test: 70.13%
    Epoch: 84, Loss: 0.9459, Train: 72.95%, Valid: 71.17% Test: 69.93%
    Epoch: 85, Loss: 0.9414, Train: 73.03%, Valid: 71.21% Test: 70.01%
    Epoch: 86, Loss: 0.9431, Train: 73.12%, Valid: 71.48% Test: 70.46%
    Epoch: 87, Loss: 0.9416, Train: 73.13%, Valid: 71.38% Test: 70.13%
    Epoch: 88, Loss: 0.9403, Train: 73.22%, Valid: 71.48% Test: 70.44%
    Epoch: 89, Loss: 0.9388, Train: 73.09%, Valid: 71.58% Test: 70.74%
    Epoch: 90, Loss: 0.9381, Train: 72.96%, Valid: 71.49% Test: 70.93%
    Epoch: 91, Loss: 0.9347, Train: 73.26%, Valid: 71.59% Test: 70.70%
    Epoch: 92, Loss: 0.9320, Train: 73.25%, Valid: 71.31% Test: 70.03%
    Epoch: 93, Loss: 0.9316, Train: 73.17%, Valid: 71.03% Test: 69.63%
    Epoch: 94, Loss: 0.9325, Train: 73.39%, Valid: 71.52% Test: 70.41%
    Epoch: 95, Loss: 0.9309, Train: 73.37%, Valid: 71.75% Test: 71.18%
    Epoch: 96, Loss: 0.9268, Train: 73.21%, Valid: 71.80% Test: 71.35%
    Epoch: 97, Loss: 0.9271, Train: 73.38%, Valid: 71.67% Test: 71.22%
    Epoch: 98, Loss: 0.9239, Train: 73.50%, Valid: 71.62% Test: 71.12%
    Epoch: 99, Loss: 0.9245, Train: 73.62%, Valid: 71.61% Test: 70.90%
    Epoch: 100, Loss: 0.9250, Train: 73.64%, Valid: 71.55% Test: 70.65%

##### Question 5: 验证和测试中最好的模型是什么？

Run the cell below to see the results of your best of model and save your model's predictions to a file named *ogbn-arxiv_node.csv*. 

You can view this file by clicking on the *Folder* icon on the left side pannel. As in Colab 1, when you sumbit your assignment, you will have to download this file and attatch it to your submission.


```python
if 'IS_GRADESCOPE_ENV' not in os.environ:
  best_result = test(best_model, data, split_idx, evaluator, save_model_results=True)
  train_acc, valid_acc, test_acc = best_result
  print(f'Best model: '
        f'Train: {100 * train_acc:.2f}%, '
        f'Valid: {100 * valid_acc:.2f}% '
        f'Test: {100 * test_acc:.2f}%')
```

    Saving Model Predictions
    Best model: Train: 73.21%, Valid: 71.80% Test: 71.35%


    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:78: UserWarning: Implicit dimension choice for log_softmax has been deprecated. Change the call to include dim=X as an argument.

###  GNN: 图类别预测

In this section we will create a graph neural network for graph property prediction (graph classification).

#### Load and preprocess the dataset


```python
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.data import DataLoader
from tqdm.notebook import tqdm

if 'IS_GRADESCOPE_ENV' not in os.environ:
  # Load the dataset 
  dataset = PygGraphPropPredDataset(name='ogbg-molhiv')

  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  print('Device: {}'.format(device))

  split_idx = dataset.get_idx_split()

  # Check task type
  print('Task type: {}'.format(dataset.task_type))
```

    Downloading http://snap.stanford.edu/ogb/data/graphproppred/csv_mol_download/hiv.zip


    Downloaded 0.00 GB: 100%|â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆ| 3/3 [00:02<00:00,  1.33it/s]
    Processing...


    Extracting dataset/hiv.zip
    Loading necessary files...
    This might take a while.
    Processing graphs...


    100%|â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆ| 41127/41127 [00:00<00:00, 102734.99it/s]


    Converting graphs into PyG objects...


    100%|â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆ| 41127/41127 [00:00<00:00, 43779.36it/s]


    Saving...
    Device: cuda
    Task type: binary classification


    Done!

```python
# Load the dataset splits into corresponding dataloaders
# We will train the graph classification task on a batch of 32 graphs
# Shuffle the order of graphs for training set
if 'IS_GRADESCOPE_ENV' not in os.environ:
  train_loader = DataLoader(dataset[split_idx["train"]], batch_size=32, shuffle=True, num_workers=0)
  valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=32, shuffle=False, num_workers=0)
  test_loader = DataLoader(dataset[split_idx["test"]], batch_size=32, shuffle=False, num_workers=0)
```

    /usr/local/lib/python3.7/dist-packages/torch_geometric/deprecation.py:12: UserWarning: 'data.DataLoader' is deprecated, use 'loader.DataLoader' instead
      warnings.warn(out)

```python
if 'IS_GRADESCOPE_ENV' not in os.environ:
  # Please do not change the args
  args = {
      'device': device,
      'num_layers': 5,
      'hidden_dim': 256,
      'dropout': 0.5,
      'lr': 0.001,
      'epochs': 30,
  }
  args
```


```python
test = dataset[1:10]
```


```python
test[1]
```


    Data(edge_index=[2, 48], edge_attr=[48, 3], x=[21, 9], y=[1, 1], num_nodes=21)

#### Graph Prediction Model

使用上面的 GCN 模型产生图中节点的 embedding，然后使用池化操作（这里是平均）得到每个图的 embedding（对每个节点的 embedding 进行按元素平均），`torch_geometric.data.Batch` 中的 batch 有利于我们做这个池化操作。


```python
model
```


    GCN_Graph(
      (node_encoder): AtomEncoder(
        (atom_embedding_list): ModuleList(
          (0): Embedding(119, 256)
          (1): Embedding(4, 256)
          (2): Embedding(12, 256)
          (3): Embedding(12, 256)
          (4): Embedding(10, 256)
          (5): Embedding(6, 256)
          (6): Embedding(6, 256)
          (7): Embedding(2, 256)
          (8): Embedding(2, 256)
        )
      )
      (gnn_node): GCN(
        (convs): ModuleList(
          (0): GCNConv(256, 256)
          (1): GCNConv(256, 256)
          (2): GCNConv(256, 256)
          (3): GCNConv(256, 256)
          (4): GCNConv(256, 256)
        )
        (bns): ModuleList(
          (0): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (1): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (3): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (softmax): LogSoftmax(dim=None)
      )
      (linear): Linear(in_features=256, out_features=1, bias=True)
    )


```python
from ogb.graphproppred.mol_encoder import AtomEncoder
from torch_geometric.nn import global_add_pool, global_mean_pool

### GCN to predict graph property
class GCN_Graph(torch.nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers, dropout):
        super(GCN_Graph, self).__init__()

        # Load encoders for Atoms in molecule graphs
        self.node_encoder = AtomEncoder(hidden_dim)

        # Node embedding model
        # Note that the input_dim and output_dim are set to hidden_dim
        self.gnn_node = GCN(hidden_dim, hidden_dim,
            hidden_dim, num_layers, dropout, return_embeds=True)

        self.pool = None

        ############# Your code here ############
        ## Note:
        ## 1. Initialize self.pool as a global mean pooling layer
        ## For more information please refer to the documentation:
        ## https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#global-pooling-layers
        self.pool = global_mean_pool
        #########################################

        # Output layer
        self.linear = torch.nn.Linear(hidden_dim, output_dim)


    def reset_parameters(self):
      self.gnn_node.reset_parameters()
      self.linear.reset_parameters()

    def forward(self, batched_data):
        # TODO: Implement a function that takes as input a 
        # mini-batch of graphs (torch_geometric.data.Batch) and 
        # returns the predicted graph property for each graph. 
        #
        # NOTE: Since we are predicting graph level properties,
        # your output will be a tensor with dimension equaling
        # the number of graphs in the mini-batch

    
        # Extract important attributes of our mini-batch
        x, edge_index, batch = batched_data.x, batched_data.edge_index, batched_data.batch
        embed = self.node_encoder(x)

        out = None

        ############# Your code here ############
        ## Note:
        ## 1. Construct node embeddings using existing GCN model
        ## 2. Use the global pooling layer to aggregate features for each individual graph
        ## For more information please refer to the documentation:
        ## https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#global-pooling-layers
        ## 3. Use a linear layer to predict each graph's property
        ## (~3 lines of code)
        x = self.gnn_node(embed,edge_index)
        x = self.pool(x,batch)
        out = self.linear(x)
        #########################################

        return out
```


```python
def train(model, device, data_loader, optimizer, loss_fn):
    # TODO: Implement a function that trains your model by 
    # using the given optimizer and loss_fn.
    model.train()
    loss = 0

    for step, batch in enumerate(tqdm(data_loader, desc="Iteration")):
      batch = batch.to(device)

      if batch.x.shape[0] == 1 or batch.batch[-1] == 0:##å�ªæœ‰ä¸€ä¸ªå›¾
          pass
      else:
        ## ignore nan targets (unlabeled) when computing training loss.
        is_labeled = batch.y == batch.y

        ############# Your code here ############
        ## Note:
        ## 1. Zero grad the optimizer
        ## 2. Feed the data into the model
        ## 3. Use `is_labeled` mask to filter output and labels
        ## 4. You may need to change the type of label to torch.float32
        ## 5. Feed the output and label to the loss_fn
        ## (~3 lines of code)
        optimizer.zero_grad()
        out = model(batch)
        loss = loss_fn(out[is_labeled], batch.y[is_labeled].type(torch.float32).reshape(-1))
        #########################################

        loss.backward()
        optimizer.step()

    return loss.item()
```


```python
# The evaluation function
def eval(model, device, loader, evaluator, save_model_results=False, save_file=None):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    if save_model_results:
        print ("Saving Model Predictions")
        
        # Create a pandas dataframe with a two columns
        # y_pred | y_true
        data = {}
        data['y_pred'] = y_pred.reshape(-1)
        data['y_true'] = y_true.reshape(-1)

        df = pd.DataFrame(data=data)
        # Save to csv
        df.to_csv('ogbg-molhiv_graph_' + save_file + '.csv', sep=',', index=False)

    return evaluator.eval(input_dict)
```


```python
if 'IS_GRADESCOPE_ENV' not in os.environ:
  model = GCN_Graph(args['hidden_dim'],
              dataset.num_tasks, args['num_layers'],
              args['dropout']).to(device)
  evaluator = Evaluator(name='ogbg-molhiv')
```


```python
import copy

if 'IS_GRADESCOPE_ENV' not in os.environ:
  model.reset_parameters()

  optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
  loss_fn = torch.nn.BCEWithLogitsLoss()

  best_model = None
  best_valid_acc = 0

  for epoch in range(1, 1 + args["epochs"]):
    print('Training...')
    loss = train(model, device, train_loader, optimizer, loss_fn)

    print('Evaluating...')
    train_result = eval(model, device, train_loader, evaluator)
    val_result = eval(model, device, valid_loader, evaluator)
    test_result = eval(model, device, test_loader, evaluator)

    train_acc, valid_acc, test_acc = train_result[dataset.eval_metric], val_result[dataset.eval_metric], test_result[dataset.eval_metric]
    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        best_model = copy.deepcopy(model)
    print(f'Epoch: {epoch:02d}, '
          f'Loss: {loss:.4f}, '
          f'Train: {100 * train_acc:.2f}%, '
          f'Valid: {100 * valid_acc:.2f}% '
          f'Test: {100 * test_acc:.2f}%')
```

    Training...


    Epoch: 30, Loss: 0.0200, Train: 84.28%, Valid: 79.55% Test: 75.39%

##### Question 6: 验证和测试集中 AUC 最高的模型是哪个？


```python
if 'IS_GRADESCOPE_ENV' not in os.environ:
  train_acc = eval(best_model, device, train_loader, evaluator)[dataset.eval_metric]
  valid_acc = eval(best_model, device, valid_loader, evaluator, save_model_results=True, save_file="valid")[dataset.eval_metric]
  test_acc  = eval(best_model, device, test_loader, evaluator, save_model_results=True, save_file="test")[dataset.eval_metric]

  print(f'Best model: '
      f'Train: {100 * train_acc:.2f}%, '
      f'Valid: {100 * valid_acc:.2f}% '
      f'Test: {100 * test_acc:.2f}%')
```


    Iteration:   0%|          | 0/1029 [00:00<?, ?it/s]

    Iteration:   0%|          | 0/129 [00:00<?, ?it/s]


    Saving Model Predictions

    Iteration:   0%|          | 0/129 [00:00<?, ?it/s]


    Saving Model Predictions
    Best model: Train: 84.10%, Valid: 80.21% Test: 74.61%

