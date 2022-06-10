---
title: NetworkX 网络分析
date: 2022-03-31 19:14:18
tags: 网络分析
index_img: img/networkx.png
categories:
  - python
---



NetworkX 入门 [Tutorial — NetworkX 2.7.1 documentation](https://networkx.org/documentation/stable/tutorial.html)

<!-- more -->

NetworkX 是 python 中一个进行网络分析的包；该包提供了图形图对象的类、创建标准网络图的生成器、读取现有数据集、网络分析的算法和一些基本的可视化工具，这篇文章提供了 NetworkX 的基本介绍。

### 安装

NetworkX 需要 python 3.8 以上的版本，使用 pip 安装：

``` bash
pip install networkx[default]

##如果不想要安装依赖，如numpy，scipy等可以使用
pip install networkx

##安装最新的开发版：
git clone https://github.com/networkx/networkx.git
cd networkx
pip install -e .[default]
```

### 创建图

创建一个没有节点和边的空图：

``` python
import networkx as nx
G = nx.Graph()
```

图是节点和节点对之间关系（边/连接）的集合，在 NetworkX 中节点可以是任何可哈希的对象（不可变数据类型），如字符串，图片，甚至是另一个图。

### 节点

可以使用 `add_node` 方法来添加节点：

``` python
G.add_node(1)
print(G)
>> Graph with 1 nodes and 0 edges
```

或者使用 `add_nodes_from` 从一个可迭代容器中获取数据添加节点：

``` python
G.add_nodes_from([2, 3])
print(G)
>> Graph with 3 nodes and 0 edges
G.add_nodes_from((4,5))
print(G)
>> Graph with 5 nodes and 0 edges
```

我们也可以在添加节点时给节点附加属性，此时节点要以元组的形式添加：`(节点，表示节点属性的字典)`：

``` python
G.add_nodes_from([
  ("n6",{"color":"red"}),
  (7,{"color":"green"})
])
print(G)
>> Graph with 7 nodes and 0 edges
```

也可以将另一个图中节点全部添加到现有的图中：

``` python
H = nx.path_graph(10) ##10个节点的线性图
print(H)
>> Graph with 10 nodes and 9 edges
G.add_nodes_from(H)
print(G)
>> Graph with 11 nodes and 0 edges
```

可以看到 G 的节点只增加了 4 个，这是因为之前 G 就有 `1,2,3,4,5,7` 的节点了，而 H 里面节点是 1-10，所以不一样的节点只有 4 个了。前面说过可以使用整个图作为另一个图的节点，所以我们可以把 H 作为一个节点添加到 G 里面：

``` python
G.add_node(H)
print(G)
>> Graph with 12 nodes and 0 edges
```

### 边

使用 `add_edge` 往图里面添加边，参数是：

``` python
G.add_edge(1, 2)
e = (2, 3)
G.add_edge(*e)##元组需要拆开
```

添加多个边需要用列表：

``` python
G.add_edges_from([(1, 2), (1, 3)])
print(G)
>> Graph with 12 nodes and 3 edges
```

边可以像节点一样添加属性：`(node1,node2,{属性字典})`:

``` python
G.add_edges_from([
  (3,4,{"weight":3})
])
```

也可以直接使用其他图中的边：

``` python
G.add_edges_from(H.edges)
print(G)
>> Graph with 12 nodes and 10 edges
```

### 检查图中的元素

有四种基本的图属性：`G.nodes`, `G.edges`, `G.adj`, `G.degree`:

``` python
G.nodes
>> NodeView((1, 2, 3, 4, 5, 'n6', 7, 0, 6, 8, 9, <networkx.classes.graph.Graph object at 0x0000022A0F0C62E0>))
G.edges
>> EdgeView([(1, 2), (1, 3), (1, 0), (2, 3), (3, 4), (4, 5), (5, 6), (7, 6), (7, 8), (8, 9)])
G.adj##每个节点的邻居节点及其属性
>> AdjacencyView({1: {2: {}, 3: {}, 0: {}}, 2: {1: {}, 3: {}}, 3: {2: {}, 1: {}, 4: {'weight': 3}}, 4: {3: {'weight': 3}, 5: {}}, 5: {4: {}, 6: {}}, 'n6': {}, 7: {6: {}, 8: {}}, 0: {1: {}}, 6: {5: {}, 7: {}}, 8: {7: {}, 9: {}}, 9: {8: {}}, <networkx.classes.graph.Graph object at 0x0000022A0F0C62E0>: {}})
G.degree##每个节点的度
>> DegreeView({1: 3, 2: 2, 3: 3, 4: 2, 5: 2, 'n6': 0, 7: 2, 0: 1, 6: 2, 8: 2, 9: 1, <networkx.classes.graph.Graph object at 0x0000022A0F0C62E0>: 0})
```

可以看到这些都是一种 `view` 的数据格式，我们可以将其转化为列表，字典或者其他的容器数据类型：

``` python
list(G.nodes)
>> [1, 2, 3, 4, 5, 'n6', 7, 0, 6, 8, 9, <networkx.classes.graph.Graph object at 0x0000022A0F0C62E0>]
list(G.edges)
>> [(1, 2), (1, 3), (1, 0), (2, 3), (3, 4), (4, 5), (5, 6), (7, 6), (7, 8), (8, 9)]
```

我们也可以将其视作字典来根据键选择值：

``` python
list(G.adj)##只是把类似字典的值给拿出来了
>> [1, 2, 3, 4, 5, 'n6', 7, 0, 6, 8, 9, <networkx.classes.graph.Graph object at 0x0000022A0F0C62E0>]
list(G.adj)[1]
>> 2
list(G.adj[1])##注意，这才是我们想要的第二个节点的邻居节点
>> [2, 3, 0]
```

还可以展示一部分节点的边和度：

``` python
G.edges([2, 3])##和节点2 3 有连接的边
>> EdgeDataView([(2, 1), (2, 3), (3, 1), (3, 4)])
G.degree([2, 3])
>> DegreeView({2: 2, 3: 3})
```

### 从图中移除元素

移除元素和添加元素的方法类似，使用 `remove_node`, `remove_nodes_from`, `remove_edge`, `remove_edges_from`：

``` python
list(G.nodes)
>> [1, 2, 3, 4, 5, 'n6', 7, 0, 6, 8, 9, <networkx.classes.graph.Graph object at 0x0000022A0F0C62E0>]
G.remove_node(2)
G.add_nodes_from(["q","m"])
list(G.nodes)
>> [1, 3, 4, 5, 'n6', 7, 0, 6, 8, 9, <networkx.classes.graph.Graph object at 0x0000022A0F0C62E0>, 'q', 'm']
G.remove_nodes_from("qm")
list(G.nodes)
>> [1, 3, 4, 5, 'n6', 7, 0, 6, 8, 9, <networkx.classes.graph.Graph object at 0x0000022A0F0C62E0>]
list(G.edges)
>> [(1, 3), (1, 0), (3, 4), (4, 5), (5, 6), (7, 6), (7, 8), (8, 9)]
G.remove_edge(1, 3)
list(G.edges)
>> [(1, 0), (3, 4), (4, 5), (5, 6), (7, 6), (7, 8), (8, 9)]
```

### 使用图构造器

创建图不一定需要像上面那么逐步的加入点和边，我们可以先创建图的结构然后使用特定图类型的构造器来基于这些预先创建的结构来生成图：

``` python
G.clear()##清空图
print(G)
>> Graph with 0 nodes and 0 edges
G.add_edge(1, 2)
H = nx.DiGraph(G) ##基于已有的图创建一个有向图
list(H.edges())
>> [(1, 2), (2, 1)]
edgelist = [(0, 1), (1, 2), (2, 3)]
H = nx.Graph(edgelist)
list(H.edges())
>> [(0, 1), (1, 2), (2, 3)]
adjacency_dict = {0: (1, 2), 1: (0, 2), 2: (0, 1)}##基于邻接关系创建图
H = nx.Graph(adjacency_dict)
list(H.edges())
>> [(0, 1), (0, 2), (1, 2)]
```

### 获取边和邻居节点

除了使用上面的 view 方法（Graph.edges, Graph.adj）来查看节点和邻居，也可以使用类似下标的方法：

``` python
G = nx.Graph([(1, 5, {"color": "yellow"})])
G.adj
>> AdjacencyView({1: {5: {'color': 'yellow'}}, 5: {1: {'color': 'yellow'}}})
G.adj[5]
>> AtlasView({1: {'color': 'yellow'}})
G[5]##即把与节点5连接的边拿出来
>> AtlasView({1: {'color': 'yellow'}})
G.edges[1,5]
>> {'color': 'yellow'}
```

我们也可以通过这种方法来改变边的属性：

``` python
G.add_edge(1, 3)
G[1][3]['color'] = "blue"
G.edges[1, 5]['color'] = "red"
G.adj
>> AdjacencyView({1: {5: {'color': 'red'}, 3: {'color': 'blue'}}, 5: {1: {'color': 'red'}}, 3: {1: {'color': 'blue'}}})
```

可以通过 `G.adjacency` 或者 `G.adj.items` 来遍历节点对：

``` python
FG = nx.Graph()
FG.add_weighted_edges_from([(1, 2, 0.125), (1, 3, 0.75), (2, 4, 1.2), (3, 4, 0.375)])

for n, nbrs in FG.adj.items():
  for nbr, eattr in nbrs.items():
    wt = eattr['weight']
    if wt < 0.5: print(f"({n}, {nbr}, {wt:.3})")
>> (1, 2, 0.125)
>> (2, 1, 0.125)
>> (3, 4, 0.375)
>> (4, 3, 0.375)
```

更简洁的方法是通过 `edges.data` 来展现属性：

``` python
for (u, v, wt) in FG.edges.data('weight'):
  if wt < 0.5:
    print(f"({u}, {v}, {wt:.3})")
>> (1, 2, 0.125)
>> (3, 4, 0.375)
```

### 添加属性

可以向图，节点或边添加属性，属性可以是权重，标签，颜色或者任意 python 对象，这些属性都是以键值对形式指定的。

创建图时就可以直接赋予图属性：

``` python
G = nx.Graph(day="Friday")
G.graph
>> {'day': 'Friday'}
```

也可以后来更改：

``` python
G.graph['day'] = "Monday"
G.graph
>> {'day': 'Monday'}
```

节点和边也是类似：

``` python
##节点
G.add_node(1, time='5pm')
G.add_nodes_from([3], time='2pm')
G.add_nodes_from([
  (4,{"time":"4pm"}),
  (6,{"time":"5pm"})
])

G.nodes[6]
>> {'time': '5pm'}
G.nodes[1]['room'] = 714##添加属性
G.nodes.data()

##边
>> NodeDataView({1: {'time': '5pm', 'room': 714}, 3: {'time': '2pm'}, 4: {'time': '4pm'}, 6: {'time': '5pm'}})
G.add_edge(1, 2, weight=4.7 )
G.add_edges_from([(3, 4), (4, 5)], color='red')
G.add_edges_from([
  (1, 2, {'color': 'blue'}), 
  (2, 3, {'weight': 8})
])
G[1][2]['weight'] = 4.7
G.edges[3, 4]['weight'] = 4.2

G.edges.data()
>> EdgeDataView([(1, 2, {'weight': 4.7, 'color': 'blue'}), (3, 4, {'color': 'red', 'weight': 4.2}), (3, 2, {'weight': 8}), (4, 5, {'color': 'red'})])
```

### 有向图

`DiGraph` 类表示有向边，有向图有一些特殊的特征，比如度可以分成入度和出度：

``` python
DG = nx.DiGraph()
DG.add_weighted_edges_from([(1, 2, 0.5), (3, 1, 0.75)])
DG.out_degree(1, weight='weight')
>> 0.5
DG.out_degree(3, weight='weight')
>> 0.75
DG.degree(1, weight='weight')##入度+出度
>> 1.25
DG.degree(1)##不考虑边权重
>> 2
```

注意有向图里面的 `neighbor` 和 `successor` 是一样的，`successor` 是后继节点，也就是如果 n 指向 m ，那么 m 是 n 的 \``successor`:

``` python
list(DG.successors(1))
>> [2]
list(DG.neighbors(1))
>> [2]
```

可以使用 `to_undirected` 将有向图转化为无向图或者直接从有向图构建无向图：

``` python
G  = DG.to_undirected()
G.adj

##or
>> AdjacencyView({1: {2: {'weight': 0.5}, 3: {'weight': 0.75}}, 2: {1: {'weight': 0.5}}, 3: {1: {'weight': 0.75}}})
H = nx.Graph(DG)
H.adj
>> AdjacencyView({1: {2: {'weight': 0.5}, 3: {'weight': 0.75}}, 2: {1: {'weight': 0.5}}, 3: {1: {'weight': 0.75}}})
```

### 多图

多图指的是节点对之间有多条边：

``` python
MG = nx.MultiGraph()
MG.add_weighted_edges_from([(1, 2, 0.5), (1, 2, 0.75), (2, 3, 0.5)])
dict(MG.degree(weight='weight'))
>> {1: 1.25, 2: 1.75, 3: 0.5}
MG.adj
##将多图转化为标准的图
>> MultiAdjacencyView({1: {2: {0: {'weight': 0.5}, 1: {'weight': 0.75}}}, 2: {1: {0: {'weight': 0.5}, 1: {'weight': 0.75}}, 3: {0: {'weight': 0.5}}}, 3: {2: {0: {'weight': 0.5}}}})
GG = nx.Graph()
for n, nbrs in MG.adjacency():
  for nbr, edict in nbrs.items():
    minvalue = min([d['weight'] for d in edict.values()])
    GG.add_edge(n, nbr, weight = minvalue)
```

### 可视化

NetworkX 不是专业绘图的包，但是提供了一些简单的绘图函数。

``` python
import matplotlib.pyplot as plt
nx.draw(G, with_labels=True, font_weight='bold')
plt.show()  
```

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/unnamed-chunk-2-1.png" width="672" />

``` python
G = nx.petersen_graph()
nx.draw(G, with_labels=True, font_weight='bold')
plt.show() 
```

<img src="https://picgo-wutao.oss-cn-shanghai.aliyuncs.com/img/unnamed-chunk-2-2.png" width="672" />

### 图生成和操作

#### 使用经典图操作器

| [`subgraph`](https://networkx.org/documentation/stable/reference/generated/networkx.classes.function.subgraph.html#networkx.classes.function.subgraph)(G, nbunch) | Returns the subgraph induced on nodes in nbunch.             |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`union`](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.operators.binary.union.html#networkx.algorithms.operators.binary.union)(G, H[, rename, name]) | Return the union of graphs G and H.                          |
| [`disjoint_union`](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.operators.binary.disjoint_union.html#networkx.algorithms.operators.binary.disjoint_union)(G, H) | Return the disjoint union of graphs G and H.                 |
| [`cartesian_product`](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.operators.product.cartesian_product.html#networkx.algorithms.operators.product.cartesian_product)(G, H) | Returns the Cartesian product of G and H.                    |
| [`compose`](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.operators.binary.compose.html#networkx.algorithms.operators.binary.compose)(G, H) | Returns a new graph of G composed with H.                    |
| [`complement`](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.operators.unary.complement.html#networkx.algorithms.operators.unary.complement)(G) | Returns the graph complement of G.                           |
| [`create_empty_copy`](https://networkx.org/documentation/stable/reference/generated/networkx.classes.function.create_empty_copy.html#networkx.classes.function.create_empty_copy)(G[, with_data]) | Returns a copy of the graph G with all of the edges removed. |
| [`to_undirected`](https://networkx.org/documentation/stable/reference/generated/networkx.classes.function.to_undirected.html#networkx.classes.function.to_undirected)(graph) | Returns an undirected view of the graph `graph`.             |
| [`to_directed`](https://networkx.org/documentation/stable/reference/generated/networkx.classes.function.to_directed.html#networkx.classes.function.to_directed)(graph) | Returns a directed view of the graph `graph`.                |

#### 调用一些预定义的小型图

| [`petersen_graph`](https://networkx.org/documentation/stable/reference/generated/networkx.generators.small.petersen_graph.html#networkx.generators.small.petersen_graph)([create_using]) | Returns the Petersen graph.                       |
| ------------------------------------------------------------ | ------------------------------------------------- |
| [`tutte_graph`](https://networkx.org/documentation/stable/reference/generated/networkx.generators.small.tutte_graph.html#networkx.generators.small.tutte_graph)([create_using]) | Returns the Tutte graph.                          |
| [`sedgewick_maze_graph`](https://networkx.org/documentation/stable/reference/generated/networkx.generators.small.sedgewick_maze_graph.html#networkx.generators.small.sedgewick_maze_graph)([create_using]) | Return a small maze with a cycle.                 |
| [`tetrahedral_graph`](https://networkx.org/documentation/stable/reference/generated/networkx.generators.small.tetrahedral_graph.html#networkx.generators.small.tetrahedral_graph)([create_using]) | Returns the 3-regular Platonic Tetrahedral graph. |

#### 经典图的构造器

| [`complete_graph`](https://networkx.org/documentation/stable/reference/generated/networkx.generators.classic.complete_graph.html#networkx.generators.classic.complete_graph)(n[, create_using]) | Return the complete graph `K_n` with n nodes.                |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`complete_bipartite_graph`](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.bipartite.generators.complete_bipartite_graph.html#networkx.algorithms.bipartite.generators.complete_bipartite_graph)(n1, n2[, create_using]) | Returns the complete bipartite graph `K_{n_1,n_2}`.          |
| [`barbell_graph`](https://networkx.org/documentation/stable/reference/generated/networkx.generators.classic.barbell_graph.html#networkx.generators.classic.barbell_graph)(m1, m2[, create_using]) | Returns the Barbell Graph: two complete graphs connected by a path. |
| [`lollipop_graph`](https://networkx.org/documentation/stable/reference/generated/networkx.generators.classic.lollipop_graph.html#networkx.generators.classic.lollipop_graph)(m, n[, create_using]) | Returns the Lollipop Graph; `K_m` connected to `P_n`.        |

#### 随机图生成器

| [`erdos_renyi_graph`](https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_graphs.erdos_renyi_graph.html#networkx.generators.random_graphs.erdos_renyi_graph)(n, p[, seed, directed]) | Returns a Gn,p random graph, also known as an Erdős-Rényi graph or a binomial graph. |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`watts_strogatz_graph`](https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_graphs.watts_strogatz_graph.html#networkx.generators.random_graphs.watts_strogatz_graph)(n, k, p[, seed]) | Returns a Watts–Strogatz small-world graph.                  |
| [`barabasi_albert_graph`](https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_graphs.barabasi_albert_graph.html#networkx.generators.random_graphs.barabasi_albert_graph)(n, m[, seed, ...]) | Returns a random graph using Barabási–Albert preferential attachment |
| [`random_lobster`](https://networkx.org/documentation/stable/reference/generated/networkx.generators.random_graphs.random_lobster.html#networkx.generators.random_graphs.random_lobster)(n, p1, p2[, seed]) | Returns a random lobster graph.                              |
