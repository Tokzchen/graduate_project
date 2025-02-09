import pickle

import matplotlib.pyplot as plt
import networkx as nx


class TreeNode:
    def __init__(self, attributes):
        self.attributes = attributes
        self.children = []

    def add_child(self, child):
        self.children.append(child)


def build_graph(node, graph=None, parent=None, layer=0):
    if graph is None:
        graph = nx.DiGraph()
    graph.add_node(node, layer=layer)
    if parent is not None:
        graph.add_edge(parent, node)
    for child in node.children:
        build_graph(child, graph, parent=node, layer=layer + 1)
    return graph


def visualize_tree(root):
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    G = build_graph(root)
    pos = nx.multipartite_layout(G, subset_key="layer", align='vertical')
    # 反转y轴使得根节点在顶部
    for node in pos:
        pos[node][1] = -pos[node][1]
    labels = {node: node.attributes[0] for node in G.nodes()}
    nx.draw(G, pos, labels=labels, with_labels=True, node_size=2000,
            node_color='white', arrows=False, edge_color='gray')
    plt.show()


with open('../src/graph_nodes.pkl','rb')as f:
    nodes=pickle.load(f)
    visualize_tree(nodes[1])