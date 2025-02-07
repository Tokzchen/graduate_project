# 定义节点类来存储每个节点的信息
import collections
from collections import defaultdict
import pickle
from rdflib import RDFS


class Node:
    def __init__(self, node_id, attribute):
        self.node_id = node_id  # 节点编号
        self.attributes = [] # 节点属性
        self.attributes.append(attribute)
        self.children = []  # 子节点
        self.parent = []  # 父节点

    def add_child(self, child_node):
        self.children.append(child_node)
        child_node.parent.append(self)

    def __repr__(self):
        return f"Node(id={self.node_id}, name={self.name})"


# 解析树的描述文件
def parse_tree_file(file_path):
    nodes = {}  # 存储所有的节点
    edges = []  # 存储所有的边
    node_attributes = defaultdict(list)  # 存储节点的属性

    # 读取文件
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith("Node"):
                parts = line.split(", ")
                node_id = int(parts[0].split(":")[1].strip())
                value_1 = float(parts[1].strip())
                value_2 = float(parts[2].strip())
                node = Node(node_id)
                nodes[node_id] = node
            elif line.startswith("Edge"):
                parts = line.split(", ")
                node_1 = int(parts[0].split(":")[1].strip())
                node_2 = int(parts[1].strip())
                edges.append((node_1, node_2))
            elif line.startswith("Attribute"):
                parts = line.split(", ")
                node_id = int(parts[0].split(":")[1].strip())
                attribute = parts[1].strip()
                if node_id not in node_attributes:
                    node_attributes[node_id] = []
                node_attributes[node_id].append(attribute)

    # 给节点添加属性
    for node_id, attributes in node_attributes.items():
        node = nodes[node_id]
        for attribute in attributes:
            node.add_attribute(attribute)

    # 构建边
    for node_1, node_2 in edges:
        if node_1 in nodes and node_2 in nodes:
            nodes[node_2].add_child(nodes[node_1])

    # 剪枝
    for node_id,node in nodes.items():
        if len(node.attributes)==0:
            for p in node.parent:
                p.children.remove(node)

    return nodes


# 打印节点及其连接情况
def print_tree(node):
    que=collections.deque()
    que.append(node)
    while que:
        item=que.popleft()
        print(item.attributes)
        for c in item.children:
            que.append(c)
from rdflib import Graph, Namespace, URIRef, RDF, OWL, Literal
def convert_to_owl(graph,node,parent):
    """

    :param graph: rdflib中的Graph()
    :param node: 当前节点
    :param parent: 父节点，没有则传None
    :return:
    """
    EX = Namespace("")

    # 创建一个空的 RDF 图
    g = graph

    # 绑定命名空间前缀
    g.bind("ex", EX)
    g.bind("owl", OWL)

    # 定义类（OWL 类）
    cur_class = EX[node.attributes[0]]
    g.add((cur_class,RDF.type,OWL.Class))
    if parent is not None:
        parent_class = EX[parent.attributes[0]]
        g.add((parent_class,RDF.type,OWL.Class))
        # 定义子类关系
        g.add((cur_class, RDFS.subClassOf, parent_class))

    # 递归进行关系绑定
    for c in node.children:
        convert_to_owl(graph,c,node)
    return graph

if __name__ == '__main__':
    # 测试解析
    file_path = 'how.txt'  # 假设文件路径为tree_description.txt
    nodes = parse_tree_file(file_path)
    print_tree(nodes[1])
    with open('graph_nodes.pkl','wb') as f:
        pickle.dump(nodes,f)