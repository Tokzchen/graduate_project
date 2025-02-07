# 创建所有节点（共20个）
from src.read_knowledge_graph_from_conexp import Node

# 创建所有节点（共20个）
nodes = {}

# 1号节点：根节点
nodes[1] = Node(1, "暴雨灾害")
root = nodes[1]

# 2-6号节点：第二层核心维度
nodes[2] = Node(2, "原因")
nodes[3] = Node(3, "灾害")
nodes[4] = Node(4, "应急响应")
nodes[5] = Node(5, "灾损")
nodes[6] = Node(6, "责任方")

root.add_child(nodes[2])
root.add_child(nodes[3])
root.add_child(nodes[4])
root.add_child(nodes[5])
root.add_child(nodes[6])

# 7-20号节点：第三层及以下节点
# 致灾因子分支
nodes[7] = Node(7, "降水")
nodes[8] = Node(8, "地形")
nodes[9] = Node(9, "城市设施")

nodes[2].add_child(nodes[7])
nodes[2].add_child(nodes[8])
nodes[2].add_child(nodes[9])

# 灾害链分支
nodes[10] = Node(10, "洪水")
nodes[11] = Node(11, "内涝")
nodes[12] = Node(12, "滑坡")

nodes[3].add_child(nodes[10])
nodes[3].add_child(nodes[11])
nodes[3].add_child(nodes[12])

# 应急响应分支
nodes[13] = Node(13, "预警")
nodes[14] = Node(14, "救援")

nodes[4].add_child(nodes[13])
nodes[4].add_child(nodes[14])

# 灾损分支
nodes[15] = Node(15, "伤亡")
nodes[16] = Node(16, "损失")

nodes[5].add_child(nodes[15])
nodes[5].add_child(nodes[16])

# 责任方分支
nodes[17] = Node(17, "气象局")
nodes[18] = Node(18, "市政")

nodes[6].add_child(nodes[17])
nodes[6].add_child(nodes[18])

# 第四层：末端节点（仅在最必要处展开）
# 降水子节点
nodes[19] = Node(19, "强度")
nodes[20] = Node(20, "时长")

nodes[7].add_child(nodes[19])
nodes[7].add_child(nodes[20])

# 检查节点总数
print(f"总节点数：{len(nodes)}")  # 应输出20

# 持久化存储
import pickle
with open('graph_nodes.pkl', 'wb') as f:
    pickle.dump(nodes, f)