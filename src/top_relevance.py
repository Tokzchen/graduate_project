# 主题相关度计算
# 对本体（知识图谱）树的相关值的操作
# 主题是根节点，该类是计算其他概念（叶子节点）与主题之间的相关度计算
# 最终得到的是主题权重向量
import pickle
from typing import Tuple
from read_knowledge_graph_from_conexp import  Node
from src.link_rating import LinkRating


class TopicRelevance:

    def __init__(self,nodes,root):
        self.nodes=nodes
        self.root=root
        self.max_child_cnt=None
        self.tree_height=None

    def _count_tree_height(self,node):
        def count_tree_height(root):
            if root is None:
                return 0
            child_max_height = 0
            for c in root.children:
                temp = count_tree_height(c)
                if temp > child_max_height:
                    child_max_height = temp
            return child_max_height + 1
        return count_tree_height(node)

    def count_dis(self,root,node,x):
        """
        计算语义距离因子，dis=x/(x+pow(distance(root,node),2))
        :param root:根节点，即主题节点
        :param node:即概念叶子节点
        :param x: 指定调节因子，大于0的实数，由专家意见决定
        :return:
        """
        res=-1
        def dfs(s,d,dis,path_set):
            if s==d:
                nonlocal res
                res=dis
                return
            path_set.add(s.node_id)
            for c in s.children:
                if c.node_id in path_set:
                    continue
                else:
                    dfs(c,d,dis+1,path_set)
            path_set.remove(s.node_id)
        my_set=set()
        dfs(root,node,0,my_set)
        if res==-1:
            raise RuntimeError(f'can not find such node, node_id:{node.node_id}')
        return x/(x+pow(res,2))


    def count_den(self):
        """
        概念密度因子，因为计算的都是与根节点的概念密度相似度，所以能直接返回即可
        所有的概念子节点与根节点的最近公共祖先都是根节点，因此：
        :return:
        """
        if self.max_child_cnt is None:
            max_child_cnt=0
            for nid,n in self.nodes.items():
                if len(n.children)>max_child_cnt:
                    max_child_cnt=len(n.children)
            self.max_child_cnt=max_child_cnt
        root_child_cnt=len(self.root.children)
        return (root_child_cnt+1)/(self.max_child_cnt+1)

    def count_dep(self,root,node):
        """
        概念深度因子，根节点深度为1，
        :param root:
        :param node:
        :return:
        """

        def count_tree_height(root):
            if root is None:
                return 0
            child_max_height = 0
            for c in root.children:
                temp = count_tree_height(c)
                if temp > child_max_height:
                    child_max_height = temp
            return child_max_height + 1
        if self.tree_height is None:
            # 首先需要计算树高度
            th=count_tree_height(self.root)
            self.tree_height=th
        # root和node的最近公共祖先是root，node的深度=self.tree_height-count_tree_height(node)
        node_dep=self.tree_height-count_tree_height(node)
        part_one=(1+node_dep)/((1+node_dep)+(2*self.tree_height))
        part_two=1/self.tree_height
        return 0.5*(part_one+part_two)

    def count_coi(self,root,node):
        return 1/((self.tree_height-self._count_tree_height(node))+1)


    def count_rel(self):
        # 根据专家意见取1，0.5，0.33
        return 0.5

    def main_generate(self,root,nodes,settings:Tuple):
        """
        产生主题语义向量的主逻辑方法【主题权重向量】
        对于每一个节点分别计算一个最终的权重值，存入新的字典
        :param root:
        :param nodes:
        :param settings-设置的五个影响因子的权重，和为1，例如(0.2,0.2,0.2,0.2,0.2)
        :return:
        """
        final_score_dict={}
        for node_id,node in self.nodes.items():
            if len(node.attributes)==0:
                continue
            # 计算dis,x取1
            dis=self.count_dis(self.root,node,1)
            # 计算den
            den=self.count_den()
            # 计算dep
            dep=self.count_dep(self.root,node)
            # 计算coi
            coi=self.count_coi(self.root,node)
            # 计算rel
            rel=self.count_rel()
            # 加权求和
            a,b,c,d,e=settings
            scores=a*dis+b*den+c*dep+d*coi+e*rel
            final_score_dict[node_id]=scores
        dict_list=final_score_dict.items()
        dict_list=sorted(dict_list,key=lambda item:item[0])# 排序
        final_score_dict=dict(dict_list) 
        self.final_score_dict = final_score_dict
        # 根据主题语义向量，构建主题关键词集合
        self.topic_meaning_matrix=[i[1] for i in dict_list]
        self.keyword_list=[]
        for node_id,value in dict_list:
            self.keyword_list.append(self.nodes[node_id].attributes[0])

if __name__ == '__main__':
    with open('graph_nodes.pkl','rb')as f:
        nodes=pickle.load(f)
        topic_relevance=TopicRelevance(nodes,nodes[1])
        topic_relevance.main_generate(nodes[1],nodes,(0.2,0.2,0.2,0.2,0.2))
        print(topic_relevance.topic_meaning_matrix)
        print(topic_relevance.keyword_list)
        graph=LinkRating(topic_matrix=topic_relevance.topic_meaning_matrix)
        # 模拟网页内容
        html_content_A = """
           <html>
               <body>
                   <a href="B.html">气象灾害监测</a>
                   <a href="C.html">天气预警服务</a>4
                   <a href="http://www.baidu.com">百度一下，看看监测结果</a>
               </body>
           </html>
           """
        html_content_B = """
           <html>
               <body>
                   <a href="C.html">低温干旱警告</a>
                   <a href="D.html">气象服务平台</a>
               </body>
           </html>
           """
        html_content_C = """
           <html>
               <body>
                   <a href="A.html">天气数据分析</a>
               </body>
           </html>
           """

        # 定义关键词列表和底数
        keyword_list = topic_relevance.keyword_list
        base = 2

        # 添加网页到图中
        graph.add_page_to_graph("http://www.first.com", html_content_A, keyword_list, base)
        graph.add_page_to_graph("http://www.second.com", html_content_B, keyword_list, base)
        graph.add_page_to_graph("http://www.third.com", html_content_C, keyword_list, base)

        # 打印链接关系和锚文本相关度
        print("Out Links:", dict(graph.out_links))
        print("In Links:", dict(graph.in_links))
        print("Anchor Scores:", graph.anchor_scores)
        pages = list(graph.out_links.keys() | graph.in_links.keys())
        damping=0.85
        omega=1
        for p in pages:
            r=graph.get_pagerank(p,damping,omega)
            print(f'{p}的pagerank={r}')