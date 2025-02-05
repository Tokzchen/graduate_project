# 链接优先级排序-使用主题相关度
import math
import pickle
from collections import defaultdict, Counter
import numpy as np
import jieba
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from src.read_knowledge_graph_from_conexp import Node

class LinkRating:
    """
        链接评分排序由三部分组成
        1.链接所在的网页的网页相关度，可以用html_feature_matrix.py进行获取，取余弦相似度
        2. 链接锚文本分析相关度，
        3. 链接所在网页的pagerank值
        LinkRating类主要是及进行2、3的分析
    """
    def __init__(self,topic_matrix):
        """
        :topic_matrix:主题语义向量
            在这里存储记录已爬取的网页的相关数据，记录的内容包括：
            -已爬取的网页的url数，url列表，page_rank值，分别使用字典进行记录
        """
        # 初始化图结构
        self.out_links = defaultdict(set)  # 出链接：每个网页指向的其他网页
        self.in_links = defaultdict(set)  # 入链接：每个网页被哪些网页指向
        self.anchor_scores=defaultdict(dict)  # 保存锚文本的主题相关度
        self.topic_matrix=topic_matrix

        self.pagerank = defaultdict(float)  # {url: rank}
        self.pending_updates = set()  # 待更新的 URL 队列
        self.damping = 0.85  # 阻尼系数
        self.omega = 0.5  # 调节因子
        self.tol = 1e-4  # 收敛阈值

    def _cosine_similarity(self,vec1, vec2):
        """
        计算两个向量的余弦相似度
        :param vec1: 向量1 (list or numpy array)
        :param vec2: 向量2 (list or numpy array)
        :return: 余弦相似度 (float)
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        dot_product = np.dot(vec1, vec2)  # 点积
        norm_vec1 = np.linalg.norm(vec1)  # vec1 的模
        norm_vec2 = np.linalg.norm(vec2)  # vec2 的模
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0  # 如果有零向量，余弦相似度为 0
        return dot_product / (norm_vec1 * norm_vec2)

    def extract_links(self,html_content, base_url):
        """
        从网页中提取所有超链接，并将其转换为完整 URL。
        :param html_content: 网页内容 (str)
        :param base_url: 网页的基础 URL，用于处理相对链接 (str)
        :return: 一组完整链接 (set)
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        links = set()
        for tag in soup.find_all('a', href=True):  # 找到所有带 href 的 <a> 标签
            href = tag['href']
            # 将相对链接转换为绝对链接,如果是完整的地址则会直接返回原来的地址
            full_url = urljoin(base_url, href)
            # 忽略无效链接和锚点
            parsed_url = urlparse(full_url)
            if parsed_url.scheme in ['http', 'https']:
                links.add(full_url)
        return links

    def normalize_url(self,url):
        """
        标准化 URL，移除尾斜杠和查询参数等。
        :param url: 原始 URL (str)
        :return: 标准化后的 URL (str)
        """
        parsed = urlparse(url)
        # 移除查询参数和 fragment（锚点）
        normalized_url = parsed.scheme + "://" + parsed.netloc + parsed.path
        # 去掉尾部的斜杠
        if normalized_url.endswith('/'):
            normalized_url = normalized_url[:-1]
        return normalized_url

    def add_page_to_graph(self, base_url, html_content, keyword_list, base):
        """
        动态添加 URL 并触发增量 PageRank 更新
        """
        normalized_base_url = self.normalize_url(base_url)
        links = self.extract_links(html_content, base_url)

        # 分析锚文本主题相关度
        anchor_scores = self.url_anchor_analyze_relevance_scores(
            base_url, html_content, keyword_list, base, self.topic_matrix
        )

        # 初始化当前 URL 的 PageRank（如果未存在）
        if normalized_base_url not in self.pagerank:
            self.pagerank[normalized_base_url] = 1.0

        for link in links:
            normalized_link = self.normalize_url(link)

            # 更新出链和入链关系
            self.out_links[normalized_base_url].add(normalized_link)
            self.in_links[normalized_link].add(normalized_base_url)

            # 保存锚文本相关度
            self.anchor_scores[normalized_base_url][normalized_link] = anchor_scores.get(link, 0)

            # 初始化出链 URL 的 PageRank（如果未存在）
            if normalized_link not in self.pagerank:
                self.pagerank[normalized_link] = 1.0

        # 标记受影响节点：当前 URL 及其所有出链 URL
        self.pending_updates.add(normalized_base_url)
        self.pending_updates.update([self.normalize_url(link) for link in links])

        # 触发增量 PageRank 更新
        self._batch_update()

    def _batch_update(self, max_iter=3):
        """批量更新受影响的 PageRank 值"""
        for _ in range(max_iter):
            changes = {}
            for url in self.pending_updates:
                old_rank = self.pagerank[url]

                # 计算入链贡献（公式中的求和部分）
                rank_sum = 0.0
                for in_page in self.in_links.get(url, []):
                    # 跳过无出链的页面（悬挂节点）
                    if len(self.out_links.get(in_page, [])) == 0:
                        continue

                    # 获取语义相关度权重
                    sem_score = self.anchor_scores.get(in_page, {}).get(url, 0)
                    weight = 1 + self.omega * sem_score

                    # 贡献值 = PR(in_page) / C(in_page) * weight
                    rank_sum += (self.pagerank[in_page] / len(self.out_links[in_page])) * weight

                # 更新 PageRank
                new_rank = (1 - self.damping) + self.damping * rank_sum
                changes[url] = abs(new_rank - old_rank)
                self.pagerank[url] = new_rank

            # 判断是否收敛
            if max(changes.values(), default=0) < self.tol:
                break

        # 清空待更新队列
        self.pending_updates.clear()

        # 处理悬挂节点（可选）
        self._handle_dangling_nodes()

    def _handle_dangling_nodes(self):
        """处理无出链的悬挂节点贡献"""
        dangling_nodes = [url for url in self.pagerank if len(self.out_links.get(url, [])) == 0]
        if not dangling_nodes:
            return

        # 计算悬挂节点的总贡献
        total_dangling_rank = sum(self.pagerank[url] for url in dangling_nodes)
        dangling_contribution = self.damping * total_dangling_rank / len(self.pagerank)

        # 将贡献分配给所有节点
        for url in self.pagerank:
            self.pagerank[url] += dangling_contribution

    def url_anchor_analyze(self,base_url,html_text,keyword_list,base):
        """
        当爬取到一个html网页时，会对网页中所有的url(即外链)进行分析，根据锚文本进行评分，排序
        :param html_text:
        :param keyword_list:关键词列表
        :param base: 计算权重时的底数
        :return:
        """
        # 初始化 BeautifulSoup
        soup = BeautifulSoup(html_text, 'html.parser')
        anchor_text_and_urls = [
            (a['href'],a.get_text(strip=True))
            for a in soup.find_all('a', href=True)
        ]
        # 对url进行标准化,拼接等操作
        temp_urls=[]
        for url,text in anchor_text_and_urls:
            full_url = urljoin(base_url, url)
            # 忽略无效链接和锚点
            parsed_url = urlparse(full_url)
            if parsed_url.scheme in ['http', 'https']:
                temp_urls.append((full_url,text))
        anchor_text_and_urls=temp_urls
        anchor_text_and_urls=[(self.normalize_url(url),text) for url,text in anchor_text_and_urls]
        # 根据第一个值去重
        anchor_text_and_urls = list({item[0]: item for item in anchor_text_and_urls}.values())
        exist_dict=defaultdict(list)
        results={key:defaultdict(int) for key,_ in anchor_text_and_urls}
        max_freq=defaultdict(int)
        for url,text in anchor_text_and_urls:
            words=jieba.lcut(text)
            words_cnt=Counter(words)
            for kw in keyword_list:
                if words_cnt[kw]>max_freq[url]:
                    max_freq[url]=words_cnt[kw]
                results[url][kw]=words_cnt[kw]
                exist_dict[kw].append(url)
        #进行每个url的评分统计
        matrix_res=defaultdict(list)
        for url,cnt in results.items():
            for kw in keyword_list:
                # 为防止分母为0，分子分母同时加1
                value=((results[url][kw]+1)/(max_freq[url]+1))*(math.log((len(anchor_text_and_urls)/len(exist_dict[kw]))+0.01,base))
                matrix_res[url].append(value)
        return matrix_res

    def url_anchor_analyze_relevance_scores(self,base_url,html_text,keyword_list,base,topic_matrix):
        """
        计算该网页下各锚文本的主题相关度，主要是使用余弦相似度，参数与url_anchor_analyze一致
        :param base_url:
        :param html_text:
        :param keyword_list:
        :param base:
        :param topic_matrix: 主题语义权重向量，根据本体树（知识图谱）计算获取，使用TopicRelevance.main_generate()获取
        :return:
        """
        res={}
        anchor_matrix_dict=self.url_anchor_analyze(base_url,html_text,keyword_list,base)
        for url,vector in anchor_matrix_dict.items():
            cos_sim=self._cosine_similarity(vector,topic_matrix)
            res[url]=cos_sim
        return res





if __name__ == '__main__':
    # 假设已初始化爬虫对象 crawler
    with open('graph_nodes.pkl', 'rb') as f:
        nodes = pickle.load(f)
        from src.top_relevance import TopicRelevance
        topic_relevance = TopicRelevance(nodes, nodes[1])
        topic_relevance.main_generate(nodes[1], nodes, (0.2, 0.2, 0.2, 0.2, 0.2))
        print(topic_relevance.topic_meaning_matrix)
        print(topic_relevance.keyword_list)
        crawler=LinkRating(topic_relevance.topic_meaning_matrix)
        crawler.add_page_to_graph(
            base_url="http://example.com/A",
            html_content="<a href='http://example.com/B'>link</a>...",
            keyword_list=topic_relevance.keyword_list,
            base=2.0
        )

        # 查看 PageRank 结果
        print(crawler.pagerank)

