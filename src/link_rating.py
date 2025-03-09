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
        with open('../src/graph_nodes.pkl', 'rb') as f:
            from src.top_relevance import TopicRelevance
            nodes = pickle.load(f)
            topic_relevance = TopicRelevance(nodes, nodes[1])
            topic_relevance.main_generate(nodes[1], nodes, (0.2, 0.2, 0.2, 0.2, 0.2))
            kw_list=topic_relevance.keyword_list
            for w in kw_list:
                jieba.add_word(w)
        # 初始化图结构
        self.out_links = defaultdict(set)  # 出链接：每个网页指向的其他网页
        self.in_links = defaultdict(set)  # 入链接：每个网页被哪些网页指向
        self.anchor_scores=defaultdict(dict)  # 保存锚文本的主题相关度
        self.topic_matrix=topic_matrix


        self.block_list_set = {'https://baike.baidu.com', 'https://beian.miit.gov.cn'}

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
                if any(full_url.startswith(blocked) for blocked in self.block_list_set):
                    continue
                links.add(full_url)
        return links

    def normalize_url(self,url):
        """
        标准化 URL，移除尾斜杠和查询参数等。
        :param url: 原始 URL (str)
        :return: 标准化后的 URL (str)
        """
        # update：主题爬虫不再对url进行标准化
        # parsed = urlparse(url)
        # # 移除查询参数和 fragment（锚点）
        # normalized_url = parsed.scheme + "://" + parsed.netloc + parsed.path
        # # 去掉尾部的斜杠
        # if normalized_url.endswith('/'):
        #     normalized_url = normalized_url[:-1]
        normalized_url=url
        return normalized_url

    def add_page_to_graph(self, base_url, html_content, keyword_list, base, crawl_page_total_cnt,contains_kw_url_cnt ):
        """
         处理网页中的相关链接
        :param crawl_page_total_cnt 爬到的网页总数
        :param 包含关键词的网页数字典
        """
        normalized_base_url = self.normalize_url(base_url)
        links = self.extract_links(html_content, base_url)

        # 分析锚文本主题相关度
        anchor_scores = self.url_anchor_analyze_relevance_scores(
            base_url, html_content, keyword_list, base, self.topic_matrix, crawl_page_total_cnt, contains_kw_url_cnt
        )


        for link in links:
            normalized_link = self.normalize_url(link)

            # 更新出链和入链关系
            self.out_links[normalized_base_url].add(normalized_link)
            self.in_links[normalized_link].add(normalized_base_url)

            # 保存锚文本相关度
            self.anchor_scores[normalized_base_url][normalized_link] = anchor_scores.get(link, 0)


    def url_anchor_analyze(self,base_url,html_text,keyword_list,base,crawl_page_total_cnt, contains_kw_url_cnt):
        """
        当爬取到一个html网页时，会对网页中所有的url(即外链)进行分析，根据锚文本进行评分，排序
        :param html_text:
        :param keyword_list:关键词列表
        :param base: 计算权重时的底数
        :param crawl_page_total_cnt 爬取到的网页数量
        :param 包含关键词的网页情况
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
        exist_dict=defaultdict(list) # 存在某个关键词的锚文本url {keyword:[urls]}
        results={key:defaultdict(int) for key,_ in anchor_text_and_urls} # 用于记录每个锚文本对应的关键词词频{url:{word:cnt}}
        max_freq=defaultdict(int) # 统计每个锚文本对应的最大主题关键词的词频 {url:topic_word_cnt}
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
                value=((results[url][kw])/(max_freq[url]+1))*(math.log(((crawl_page_total_cnt+1)/(contains_kw_url_cnt[kw]+1))+math.e,base))
                matrix_res[url].append(value)
        return matrix_res

    def url_anchor_analyze_relevance_scores(self,base_url,html_text,keyword_list,base,topic_matrix,crawl_page_total_cnt, contains_kw_url_cnt):
        """
        计算该网页下各锚文本的主题相关度，主要是使用余弦相似度，参数与url_anchor_analyze一致
        :param base_url:
        :param html_text:
        :param keyword_list:
        :param base:
        :param crawl_page_total_cnt 爬到的网页总数
        :param contains_kw_url_cnt 包含关键词的网页数字典
        :param topic_matrix: 主题语义权重向量，根据本体树（知识图谱）计算获取，使用TopicRelevance.main_generate()获取
        :return:
        """
        res={}
        anchor_matrix_dict=self.url_anchor_analyze(base_url,html_text,keyword_list,base,crawl_page_total_cnt, contains_kw_url_cnt)
        for url,vector in anchor_matrix_dict.items():
            cos_sim=self._cosine_similarity(vector,topic_matrix)
            if cos_sim<0:
                print(f'检测到相似度小于零，此时锚文本向量为：{vector}')
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

