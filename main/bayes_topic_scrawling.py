# @author: 陈德枝，广东外语外贸大学
# 整合所有源代码，进行最终的基于贝叶斯分类器的主题爬虫,收集实验数据
# 算法流程:
"""
a) 根据主题，使用手动或半自动方式构建本体树(已构建完成)
b) 根据本体树，使用向量空间模型计算主题语义向量 `tp_vec` 和概念关键词列表，表示向量空间模型上的每一维度的含义。(已构建完成)
   - 设定网页相似度阈值 `p1`。【需指定】
   - 设定链接优先度阈值 `p2`。【需指定】
   - 实现模块 `Link.get_priority(url)` 用于计算链接的综合优先度。(已完成)
   - 使用改进的TF-IDF模型实现模块 ‘Web.convert_to_vec(url)’,用于映射网页向量(已完成)
c) 使用手动方式或搜索引擎生成初始种子 URL 列表 `url_queue`。【需使用自动化程序完成】
   - 初始化空的目标 URL 列表 `url_target`。【需定义】
d) 使用训练集训练贝叶斯分类器 `Bayes`。(已完成)
   - 使用 `Bayes.predict(vector)` 进行预测。【已完成，需调用】
e) 遍历 `url_queue` 中的 URL：【需完成】
   While `url_queue` 不为空：
      - 使用Web.convert_to_vec(url), 将当前 URL 对应的页面转换为向量 `v1`
      - 使用 `Bayes.predict(v1)` 预测 `v1` 的标签：
      - 如果预测标签不等于主题，则跳过当前 URL。
      - 计算 `v1` 与 `tp_vec` 的余弦相似度：
      - 如果 `Similarity < p1`，则跳过当前 URL。
      - 将当前 URL 添加到 `url_target`。
      - 使用 `get_new_urls_from_web(url)` 提取未访问的 URL 列表 `new_urls`：
         - 更新 `new_urls` 和当前 URL 的图信息，包括入链图和出链图。
         - 对于每个 `new_url`：
             - 计算其优先度 `Priority = Link.get_priority(new_url)`。
             - 如果 `Priority > p2`，将 `new_url` 添加到 `url_queue`。
      - 如果 `url_target` 的长度超过目标阈值，则退出循环。
f)	返回 `url_target` 并进行相关性能检验。【需完成】

"""
import collections
import math
import pickle
import time

import numpy as np
import requests

from src.html_feature_matrix import HtmlFeatureMatrix
from src.link_rating import LinkRating
from src.read_knowledge_graph_from_conexp import Node
from main.web_driver import WebAutomation
from src.bayes_predict import Bayes_Predict
from src.top_relevance import TopicRelevance

from collections import defaultdict
import pyhttpx


# 模型定义变量，不定义的话无法加载二进制文件model
def default_class_stats():
    return {
        "docs_count": 0,
        "weighted_word_freq": defaultdict(float),
        "total_weighted_words": 0.0,
        "doc_contain_word": defaultdict(int)
    }


class BayesTopicScrawling:
    def __init__(self):
        """
            加载相关模块，包括：
            主题语义向量，关键词列表，贝叶斯分类器模型
            指定网页相关度阈值p1,链接主题相关度阈值p2
            初始化目标url队列url_target
            以及需要处理的url队列 url_queue
            定义url处理计数url_process_cnt, 用以计算爬准率
        """
        # 加载本体，获取主题语义向量与关键词列表
        with open('../src/graph_nodes.pkl', 'rb') as f:
            nodes = pickle.load(f)
            topic_relevance = TopicRelevance(nodes, nodes[1])
            topic_relevance.main_generate(nodes[1], nodes, (0.2, 0.2, 0.2, 0.2, 0.2))
            print(topic_relevance.topic_meaning_matrix)
            print(topic_relevance.keyword_list)
            self.topic_meaning_vector = topic_relevance.topic_meaning_matrix  # 主题语义向量
            self.keyword_list = topic_relevance.keyword_list  # 关键词列表
            self.bayes = Bayes_Predict()  # 贝叶斯分类器模型
            self.p1 = 0.94  # 网页相关度阈值p1
            self.p2 = 2  # 链接相关度阈值p2
            self.url_target = []  # 目标url列表
            self.url_queue = collections.deque([])  # 需要处理的url队列
            self.url_process_cnt = 0  # 处理过的url计数，用以计算爬准率
            self.driver_path = '../diver/132.0.6834.160/chromedriver.exe'  # chrome_driver路径，用以自动化
            self.web = None  # 自动化浏览器对象
            self.topic = '暴雨灾害'
            self.seed_num = 50  # 种子链接数
            self.target_num = 200  # 目标收集的url数目
            self.request_html_retry_cnt = 3  # 请求url的html的重试次数
            self.link_graph = LinkRating(self.topic_meaning_vector)
            self.base = math.e  # 计算链接优先相关度时的对数底数
            self.link_analyze_setting = [0.33, 0.33, 0.33]  # 计算链接综合优先度时三个变量的权重值
            self.link_graph.damping = 0.85  # pagerank 阻尼系数
            self.link_graph.omega = 0.5  # pagerank 调节因子


    def __del__(self):
        # 对象摧毁前关闭浏览器
        if self.web is not None:
            self.web.close()

    def get_url_seeds_from_baidu(self, topic, url_cnt):
        """
        使用自动化程序从百度搜索引擎获取主题相关的种子url,标准化，并持久化为pkl文件
        :param topic: 主题名称
        :param url_cnt: 需获取的种子url数目
        :return: 种子url列表
        """
        res = set()
        baidu_homepage_url = 'https://www.baidu.com'
        homepage_input_selector = '//input[contains(@class,"s_ipt")]'
        homepage_search_btn_selector = '//input[@value="百度一下"]'
        search_result_item_a_selector = '//div[contains(@class,"result") and contains(@class,"c-container")]//h3//a[1]'
        next_search_page_btn_selector = '//a[contains(text(),"下一页")]'
        self.web = WebAutomation(browser='chrome', driver_path=self.driver_path)
        self.web.open_url(baidu_homepage_url)
        time.sleep(1)
        self.web.maximize_window()
        time.sleep(1)
        self.web.type(homepage_input_selector, topic)
        time.sleep(1)
        self.web.click(homepage_search_btn_selector)
        time.sleep(1)
        while len(res) < url_cnt:
            search_result_item_num = self.web.get_element_num(search_result_item_a_selector)
            # 获取当前页面的句柄
            search_result_page_handle = self.web.get_current_window_handle()
            self.web.web_refresh()
            time.sleep(1)
            for i in range(1, search_result_item_num + 1):
                time.sleep(1)
                try:
                    self.web.web_click_js(f'({search_result_item_a_selector})[{i}]')
                except Exception as e:
                    continue
                # 移动到新打开的搜索页获取url
                self.web.switch_to_window(self.web.get_window_handles()[-1])
                try:
                    cur_url = self.web.get_web_content('current_url')
                except Exception as e:
                    self.web.close_window()
                    continue
                # Todo url是否需要标准化, 这里先统一标准化了？
                res.add(self.link_graph.normalize_url(cur_url))
                if len(res) >= url_cnt:
                    break
                self.web.close_window()
                time.sleep(1)
                self.web.switch_to_window(search_result_page_handle)
            else:
                self.web.click(next_search_page_btn_selector)
                time.sleep(3)
        self.web.close()
        self.web = None
        with open('./seed_urls.pkl','wb') as f:
            pickle.dump(list(res),f)
        return list(res)

    def load_seed_urls_from_pickle(self):
        """
        加载种子url文件
        :return:
        """
        with open('./seed_urls.pkl','rb')as f:
            seed_urls=pickle.load(f)
            return seed_urls

    def get_html_text_from_url(self, url, retry):
        """
        从url获取html文本
        :param url: 目标url
        :param retry: 重试次数
        :return:
        """

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36",
        }
        session = pyhttpx.HttpSession()

        cnt = retry
        while cnt > 0:
            # 发送 GET 请求
            response = session.get(url='https://www.baidu.com/', headers=headers)
            # 检查请求是否成功
            if response.status_code == 200:
                # 获取 HTML 文本
                html_text = response.text
                return html_text
            else:
                print(f"请求失败，状态码: {response.status_code}")
                cnt -= 1
        else:
            raise RuntimeError(f'请求{url}失败')

    def _cosine_similarity(self, vec1, vec2):
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

    def run(self):
        """
        主题爬虫的主逻辑
        :param seed_num: 种子链接的数量
        :param target_num 目标需要获取的url数量
        :return:
        """
        # 自动化从百度引擎获取种子url
        # 记录一下主题相关度与综合优先度
        web_sim_record=[]
        link_sim_record=[]
        total_web_sim=0
        total_web_cnt=0 # 通过筛选的网页数目
        total_link_sim=0
        total_link_cnt=0 # 通过筛选的链接数目
        seed_urls = self.load_seed_urls_from_pickle() # 加载种子url
        self.url_queue.extend(seed_urls)  # 种子链接加入到待处理队列
        while self.url_queue:
            cur_url = self.url_queue.popleft()
            cur_html_text = self.get_html_text_from_url(cur_url, self.request_html_retry_cnt)
            self.url_process_cnt += 1  # 下载一次网页则计数加1
            cur_label = self.bayes.classify_with_weight(cur_html_text)
            if cur_label != self.topic:
                continue
            # 获取网页的加权向量表示
            cur_web_vector = HtmlFeatureMatrix().main_generate(cur_html_text, self.keyword_list)
            # 获取网页的向量与主题语义向量的余弦相似度（网页相似度)
            cur_web_similarity = self._cosine_similarity(cur_web_vector, self.topic_meaning_vector)
            web_sim_record.append(cur_web_similarity) #记录一下网页相似度方便调整
            # 网页文本主题相关度则跳过
            if cur_web_similarity < self.p1:
                continue
            total_web_sim+=cur_web_similarity
            total_web_cnt+=1
            # 进行网页处理，提取网页的链接，根据链接优先度进行分析与筛选
            self.link_graph.add_page_to_graph(cur_url, cur_html_text, self.keyword_list, self.base)
            # 获取网页链接出去的链接
            all_outer_urls = self.link_graph.extract_links(cur_html_text, cur_url)
            # 综合链接锚文本优先度、当前网页的主题相关度、pr值综合计算链接的优先级相关度
            for outer_url in all_outer_urls:
                a, b, c = self.link_analyze_setting
                cur_outer_url_prior = a * self.link_graph.anchor_scores[self.link_graph.normalize_url(cur_url)][
                    self.link_graph.normalize_url(
                        outer_url)] + b * cur_web_similarity + c * self.link_graph.pagerank[outer_url]
                # 如果链接综合优先度小于p2则跳过
                link_sim_record.append(cur_outer_url_prior) # 记录一下综合优先度方便进行调整
                if cur_outer_url_prior < self.p2:
                    continue
                total_link_sim+=cur_outer_url_prior
                total_link_cnt+=1
                self.url_process_cnt += 1  # 进入到需要下载处理的计数
                self.url_queue.append(outer_url)
            self.url_target.append(cur_url)  # 将处理完的url加入到目标列表
            # 检查目标列表是否达到需要收集的条目数
            if len(self.url_target) >= self.target_num:
                break
        print(f'共收集到{len(self.url_target)}条url，共下载网页{self.url_process_cnt}个，爬准率为:{self.target_num / self.url_process_cnt}')
        # print(f'网页相似度记录：{web_sim_record}')
        print(f'通过网页筛选共{total_web_cnt}条，平均网页相似度为{total_web_sim/(total_web_cnt+1)}')
        # print(f'链接优先度记录:{link_sim_record}')
        print(f'通过链接筛选共{total_link_cnt}条，平均综合优先度为{total_link_sim/(total_link_cnt+1)}')
        return self.url_target


if __name__ == '__main__':
    BayesTopicScrawling().run()