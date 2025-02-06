# 获取网页特征向量
import re
from bs4 import BeautifulSoup
import jieba
from collections import Counter, defaultdict

class HtmlFeatureMatrix:
    def __init__(self,feature_setting=(2.0,1.8,1.8,1.0,1.5,2.5)):
        self.feature_setting=feature_setting



    def main_generate(self,html_text,keyword_list):
        """
        读入html文本，返回网页文本特征向量
        :param html_text:
        :return:
        """
        webpage_matrix=[]
        results,max_freq=self.analyze_html(html_text,keyword_list)
        for kw in keyword_list:
            max_kw_freq=max_freq[kw]
            title_val=((results['title'][kw]+1)/(max_kw_freq+1))*self.feature_setting[0]
            keywords_val=((results['keywords'][kw]+1)/(max_kw_freq+1))*self.feature_setting[1]
            description_val=((results['description'][kw]+1)/(max_kw_freq+1))*self.feature_setting[2]
            body_val=((results['body'][kw]+1)/(max_kw_freq+1))*self.feature_setting[3]
            hl_val=((results['hl'][kw]+1)/(max_kw_freq+1))*self.feature_setting[4]
            anchor_val=((results['anchor'][kw]+1)/(max_kw_freq+1))*self.feature_setting[5]
            kw_val=title_val+keywords_val+description_val+body_val+hl_val+anchor_val
            webpage_matrix.append(kw_val)
        return webpage_matrix

    def analyze_html(self,html_text, keywords):
        """
        分析HTML文本中关键词在不同类型文本中的词频，并实时更新最大词频。

        :param html_text: HTML网页内容 (str)
        :param keywords: 关键词列表 (list of str)
        :return: 词频统计结果 (dict) 和 最大值统计 (dict)
        """
        # 初始化 BeautifulSoup
        soup = BeautifulSoup(html_text, 'html.parser')

        # 解析 HTML 的各类文本内容
        title = soup.title.string if soup.title else ""
        meta_keywords = soup.find("meta", {"name": re.compile(r"^keywords$", re.IGNORECASE)})
        meta_description = soup.find("meta", {"name": re.compile(r"^description$", re.IGNORECASE)})
        keywords_text = meta_keywords["content"] if meta_keywords and "content" in meta_keywords.attrs else ""
        description_text = meta_description["content"] if meta_description and "content" in meta_description.attrs else ""
        body_text = soup.body.get_text() if soup.body else ""
        hl_text = " ".join([tag.get_text() for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])])
        anchor_text = " ".join([a.get_text() for a in soup.find_all('a')])

        # 将内容分类为六类
        categories = {
            "title": title,
            "keywords": keywords_text,
            "description": description_text,
            "body": body_text,
            "hl": hl_text,
            "anchor": anchor_text,
        }

        # 初始化统计结果和最大值字典
        results = {key: defaultdict(int) for key in categories}
        max_frequencies = {key: 0 for key in keywords}

        # 分词并统计关键词词频
        for category, text in categories.items():
            words = jieba.lcut(text)  # 使用 jieba 分词
            word_counts = Counter(words)  # 统计词频

            for keyword in keywords:
                results[category][keyword] = word_counts[keyword]  # 获取当前类别的关键词词频

                # 更新最大值
                if word_counts[keyword] > max_frequencies[keyword]:
                    max_frequencies[keyword] = word_counts[keyword]

        return results, max_frequencies


