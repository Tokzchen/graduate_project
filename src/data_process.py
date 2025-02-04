# 进行数据合并处理
import csv
import os
import pickle
from src.test import Node
import numpy as np

from src.html_feature_matrix import HtmlFeatureMatrix
from src.text2html import extract_keywords, text_to_html
from src.top_relevance import TopicRelevance


class DataProcess:
    """
        进行数据处理合并
    """

    def __init__(self):
        pass

    import os
    import csv

    def text2html(self,text):
        keywords = extract_keywords(text, top_n=5)
        # 转换为HTML
        html_output = text_to_html(text, keywords)
        html_output=html_output.strip('"')
        return html_output

    # 读取并处理 cnews.val.txt
    def process_cnews(self,file_path, output_csv):
        categories = ['体育', '家居', '房产', '科技']
        category_count = {category: 0 for category in categories}

        with open(file_path, 'r', encoding='utf-8') as file, open(output_csv, 'w', newline='',
                                                                  encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['类别', 'HTML 文本'])  # 写入 CSV 文件的表头

            for line in file:
                category, content = line.strip().split('\t', 1)
                if category in categories and category_count[category] < 1200:
                    html_content = self.text2html(content)
                    writer.writerow([category, html_content])
                    category_count[category] += 1

                    # 如果所有类别的数据都已经达到1200条，提前退出
                    if all(count >= 1200 for count in category_count.values()):
                        break

    # 读取并处理暴雨灾害相关的网页文本
    def process_storm_files(self,folder_path, output_csv):
        with open(output_csv, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            for filename in os.listdir(folder_path):
                if filename.endswith('.txt'):
                    file_path = os.path.join(folder_path, filename)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        html_content = file.read()
                        writer.writerow(['暴雨灾害', html_content])

    # 主函数
    def main(self):
        cnews_file = '../data/cnews.val.txt'  # cnews.val.txt 文件路径
        storm_folder = '../data/renmin_news'  # 暴雨灾害文本文件夹路径
        output_csv = '../data/data.csv'  # 输出文件路径

        # 处理 cnews.val.txt
        self.process_cnews(cnews_file, output_csv)

        # 处理暴雨灾害文本
        self.process_storm_files(storm_folder, output_csv)

    def main_matrix(self):
        """
        构造矩阵数据集
        :return:
        """
        with open('graph_nodes.pkl', 'rb') as f:
            nodes = pickle.load(f)
            topic_relevance = TopicRelevance(nodes, nodes[1])
            topic_relevance.main_generate(nodes[1], nodes, (0.2, 0.2, 0.2, 0.2, 0.2))
            print(topic_relevance.topic_meaning_matrix)
            topic_meaning_vector=topic_relevance.topic_meaning_matrix # 主题语义向量
            keyword_list=topic_relevance.keyword_list # 主题关键词列表
            print(topic_relevance.keyword_list)
            self.process_data_vector('../data/data.csv','../data/data_vector.csv',topic_relevance.keyword_list)

    def process_data_vector(self,input_csv, output_csv, keywords_list):
        # 初始化特征提取工具,使用tf-idf生成对应的向量
        html_feature = HtmlFeatureMatrix()

        with open(input_csv, 'r', encoding='utf-8') as infile, \
                open(output_csv, 'w', newline='', encoding='utf-8') as outfile:

            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            # 跳过表头
            next(reader)
            # 写入新文件的表头：类别 + 特征维度（例如：feature_1, feature_2, ...）
            header = ["类别"] + [f"feature_{i + 1}" for i in range(len(keywords_list))]
            writer.writerow(header)

            for row in reader:
                category, html_content = row
                # 生成特征向量
                feature_vector = html_feature.main_generate(html_content, keywords_list)
                # 转换为列表（如果返回的是 numpy 数组）
                if isinstance(feature_vector, np.ndarray):
                    feature_vector = feature_vector.tolist()
                # 写入类别和特征向量
                writer.writerow([category] + feature_vector)

if __name__ == '__main__':
        DataProcess().main_matrix()